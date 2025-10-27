import requests
import json
import time
import hashlib
import hmac
import websocket
import threading
import ssl
from config import BYBIT_CONFIG
from typing import Optional, Callable, Dict, List

class BybitClient:
    def __init__(self):
        self.api_key = BYBIT_CONFIG["API_KEY"]
        self.api_secret = BYBIT_CONFIG["API_SECRET"]
        self.base_url = BYBIT_CONFIG["BASE_URL"]
        self.ws_public_url = BYBIT_CONFIG.get("WS_PUBLIC_URL", "wss://stream.bybit.com/v5/public/linear")
        self.ws_private_url = BYBIT_CONFIG.get("WS_PRIVATE_URL", "wss://stream.bybit.com/v5/private")
        self.error_handler = None
        
        # WebSocket Attributes
        self.ws_public = None
        self.ws_private = None
        self.ws_public_thread = None
        self.ws_private_thread = None
        self.ws_public_connected = False
        self.ws_private_connected = False
        self.ws_public_subscriptions = set()
        self.ws_private_subscriptions = set()
        self.ws_callback: Optional[Callable[[Dict], None]] = None
        self._ws_stop_event = threading.Event()
        self._ws_last_ping_time = {'public': 0, 'private': 0}
        self._ws_ping_interval = 20 # Bybit requires ping every 20s
        self._ws_reconnect_delay = 5

    def set_error_handler(self, error_handler):
        """Set error handler for API and WebSocket error management"""
        self.error_handler = error_handler
        
    def set_ws_callback(self, callback: Callable[[Dict], None]):
        """Set the callback function for processing WebSocket messages"""
        self.ws_callback = callback

    # --- REST API Methods (Original) ---
    def _generate_signature(self, params, timestamp):
        """Generate HMAC signature for Bybit REST API"""
        if isinstance(params, dict):
            if params:
                param_str = json.dumps(params, separators=(',', ':')) # Ensure compact JSON for POST
            else:
                param_str = ""
        else:
            param_str = params # GET request query string
            
        # Correct recv_window usage in signature
        recv_window = "5000"
        signature_payload = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            signature_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _validate_api_response(self, response) -> bool:
        """Validate REST API response structure and data"""
        if not response:
            return False
        if 'retCode' not in response:
            return False
        # Treat common success codes as valid (0: OK, 10001: Params error usually OK for gets, 10002: Kline not ready yet)
        if response.get('retCode') not in [0, 10001, 10002]:
            return False
        return True
    
    def _request(self, method, endpoint, params=None, retry_count=0):
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if params is None:
            params = {}
            
        # Prepare query string for GET/DELETE, body string for POST
        if method.upper() in ["GET", "DELETE"]:
            # Sort params alphabetically for consistent signature
            sorted_params = sorted(params.items())
            query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
            signature = self._generate_signature(query_string, timestamp)
        else: # POST
            signature = self._generate_signature(params, timestamp)
            query_string = "" # Query string not used in POST signature payload

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == "POST":
                # For POST, pass params as json body, not query params
                response = requests.post(url, headers=headers, json=params, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, params=params, timeout=10)
            else:
                raise ValueError("Unsupported HTTP method")
                
            response_data = response.json()
            
            if not self._validate_api_response(response_data):
                error_msg = response_data.get('retMsg', 'Invalid API response format')
                ret_code = response_data.get('retCode', -1)
                
                # Handle specific error codes if needed
                if ret_code == 10001 and method.upper() == "GET": # Often ignorable for GET requests asking for data that doesn't exist yet
                     pass # Don't treat as critical error immediately
                elif self.error_handler:
                    error_result = self.error_handler.handle_api_error(
                        Exception(f"API Error ({ret_code}): {error_msg}"), endpoint, retry_count
                    )
                    if error_result.get('should_retry', False):
                        time.sleep(error_result.get('retry_after', 2))
                        return self._request(method, endpoint, params, retry_count + 1)
                
                # Return None for invalid responses after handling/retry logic
                return None
                
            return response_data
            
        except requests.exceptions.Timeout as e:
            if self.error_handler:
                error_result = self.error_handler.handle_api_error(e, endpoint, retry_count)
                if error_result.get('should_retry', False):
                    time.sleep(error_result.get('retry_after', 2))
                    return self._request(method, endpoint, params, retry_count + 1)
            return None
            
        except requests.exceptions.ConnectionError as e:
            if self.error_handler:
                error_result = self.error_handler.handle_api_error(e, endpoint, retry_count)
                if error_result.get('should_retry', False):
                    time.sleep(error_result.get('retry_after', 2))
                    return self._request(method, endpoint, params, retry_count + 1)
            return None
            
        except Exception as e:
            if self.error_handler:
                # Let error handler decide if retry is needed for generic exceptions
                 error_result = self.error_handler.handle_api_error(e, endpoint, retry_count)
                 if error_result.get('should_retry', False):
                    time.sleep(error_result.get('retry_after', 2))
                    return self._request(method, endpoint, params, retry_count + 1)
            return None

    # --- Original Account/Market/Order Methods ---
    def get_wallet_balance(self):
        return self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    
    def set_leverage(self, symbol, leverage):
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
        return self._request("POST", "/v5/position/set-leverage", params)
    
    def get_kline(self, symbol, interval, limit=200):
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._request("GET", "/v5/market/kline", params)
    
    def place_order(self, symbol, side, order_type, qty, price=None, stop_loss=None, take_profit=None, time_in_force=None):
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty)
        }
        
        if price and order_type == "Limit":
            params["price"] = str(price)
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        if time_in_force:
            params["timeInForce"] = time_in_force
        else:
             params["timeInForce"] = "GTC" # Default to GTC if not specified
            
        return self._request("POST", "/v5/order/create", params)
    
    def get_open_orders(self, symbol=None):
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/v5/order/realtime", params)
    
    def cancel_order(self, symbol, order_id):
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id
        }
        return self._request("POST", "/v5/order/cancel", params)
    
    def get_position_info(self, symbol: Optional[str] = None):
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
            else:
                params["settleCoin"] = "USDT"
            response = self._request("GET", "/v5/position/list", params)
            return response
        except Exception as e:
            if self.error_handler:
                # Pass context to error handler
                self.error_handler.handle_api_error(e, f"get_position_info ({symbol or 'all'})")
            return None

    # --- NEW WebSocket Methods ---

    def start_websockets(self):
        """Starts public and private WebSocket connections in separate threads."""
        if not self.ws_callback:
            print("Error: WebSocket callback not set. Use set_ws_callback().")
            return

        self._ws_stop_event.clear()

        # Start Public WebSocket
        if not self.ws_public_thread or not self.ws_public_thread.is_alive():
            self.ws_public_thread = threading.Thread(target=self._ws_run_loop, args=(self.ws_public_url, "public"), daemon=True)
            self.ws_public_thread.start()
            print("Public WebSocket thread started.")

        # Start Private WebSocket
        if not self.ws_private_thread or not self.ws_private_thread.is_alive():
            self.ws_private_thread = threading.Thread(target=self._ws_run_loop, args=(self.ws_private_url, "private"), daemon=True)
            self.ws_private_thread.start()
            print("Private WebSocket thread started.")

    def stop_websockets(self):
        """Signals WebSocket threads to stop and closes connections."""
        print("Stopping WebSocket connections...")
        self._ws_stop_event.set() # Signal threads to stop

        if self.ws_public:
            try:
                self.ws_public.close()
            except Exception as e:
                print(f"Error closing public WS: {e}")
        if self.ws_private:
            try:
                self.ws_private.close()
            except Exception as e:
                print(f"Error closing private WS: {e}")

        # Wait for threads to finish
        if self.ws_public_thread and self.ws_public_thread.is_alive():
            self.ws_public_thread.join(timeout=5)
        if self.ws_private_thread and self.ws_private_thread.is_alive():
            self.ws_private_thread.join(timeout=5)
            
        self.ws_public_connected = False
        self.ws_private_connected = False
        print("WebSocket connections stopped.")

    def _ws_run_loop(self, url: str, stream_type: str):
        """Main loop for handling WebSocket connection, messages, and reconnection."""
        while not self._ws_stop_event.is_set():
            try:
                print(f"Connecting to {stream_type} WebSocket at {url}...")
                ws = websocket.WebSocketApp(url,
                                          on_open=lambda ws_app: self._ws_on_open(ws_app, stream_type),
                                          on_message=lambda ws_app, msg: self._ws_on_message(ws_app, msg, stream_type),
                                          on_error=lambda ws_app, err: self._ws_on_error(ws_app, err, stream_type),
                                          on_close=lambda ws_app, status, msg: self._ws_on_close(ws_app, status, msg, stream_type))
                
                if stream_type == "public":
                    self.ws_public = ws
                else:
                    self.ws_private = ws

                # Run forever handles automatic pinging internally if keepalive is set
                # However, Bybit requires explicit pings, so we manage it manually.
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=0, ping_timeout=None) # Disable automatic pings

            except Exception as e:
                print(f"WebSocket run_forever error ({stream_type}): {e}")
                if self.error_handler:
                    self.error_handler.handle_api_error(e, f"ws_{stream_type}_run_loop")

            # If loop exits (due to error or close), wait before reconnecting unless stopping
            if not self._ws_stop_event.is_set():
                print(f"WebSocket ({stream_type}) disconnected. Reconnecting in {self._ws_reconnect_delay} seconds...")
                if stream_type == "public":
                    self.ws_public_connected = False
                else:
                    self.ws_private_connected = False
                time.sleep(self._ws_reconnect_delay)

        print(f"WebSocket loop stopped for {stream_type}.")

    def _ws_on_open(self, ws_app, stream_type: str):
        """Callback executed when WebSocket connection is opened."""
        print(f"{stream_type.capitalize()} WebSocket connection opened.")
        if stream_type == "public":
            self.ws_public_connected = True
            self._ws_last_ping_time['public'] = time.time()
            # Resubscribe to topics if needed
            if self.ws_public_subscriptions:
                self.ws_subscribe("public", list(self.ws_public_subscriptions))
        else: # private
             self.ws_private_connected = True
             self._ws_last_ping_time['private'] = time.time()
             self._ws_authenticate()
             # Resubscribe after authentication success (handled in on_message)


    def _ws_on_message(self, ws_app, message: str, stream_type: str):
        """Callback executed when a WebSocket message is received."""
        now = time.time()
        # Handle periodic ping
        if now - self._ws_last_ping_time[stream_type] > self._ws_ping_interval:
             self._ws_send_ping(stream_type)

        try:
            data = json.loads(message)
            
            # Handle Ping/Pong
            if "op" in data and data["op"] == "pong":
                # print(f"Received pong from {stream_type}")
                return
            if "ping" in data:
                 ws_app.send(json.dumps({"op": "pong"}))
                 # print(f"Responded to ping on {stream_type}")
                 return

            # Handle Authentication Response (Private Stream)
            if stream_type == "private" and data.get("op") == "auth":
                if data.get("success"):
                    print("Private WebSocket authenticated successfully.")
                    # Subscribe to private topics after successful auth
                    if self.ws_private_subscriptions:
                        self.ws_subscribe("private", list(self.ws_private_subscriptions))
                else:
                    print(f"Private WebSocket authentication failed: {data.get('ret_msg')}")
                    if self.error_handler:
                         self.error_handler.handle_api_error(Exception(f"WS Auth Failed: {data.get('ret_msg')}"), "ws_private_auth")
                    # Consider closing connection or retrying auth
                return

            # Handle Subscription Response
            if data.get("op") == "subscribe":
                if data.get("success"):
                    subs = data.get("args", [])
                    print(f"Successfully subscribed to {subs} on {stream_type} stream.")
                else:
                    print(f"Subscription failed on {stream_type}: {data.get('ret_msg')}")
                    if self.error_handler:
                         self.error_handler.handle_api_error(Exception(f"WS Sub Failed: {data.get('ret_msg')}"), f"ws_{stream_type}_subscribe")
                return
                
            if data.get("op") == "unsubscribe":
                 if data.get("success"):
                     subs = data.get("args", [])
                     print(f"Successfully unsubscribed from {subs} on {stream_type} stream.")
                 else:
                     print(f"Unsubscription failed on {stream_type}: {data.get('ret_msg')}")
                 return

            # Process Data Messages
            if "topic" in data and self.ws_callback:
                self.ws_callback(data) # Pass the full message to the handler

        except json.JSONDecodeError:
            print(f"Received non-JSON message on {stream_type}: {message}")
        except Exception as e:
            print(f"Error processing WebSocket message ({stream_type}): {e} | Message: {message[:200]}") # Log first 200 chars
            if self.error_handler:
                 self.error_handler.handle_api_error(e, f"ws_{stream_type}_on_message")

    def _ws_on_error(self, ws_app, error, stream_type: str):
        """Callback executed when a WebSocket error occurs."""
        print(f"{stream_type.capitalize()} WebSocket error: {error}")
        if self.error_handler:
            self.error_handler.handle_api_error(error, f"ws_{stream_type}_error")
        # The run_forever loop will exit, triggering reconnection logic in _ws_run_loop

    def _ws_on_close(self, ws_app, close_status_code, close_msg, stream_type: str):
        """Callback executed when WebSocket connection is closed."""
        print(f"{stream_type.capitalize()} WebSocket connection closed. Code: {close_status_code}, Msg: {close_msg}")
        if stream_type == "public":
            self.ws_public_connected = False
        else:
            self.ws_private_connected = False
        # Reconnection logic is handled in _ws_run_loop after run_forever exits

    def _ws_authenticate(self):
        """Sends authentication request to the private WebSocket."""
        if not self.ws_private_connected:
            print("Cannot authenticate: Private WebSocket not connected.")
            return

        expires = int((time.time() + 10) * 1000) # Expires in 10 seconds
        signature_payload = f"GET/realtime{expires}"
        signature = hmac.new(bytes(self.api_secret, "utf-8"), signature_payload.encode("utf-8"), hashlib.sha256).hexdigest()

        auth_payload = {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }
        try:
             self.ws_private.send(json.dumps(auth_payload))
             print("Sent authentication request to private WebSocket.")
        except Exception as e:
             print(f"Failed to send WS authentication: {e}")
             if self.error_handler:
                  self.error_handler.handle_api_error(e, "ws_private_auth_send")


    def ws_subscribe(self, stream_type: str, topics: List[str]):
        """Subscribes to specified topics on the public or private WebSocket."""
        ws = self.ws_public if stream_type == "public" else self.ws_private
        connected = self.ws_public_connected if stream_type == "public" else self.ws_private_connected
        
        if not connected or not ws:
            print(f"Cannot subscribe: {stream_type.capitalize()} WebSocket not connected.")
            # Store subscription request to attempt upon reconnection
            if stream_type == "public":
                self.ws_public_subscriptions.update(topics)
            else:
                self.ws_private_subscriptions.update(topics)
            return

        payload = {
            "op": "subscribe",
            "args": topics
        }
        try:
             ws.send(json.dumps(payload))
             if stream_type == "public":
                 self.ws_public_subscriptions.update(topics)
             else:
                 self.ws_private_subscriptions.update(topics)
        except Exception as e:
             print(f"Failed to send WS subscription ({stream_type}): {e}")
             if self.error_handler:
                 self.error_handler.handle_api_error(e, f"ws_{stream_type}_subscribe_send")

    def ws_unsubscribe(self, stream_type: str, topics: List[str]):
        """Unsubscribes from specified topics."""
        ws = self.ws_public if stream_type == "public" else self.ws_private
        connected = self.ws_public_connected if stream_type == "public" else self.ws_private_connected

        if not connected or not ws:
            print(f"Cannot unsubscribe: {stream_type.capitalize()} WebSocket not connected.")
            return

        payload = {
            "op": "unsubscribe",
            "args": topics
        }
        try:
            ws.send(json.dumps(payload))
            if stream_type == "public":
                self.ws_public_subscriptions.difference_update(topics)
            else:
                self.ws_private_subscriptions.difference_update(topics)
        except Exception as e:
            print(f"Failed to send WS unsubscription ({stream_type}): {e}")

    def _ws_send_ping(self, stream_type: str):
        """Sends a ping message to keep the connection alive."""
        ws = self.ws_public if stream_type == "public" else self.ws_private
        connected = self.ws_public_connected if stream_type == "public" else self.ws_private_connected

        if connected and ws:
            try:
                # print(f"Sending ping to {stream_type}...")
                ws.send(json.dumps({"op": "ping"}))
                self._ws_last_ping_time[stream_type] = time.time()
            except Exception as e:
                print(f"Failed to send ping to {stream_type}: {e}")
                # Trigger reconnection by marking as disconnected
                if stream_type == "public":
                    self.ws_public_connected = False
                else:
                    self.ws_private_connected = False
                ws.close() # Force close to trigger reconnection loop

    # --- Helper to get ticker data (needed for WS examples) ---
    def get_ticker(self, symbol: str):
         params = {"category": "linear", "symbol": symbol}
         return self._request("GET", "/v5/market/tickers", params)
         
    # --- Helper to get orderbook data (needed for WS examples) ---
    def get_orderbook(self, symbol: str, limit: int = 1):
        # Limit=1 gets top bid/ask, use larger for more depth
         params = {"category": "linear", "symbol": symbol, "limit": limit}
         return self._request("GET", "/v5/market/orderbook", params)