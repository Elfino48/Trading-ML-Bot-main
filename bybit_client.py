import requests
import json
import time
import hashlib
import hmac
import websocket
import threading
import ssl
from config import BYBIT_CONFIG # Assuming config.py has BYBIT_CONFIG dictionary
from typing import Optional, Callable, Dict, List
import logging # Added for logging potential errors

# Configure logger for this module if needed, or rely on root logger
logger = logging.getLogger('BybitClient')
# Basic logging config if running standalone or if root logger isn't set up elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BybitClient:
    def __init__(self):
        # --- Use .get for safer access to config ---
        self.api_key = BYBIT_CONFIG.get("API_KEY")
        self.api_secret = BYBIT_CONFIG.get("API_SECRET")
        self.base_url = BYBIT_CONFIG.get("BASE_URL")
        self.ws_public_url = BYBIT_CONFIG.get("WS_PUBLIC_URL", "wss://stream.bybit.com/v5/public/linear")
        self.ws_private_url = BYBIT_CONFIG.get("WS_PRIVATE_URL", "wss://stream.bybit.com/v5/private")
        self.error_handler = None
        # --- Store recv_window from config or default ---
        self.recv_window = str(BYBIT_CONFIG.get("RECV_WINDOW", "5000"))

        # WebSocket Attributes with enhanced stability features
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
        self._ws_ping_interval = 15  # Changed to 15 seconds for safer buffer (was 20)
        
        # Enhanced reconnection settings
        self._ws_reconnect_attempts = {'public': 0, 'private': 0}
        self._max_reconnect_delay = 60  # Maximum delay in seconds
        self._base_reconnect_delay = 5  # Base delay in seconds
        
        # Connection health monitoring (optional - monitor if this causes issues)
        self._ws_last_message_time = {'public': 0, 'private': 0}
        self._ws_max_silence = 45  # Increased to 45 seconds to be less aggressive (was 30)

        # --- Input Validation ---
        if not self.api_key or not self.api_secret:
             logger.warning("API Key or Secret is missing in config.")
        if not self.base_url:
             logger.error("BASE_URL is missing in config.")
             raise ValueError("Bybit Base URL not configured")

    def set_error_handler(self, error_handler):
        """Set error handler for API and WebSocket error management"""
        self.error_handler = error_handler

    def set_ws_callback(self, callback: Callable[[Dict], None]):
        """Set the callback function for processing WebSocket messages"""
        self.ws_callback = callback

    # --- REST API Methods ---

    # --- CORRECTED _generate_signature (Matches Original Logic Confirmed by Java) ---
    def _generate_signature(self, params_or_body_str: str, timestamp: str) -> str:
        """
        Generate HMAC signature for Bybit REST API V5.
        Includes EITHER the query string (GET/DELETE) OR the body string (POST)
        in the signature payload, matching the confirmed Java logic.
        """
        if not self.api_secret:
             logger.error("Cannot generate signature: API Secret is missing.")
             raise ValueError("API Secret not configured")

        # The payload string to sign includes timestamp, api_key, recv_window,
        # and the params_or_body_str (which is the query string OR the JSON body string).
        signature_payload = f"{timestamp}{self.api_key}{self.recv_window}{params_or_body_str}"

        # Calculate the signature
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            signature_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    # --- END CORRECTION ---

    # --- Kept original _validate_api_response ---
    def _validate_api_response(self, response) -> bool:
        """Validate REST API response structure and data"""
        if not response:
            return False
        if 'retCode' not in response:
            return False
        # Treat common success codes as valid (0: OK, 10001: Params error usually OK for gets, 10002: Kline not ready yet)
        # V5 API generally uses 0 for success. Be stricter if needed.
        # if response.get('retCode') != 0:
        if response.get('retCode') not in [0, 10001, 10002]: # Kept original logic
            logger.warning(f"API call returned potentially problematic retCode: {response.get('retCode')} - {response.get('retMsg')}")
            # return False # Uncomment for stricter checking
        return True

    # --- CORRECTED _request Method ---
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, retry_count: int = 0):
        """Makes an authenticated HTTP request to the Bybit API (V5 corrected)."""
        timestamp = str(int(time.time() * 1000))
        http_method = method.upper()

        if params is None:
            params = {}

        params_or_body_for_sig = ""
        query_string_for_req = ""
        request_body_for_req = "" # Use this for POST body

        try:
            # Prepare signature string and request components based on method
            if http_method in ["GET", "DELETE"]:
                # Sort params alphabetically for consistent signature
                sorted_params = sorted(params.items())
                # Create query string: key=value pairs joined by '&'
                query_string_for_req = "&".join([f"{k}={v}" for k, v in sorted_params if v is not None])
                params_or_body_for_sig = query_string_for_req # Signature uses the query string

            elif http_method in ["POST", "PUT"]:
                # Use compact JSON encoding for the request body AND the signature
                if params:
                    # separators=(',', ':') removes whitespace, matching Java's likely output
                    params_or_body_for_sig = json.dumps(params, separators=(',', ':'))
                else:
                    params_or_body_for_sig = "" # Empty body string if no params
                request_body_for_req = params_or_body_for_sig # Body to send is the same string used for sig

            else:
                logger.error(f"Unsupported HTTP method: {http_method}")
                raise ValueError("Unsupported HTTP method")

            # Generate signature using the appropriate string (query or body)
            signature = self._generate_signature(params_or_body_for_sig, timestamp)

            # Prepare Headers
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": self.recv_window,
                # Content-Type added only if there's a body
            }
            if http_method in ["POST", "PUT"]:
                 headers["Content-Type"] = "application/json"

            # Prepare URL
            url = f"{self.base_url}{endpoint}"
            
            # --- FIX: Append the *exact* sorted query string to the URL ---
            # 'query_string_for_req' was already built from sorted params
            full_url_for_req = f"{url}?{query_string_for_req}" if query_string_for_req else url
            # --- END FIX ---

            response = None
            response_data = None
            start_time = time.time()

            logger.debug(f"Making API request: {http_method} {url}")
            # Log headers safely (excluding API key if desired, although sign is more sensitive)
            log_headers = {k: (v if k != 'X-BAPI-API-KEY' else '***') for k, v in headers.items()}
            logger.debug(f"Headers: {log_headers}")
            if request_body_for_req: logger.debug(f"Body: {request_body_for_req}")
            elif query_string_for_req: logger.debug(f"Query: {query_string_for_req}")


            # --- Send Request ---
            if http_method == "GET":
                 # Use the full URL with the sorted query string.
                 # Pass params=None to prevent 'requests' from re-encoding.
                 response = requests.get(full_url_for_req, headers=headers, params=None, timeout=10)
            elif http_method == "POST":
                 # Use `data` argument with encoded string body for exact control
                 response = requests.post(url, headers=headers, data=request_body_for_req.encode('utf-8'), timeout=10)
            elif http_method == "DELETE":
                 # DELETE in Bybit V5 often uses query params like GET
                 # Use the full URL with the sorted query string.
                 response = requests.delete(full_url_for_req, headers=headers, params=None, timeout=10)
            elif http_method == "PUT":
                 response = requests.put(url, headers=headers, data=request_body_for_req.encode('utf-8'), timeout=10)


            latency = time.time() - start_time
            logger.debug(f"API response received. Status: {response.status_code}, Latency: {latency:.3f}s")

            # Try parsing JSON, handle potential errors
            try:
                response_data = response.json()
                logger.debug(f"Response Body: {str(response_data)[:500]}") # Log start of response
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response. Status: {response.status_code}, Body: {response.text[:500]}")
                # Raise an exception or handle based on status code if needed
                if response.ok: # If status was OK but JSON failed, that's odd
                     raise ValueError("Received non-JSON response with OK status")
                else:
                     response_data = None # Proceed to error handling based on status

            # --- Error Handling (Status Code and retCode) ---
            if not response.ok: # Handles 4xx/5xx errors
                 error_msg = f"HTTP Error {response.status_code}"
                 ret_msg = response_data.get('retMsg', response.text[:200]) if response_data else response.text[:200]
                 ret_code = response_data.get('retCode', -response.status_code) if response_data else -response.status_code
                 full_error = f"{error_msg} (retCode: {ret_code}): {ret_msg}"
                 logger.error(f"API request failed: {full_error} | Endpoint: {http_method} {endpoint}")

                 # Signature error specific logging
                 if ret_code == 10004:
                     logger.error(f"SIGNATURE ERROR (10004) Details:")
                     logger.error(f"  Timestamp Sent: {timestamp}")
                     logger.error(f"  Recv Window: {self.recv_window}")
                     logger.error(f"  String Signed: {timestamp}{self.api_key}{self.recv_window}{params_or_body_for_sig}")
                     # Consider logging `params_or_body_for_sig` itself if safe

                 if self.error_handler:
                     api_exception = requests.exceptions.HTTPError(full_error, response=response)
                     # Add extra info if needed by handler
                     setattr(api_exception, 'ret_code', ret_code)
                     setattr(api_exception, 'response_data', response_data)

                     error_result = self.error_handler.handle_api_error(
                         api_exception, f"{http_method}_{endpoint}", retry_count
                     )
                     if error_result.get('should_retry', False):
                         time.sleep(error_result.get('retry_after', 2))
                         return self._request(method, endpoint, params, retry_count + 1)
                 return None # No retry or no handler

            # Check Bybit's retCode even if HTTP status is OK
            # Use original validation logic for retCode checks
            if not self._validate_api_response(response_data):
                error_msg = response_data.get('retMsg', 'Invalid API response format or non-zero retCode')
                ret_code = response_data.get('retCode', -1)
                logger.warning(f"API call successful (HTTP {response.status_code}) but failed validation: retCode={ret_code}, retMsg='{error_msg}'")

                # Handle specific non-zero retCodes if needed (like original 10001 check)
                if ret_code == 10001 and http_method == "GET":
                     logger.info("Ignoring retCode 10001 for GET request (likely no data).")
                     # Pass # Don't retry, let caller handle potentially empty/null data
                elif self.error_handler:
                    # Treat other validation failures as errors for retry logic
                    api_logic_exception = Exception(f"API Logic Error ({ret_code}): {error_msg}")
                    setattr(api_logic_exception, 'ret_code', ret_code)
                    setattr(api_logic_exception, 'response_data', response_data)
                    error_result = self.error_handler.handle_api_error(
                        api_logic_exception, f"{http_method}_{endpoint}_logic_error", retry_count
                    )
                    if error_result.get('should_retry', False):
                        time.sleep(error_result.get('retry_after', 2))
                        return self._request(method, endpoint, params, retry_count + 1)
                # If not retrying or no handler, decide what to return.
                # Returning response_data allows caller to inspect the specific error code.
                # return None # Or return None if validation failure means unusable data
                return response_data # Let caller check retCode != 0

            # Success case: HTTP OK and retCode validation passed
            return response_data

        # --- Exception Handling (Network issues, Timeouts) ---
        except requests.exceptions.Timeout as e:
             logger.error(f"API request timed out: {http_method} {endpoint} - {e}")
             if self.error_handler:
                 error_result = self.error_handler.handle_api_error(e, f"{http_method}_{endpoint}_timeout", retry_count)
                 if error_result.get('should_retry', False):
                     time.sleep(error_result.get('retry_after', 2))
                     return self._request(method, endpoint, params, retry_count + 1)
             return None

        except requests.exceptions.ConnectionError as e:
             logger.error(f"API connection error: {http_method} {endpoint} - {e}")
             if self.error_handler:
                 error_result = self.error_handler.handle_api_error(e, f"{http_method}_{endpoint}_connect_error", retry_count)
                 if error_result.get('should_retry', False):
                     time.sleep(error_result.get('retry_after', 5)) # Longer wait
                     return self._request(method, endpoint, params, retry_count + 1)
             return None

        except ValueError as e: # Catch unsupported method or config errors
            logger.critical(f"Configuration or usage error for API request: {e}")
            if self.error_handler:
                 self.error_handler.handle_api_error(e, f"{http_method}_{endpoint}_value_error", retry_count, is_fatal=True)
            return None

        except Exception as e:
             logger.error(f"Unexpected error during API request: {http_method} {endpoint} - {e}", exc_info=True)
             if self.error_handler:
                  error_result = self.error_handler.handle_api_error(e, f"{http_method}_{endpoint}_unexpected", retry_count)
                  if error_result.get('should_retry', False):
                      time.sleep(error_result.get('retry_after', 2))
                      return self._request(method, endpoint, params, retry_count + 1)
             return None
    # --- END CORRECTED _request Method ---


    # --- Specific API Call Methods (Use _request, match original signatures) ---

    def get_wallet_balance(self, account_type: str = "UNIFIED"): # Keep original signature if 'coin' isn't used elsewhere
        """Fetches wallet balance."""
        params = {"accountType": account_type}
        return self._request("GET", "/v5/account/wallet-balance", params)

    def set_leverage(self, symbol: str, leverage: str, category: str = "linear"): # Match original signature
        """Sets leverage for a symbol."""
        params = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(leverage), # Ensure string
            "sellLeverage": str(leverage) # Ensure string
        }
        return self._request("POST", "/v5/position/set-leverage", params)

    def get_kline(self, symbol: str, interval: str, limit: int = 200, category: str = "linear", start: Optional[int] = None, end: Optional[int] = None): # Match original + added category/start/end
        """Fetches kline (candlestick) data."""
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start: params["start"] = start
        if end: params["end"] = end
        return self._request("GET", "/v5/market/kline", params)

    def place_order(self, symbol: str, side: str, order_type: str, qty: str, # Match original signature
                    price: Optional[str] = None, stop_loss: Optional[str] = None,
                    take_profit: Optional[str] = None, time_in_force: Optional[str] = None, # Make TIF optional like original
                    category: str = "linear", # Add category default
                    order_link_id: Optional[str] = None, reduce_only: Optional[bool] = None,
                    position_idx: Optional[int] = None):
        """Places an order."""
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty), # Ensure qty is string
        }
        # Add optional parameters only if provided
        if price and order_type in ["Limit", "LimitMaker"]: params["price"] = str(price)
        if stop_loss: params["stopLoss"] = str(stop_loss)
        if take_profit: params["takeProfit"] = str(take_profit)
        # Use GTC default logic like original
        params["timeInForce"] = time_in_force if time_in_force else "GTC"
        if order_link_id: params["orderLinkId"] = order_link_id
        if reduce_only is not None: params["reduceOnly"] = reduce_only
        if position_idx is not None: params["positionIdx"] = position_idx

        return self._request("POST", "/v5/order/create", params)

    def get_open_orders(self, symbol: Optional[str] = None, category: str = "linear"): # Match original signature
        """Retrieves open orders, optionally filtered by symbol."""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        # V5 /v5/order/realtime defaults to open orders
        return self._request("GET", "/v5/order/realtime", params)

    def cancel_order(self, symbol: str, order_id: str, category: str = "linear"): # Match original signature
        """Cancels an existing order by orderId."""
        params = {
            "category": category,
            "symbol": symbol,
            "orderId": order_id
        }
        # Original didn't support orderLinkId, stick to that unless needed
        return self._request("POST", "/v5/order/cancel", params)

    # Add cancel_all_orders if it exists in your ExecutionEngine/EmergencyProtocols
    def cancel_all_orders(self, category: str = "linear", symbol: Optional[str] = None, base_coin: Optional[str] = None, settle_coin: Optional[str] = None):
        """Cancels all open orders, optionally filtered."""
        params = {"category": category}
        if symbol: params["symbol"] = symbol
        if base_coin: params["baseCoin"] = base_coin
        if settle_coin: params["settleCoin"] = settle_coin
        return self._request("POST", "/v5/order/cancel-all", params)

    def is_ws_really_connected(self, stream_type: str) -> bool:
        """More robust check: flag is True AND WS object exists"""
        if stream_type == "public":
            # Check flag AND if the WebSocketApp object itself is still assigned
            return self.ws_public_connected and self.ws_public is not None
        elif stream_type == "private":
            # Check flag AND if the WebSocketApp object itself is still assigned
            return self.ws_private_connected and self.ws_private is not None
        return False

    def get_position_info(self, symbol: Optional[str] = None, category: str = "linear", settleCoin: Optional[str] = None): # Added settleCoin
        """Retrieves position information."""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        # --- FIX: Include settleCoin in params if provided ---
        if settleCoin:
            params["settleCoin"] = settleCoin
        # --- END FIX ---
        # V5 /v5/position/list returns all for category if no symbol/settleCoin
        return self._request("GET", "/v5/position/list", params)

    def get_ticker(self, symbol: str, category: str = "linear"): # Match original signature + default category
         """Gets ticker information for a specific symbol."""
         params = {"category": category, "symbol": symbol}
         return self._request("GET", "/v5/market/tickers", params)

    def get_orderbook(self, symbol: str, limit: int = 1, category: str = "linear"): # Match original signature + default category
        """Gets order book depth."""
        params = {"category": category, "symbol": symbol, "limit": limit}
        return self._request("GET", "/v5/market/orderbook", params)


    # --- Enhanced WebSocket Methods with Security Fixes ---

    def start_websockets(self):
        """Starts public and private WebSocket connections in separate threads."""
        if not self.ws_callback:
            logger.error("WebSocket callback not set. Use set_ws_callback().")
            return

        self._ws_stop_event.clear()
        logger.info("Starting WebSocket threads...")

        # Start Public WebSocket
        if not self.ws_public_thread or not self.ws_public_thread.is_alive():
            self.ws_public_thread = threading.Thread(target=self._ws_run_loop, args=(self.ws_public_url, "public"), daemon=True, name="PublicWSThread")
            self.ws_public_thread.start()
            logger.info("Public WebSocket thread initiated.")

        # Start Private WebSocket only if keys exist
        if self.api_key and self.api_secret:
            if not self.ws_private_thread or not self.ws_private_thread.is_alive():
                self.ws_private_thread = threading.Thread(target=self._ws_run_loop, args=(self.ws_private_url, "private"), daemon=True, name="PrivateWSThread")
                self.ws_private_thread.start()
                logger.info("Private WebSocket thread initiated.")
        else:
            logger.warning("API Key/Secret not provided, skipping Private WebSocket.")


    def stop_websockets(self):
        """Signals WebSocket threads to stop and closes connections."""
        logger.info("Stopping WebSocket connections...")
        self._ws_stop_event.set() # Signal threads to stop

        threads_to_join = []
        # Close WS objects which should interrupt run_forever
        if self.ws_public:
            try:
                self.ws_public.close()
                if self.ws_public_thread and self.ws_public_thread.is_alive():
                     threads_to_join.append(self.ws_public_thread)
            except Exception as e:
                logger.error(f"Error closing public WS: {e}")
        if self.ws_private:
            try:
                self.ws_private.close()
                if self.ws_private_thread and self.ws_private_thread.is_alive():
                     threads_to_join.append(self.ws_private_thread)
            except Exception as e:
                logger.error(f"Error closing private WS: {e}")

        # Wait for threads to finish
        for thread in threads_to_join:
             logger.debug(f"Waiting for thread {thread.name} to join...")
             thread.join(timeout=5)
             if thread.is_alive():
                 logger.warning(f"WebSocket thread {thread.name} did not terminate gracefully.")

        self.ws_public_connected = False
        self.ws_private_connected = False
        self.ws_public = None # Clear WS objects
        self.ws_private = None
        logger.info("WebSocket connections stopped.")

    def _ws_run_loop(self, url: str, stream_type: str):
        """Main loop for handling WebSocket connection, messages, and reconnection with exponential backoff."""
        while not self._ws_stop_event.is_set():
            ws = None # Initialize ws to None at the start of each loop iteration
            try:
                # Calculate reconnect delay with exponential backoff
                current_attempt = self._ws_reconnect_attempts.get(stream_type, 0)
                if current_attempt > 0:
                    delay = min(self._base_reconnect_delay * (2 ** (current_attempt - 1)), 
                                self._max_reconnect_delay)
                    logger.info(f"Waiting {delay}s before reconnection attempt {current_attempt} for {stream_type}")
                    time.sleep(delay)
                
                logger.info(f"Connecting to {stream_type} WebSocket at {url}...")
                # Note: lambda functions capture variables from the surrounding scope
                ws = websocket.WebSocketApp(url,
                                          on_open=lambda ws_app: self._ws_on_open(ws_app, stream_type),
                                          on_message=lambda ws_app, msg: self._ws_on_message(ws_app, msg, stream_type),
                                          on_error=lambda ws_app, err: self._ws_on_error(ws_app, err, stream_type),
                                          on_close=lambda ws_app, status, msg: self._ws_on_close(ws_app, status, msg, stream_type))

                if stream_type == "public":
                    self.ws_public = ws # Store the WebSocketApp object
                else:
                    self.ws_private = ws

                # --- FIX: Enable automatic ping ---
                ws.run_forever(
                    ping_interval=18, # Changed from 0 to 18 (must be < 20)
                    ping_timeout=10,  # Add a 10-second timeout for the pong response
                    skip_utf8_validation=False
                )
                # --- END FIX ---
                
                # Reset reconnect attempts on clean exit
                self._ws_reconnect_attempts[stream_type] = 0

            # --- Exception Handling within run_loop ---
            except websocket.WebSocketException as e:
                 # More specific handling for websocket library errors
                 logger.error(f"WebSocketException in run_forever ({stream_type}): {e}")
                 self._ws_reconnect_attempts[stream_type] = self._ws_reconnect_attempts.get(stream_type, 0) + 1
            except Exception as e:
                # Catch unexpected errors during setup or run_forever itself
                logger.error(f"Unexpected error in WebSocket run_loop ({stream_type}): {e}", exc_info=True)
                self._ws_reconnect_attempts[stream_type] = self._ws_reconnect_attempts.get(stream_type, 0) + 1
                if self.error_handler: # Log via central handler if available
                    self.error_handler.handle_api_error(e, f"ws_{stream_type}_run_loop_unexpected", is_fatal=True)
                # Prevent immediate tight loop on persistent errors
                time.sleep(self._base_reconnect_delay)

            # --- Cleanup and mark as disconnected ---
            finally:
                if stream_type == "public":
                    self.ws_public_connected = False
                    self.ws_public = None # Clear object on disconnect
                else:
                    self.ws_private_connected = False
                    self.ws_private = None # Clear object on disconnect

        logger.info(f"WebSocket run loop stopped permanently for {stream_type}.")


    def _ws_on_open(self, ws_app, stream_type: str):
        """Callback executed when WebSocket connection is opened."""
        logger.info(f"{stream_type.capitalize()} WebSocket connection opened.")
        # Reset reconnect attempts on successful connection
        self._ws_reconnect_attempts[stream_type] = 0

        if stream_type == "public":
            self.ws_public_connected = True
            self._ws_last_ping_time['public'] = time.time()
            self._ws_last_message_time['public'] = time.time()
            # Resubscribe to topics stored in the set
            if self.ws_public_subscriptions:
                 logger.info(f"Resubscribing to public topics: {list(self.ws_public_subscriptions)}")
                 # Call ws_subscribe which handles sending only if connected
                 self.ws_subscribe("public", list(self.ws_public_subscriptions))
        else: # private
             self.ws_private_connected = True
             self._ws_last_ping_time['private'] = time.time()
             self._ws_last_message_time['private'] = time.time()
             # Authenticate first, subscriptions happen *after* successful auth response in _ws_on_message
             logger.info("Private WS opened, attempting authentication...")
             self._ws_authenticate()


    def _ws_on_message(self, ws_app, message: str, stream_type: str):
        """Callback executed when a WebSocket message is received."""
        # Update last message time for connection health monitoring
        current_time = time.time()
        self._ws_last_message_time[stream_type] = current_time
        
        # --- MANUAL PING LOGIC REMOVED ---
        # The websocket-client library now handles pings automatically
        # because ping_interval=18 (or similar) should be set in ws.run_forever()
        # --- END REMOVAL ---

        # logger.debug(f"WS message received ({stream_type}): {message[:200]}") # DEBUG: Log start of message

        try:
            data = json.loads(message)
            op = data.get("op") # Check 'op' field for commands

            # --- Handle Ping/Pong/Ops ---
            if op == "pong":
                # The library handles pongs automatically when ping_interval is set
                 logger.debug(f"Received pong from {stream_type}")
                 return # Successful pong received, confirms connection is alive

            elif op == "ping": # Handle server-initiated ping
                 logger.info(f"Received server ping on {stream_type}, sending pong response.")
                 try:
                     ws_app.send(json.dumps({"op": "pong"}))
                 except Exception as e:
                     logger.error(f"Failed to send pong response on {stream_type}: {e}")
                 return

            # Handle Authentication Response (Private Stream Only)
            elif stream_type == "private" and op == "auth":
                if data.get("success"):
                    logger.info("Private WebSocket authenticated successfully.")
                    # Subscribe to private topics *after* successful auth
                    if self.ws_private_subscriptions:
                        logger.info(f"Attempting queued private subscriptions: {list(self.ws_private_subscriptions)}")
                        # Call ws_subscribe to handle sending the request
                        self.ws_subscribe("private", list(self.ws_private_subscriptions))
                else:
                    auth_err_msg = data.get('ret_msg', 'Authentication failed')
                    logger.error(f"Private WebSocket authentication failed: {auth_err_msg}")
                    if self.error_handler:
                         self.error_handler.handle_api_error(Exception(f"WS Auth Failed: {auth_err_msg}"), "ws_private_auth_response", is_fatal=True)
                    # Close connection on auth failure
                    logger.warning("Closing private WebSocket due to authentication failure.")
                    self.ws_private_connected = False
                    ws_app.close()
                return # Auth message processed

            # Handle Subscription Response
            elif op == "subscribe":
                success = data.get("success", False)
                args = data.get("args", [])
                ret_msg = data.get("ret_msg", "")
                if success:
                    logger.info(f"Successfully subscribed to {args} on {stream_type}.")
                else:
                    logger.error(f"Subscription failed on {stream_type}: {ret_msg}. Args: {args}")
                    if self.error_handler:
                         self.error_handler.handle_api_error(Exception(f"WS Sub Failed: {ret_msg} on {args}"), f"ws_{stream_type}_subscribe_response")
                return # Subscription message processed

            elif op == "unsubscribe":
                 success = data.get("success", False)
                 args = data.get("args", [])
                 ret_msg = data.get("ret_msg", "")
                 if success:
                     logger.info(f"Successfully unsubscribed from {args} on {stream_type}.")
                     # Already removed from set in ws_unsubscribe call
                 else:
                     logger.warning(f"Unsubscription failed on {stream_type}: {ret_msg}. Args: {args}")
                 return # Unsubscribe message processed

            # --- Process Actual Data Messages (topic-based) ---
            elif "topic" in data:
                if self.ws_callback:
                    try:
                        # --- Pass to DataEngine's callback ---
                        self.ws_callback(data)
                        # --- ---
                    except Exception as callback_e:
                         logger.error(f"Error within ws_callback function ({stream_type}) for topic {data.get('topic')}: {callback_e}", exc_info=True)
                else:
                     logger.warning(f"Received topic message on {stream_type} but no ws_callback is set.")

            # Handle other message structures if necessary (e.g., initial snapshots)
            elif data.get("type") == "snapshot":
                 logger.info(f"Received snapshot on {stream_type}: {str(data)[:200]}...")
                 # Pass to callback if needed, maybe with type info
                 if self.ws_callback: self.ws_callback(data)

            else:
                 # Log messages that weren't ping/pong, auth, sub, or topic-based
                 logger.debug(f"Received unhandled WS message structure on {stream_type}: {str(data)[:200]}")


        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message on {stream_type}: {message[:200]}")
        except Exception as e:
            # General catch-all for errors during message processing
            logger.error(f"Unexpected error processing WebSocket message ({stream_type}): {e} | Message: {message[:200]}", exc_info=True)
            if self.error_handler:
                 self.error_handler.handle_api_error(e, f"ws_{stream_type}_on_message_general")

    def _ws_on_error(self, ws_app, error, stream_type: str):
        """Callback executed when the websocket library detects an error."""
        # Increment reconnect attempts for exponential backoff
        self._ws_reconnect_attempts[stream_type] = self._ws_reconnect_attempts.get(stream_type, 0) + 1
        
        error_str = str(error).lower()
        
        # Check if this is a benign, ignorable network error
        is_ignorable_error = "10054" in error_str or \
                             "remote host" in error_str or \
                             "connection lost" in error_str or \
                             "forcibly closed" in error_str

        if is_ignorable_error:
            logger.warning(f"WebSocket {stream_type} connection lost (Ignorable network issue): {error}")
        else:
            logger.error(f"WebSocket {stream_type} error (Non-network): {error}")

        # Mark as disconnected
        if stream_type == "public":
            self.ws_public_connected = False
        else:
            self.ws_private_connected = False

        # Notify error handler ONLY IF it's NOT an ignorable network error
        if not is_ignorable_error and self.error_handler:
            logger.error(f"Escalating non-network WS error to ErrorHandler: {error}")
            self.error_handler.handle_api_error(error, f"ws_{stream_type}_error")
        elif is_ignorable_error:
            logger.info(f"Ignoring benign WS disconnect for {stream_type}. Reconnection loop will handle.")


    def _ws_on_close(self, ws_app, close_status_code, close_msg, stream_type: str):
        """Callback executed when WebSocket connection is closed FOR ANY REASON."""
        # Check if the closure was initiated by stop_websockets()
        if self._ws_stop_event.is_set():
             logger.info(f"{stream_type.capitalize()} WebSocket connection closed intentionally by stop signal.")
        else:
             # Log unexpected closures
             logger.warning(f"{stream_type.capitalize()} WebSocket connection closed unexpectedly. Code: {close_status_code}, Msg: '{close_msg}'")

        # Ensure connection status is always updated on close
        if stream_type == "public":
            self.ws_public_connected = False
            self.ws_public = None # Clear object ref
        else:
            self.ws_private_connected = False
            self.ws_private = None # Clear object ref


    def _ws_authenticate(self):
        """Sends authentication request to the private WebSocket."""
        # Check prerequisites
        if not self.api_key or not self.api_secret:
             logger.error("Cannot authenticate WebSocket: API Key or Secret not available.")
             return
        if not self.ws_private:
             logger.warning("Cannot authenticate: Private WebSocket object does not exist.")
             return
        if not self.ws_private_connected:
            logger.warning("Cannot authenticate: Private WebSocket is not marked as connected.")
            return

        try:
            expires = int((time.time() + 10) * 1000) # Expires in 10 seconds
            signature_payload = f"GET/realtime{expires}" # V5 WS Auth format
            signature = hmac.new(bytes(self.api_secret, "utf-8"), signature_payload.encode("utf-8"), hashlib.sha256).hexdigest()

            auth_payload = {
                "op": "auth",
                "args": [self.api_key, expires, signature]
            }
            auth_json = json.dumps(auth_payload)
            logger.info("Sending authentication request to private WebSocket...")
            self.ws_private.send(auth_json)

        except websocket.WebSocketConnectionClosedException:
             logger.warning("Failed to send WS authentication: Connection closed before send.")
             self.ws_private_connected = False
        except Exception as e:
             logger.error(f"Exception during WS authentication send: {e}", exc_info=True)
             if self.error_handler:
                  self.error_handler.handle_api_error(e, "ws_private_auth_send_exception")


    def ws_subscribe(self, stream_type: str, topics: List[str]):
        """Subscribes to specified topics. Manages desired state."""
        if not topics:
             logger.warning(f"No topics provided for {stream_type} subscription attempt.")
             return

        ws = self.ws_public if stream_type == "public" else self.ws_private
        is_connected = self.ws_public_connected if stream_type == "public" else self.ws_private_connected
        current_subs_set = self.ws_public_subscriptions if stream_type == "public" else self.ws_private_subscriptions

        # Identify topics that are not already in the desired state
        newly_requested_topics = [t for t in topics if t not in current_subs_set]

        # Update the desired state immediately
        if newly_requested_topics:
             current_subs_set.update(newly_requested_topics)
             logger.info(f"Adding {newly_requested_topics} to desired {stream_type} subscriptions. Full set: {list(current_subs_set)}")
        else:
             logger.info(f"All requested {stream_type} topics {topics} are already in the desired subscription set.")
             return

        # If not connected, the topics are queued in the set. Subscription will be attempted on_open.
        if not is_connected or not ws:
            logger.warning(f"Cannot send subscribe request now: {stream_type.capitalize()} WebSocket not connected. Topics remain in desired set.")
            return

        # Special handling for private: Ensure authentication is likely complete.
        if stream_type == "private":
             logger.debug("Proceeding with private subscription attempt (assuming auth succeeded or is pending).")

        # Send subscription request ONLY for the newly requested topics
        payload = { "op": "subscribe", "args": newly_requested_topics }
        try:
             sub_json = json.dumps(payload)
             logger.info(f"Sending subscription request ({stream_type}): {sub_json}")
             ws.send(sub_json)

        except websocket.WebSocketConnectionClosedException:
             logger.warning(f"Failed to send WS subscription ({stream_type}): Connection closed. Topics remain desired.")
             if stream_type == "public": self.ws_public_connected = False
             else: self.ws_private_connected = False
        except Exception as e:
             logger.error(f"Failed to send WS subscription ({stream_type}): {e}", exc_info=True)
             if self.error_handler:
                 self.error_handler.handle_api_error(e, f"ws_{stream_type}_subscribe_send")

    def ws_unsubscribe(self, stream_type: str, topics: List[str]):
        """Unsubscribes from specified topics. Manages desired state."""
        if not topics:
             logger.warning(f"No topics provided for {stream_type} unsubscription attempt.")
             return

        ws = self.ws_public if stream_type == "public" else self.ws_private
        is_connected = self.ws_public_connected if stream_type == "public" else self.ws_private_connected
        current_subs_set = self.ws_public_subscriptions if stream_type == "public" else self.ws_private_subscriptions

        # Identify topics that are actually in the current desired state
        topics_to_remove = [t for t in topics if t in current_subs_set]

        # Update the desired state immediately
        if topics_to_remove:
            current_subs_set.difference_update(topics_to_remove)
            logger.info(f"Removing {topics_to_remove} from desired {stream_type} subscriptions. Full set: {list(current_subs_set)}")
        else:
            logger.info(f"None of the requested {stream_type} topics {topics} were in the desired subscription set.")
            return

        # If not connected, just update the desired state (already done)
        if not is_connected or not ws:
            logger.warning(f"Cannot send unsubscribe request now: {stream_type.capitalize()} WebSocket not connected. Topics removed from desired set.")
            return

        # Send unsubscribe request
        payload = { "op": "unsubscribe", "args": topics_to_remove }
        try:
            unsub_json = json.dumps(payload)
            logger.info(f"Sending unsubscription request ({stream_type}): {unsub_json}")
            ws.send(unsub_json)

        except websocket.WebSocketConnectionClosedException:
             logger.warning(f"Failed to send WS unsubscription ({stream_type}): Connection closed.")
             if stream_type == "public": self.ws_public_connected = False
             else: self.ws_private_connected = False
        except Exception as e:
            logger.error(f"Failed to send WS unsubscription ({stream_type}): {e}", exc_info=True)
            if self.error_handler:
                 self.error_handler.handle_api_error(e, f"ws_{stream_type}_unsubscribe_send")


    def _ws_send_ping(self, stream_type: str):
        """Sends a ping message with better error handling."""
        ws = self.ws_public if stream_type == "public" else self.ws_private
        is_connected = self.ws_public_connected if stream_type == "public" else self.ws_private_connected
        
        if not is_connected or not ws:
            return
            
        try:
            ping_payload = json.dumps({"op": "ping"})
            ws.send(ping_payload)
            logger.debug(f"Ping sent to {stream_type} WebSocket")
            
        except websocket.WebSocketConnectionClosedException:
            logger.warning(f"Failed to send ping to {stream_type}: Connection closed")
            if stream_type == "public": 
                self.ws_public_connected = False
            else: 
                self.ws_private_connected = False
        except Exception as e:
            logger.error(f"Error sending ping to {stream_type}: {e}")
            if stream_type == "public": 
                self.ws_public_connected = False
            else: 
                self.ws_private_connected = False

    # --- Enhanced WebSocket Status Monitoring ---

    def get_websocket_status(self):
        """Returns the current status of WebSocket connections."""
        current_time = time.time()
        status = {
            'public': {
                'connected': self.ws_public_connected,
                'last_message_ago': current_time - self._ws_last_message_time.get('public', 0),
                'reconnect_attempts': self._ws_reconnect_attempts.get('public', 0),
                'subscriptions': list(self.ws_public_subscriptions),
                'last_ping_ago': current_time - self._ws_last_ping_time.get('public', 0)
            },
            'private': {
                'connected': self.ws_private_connected,
                'last_message_ago': current_time - self._ws_last_message_time.get('private', 0),
                'reconnect_attempts': self._ws_reconnect_attempts.get('private', 0),
                'subscriptions': list(self.ws_private_subscriptions),
                'last_ping_ago': current_time - self._ws_last_ping_time.get('private', 0)
            }
        }
        return status

    def restart_websocket(self, stream_type: str):
        """Manually restart a specific WebSocket connection."""
        logger.info(f"Manually restarting {stream_type} WebSocket...")
        
        if stream_type == "public" and self.ws_public:
            self.ws_public.close()
        elif stream_type == "private" and self.ws_private:
            self.ws_private.close()
        
        # Reset reconnect attempts to allow immediate reconnection
        self._ws_reconnect_attempts[stream_type] = 0
        logger.info(f"{stream_type} WebSocket restart initiated.")

    def set_trading_stop(self, symbol: str, stop_loss: Optional[str] = None, take_profit: Optional[str] = None, category: str = "linear"):
        """
        Sets a Stop Loss and/or Take Profit on an existing open position.
        """
        params = {
            "category": category,
            "symbol": symbol,
        }
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        
        return self._request("POST", "/v5/position/trading-stop", params)

    def is_websocket_healthy(self, stream_type: str, max_silence: Optional[int] = None):
        """Check if WebSocket connection is healthy based on recent activity."""
        if max_silence is None:
            max_silence = self._ws_max_silence
            
        current_time = time.time()
        last_message_ago = current_time - self._ws_last_message_time.get(stream_type, 0)
        
        if stream_type == "public":
            connected = self.ws_public_connected
        else:
            connected = self.ws_private_connected
            
        return connected and last_message_ago <= max_silence
    
    def get_closed_pnl_history(self, category: str = "linear", start_time_ms: int = None, limit: int = 50) -> Optional[Dict]:
        """
        Fetches closed PnL records (closed positions) from Bybit.
        """
        params = {
            "category": category,
            "limit": limit
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        
        return self._request("GET", "/v5/position/closed-pnl", params)