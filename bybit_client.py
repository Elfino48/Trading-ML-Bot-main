import requests
import json
import time
import hashlib
import hmac
from config import BYBIT_CONFIG
from typing import Optional

class BybitClient:
    def __init__(self):
        self.api_key = BYBIT_CONFIG["API_KEY"]
        self.api_secret = BYBIT_CONFIG["API_SECRET"]
        self.base_url = BYBIT_CONFIG["BASE_URL"]
        self.error_handler = None
        
    def set_error_handler(self, error_handler):
        """Set error handler for API error management"""
        self.error_handler = error_handler
        
    def _generate_signature(self, params, timestamp):
        """Generate HMAC signature for Bybit API"""
        if isinstance(params, dict):
            # For POST requests, params is a dict that needs to be JSON stringified
            if params:
                param_str = json.dumps(params)
            else:
                param_str = ""
        else:
            # For GET requests, params is already a query string
            param_str = params
            
        signature_payload = f"{timestamp}{self.api_key}{5000}{param_str}"
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            signature_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _validate_api_response(self, response) -> bool:
        """Validate API response structure and data"""
        if not response:
            return False
            
        if 'retCode' not in response:
            return False
            
        # Check for common API errors
        if response.get('retCode') not in [0, 10001, 10002]:  # Add known success codes
            return False
            
        return True
    
    def _request(self, method, endpoint, params=None, retry_count=0):
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if params is None:
            params = {}
            
        # For GET requests, parameters are in query string
        if method.upper() == "GET":
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string, timestamp)
        else:
            signature = self._generate_signature(params, timestamp)
            
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
                response = requests.post(url, headers=headers, json=params, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, params=params, timeout=10)
                
            response_data = response.json()
            
            # Validate response
            if not self._validate_api_response(response_data):
                if self.error_handler:
                    error_msg = response_data.get('retMsg', 'Invalid API response format')
                    error_result = self.error_handler.handle_api_error(
                        Exception(error_msg), endpoint, retry_count
                    )
                    
                    # Retry if recommended
                    if error_result.get('should_retry', False):
                        time.sleep(error_result.get('retry_after', 2))
                        return self._request(method, endpoint, params, retry_count + 1)
                
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
                self.error_handler.handle_api_error(e, endpoint, retry_count)
            return None
    
    # Account Methods
    def get_wallet_balance(self):
        # Use UNIFIED for UTA accounts instead of CONTRACT
        return self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    
    def set_leverage(self, symbol, leverage):
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
        return self._request("POST", "/v5/position/set-leverage", params)
    
    # Market Data Methods
    def get_kline(self, symbol, interval, limit=200):
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._request("GET", "/v5/market/kline", params)
    
    # Order Methods
    def place_order(self, symbol, side, order_type, qty, price=None, stop_loss=None, take_profit=None):
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GTC"
        }
        
        if price and order_type == "Limit":
            params["price"] = str(price)
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
            
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
        """Get current position information"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
                
            response = self._request("GET", "/v5/position/list", params)
            return response
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_api_error(e, "get_position_info")
            return None