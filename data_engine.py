import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Optional
from bybit_client import BybitClient
from config import SYMBOLS, TIMEFRAME
from datetime import datetime
import logging # Added for logging WS messages

class DataEngine:
    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.latest_tickers: Dict[str, Dict] = {} # Store latest ticker data
        self.error_handler = None
        self.data_lock = threading.Lock() # Lock for thread safety
        self.logger = logging.getLogger('DataEngine') # Added logger

        # Set the callback in the BybitClient
        self.client.set_ws_callback(self._handle_ws_message)
        
        # --- DO NOT START STREAMS HERE ---
        # self.start_streams() # <<< COMMENT OUT OR DELETE THIS LINE
        # --- END ---
        
        # Fetch initial historical data base (This uses REST, it's fine)
        self.initialize_historical_data()

    def set_error_handler(self, error_handler):
        """Set error handler for data validation and WS errors"""
        self.error_handler = error_handler
        
    def start_streams(self):
        """Initialize WebSocket connections and subscribe to streams."""
        print("🔌 Starting WebSocket streams...")
        try:
            self.client.start_websockets()
            
            # Wait briefly for connections to establish (optional but can help)
            time.sleep(5) 
            
            kline_interval_map = {
                "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, 
                "60": 60, "120": 120, "240": 240, "360": 360, "720": 720,
                "D": "D", "W": "W", "M": "M" 
            }
            ws_interval = kline_interval_map.get(TIMEFRAME, TIMEFRAME) # Use mapped interval if numeric
            
            kline_topics = [f"kline.{ws_interval}.{symbol}" for symbol in SYMBOLS]
            ticker_topics = [f"tickers.{symbol}" for symbol in SYMBOLS]
            
            all_topics = kline_topics + ticker_topics
            
            if self.client.ws_public_connected:
                print(f"   Subscribing to public topics: {all_topics}")
                self.client.ws_subscribe("public", all_topics)
            else:
                 print("   Public WebSocket not connected yet, subscriptions queued.")
                 # Store intended subscriptions for on_open handler
                 self.client.ws_public_subscriptions.update(all_topics)
                 
            # Example for private stream subscription (if needed later)
            # private_topics = ["order", "position"]
            # if self.client.ws_private_connected:
            #     print(f"   Subscribing to private topics: {private_topics}")
            #     self.client.ws_subscribe("private", private_topics)
            # else:
            #      print("   Private WebSocket not connected yet, subscriptions queued.")
            #      self.client.ws_private_subscriptions.update(private_topics)
                 
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket streams: {e}")
            if self.error_handler:
                self.error_handler.handle_api_error(e, "start_streams")

    def _handle_ws_message(self, message: Dict):
        topic = message.get("topic", "")
        data = message.get("data", {})
        ts_message = message.get('ts', int(time.time() * 1000))

        try:
            with self.data_lock:
                if topic.startswith("tickers."):
                    symbol = topic.split(".")[-1]
                    if isinstance(data, dict):
                        self.latest_tickers[symbol] = {
                            'timestamp': ts_message,
                            'lastPrice': float(data.get('lastPrice', 0)),
                            'bid1Price': float(data.get('bid1Price', 0)),
                            'ask1Price': float(data.get('ask1Price', 0)),
                            'volume24h': float(data.get('volume24h', 0)),
                            'turnover24h': float(data.get('turnover24h', 0))
                        }
                elif topic.startswith("kline."):
                    if isinstance(data, list) and len(data) > 0:
                        parts = topic.split('.')
                        interval_str = parts[1]
                        symbol = parts[2]

                        if symbol in self.historical_data:
                            df = self.historical_data[symbol]

                            for kline_dict in data:
                                self.logger.debug(f"WS Kline Received Dict: {kline_dict}") # ADDED DEBUG LOG
                                ts = pd.to_datetime(int(kline_dict.get('start', 0)), unit='ms')
                                if ts == 0: continue

                                close_price_str = kline_dict.get('close')
                                if close_price_str is None or float(close_price_str) <= 0:
                                    self.logger.warning(f"WS received invalid close price {close_price_str} for {symbol} at {ts}. Skipping update for this kline.")
                                    continue # Indent this line
                                
                                new_kline_data = {
                                    'timestamp': ts,
                                    'open': float(kline_dict.get('open', 0)),
                                    'high': float(kline_dict.get('high', 0)),
                                    'low': float(kline_dict.get('low', 0)),
                                    'close': float(close_price_str), # Use the validated price
                                    'volume': float(kline_dict.get('volume', 0)),
                                    'turnover': float(kline_dict.get('turnover', 0))
                                }
                                
                                self.logger.debug(f"WS Kline Parsed Data for {ts}: {new_kline_data}") # ADDED DEBUG LOG

                                if not df.empty and isinstance(df.index, pd.DatetimeIndex) and df.index[-1] == ts:
                                    last_index_ts = df.index[-1]
                                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                                        if col in new_kline_data:
                                            df.loc[last_index_ts, col] = new_kline_data[col]
                                    df_updated = df # Reference updated df

                                elif df.empty or (isinstance(df.index, pd.DatetimeIndex) and ts > df.index[-1]):
                                    new_row_df = pd.DataFrame([new_kline_data])
                                    was_indexed = isinstance(df.index, pd.DatetimeIndex)
                                    if was_indexed:
                                        df_temp = df.reset_index()
                                    else:
                                        df_temp = df

                                    if not df_temp.empty:
                                        missing_cols = set(df_temp.columns) - set(new_row_df.columns)
                                        for c in missing_cols: new_row_df[c] = np.nan
                                        missing_cols_new = set(new_row_df.columns) - set(df_temp.columns)
                                        for c in missing_cols_new: df_temp[c] = np.nan

                                    df_combined = pd.concat([df_temp, new_row_df], ignore_index=True)

                                    if 'timestamp' in df_combined.columns:
                                        df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
                                        df_combined = df_combined.set_index('timestamp')
                                    else:
                                        self.logger.error(f"Critical error: 'timestamp' column missing during concat for {symbol}")
                                        continue

                                    if len(df_combined) > 1000:
                                        df_combined = df_combined.iloc[-1000:]

                                    self.historical_data[symbol] = df_combined
                                    df_updated = df_combined # Reference updated df
                                else:
                                    df_updated = df # No update happened for this kline_dict


                                # --- ADDED DEBUG LOG (runs after potential update/append) ---
                                # Check df_updated exists and has data before logging tail
                                if 'df_updated' in locals() and df_updated is not None and not df_updated.empty:
                                    self.logger.debug(f"WS Kline Post-Update DF Tail for {symbol}:\n{df_updated.tail(3)}")
                                # --- END ADDED DEBUG LOG ---
        except Exception as e:
            self.logger.error(f"Error handling WS message: {type(e).__name__} - {e} | Topic: {topic}", exc_info=True)

    def validate_market_data(self, df: pd.DataFrame) -> bool:
        """Comprehensive market data validation (mostly unchanged)"""
        if df is None or df.empty:
            if self.error_handler:
                self.error_handler.handle_data_error(Exception("Empty DataFrame"), "data_validation")
            return False
            
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception(f"Missing columns: {missing_columns}"), "data_validation"
                )
            return False
        
        # Check for NaN values (excluding turnover if it exists)
        check_cols = [col for col in required_columns if col in df.columns]
        if df[check_cols].isnull().any().any():
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("NaN values detected in essential data"), "data_validation"
                )
            return False
        
        # Check specific columns are numeric before price checks
        for col in ['open', 'high', 'low', 'close', 'volume']:
             if not pd.api.types.is_numeric_dtype(df[col]):
                 if self.error_handler:
                     self.error_handler.handle_data_error(
                         Exception(f"Column '{col}' is not numeric"), "data_validation"
                     )
                 return False

        # Check for valid price relationships (ensure no NaNs first)
        df_no_nan = df[check_cols].dropna()
        if df_no_nan.empty: # If all rows had NaNs
             return False # Already caught above, but safe check
             
        price_checks = [
            (df_no_nan['high'] >= df_no_nan['low']).all(),
            (df_no_nan['high'] >= df_no_nan['close']).all(),
            (df_no_nan['low'] <= df_no_nan['close']).all(),
            (df_no_nan['high'] >= df_no_nan['open']).all(),
            (df_no_nan['low'] <= df_no_nan['open']).all()
        ]
        if not all(price_checks):
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("Invalid price relationships"), "data_validation"
                )
            return False
        
        if (df_no_nan['volume'] < 0).any():
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("Negative volume detected"), "data_validation"
                )
            return False
        
        price_changes = df_no_nan['close'].pct_change().abs()
        if (price_changes > 5.0).any(): # 500% change threshold
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("Abnormal price movements detected"), "data_validation"
                )
            return False
        
        # Allow timestamp column to be index or regular column
        ts_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('timestamp')
        if ts_col is not None and not ts_col.is_monotonic_increasing:
             # Check if it's just the last row being updated
             if len(ts_col) > 1 and len(df) > 1 and ts_col[-1] == ts_col[-2] and df.index[-1] > df.index[-2]:
                 pass # Allow last row timestamp repeat if index is increasing (WS update)
             else:
                if self.error_handler:
                    self.error_handler.handle_data_error(
                        Exception("Timestamps not in increasing order"), "data_validation"
                    )
                return False
        
        return True
    
    def get_historical_data(self, symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
            """Fetch historical kline data via REST API - used mainly for initialization."""
            try:
                response = self.client.get_kline(symbol, interval, limit)

                if response and response.get('retCode') == 0:
                    klines = response['result']['list']
                    if not klines:
                        self.logger.warning(f"No historical klines returned for {symbol} interval {interval}")
                        # Store an empty DataFrame with index to avoid errors later
                        with self.data_lock:
                            self.historical_data[symbol] = pd.DataFrame().set_index(pd.to_datetime([]))
                        return pd.DataFrame() # Return empty DataFrame if no data

                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                    ])

                    # Convert data types safely
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float).astype(int), unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

                    # Sort by timestamp (oldest first) and handle potential duplicates from API
                    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)

                    # --- FIX: Ensure Index is Set Before Storing ---
                    if 'timestamp' not in df.columns:
                        self.logger.error(f"Timestamp column missing after fetch/sort for {symbol}")
                        return None # Cannot proceed without timestamp

                    # Set index on the fetched data *before* validation/merge check
                    df_indexed = df.set_index('timestamp')

                    # Validate data before storing
                    if self.validate_market_data(df_indexed.reset_index()): # Validate needs timestamp as column temporarily
                        final_df_to_store = None
                        with self.data_lock:
                            # Merge with existing data if WS already added some
                            if symbol in self.historical_data and not self.historical_data[symbol].empty:
                                existing_df = self.historical_data[symbol] # Has DatetimeIndex
                                
                                # Combine ensuring index is kept
                                combined_df = pd.concat([existing_df, df_indexed])
                                # Drop duplicates based on the index (timestamp)
                                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                                # Sort by index (timestamp)
                                combined_df = combined_df.sort_index()

                                # Limit size after merge
                                if len(combined_df) > 1000:
                                    combined_df = combined_df.iloc[-1000:]
                                final_df_to_store = combined_df
                                self.logger.info(f"Merged fetched data for {symbol}. New length: {len(final_df_to_store)}")
                            else:
                                # Store initial fetch (already indexed)
                                final_df_to_store = df_indexed.iloc[-1000:] # Limit initial fetch size
                                self.logger.info(f"Stored initial fetched data for {symbol}. Length: {len(final_df_to_store)}")

                            self.historical_data[symbol] = final_df_to_store # Store the indexed DataFrame

                        self.logger.info(f"Historical data updated for {symbol}: {len(final_df_to_store)} rows. Index type: {type(final_df_to_store.index)}")
                        return final_df_to_store.copy() # Return a copy
                    else:
                        self.logger.warning(f"Invalid historical data fetched for {symbol} after indexing, returning None")
                        return None
                else:
                    error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                    self.logger.error(f"Error fetching historical data for {symbol}: {error_msg}")
                    if self.error_handler:
                        self.error_handler.handle_api_error(Exception(error_msg), f"get_historical_data_{symbol}")
                    return None

            except Exception as e:
                self.logger.error(f"Exception in get_historical_data for {symbol}: {e}", exc_info=True)
                if self.error_handler:
                    self.error_handler.handle_api_error(e, f"get_historical_data_{symbol}")
                return None
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price: prioritizes WebSocket ticker, falls back to last kline close."""
        with self.data_lock:
            try:
                # 1. Try latest ticker data from WebSocket
                if symbol in self.latest_tickers:
                    ticker_data = self.latest_tickers[symbol]
                    # Check if data is recent (e.g., within last 60 seconds)
                    if time.time() * 1000 - ticker_data.get('timestamp', 0) < 60000:
                        last_price = ticker_data.get('lastPrice', 0)
                        if last_price > 0:
                            # self.logger.debug(f"Using WS ticker price for {symbol}: {last_price}")
                            return last_price
                
                # 2. Try last close price from historical data (updated by WS kline)
                if symbol in self.historical_data and not self.historical_data[symbol].empty:
                    last_close = self.historical_data[symbol]['close'].iloc[-1]
                    # self.logger.debug(f"Using WS kline last close for {symbol}: {last_close}")
                    return last_close

                # 3. Fallback: Fetch latest kline via REST (should be rare if WS is working)
                self.logger.warning(f"No WS ticker or kline data for {symbol}, attempting REST fallback.")
                df = self.get_historical_data(symbol, TIMEFRAME, limit=1)
                if df is not None and not df.empty:
                    last_close_rest = df['close'].iloc[-1]
                    self.logger.warning(f"Using REST fallback price for {symbol}: {last_close_rest}")
                    return last_close_rest
                else:
                    self.logger.error(f"Could not get current price for {symbol} from any source.")
                    return 0.0
            except Exception as e:
                self.logger.error(f"Error getting current price for {symbol}: {e}")
                if self.error_handler:
                    self.error_handler.handle_data_error(e, "get_current_price", symbol)
                return 0.0 # Return 0 on error
    
    def initialize_historical_data(self, limit: int = 200):
        """Fetch initial historical data for all symbols via REST API."""
        print(f"⏳ Initializing historical data cache ({limit} candles per symbol)...")
        successful_initializations = 0
        for symbol in SYMBOLS:
            try:
                # Fetch only if not already populated (e.g., by previous run or WS)
                with self.data_lock:
                     needs_fetch = symbol not in self.historical_data or self.historical_data[symbol].empty
                
                if needs_fetch:
                    df = self.get_historical_data(symbol, TIMEFRAME, limit=limit)
                    if df is not None and not df.empty:
                        successful_initializations += 1
                    else:
                         self.logger.warning(f"Failed initial data load for {symbol}")
                else:
                     self.logger.info(f"Historical data for {symbol} already present, skipping initial fetch.")
                     successful_initializations += 1 # Count existing data as success

                time.sleep(0.2) # Rate limiting
            except Exception as e:
                self.logger.error(f"Error initializing data for {symbol}: {e}")
                if self.error_handler:
                    self.error_handler.handle_data_error(e, "initialize_historical_data", symbol)
        
        print(f"✅ Initialized historical base for {successful_initializations}/{len(SYMBOLS)} symbols")
        
    def get_market_data_for_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        with self.data_lock:
            if symbol in self.historical_data:
                df_to_return = self.historical_data[symbol]

                # --- ADDED DEBUG LOG ---
                if df_to_return is None:
                    self.logger.warning(f"get_market_data: Found None for {symbol} in cache.")
                    return None
                elif df_to_return.empty:
                    self.logger.warning(f"get_market_data: Found empty DataFrame for {symbol} in cache.")
                    return None
                elif not isinstance(df_to_return.index, pd.DatetimeIndex):
                    self.logger.warning(f"get_market_data: DataFrame for {symbol} exists but index is NOT DatetimeIndex. Type: {type(df_to_return.index)}")
                    self.logger.debug(f"get_market_data: Returning CACHED data for {symbol}. Tail BEFORE copy (Invalid Index):\n{df_to_return.tail(3)}") # Log before copy even if index is wrong
                    return df_to_return.copy()
                else:
                    self.logger.debug(f"get_market_data: Returning CACHED data for {symbol}. Tail BEFORE copy:\n{df_to_return.tail(3)}") # Log before copy
                    return df_to_return.copy()
                # --- END ADDED DEBUG LOG ---

            else:
                self.logger.warning(f"Data requested for {symbol} but not found in cache, attempting fetch...")
                fetched_df = self.get_historical_data(symbol, TIMEFRAME, limit=200)

                # --- ADDED DEBUG LOG ---
                if fetched_df is None:
                     self.logger.warning(f"get_market_data: Fetch attempt failed for {symbol}.")
                elif fetched_df.empty:
                     self.logger.warning(f"get_market_data: Fetch attempt returned empty DF for {symbol}.")
                elif not isinstance(fetched_df.index, pd.DatetimeIndex):
                     self.logger.warning(f"get_market_data: Fetched DF for {symbol} index is NOT DatetimeIndex. Type: {type(fetched_df.index)}")
                     self.logger.debug(f"get_market_data: Returning FETCHED data for {symbol}. Tail (Invalid Index):\n{fetched_df.tail(3)}") # Log fetched data
                else:
                     self.logger.debug(f"get_market_data: Returning FETCHED data for {symbol}. Tail:\n{fetched_df.tail(3)}") # Log fetched data
                # --- END ADDED DEBUG LOG ---

                return fetched_df # Return whatever was fetched

    def get_technicals_for_symbol(self, symbol: str):
        """Get technical indicators for a symbol (interface for strategy orchestrator) - Now uses safe getter"""
        return self.get_market_data_for_analysis(symbol)
        
    def stop(self):
        """Stops the WebSocket connections."""
        print("🔌 Stopping DataEngine streams...")
        self.client.stop_websockets()
        print("✅ DataEngine streams stopped.")

    # Fallback data function remains unchanged
    def _get_fallback_data(self, symbol: str) -> pd.DataFrame:
        """Generate fallback data when API fails"""
        print(f"🔄 Using fallback data for {symbol}")
        # Create simple mock data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq=f'{TIMEFRAME}min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000.0] * 100
        })
        return df