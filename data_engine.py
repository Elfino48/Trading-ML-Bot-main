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

        # BTC data tracking for correlation features
        self.btc_data: Optional[pd.DataFrame] = None
        self.btc_correlation_cache: Dict[str, Dict] = {}
        self.btc_cache_timestamp: Dict[str, datetime] = {}
        
        # Fetch BTC data during initialization
        self.initialize_btc_data()
        self.initialize_historical_data()

    def initialize_btc_data(self, limit: int = 500):
        """Initialize BTC data for correlation calculations"""
        try:
            print("⏳ Initializing BTC data for correlation features...")
            btc_df = self.get_historical_data("BTCUSDT", TIMEFRAME, limit=limit)
            if btc_df is not None and not btc_df.empty:
                self.btc_data = btc_df
                print(f"✅ BTC data initialized: {len(btc_df)} candles")
            else:
                print("⚠️ Failed to initialize BTC data")
        except Exception as e:
            print(f"❌ Error initializing BTC data: {e}")

    def get_btc_correlation_features(self, symbol: str, current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate BTC correlation features for a symbol"""
        features = {}
        
        try:
            if self.btc_data is None or self.btc_data.empty:
                return self._get_default_btc_features()
            
            if current_data is None or current_data.empty:
                return self._get_default_btc_features()
            
            # Ensure we have enough data
            min_periods = 20
            if len(current_data) < min_periods or len(self.btc_data) < min_periods:
                return self._get_default_btc_features()
            
            # Align timestamps between symbol data and BTC data
            symbol_returns = current_data['close'].pct_change().dropna()
            btc_returns = self.btc_data['close'].pct_change().dropna()
            
            # Get common timestamps
            common_index = symbol_returns.index.intersection(btc_returns.index)
            if len(common_index) < min_periods:
                return self._get_default_btc_features()
            
            symbol_aligned = symbol_returns.loc[common_index]
            btc_aligned = btc_returns.loc[common_index]
            
            # Calculate various correlation metrics
            features.update(self._calculate_correlation_metrics(symbol_aligned, btc_aligned))
            features.update(self._calculate_relative_strength(symbol_aligned, btc_aligned))
            features.update(self._calculate_btc_dominance_metrics(symbol, current_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating BTC correlation for {symbol}: {e}")
            return self._get_default_btc_features()

    def _calculate_correlation_metrics(self, symbol_returns: pd.Series, btc_returns: pd.Series) -> Dict[str, float]:
        """Calculate various correlation metrics"""
        metrics = {}
        
        try:
            # Rolling correlations for different timeframes
            periods = [5, 10, 20, 50]
            
            for period in periods:
                if len(symbol_returns) >= period:
                    # Rolling correlation
                    rolling_corr = symbol_returns.rolling(period).corr(btc_returns)
                    metrics[f'btc_correlation_{period}'] = rolling_corr.iloc[-1] if not pd.isna(rolling_corr.iloc[-1]) else 0.0
                    
                    # Beta calculation (sensitivity to BTC moves)
                    covariance = symbol_returns.rolling(period).cov(btc_returns)
                    btc_variance = btc_returns.rolling(period).var()
                    beta = covariance / btc_variance
                    metrics[f'btc_beta_{period}'] = beta.iloc[-1] if not pd.isna(beta.iloc[-1]) else 1.0
            
            # Overall correlation
            metrics['btc_correlation_overall'] = symbol_returns.corr(btc_returns) if len(symbol_returns) > 10 else 0.0
            
            # Correlation in different market regimes
            high_vol_periods = btc_returns.abs() > btc_returns.abs().quantile(0.7)
            if high_vol_periods.sum() > 5:
                high_vol_corr = symbol_returns[high_vol_periods].corr(btc_returns[high_vol_periods])
                metrics['btc_correlation_high_vol'] = high_vol_corr if not pd.isna(high_vol_corr) else 0.0
            else:
                metrics['btc_correlation_high_vol'] = 0.0
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation metrics: {e}")
            return {f'btc_correlation_{p}': 0.0 for p in [5, 10, 20, 50]}

    def _calculate_relative_strength(self, symbol_returns: pd.Series, btc_returns: pd.Series) -> Dict[str, float]:
        """Calculate relative strength vs BTC"""
        metrics = {}
        
        try:
            # Relative returns (symbol vs BTC)
            relative_returns = symbol_returns - btc_returns
            
            # Rolling relative performance
            periods = [5, 10, 20]
            for period in periods:
                if len(relative_returns) >= period:
                    rel_perf = (1 + relative_returns).rolling(period).apply(np.prod, raw=True) - 1
                    metrics[f'relative_perf_vs_btc_{period}'] = rel_perf.iloc[-1] if not pd.isna(rel_perf.iloc[-1]) else 0.0
            
            # Relative strength momentum
            if len(relative_returns) >= 10:
                rel_strength = (1 + relative_returns.tail(5)).prod() / (1 + relative_returns.tail(10)).prod() - 1
                metrics['relative_strength_momentum'] = rel_strength if not pd.isna(rel_strength) else 0.0
            
            # Outperformance ratio
            outperformance_ratio = (symbol_returns > btc_returns).rolling(20).mean()
            metrics['outperformance_ratio_20'] = outperformance_ratio.iloc[-1] if not pd.isna(outperformance_ratio.iloc[-1]) else 0.5
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating relative strength: {e}")
            return {}

    def _calculate_btc_dominance_metrics(self, symbol: str, current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate BTC dominance and market regime metrics"""
        metrics = {}
        
        try:
            if symbol == "BTCUSDT":
                # BTC doesn't have BTC dominance metrics against itself
                return {}
            
            # BTC dominance trend (simplified - in production you'd get actual dominance data)
            if self.btc_data is not None and len(self.btc_data) >= 20:
                # Use BTC momentum as proxy for dominance trend
                btc_momentum = self.btc_data['close'].pct_change(10).iloc[-1]
                metrics['btc_dominance_trend'] = 1.0 if btc_momentum > 0 else -1.0 if btc_momentum < 0 else 0.0
                
                # Altcoin season detection
                btc_volatility = self.btc_data['close'].pct_change().rolling(20).std().iloc[-1]
                symbol_volatility = current_data['close'].pct_change().rolling(20).std().iloc[-1]
                
                # Altcoins typically outperform in low BTC volatility environments
                volatility_ratio = symbol_volatility / btc_volatility if btc_volatility > 0 else 1.0
                metrics['altcoin_season_score'] = min(2.0, volatility_ratio)  # Cap at 2.0
            
            # Market regime based on BTC behavior
            if self.btc_data is not None and len(self.btc_data) >= 50:
                btc_returns = self.btc_data['close'].pct_change().dropna()
                
                # Bull/bear market detection
                btc_sma_20 = self.btc_data['close'].rolling(20).mean().iloc[-1]
                btc_sma_50 = self.btc_data['close'].rolling(50).mean().iloc[-1]
                
                if btc_sma_20 > btc_sma_50 and btc_returns.tail(5).mean() > 0:
                    metrics['btc_market_regime'] = 1.0  # Bull market
                elif btc_sma_20 < btc_sma_50 and btc_returns.tail(5).mean() < 0:
                    metrics['btc_market_regime'] = -1.0  # Bear market
                else:
                    metrics['btc_market_regime'] = 0.0  # Neutral
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating BTC dominance metrics: {e}")
            return {}

    def _get_default_btc_features(self) -> Dict[str, float]:
        """Return default BTC correlation features when calculation fails"""
        return {
            'btc_correlation_5': 0.0,
            'btc_correlation_10': 0.0,
            'btc_correlation_20': 0.0,
            'btc_correlation_50': 0.0,
            'btc_correlation_overall': 0.0,
            'btc_correlation_high_vol': 0.0,
            'btc_beta_5': 1.0,
            'btc_beta_10': 1.0,
            'btc_beta_20': 1.0,
            'btc_beta_50': 1.0,
            'relative_perf_vs_btc_5': 0.0,
            'relative_perf_vs_btc_10': 0.0,
            'relative_perf_vs_btc_20': 0.0,
            'relative_strength_momentum': 0.0,
            'outperformance_ratio_20': 0.5,
            'btc_dominance_trend': 0.0,
            'altcoin_season_score': 1.0,
            'btc_market_regime': 0.0
        }

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

                        if symbol not in self.historical_data:
                            self.logger.warning(f"No historical data found for {symbol}, skipping WS update")
                            return
                            
                        # Get a working copy of the DataFrame
                        current_df = self.historical_data[symbol].copy()
                        
                        for kline_dict in data:
                            self.logger.debug(f"WS Kline Received Dict: {kline_dict}")
                            
                            # More robust timestamp parsing
                            start_time = kline_dict.get('start')
                            if not start_time:
                                self.logger.warning(f"No start time in kline data for {symbol}")
                                continue
                                
                            try:
                                ts = pd.to_datetime(int(start_time), unit='ms')
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Invalid timestamp {start_time} for {symbol}: {e}")
                                continue

                            # Validate price data
                            close_price_str = kline_dict.get('close')
                            if close_price_str is None or float(close_price_str) <= 0:
                                self.logger.warning(f"Invalid close price {close_price_str} for {symbol} at {ts}")
                                continue
                                
                            new_kline_data = {
                                'timestamp': ts,
                                'open': float(kline_dict.get('open', 0)),
                                'high': float(kline_dict.get('high', 0)),
                                'low': float(kline_dict.get('low', 0)),
                                'close': float(close_price_str),
                                'volume': float(kline_dict.get('volume', 0)),
                                'turnover': float(kline_dict.get('turnover', 0))
                            }
                            
                            self.logger.info(f"Processing WS kline for {symbol} at {ts}: close={new_kline_data['close']}")

                            # Check if we need to update last candle or append new one
                            if not current_df.empty and ts in current_df.index:
                                # Update existing candle
                                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                                    if col in new_kline_data:
                                        current_df.loc[ts, col] = new_kline_data[col]
                                self.logger.info(f"Updated existing candle for {symbol} at {ts}")
                                
                            elif current_df.empty or ts > current_df.index[-1]:
                                # Append new candle
                                new_row = pd.DataFrame([new_kline_data]).set_index('timestamp')
                                current_df = pd.concat([current_df, new_row])
                                current_df = current_df[~current_df.index.duplicated(keep='last')]
                                current_df = current_df.sort_index()
                                
                                # Limit size
                                if len(current_df) > 1000:
                                    current_df = current_df.iloc[-1000:]
                                    
                                self.logger.info(f"Appended new candle for {symbol} at {ts}. New length: {len(current_df)}")
                                
                            else:
                                self.logger.warning(f"Out-of-order timestamp for {symbol}: {ts} vs last {current_df.index[-1] if not current_df.empty else 'None'}")
                        
                        # Update the stored DataFrame
                        self.historical_data[symbol] = current_df
                        self.logger.info(f"Updated {symbol} data. Latest close: {current_df['close'].iloc[-1] if not current_df.empty else 'N/A'}")
                        
        except Exception as e:
            self.logger.error(f"Error handling WS message: {type(e).__name__} - {e} | Topic: {topic}", exc_info=True)

    def check_data_freshness(self):
        """Check if data is being updated properly"""
        current_time = pd.Timestamp.now()
        freshness_issues = []
        
        with self.data_lock:
            for symbol, df in self.historical_data.items():
                if df is None or df.empty:
                    freshness_issues.append(f"{symbol}: No data")
                    continue
                    
                last_timestamp = df.index[-1]
                time_diff = (current_time - last_timestamp).total_seconds() / 60  # minutes
                
                if time_diff > 10:  # More than 10 minutes old
                    freshness_issues.append(f"{symbol}: Data is {time_diff:.1f} minutes old")
                    
                # Check if data is changing
                if len(df) > 1:
                    recent_changes = df['close'].iloc[-5:].nunique()  # Check last 5 candles
                    if recent_changes == 1:
                        freshness_issues.append(f"{symbol}: No price changes in last 5 candles")
        
        if freshness_issues:
            self.logger.warning(f"Data freshness issues:\n" + "\n".join(freshness_issues))
            return False
        else:
            self.logger.info("All data streams are fresh")
            return True

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