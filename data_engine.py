import pandas as pd
import time
from typing import Dict, List, Optional
from bybit_client import BybitClient
from config import SYMBOLS, TIMEFRAME

class DataEngine:
    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.historical_data = {}
        self.error_handler = None
        
    def set_error_handler(self, error_handler):
        """Set error handler for data validation errors"""
        self.error_handler = error_handler
        
    def validate_market_data(self, df: pd.DataFrame) -> bool:
        """
        Comprehensive market data validation
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        if df is None or df.empty:
            if self.error_handler:
                self.error_handler.handle_data_error(Exception("Empty DataFrame"), "data_validation")
            return False
            
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception(f"Missing columns: {missing_columns}"), "data_validation"
                )
            return False
        
        # Check for NaN values
        if df.isnull().any().any():
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("NaN values detected in data"), "data_validation"
                )
            return False
        
        # Check for valid price relationships
        price_checks = [
            (df['high'] >= df['low']).all(),
            (df['high'] >= df['close']).all(),
            (df['low'] <= df['close']).all(),
            (df['high'] >= df['open']).all(),
            (df['low'] <= df['open']).all()
        ]
        
        if not all(price_checks):
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("Invalid price relationships"), "data_validation"
                )
            return False
        
        # Check for non-negative volume
        if (df['volume'] < 0).any():
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("Negative volume detected"), "data_validation"
                )
            return False
        
        # Check for reasonable price movements (no 100x changes in one candle)
        price_changes = df['close'].pct_change().abs()
        if (price_changes > 5.0).any():  # 500% change threshold
            if self.error_handler:
                self.error_handler.handle_data_error(
                    Exception("Abnormal price movements detected"), "data_validation"
                )
            return False
        
        # Check timestamp order (should be increasing)
        if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if not df['timestamp'].is_monotonic_increasing:
                if self.error_handler:
                    self.error_handler.handle_data_error(
                        Exception("Timestamps not in increasing order"), "data_validation"
                    )
                return False
        
        return True
    
    def get_historical_data(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Fetch historical kline data and convert to DataFrame with validation"""
        try:
            response = self.client.get_kline(symbol, interval, limit)
            
            if response and response.get('retCode') == 0:
                klines = response['result']['list']
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'turnover'
                ])
                
                # Convert data types
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Sort by timestamp (oldest first)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Validate data before storing
                if self.validate_market_data(df):
                    self.historical_data[symbol] = df
                    return df
                else:
                    print(f"⚠️ Invalid data for {symbol}, returning None")
                    return None
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                print(f"Error fetching data for {symbol}: {error_msg}")
                if self.error_handler:
                    self.error_handler.handle_api_error(Exception(error_msg), f"get_kline_{symbol}")
                return None
                
        except Exception as e:
            print(f"Exception in get_historical_data for {symbol}: {e}")
            if self.error_handler:
                self.error_handler.handle_api_error(e, f"get_historical_data_{symbol}")
            return None
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with fallback"""
        try:
            # Use the latest close price from historical data
            if symbol in self.historical_data and not self.historical_data[symbol].empty:
                return self.historical_data[symbol]['close'].iloc[-1]
            else:
                # Fallback: fetch minimal historical data
                df = self.get_historical_data(symbol, TIMEFRAME, limit=1)
                if df is not None and not df.empty:
                    return df['close'].iloc[-1]
                else:
                    return 0.0
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            if self.error_handler:
                self.error_handler.handle_data_error(e, "get_current_price", symbol)
            return 0.0
    
    def update_all_data(self):
        """Update historical data for all symbols with error handling"""
        successful_updates = 0
        for symbol in SYMBOLS:
            try:
                df = self.get_historical_data(symbol, TIMEFRAME)
                if df is not None and self.validate_market_data(df):
                    successful_updates += 1
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error updating data for {symbol}: {e}")
                if self.error_handler:
                    self.error_handler.handle_data_error(e, "update_all_data", symbol)
        
        print(f"✅ Updated data for {successful_updates}/{len(SYMBOLS)} symbols")
    
    def get_technicals_for_symbol(self, symbol: str):
        """Get technical indicators for a symbol (interface for strategy orchestrator)"""
        if symbol not in self.historical_data:
            self.get_historical_data(symbol, TIMEFRAME)
        return self.historical_data.get(symbol)
    
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