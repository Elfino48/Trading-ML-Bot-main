import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

class TestTradingBotCore(unittest.TestCase):
    """Unit tests for core trading bot functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = self._create_test_data()
        self.temp_db_path = tempfile.mktemp(suffix='.db')
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _create_test_data(self, periods: int = 100) -> pd.DataFrame:
        """Create realistic test market data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=periods//24),
            periods=periods,
            freq='1H'
        )
        
        # Generate realistic price data with some volatility
        prices = [100.0]
        for i in range(periods - 1):
            change = np.random.normal(0, 2)  # 2% volatility
            new_price = max(1.0, prices[-1] + change)
            prices.append(new_price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],  # High is 1% above open
            'low': [p * 0.99 for p in prices],   # Low is 1% below open
            'close': [p * 1.005 for p in prices], # Close near high
            'volume': np.random.uniform(1000, 10000, periods)
        })
    
    def test_technical_analyzer_calculation(self):
        """Test technical analyzer calculations"""
        from enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
        
        analyzer = EnhancedTechnicalAnalyzer()
        indicators = analyzer.calculate_regime_indicators(self.test_data)
        
        # Verify indicators are calculated
        self.assertIsInstance(indicators, dict)
        self.assertGreater(len(indicators), 0)
        
        # Verify key indicators exist
        expected_indicators = ['rsi_14', 'atr', 'market_regime']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators)
    
    def test_risk_manager_position_sizing(self):
        """Test risk manager position sizing calculations"""
        from advanced_risk_manager import AdvancedRiskManager
        
        # Mock client
        class MockClient:
            def get_wallet_balance(self):
                return {'retCode': 0, 'result': {'list': [{'totalEquity': '10000'}]}}
        
        risk_manager = AdvancedRiskManager(MockClient(), "conservative")
        
        # Test position sizing
        position_info = risk_manager.calculate_aggressive_position_size(
            'BTCUSDT', 75.0, 50000, 1000, 10000, "conservative"
        )
        
        self.assertIsInstance(position_info, dict)
        self.assertIn('size_usdt', position_info)
        self.assertIn('quantity', position_info)
        
        # Verify position size is within limits
        self.assertGreater(position_info['size_usdt'], 0)
        self.assertLessEqual(position_info['size_usdt'], 100)  # Conservative limit
    
    def test_ml_predictor_data_leakage(self):
        """Test ML predictor for data leakage"""
        from ml_predictor import MLPredictor
        
        predictor = MLPredictor()
        
        # Test feature preparation
        features = predictor.prepare_features(self.test_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        if not features.empty:
            # Verify no NaN values
            self.assertFalse(features.isnull().any().any())
            
            # Verify feature columns
            self.assertGreater(len(features.columns), 0)
    
    def test_database_operations(self):
        """Test database storage and retrieval"""
        from trading_database import TradingDatabase
        
        db = TradingDatabase(self.temp_db_path)
        
        # Test trade storage
        test_trade = {
            'timestamp': datetime.now(),
            'symbol': 'TESTUSDT',
            'action': 'BUY',
            'quantity': 1.0,
            'entry_price': 100.0,
            'position_size_usdt': 100.0,
            'stop_loss': 95.0,
            'take_profit': 110.0,
            'confidence': 75.0,
            'composite_score': 25.0,
            'risk_reward_ratio': 2.0,
            'aggressiveness': 'moderate',
            'order_id': 'TEST_123',
            'success': True
        }
        
        # Store trade
        result = db.store_trade(test_trade)
        self.assertTrue(result)
        
        # Retrieve trades
        trades = db.get_historical_trades(days=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]['symbol'], 'TESTUSDT')
        
        db.close()
    
    def test_error_handling(self):
        """Test error handler functionality"""
        from error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Test API error handling
        error_result = handler.handle_api_error(
            Exception("Test API error"), 
            "test_endpoint"
        )
        
        self.assertIsInstance(error_result, dict)
        self.assertIn('handled', error_result)
        self.assertIn('should_retry', error_result)
    
    def test_execution_engine_validation(self):
        """Test execution engine trade validation"""
        from execution_engine import ExecutionEngine
        
        # Mock dependencies
        class MockClient:
            def place_order(self, **kwargs):
                return {'retCode': 0, 'result': {'orderId': 'TEST_ORDER'}}
        
        class MockRiskManager:
            def can_trade(self, symbol, size):
                return {'approved': True, 'adjusted_size': size}
        
        engine = ExecutionEngine(MockClient(), MockRiskManager())
        
        # Test valid trade decision
        valid_decision = {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'quantity': 0.1,
            'position_size': 1000,
            'current_price': 50000,
            'stop_loss': 49000,
            'take_profit': 52000
        }
        
        is_valid = engine._validate_trade_decision(valid_decision)
        self.assertTrue(is_valid)
        
        # Test invalid trade decision
        invalid_decision = {
            'symbol': 'BTCUSDT',
            'action': 'INVALID_ACTION',  # Invalid action
            'quantity': -0.1,  # Invalid quantity
            'position_size': 1000,
            'current_price': 50000
        }
        
        is_valid = engine._validate_trade_decision(invalid_decision)
        self.assertFalse(is_valid)

class TestStressScenarios(unittest.TestCase):
    """Test edge cases and stress scenarios"""
    
    def test_empty_market_data(self):
        """Test handling of empty market data"""
        from enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
        
        analyzer = EnhancedTechnicalAnalyzer()
        empty_data = pd.DataFrame()
        
        indicators = analyzer.calculate_regime_indicators(empty_data)
        self.assertEqual(indicators, {})
    
    def test_extreme_market_conditions(self):
        """Test handling of extreme market data"""
        from enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
        
        # Create data with extreme moves
        dates = pd.date_range(start=datetime.now(), periods=50, freq='1H')
        extreme_data = pd.DataFrame({
            'timestamp': dates,
            'open': [1000] * 50,
            'high': [10000] * 50,  # Extreme high
            'low': [1] * 50,       # Extreme low
            'close': [5000] * 50,
            'volume': [1000000] * 50
        })
        
        analyzer = EnhancedTechnicalAnalyzer()
        indicators = analyzer.calculate_regime_indicators(extreme_data)
        
        # Should handle extreme data without crashing
        self.assertIsInstance(indicators, dict)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)