import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

class StressTester:
    """
    Stress testing framework for the trading bot
    Tests system under extreme market conditions
    """
    
    def __init__(self, trading_bot):
        self.bot = trading_bot
        self.logger = logging.getLogger('StressTester')
        self.test_results = []
        
    def generate_stress_market_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Generate extreme market conditions for stress testing"""
        try:
            # Create volatile market data
            base_price = 50000  # Starting price
            prices = [base_price]
            
            for i in range(periods - 1):
                # Extreme volatility: up to 20% moves
                volatility = random.uniform(0.01, 0.20)
                direction = random.choice([-1, 1])
                price_change = prices[-1] * volatility * direction
                new_price = max(1, prices[-1] + price_change)  # Prevent negative prices
                prices.append(new_price)
            
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * random.uniform(1.0, 1.1) for p in prices],
                'low': [p * random.uniform(0.9, 1.0) for p in prices],
                'close': prices,
                'volume': [random.uniform(1000, 100000) for _ in prices]
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating stress market data: {e}")
            return pd.DataFrame()
    
    def run_market_crash_test(self, duration_minutes: int = 10):
        """Test bot performance during market crash conditions"""
        self.logger.info(f"Starting market crash stress test for {duration_minutes} minutes")
        
        start_time = time.time()
        test_metrics = {
            'test_type': 'market_crash',
            'start_time': datetime.now().isoformat(),
            'cycles_completed': 0,
            'errors_encountered': 0,
            'emergency_stops': 0,
            'performance_metrics': []
        }
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                cycle_start = time.time()
                
                # Simulate crash conditions
                crash_data = {}
                for symbol in self.bot.SYMBOLS:
                    crash_data[symbol] = self.generate_stress_market_data(symbol, 50)
                
                # Run analysis under stress
                decisions = []
                for symbol in self.bot.SYMBOLS:
                    try:
                        decision = self.bot.strategy_orchestrator.analyze_symbol(
                            symbol, crash_data[symbol], 10000
                        )
                        decisions.append(decision)
                    except Exception as e:
                        test_metrics['errors_encountered'] += 1
                        self.logger.error(f"Error in crash test for {symbol}: {e}")
                
                # Check emergency protocols
                emergency_status = self.bot.emergency_protocols.check_emergency_conditions(
                    5000, -15.0, []  # Simulate 15% loss
                )
                
                if emergency_status['emergency']:
                    test_metrics['emergency_stops'] += 1
                
                # Record cycle metrics
                cycle_metrics = {
                    'cycle_duration': time.time() - cycle_start,
                    'decisions_made': len(decisions),
                    'emergency_triggered': emergency_status['emergency'],
                    'timestamp': datetime.now().isoformat()
                }
                test_metrics['performance_metrics'].append(cycle_metrics)
                test_metrics['cycles_completed'] += 1
                
                time.sleep(5)  # Short delay between cycles
            
            # Finalize test results
            test_metrics['end_time'] = datetime.now().isoformat()
            test_metrics['total_duration'] = time.time() - start_time
            
            self._log_stress_test_results(test_metrics)
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"Market crash test failed: {e}")
            test_metrics['error'] = str(e)
            return test_metrics
    
    def run_high_frequency_test(self, cycles: int = 50, delay: float = 0.1):
        """Test bot under high-frequency trading conditions"""
        self.logger.info(f"Starting high-frequency stress test: {cycles} cycles")
        
        test_metrics = {
            'test_type': 'high_frequency',
            'start_time': datetime.now().isoformat(),
            'cycle_times': [],
            'memory_usage': [],
            'errors': []
        }
        
        try:
            for i in range(cycles):
                cycle_start = time.time()
                memory_before = self._get_memory_usage()
                
                try:
                    # Run rapid trading cycles
                    decisions = self.bot.run_trading_cycle()
                    
                    cycle_time = time.time() - cycle_start
                    memory_after = self._get_memory_usage()
                    
                    test_metrics['cycle_times'].append(cycle_time)
                    test_metrics['memory_usage'].append(memory_after - memory_before)
                    
                except Exception as e:
                    test_metrics['errors'].append({
                        'cycle': i,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(delay)
            
            # Calculate performance statistics
            test_metrics['end_time'] = datetime.now().isoformat()
            test_metrics['avg_cycle_time'] = np.mean(test_metrics['cycle_times'])
            test_metrics['max_cycle_time'] = max(test_metrics['cycle_times'])
            test_metrics['total_memory_increase'] = sum(test_metrics['memory_usage'])
            
            self._log_stress_test_results(test_metrics)
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"High frequency test failed: {e}")
            test_metrics['error'] = str(e)
            return test_metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _log_stress_test_results(self, test_metrics: Dict):
        """Log stress test results comprehensively"""
        self.logger.info(
            "Stress test completed",
            extra={'test_metrics': test_metrics}
        )
        
        # Store in database
        if hasattr(self.bot, 'database') and self.bot.database:
            self.bot.database.store_system_event(
                "STRESS_TEST_COMPLETED",
                test_metrics,
                "INFO",
                "Stress Testing"
            )
        
        # Print summary
        print(f"\n🧪 STRESS TEST SUMMARY: {test_metrics['test_type']}")
        print(f"   • Duration: {test_metrics.get('total_duration', 0):.1f}s")
        print(f"   • Cycles: {test_metrics.get('cycles_completed', len(test_metrics.get('cycle_times', [])))}")
        print(f"   • Errors: {len(test_metrics.get('errors', []))}")
        
        if 'avg_cycle_time' in test_metrics:
            print(f"   • Avg Cycle Time: {test_metrics['avg_cycle_time']:.3f}s")
            print(f"   • Max Cycle Time: {test_metrics['max_cycle_time']:.3f}s")