import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from enhanced_strategy_orchestrator import EnhancedStrategyOrchestrator
from advanced_risk_manager import AdvancedRiskManager

class AdvancedBacktester:
    """
    Sophisticated backtesting engine with walk-forward analysis
    Includes transaction costs, slippage, and detailed performance metrics
    """
    
    def __init__(self, strategy_orchestrator: EnhancedStrategyOrchestrator):
        self.strategy_orchestrator = strategy_orchestrator
        self.results = {}
        self.backtest_history = []
        
        # Backtesting parameters
        self.initial_capital = 10000
        self.transaction_cost = 0.001  # 0.1% per trade
        self.slippage = 0.0005  # 0.05% slippage
        self.min_trade_size = 10  # Minimum trade size in USDT
        
        self.logger = logging.getLogger('AdvancedBacktester')
        
        print("🧪 Advanced Backtester initialized")
    
    def run_walk_forward_analysis(self, historical_data: Dict[str, pd.DataFrame], 
                                 periods: int = 100, train_size: int = 200, 
                                 test_size: int = 50) -> Dict:
        """
        Run walk-forward backtesting analysis
        
        Args:
            historical_data: Dictionary of DataFrames keyed by symbol
            periods: Number of walk-forward periods
            train_size: Training window size
            test_size: Testing window size
            
        Returns:
            Backtest results
        """
        all_results = []
        
        for symbol, data in historical_data.items():
            self.logger.info(f"Running walk-forward analysis for {symbol}")
            
            symbol_results = self._walk_forward_symbol(symbol, data, periods, train_size, test_size)
            all_results.extend(symbol_results)
        
        # Aggregate results
        aggregated_results = self._aggregate_backtest_results(all_results)
        
        self.results = aggregated_results
        self.backtest_history.append({
            'timestamp': datetime.now(),
            'parameters': {
                'periods': periods,
                'train_size': train_size,
                'test_size': test_size
            },
            'results': aggregated_results
        })
        
        return aggregated_results
    
    def _walk_forward_symbol(self, symbol: str, data: pd.DataFrame, 
                           periods: int, train_size: int, test_size: int) -> List[Dict]:
        """
        Run walk-forward analysis for a single symbol
        """
        results = []
        total_length = len(data)
        
        for i in range(periods):
            # Calculate train and test indices
            train_start = i * test_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            if test_end > total_length:
                break
            
            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Run backtest on test period
            period_result = self._backtest_period(symbol, train_data, test_data, i)
            results.append(period_result)
        
        return results
    
    def _backtest_period(self, symbol: str, train_data: pd.DataFrame, 
                        test_data: pd.DataFrame, period: int) -> Dict:
        """
        Backtest a single period
        """
        # Simulate trading on test data
        trades = []
        portfolio_value = self.initial_capital
        position = None
        
        for idx, (timestamp, row) in enumerate(test_data.iterrows()):
            current_price = row['close']
            
            # Create historical context up to current point
            historical_up_to_now = pd.concat([train_data, test_data.iloc[:idx+1]])
            
            # Get trading decision
            decision = self._simulate_trading_decision(symbol, historical_up_to_now, portfolio_value)
            
            # Execute trade if needed
            if decision['action'] != 'HOLD' and decision['confidence'] >= 25:
                trade_result = self._execute_simulated_trade(
                    symbol, decision, position, current_price, portfolio_value
                )
                
                if trade_result['executed']:
                    trades.append(trade_result['trade'])
                    portfolio_value = trade_result['new_portfolio_value']
                    position = trade_result['new_position']
            
            # Update portfolio value for tracking
            if idx == len(test_data) - 1:
                final_portfolio_value = portfolio_value
        
        # Calculate period performance
        period_performance = self._calculate_period_performance(trades, final_portfolio_value)
        
        return {
            'period': period,
            'symbol': symbol,
            'trades': trades,
            'performance': period_performance,
            'final_portfolio_value': final_portfolio_value,
            'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}"
        }
    
    def _simulate_trading_decision(self, symbol: str, historical_data: pd.DataFrame, 
                                 portfolio_value: float) -> Dict:
        """
        Simulate trading decision using the strategy orchestrator
        """
        try:
            # Use a copy to avoid modifying original data
            data_copy = historical_data.copy()
            
            # Get decision from strategy orchestrator
            decision = self.strategy_orchestrator.analyze_symbol(symbol, data_copy, portfolio_value)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error simulating trading decision for {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0}
    
    def _execute_simulated_trade(self, symbol: str, decision: Dict, current_position: Optional[Dict],
                               current_price: float, portfolio_value: float) -> Dict:
        """
        Execute simulated trade with costs and slippage
        """
        action = decision['action']
        quantity = decision['quantity']
        position_size = decision['position_size']
        
        # Apply slippage
        execution_price = current_price * (1 + self.slippage) if action == 'BUY' else current_price * (1 - self.slippage)
        
        # Calculate transaction costs
        trade_cost = position_size * self.transaction_cost
        
        trade_result = {
            'executed': False,
            'trade': None,
            'new_portfolio_value': portfolio_value,
            'new_position': current_position
        }
        
        if action == 'BUY' and not current_position:
            # Open long position
            if position_size >= self.min_trade_size and position_size <= portfolio_value:
                new_position = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'entry_time': datetime.now(),
                    'position_size': position_size - trade_cost
                }
                
                new_portfolio_value = portfolio_value - position_size - trade_cost
                
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'position_size': position_size,
                    'costs': trade_cost,
                    'timestamp': datetime.now()
                }
                
                trade_result.update({
                    'executed': True,
                    'trade': trade,
                    'new_portfolio_value': new_portfolio_value,
                    'new_position': new_position
                })
        
        elif action == 'SELL' and current_position and current_position['action'] == 'BUY':
            # Close long position
            entry_price = current_position['entry_price']
            position_quantity = current_position['quantity']
            
            # Calculate PnL
            pnl = (execution_price - entry_price) * position_quantity
            pnl_percent = (execution_price - entry_price) / entry_price * 100
            
            # Apply costs
            pnl_after_costs = pnl - trade_cost
            
            new_portfolio_value = portfolio_value + current_position['position_size'] + pnl_after_costs
            
            trade = {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': position_quantity,
                'entry_price': entry_price,
                'exit_price': execution_price,
                'position_size': current_position['position_size'],
                'pnl': pnl_after_costs,
                'pnl_percent': pnl_percent,
                'costs': trade_cost,
                'timestamp': datetime.now(),
                'hold_duration': (datetime.now() - current_position['entry_time']).total_seconds() / 3600
            }
            
            trade_result.update({
                'executed': True,
                'trade': trade,
                'new_portfolio_value': new_portfolio_value,
                'new_position': None  # Position closed
            })
        
        return trade_result
    
    def _calculate_period_performance(self, trades: List[Dict], final_portfolio_value: float) -> Dict:
        """
        Calculate performance metrics for a backtest period
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Extract PnL values
        pnls = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
        pnl_percents = [trade.get('pnl_percent', 0) for trade in trades if 'pnl_percent' in trade]
        
        winning_trades = [p for p in pnl_percents if p > 0]
        losing_trades = [p for p in pnl_percents if p < 0]
        
        # Calculate metrics
        total_trades = len(trades)
        winning_count = len(winning_trades)
        win_rate = (winning_count / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(pnls)
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_percents) if pnl_percents else 0
        
        # Max drawdown (simplified)
        max_drawdown = self._calculate_max_drawdown(pnl_percents) if pnl_percents else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns"""
        if not returns:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0
        
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        
        return np.max(drawdown) * 100 if len(drawdown) > 0 else 0
    
    def _aggregate_backtest_results(self, all_results: List[Dict]) -> Dict:
        """
        Aggregate results across all periods and symbols
        """
        if not all_results:
            return {}
        
        # Flatten all trades
        all_trades = []
        for result in all_results:
            all_trades.extend(result.get('trades', []))
        
        # Calculate overall metrics
        total_periods = len(all_results)
        total_trades = len(all_trades)
        
        # Aggregate performance metrics
        performance_metrics = [r.get('performance', {}) for r in all_results]
        
        aggregated = {
            'total_periods': total_periods,
            'total_trades': total_trades,
            'avg_win_rate': np.mean([p.get('win_rate', 0) for p in performance_metrics]),
            'avg_total_return': np.mean([p.get('total_return', 0) for p in performance_metrics]),
            'avg_sharpe_ratio': np.mean([p.get('sharpe_ratio', 0) for p in performance_metrics]),
            'avg_max_drawdown': np.mean([p.get('max_drawdown', 0) for p in performance_metrics]),
            'final_portfolio_value': all_results[-1].get('final_portfolio_value', self.initial_capital) if all_results else self.initial_capital,
            'total_pnl': sum([p.get('total_pnl', 0) for p in performance_metrics]),
            'period_results': all_results,
            'all_trades': all_trades
        }
        
        return aggregated
    
    def run_single_backtest(self, symbol: str, historical_data: pd.DataFrame, 
                           initial_capital: float = None) -> Dict:
        """
        Run a simple single-period backtest
        """
        if initial_capital:
            self.initial_capital = initial_capital
        
        # Split data into train/test (80/20)
        split_idx = int(len(historical_data) * 0.8)
        train_data = historical_data.iloc[:split_idx].copy()
        test_data = historical_data.iloc[split_idx:].copy()
        
        # Run backtest
        period_result = self._backtest_period(symbol, train_data, test_data, 0)
        
        # Store results
        self.results = {
            'single_backtest': period_result,
            'timestamp': datetime.now()
        }
        
        return period_result
    
    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics from trades
        """
        if not trades:
            return {}
        
        # Extract trading data
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        pnl_percents = [t.get('pnl_percent', 0) for t in trades if 'pnl_percent' in t]
        hold_times = [t.get('hold_duration', 0) for t in trades if 'hold_duration' in t]
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [p for p in pnl_percents if p > 0]
        losing_trades = [p for p in pnl_percents if p < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_percents)
        sortino_ratio = self._calculate_sortino_ratio(pnl_percents)
        calmar_ratio = self._calculate_calmar_ratio(pnl_percents)
        
        # Drawdown analysis
        max_drawdown = self._calculate_max_drawdown(pnl_percents)
        avg_drawdown = self._calculate_avg_drawdown(pnl_percents)
        
        # Trade characteristics
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        trade_frequency = total_trades / (len(set([t['timestamp'].date() for t in trades])) or 1)
        
        return {
            'basic_metrics': {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0,
                'total_return': sum(pnl_percents)
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'volatility': np.std(pnl_percents) if pnl_percents else 0
            },
            'trade_characteristics': {
                'avg_hold_time_hours': avg_hold_time,
                'trades_per_day': trade_frequency,
                'best_trade': max(pnl_percents) if pnl_percents else 0,
                'worst_trade': min(pnl_percents) if pnl_percents else 0,
                'avg_trade_result': np.mean(pnl_percents) if pnl_percents else 0
            }
        }
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (only downside risk)"""
        if not returns:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)
        
        # Only consider negative returns for downside deviation
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        if downside_deviation == 0:
            return 0
        
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        return sortino
    
    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio (return vs max drawdown)"""
        if not returns:
            return 0
        
        avg_return = np.mean(returns) * 252  # Annualized
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0
        
        return avg_return / max_dd
    
    def _calculate_avg_drawdown(self, returns: List[float]) -> float:
        """Calculate average drawdown"""
        if not returns:
            return 0
        
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / peak
        
        # Only consider actual drawdown periods
        actual_drawdowns = drawdowns[drawdowns > 0]
        
        return np.mean(actual_drawdowns) * 100 if len(actual_drawdowns) > 0 else 0
    
    def generate_backtest_report(self, results: Dict) -> str:
        """
        Generate a comprehensive backtest report
        """
        if not results:
            return "No backtest results available"
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        
        if 'single_backtest' in results:
            # Single backtest report
            single_result = results['single_backtest']
            performance = single_result.get('performance', {})
            
            report.append(f"Symbol: {single_result.get('symbol', 'N/A')}")
            report.append(f"Test Period: {single_result.get('test_period', 'N/A')}")
            report.append("")
            report.append("PERFORMANCE SUMMARY:")
            report.append(f"  Total Trades: {performance.get('total_trades', 0)}")
            report.append(f"  Win Rate: {performance.get('win_rate', 0):.1f}%")
            report.append(f"  Total Return: {performance.get('total_return', 0):.2f}%")
            report.append(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
            report.append(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2f}%")
            report.append(f"  Final Portfolio: ${single_result.get('final_portfolio_value', 0):.2f}")
        
        else:
            # Walk-forward report
            report.append("WALK-FORWARD BACKTEST RESULTS")
            report.append("")
            report.append("AGGREGATED METRICS:")
            report.append(f"  Total Periods: {results.get('total_periods', 0)}")
            report.append(f"  Total Trades: {results.get('total_trades', 0)}")
            report.append(f"  Average Win Rate: {results.get('avg_win_rate', 0):.1f}%")
            report.append(f"  Average Return: {results.get('avg_total_return', 0):.2f}%")
            report.append(f"  Average Sharpe: {results.get('avg_sharpe_ratio', 0):.2f}")
            report.append(f"  Average Max DD: {results.get('avg_max_drawdown', 0):.2f}%")
            report.append(f"  Final Portfolio: ${results.get('final_portfolio_value', 0):.2f}")
        
        report.append("")
        report.append("STRATEGY ASSESSMENT:")
        
        # Strategy assessment
        avg_win_rate = results.get('avg_win_rate', 0) if 'avg_win_rate' in results else results.get('single_backtest', {}).get('performance', {}).get('win_rate', 0)
        
        if avg_win_rate > 60:
            assessment = "EXCELLENT - High win rate strategy"
        elif avg_win_rate > 50:
            assessment = "GOOD - Profitable strategy"
        elif avg_win_rate > 40:
            assessment = "FAIR - Needs improvement"
        else:
            assessment = "POOR - Consider strategy changes"
        
        report.append(f"  {assessment}")
        report.append("")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_backtest_history(self) -> List[Dict]:
        """Get history of all backtests run"""
        return self.backtest_history
    
    def optimize_parameters(self, historical_data: Dict[str, pd.DataFrame], 
                           param_ranges: Dict) -> Dict:
        """
        Optimize strategy parameters using backtesting
        """
        best_params = {}
        best_score = -np.inf
        
        self.logger.info("Starting parameter optimization...")
        
        # Simple grid search (in practice, use more sophisticated methods)
        for param_combination in self._generate_param_combinations(param_ranges):
            try:
                # Update strategy parameters
                self._update_strategy_parameters(param_combination)
                
                # Run backtest
                results = self.run_walk_forward_analysis(historical_data, periods=20)
                
                # Calculate optimization score (customizable)
                score = self._calculate_optimization_score(results)
                
                if score > best_score:
                    best_score = score
                    best_params = param_combination.copy()
                    
                    self.logger.info(f"New best params: {best_params} (score: {score:.3f})")
                    
            except Exception as e:
                self.logger.error(f"Error testing parameters {param_combination}: {e}")
                continue
        
        self.logger.info(f"Parameter optimization complete. Best: {best_params}")
        return best_params
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate parameter combinations for grid search"""
        # Simplified implementation
        combinations = [{}]  # Start with empty combination
        
        for param, values in param_ranges.items():
            new_combinations = []
            for combo in combinations:
                for value in values:
                    new_combo = combo.copy()
                    new_combo[param] = value
                    new_combinations.append(new_combo)
            combinations = new_combinations
        
        return combinations
    
    def _update_strategy_parameters(self, params: Dict):
        """Update strategy parameters for testing"""
        # This would need to be implemented based on your strategy structure
        pass
    
    def _calculate_optimization_score(self, results: Dict) -> float:
        """Calculate optimization score from backtest results"""
        if not results:
            return -np.inf
        
        # Customizable scoring function
        win_rate = results.get('avg_win_rate', 0)
        sharpe = results.get('avg_sharpe_ratio', 0)
        max_dd = results.get('avg_max_drawdown', 100)
        
        # Penalize high drawdowns, reward high Sharpe and win rate
        score = (win_rate * 0.4) + (sharpe * 0.4) - (max_dd * 0.2)
        
        return score


# Example usage and testing
if __name__ == "__main__":
    # Mock strategy orchestrator for testing
    class MockStrategyOrchestrator:
        def analyze_symbol(self, symbol, data, portfolio_value):
            # Simple mock decision
            return {
                'action': 'BUY' if len(data) % 2 == 0 else 'SELL',
                'confidence': 65.5,
                'quantity': 0.1,
                'position_size': 1000,
                'composite_score': 25.3
            }
    
    # Test backtester
    strategy = MockStrategyOrchestrator()
    backtester = AdvancedBacktester(strategy)
    
    # Generate mock data
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='H')
    mock_data = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 10 + 100,
        'high': np.random.randn(len(dates)) * 10 + 101,
        'low': np.random.randn(len(dates)) * 10 + 99,
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randn(len(dates)) * 1000 + 10000
    }, index=dates)
    
    # Run single backtest
    results = backtester.run_single_backtest('TEST', mock_data)
    print(f"Single backtest completed: {len(results.get('trades', []))} trades")
    
    # Generate report
    report = backtester.generate_backtest_report(backtester.results)
    print(report)