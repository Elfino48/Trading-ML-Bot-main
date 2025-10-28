import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class AttributionResult:
    """Data class for performance attribution results"""
    symbol: str
    strategy: str
    market_regime: str
    timeframe: str
    pnl: float
    risk_taken: float
    trade_count: int
    win_rate: float
    sharpe_ratio: float
    contribution_percent: float

class PerformanceAttribution:
    """
    Advanced performance attribution system that breaks down performance
    by strategy, symbol, market regime, and timeframe.
    """
    
    def __init__(self, database):
        self.database = database
        self.logger = logging.getLogger('PerformanceAttribution')
        
        # Attribution categories
        self.strategy_categories = ['trend_following', 'mean_reversion', 'breakout', 'ml_prediction']
        self.regime_categories = ['bull_trend', 'bear_trend', 'ranging', 'high_volatility', 'low_volatility']
        self.timeframe_categories = ['intraday', 'short_term', 'medium_term', 'long_term']
        
        print("📊 Performance Attribution System initialized")
    
    def perform_comprehensive_attribution(self, days: int = 30, 
                                        portfolio_value: float = None) -> Dict:
        """
        Perform comprehensive performance attribution analysis.
        """
        try:
            # Get trade data
            trades_df = self.database.get_historical_trades(days=days)
            
            if trades_df.empty:
                self.logger.warning("No trade data available for attribution analysis")
                return {}
            
            attribution_results = {
                'summary': {},
                'by_strategy': {},
                'by_symbol': {},
                'by_regime': {},
                'by_timeframe': {},
                'attribution_breakdown': [],
                'performance_metrics': {},
                'attribution_insights': []
            }
            
            # Basic data validation and preparation
            trades_df = self._prepare_attribution_data(trades_df)
            
            # Calculate summary statistics
            attribution_results['summary'] = self._calculate_summary_statistics(trades_df, portfolio_value)
            
            # Strategy attribution
            attribution_results['by_strategy'] = self._attribute_by_strategy(trades_df)
            
            # Symbol attribution
            attribution_results['by_symbol'] = self._attribute_by_symbol(trades_df)
            
            # Market regime attribution
            attribution_results['by_regime'] = self._attribute_by_market_regime(trades_df)
            
            # Timeframe attribution
            attribution_results['by_timeframe'] = self._attribute_by_timeframe(trades_df)
            
            # Detailed attribution breakdown
            attribution_results['attribution_breakdown'] = self._create_detailed_breakdown(trades_df)
            
            # Advanced performance metrics
            attribution_results['performance_metrics'] = self._calculate_advanced_metrics(trades_df)
            
            # Generate insights
            attribution_results['attribution_insights'] = self._generate_attribution_insights(attribution_results)
            
            # Store attribution results
            if self.database:
                self._store_attribution_results(attribution_results)
            
            self.logger.info(f"Performance attribution completed for {len(trades_df)} trades")
            return attribution_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive attribution: {e}")
            return {}
    
    def _prepare_attribution_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for attribution analysis"""
        df = trades_df.copy()
        
        # Ensure required columns exist
        required_cols = ['symbol', 'timestamp', 'pnl_percent', 'composite_score']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Filter out trades without PnL
        df = df.dropna(subset=['pnl_percent'])
        
        # Add derived columns
        df['abs_pnl'] = abs(df['pnl_percent'])
        df['is_winning'] = df['pnl_percent'] > 0
        df['is_losing'] = df['pnl_percent'] < 0
        
        # Add strategy dominance
        df['dominant_strategy'] = self._determine_dominant_strategy(df)
        
        # Add timeframe classification
        df['timeframe_category'] = self._classify_timeframe(df)
        
        # Add risk-adjusted metrics
        df = self._add_risk_adjusted_metrics(df)
        
        return df
    
    def _determine_dominant_strategy(self, trades_df: pd.DataFrame) -> pd.Series:
        """Determine the dominant strategy for each trade"""
        strategy_columns = ['trend_score', 'mr_score', 'breakout_score', 'ml_score']
        
        # Check which strategy columns exist
        available_strategies = [col for col in strategy_columns if col in trades_df.columns]
        
        if not available_strategies:
            return pd.Series(['unknown'] * len(trades_df), index=trades_df.index)
        
        # Fill missing values with 0
        strategy_scores = trades_df[available_strategies].fillna(0)
        
        # Find the strategy with highest absolute score
        strategy_scores_abs = strategy_scores.abs()
        dominant_strategy_idx = strategy_scores_abs.idxmax(axis=1)
        
        # Map to strategy names
        strategy_mapping = {
            'trend_score': 'trend_following',
            'mr_score': 'mean_reversion', 
            'breakout_score': 'breakout',
            'ml_score': 'ml_prediction'
        }
        
        return dominant_strategy_idx.map(strategy_mapping)
    
    def _classify_timeframe(self, trades_df: pd.DataFrame) -> pd.Series:
        """Classify trades by timeframe based on holding period or signal characteristics"""
        timeframe_categories = []
        
        for idx, trade in trades_df.iterrows():
            # If we have entry/exit timestamps, use holding period
            if 'entry_timestamp' in trade and 'exit_timestamp' in trade:
                if pd.notna(trade['entry_timestamp']) and pd.notna(trade['exit_timestamp']):
                    holding_hours = (trade['exit_timestamp'] - trade['entry_timestamp']).total_seconds() / 3600
                    
                    if holding_hours < 4:
                        timeframe_categories.append('intraday')
                    elif holding_hours < 24:
                        timeframe_categories.append('short_term')
                    elif holding_hours < 168:  # 1 week
                        timeframe_categories.append('medium_term')
                    else:
                        timeframe_categories.append('long_term')
                    continue
            
            # Fallback: Use signal characteristics
            confidence = trade.get('confidence', 50)
            composite_score = trade.get('composite_score', 0)
            
            if confidence > 70 and abs(composite_score) > 30:
                timeframe_categories.append('short_term')
            elif confidence > 60:
                timeframe_categories.append('medium_term')
            else:
                timeframe_categories.append('long_term')
        
        return pd.Series(timeframe_categories, index=trades_df.index)
    
    def _add_risk_adjusted_metrics(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted performance metrics"""
        df = trades_df.copy()
        
        # Calculate Sharpe ratio (using PnL as returns)
        if len(df) > 1:
            sharpe_ratio = df['pnl_percent'].mean() / (df['pnl_percent'].std() + 1e-8) * np.sqrt(365)
            df['sharpe_ratio'] = sharpe_ratio
        else:
            df['sharpe_ratio'] = 0
        
        # Calculate Calmar ratio (max drawdown)
        if 'drawdown' in df.columns:
            df['calmar_ratio'] = df['pnl_percent'].mean() / (df['drawdown'].max() + 1e-8)
        else:
            df['calmar_ratio'] = 0
        
        # Risk-adjusted return
        df['risk_adjusted_return'] = df['pnl_percent'] / (df.get('position_size', 1) + 1e-8)
        
        return df
    
    def _calculate_summary_statistics(self, trades_df: pd.DataFrame, 
                                    portfolio_value: float = None) -> Dict:
        """Calculate overall performance summary statistics"""
        if trades_df.empty:
            return {}
        
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] < 0]
        
        summary = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'breakeven_trades': total_trades - len(winning_trades) - len(losing_trades),
            'win_rate': (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0,
            'total_pnl_percent': trades_df['pnl_percent'].sum(),
            'avg_win': winning_trades['pnl_percent'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl_percent'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': trades_df['pnl_percent'].max(),
            'largest_loss': trades_df['pnl_percent'].min(),
            'profit_factor': abs(winning_trades['pnl_percent'].sum() / losing_trades['pnl_percent'].sum()) 
                            if len(losing_trades) > 0 and losing_trades['pnl_percent'].sum() != 0 else 0,
            'expectancy': (winning_trades['pnl_percent'].mean() * (len(winning_trades)/total_trades) + 
                          losing_trades['pnl_percent'].mean() * (len(losing_trades)/total_trades)) 
                          if total_trades > 0 else 0
        }
        
        # Add portfolio value-based metrics if available
        if portfolio_value:
            summary['total_pnl_usdt'] = summary['total_pnl_percent'] * portfolio_value / 100
            summary['avg_daily_return'] = summary['total_pnl_percent'] / max(1, len(trades_df))
        
        # Risk metrics
        if len(trades_df) > 1:
            summary['sharpe_ratio'] = trades_df['pnl_percent'].mean() / (trades_df['pnl_percent'].std() + 1e-8)
            summary['volatility'] = trades_df['pnl_percent'].std()
        else:
            summary['sharpe_ratio'] = 0
            summary['volatility'] = 0
        
        return summary
    
    def _attribute_by_strategy(self, trades_df: pd.DataFrame) -> Dict:
        """Attribute performance by trading strategy"""
        attribution = {}
        
        for strategy in self.strategy_categories:
            strategy_trades = trades_df[trades_df['dominant_strategy'] == strategy]
            
            if len(strategy_trades) > 0:
                winning_trades = strategy_trades[strategy_trades['pnl_percent'] > 0]
                
                attribution[strategy] = {
                    'trade_count': len(strategy_trades),
                    'total_pnl': strategy_trades['pnl_percent'].sum(),
                    'win_rate': (len(winning_trades) / len(strategy_trades)) * 100,
                    'avg_pnl': strategy_trades['pnl_percent'].mean(),
                    'contribution_percent': (strategy_trades['pnl_percent'].sum() / 
                                           trades_df['pnl_percent'].sum() * 100) 
                                           if trades_df['pnl_percent'].sum() != 0 else 0,
                    'efficiency_ratio': self._calculate_strategy_efficiency(strategy_trades)
                }
            else:
                attribution[strategy] = {
                    'trade_count': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'contribution_percent': 0,
                    'efficiency_ratio': 0
                }
        
        return attribution
    
    def _calculate_strategy_efficiency(self, strategy_trades: pd.DataFrame) -> float:
        """Calculate strategy efficiency (risk-adjusted performance)"""
        if len(strategy_trades) < 2:
            return 0
        
        pnl = strategy_trades['pnl_percent']
        efficiency = pnl.mean() / (pnl.std() + 1e-8)
        return efficiency
    
    def _attribute_by_symbol(self, trades_df: pd.DataFrame) -> Dict:
        """Attribute performance by symbol"""
        attribution = {}
        symbols = trades_df['symbol'].unique()
        
        for symbol in symbols:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            winning_trades = symbol_trades[symbol_trades['pnl_percent'] > 0]
            
            attribution[symbol] = {
                'trade_count': len(symbol_trades),
                'total_pnl': symbol_trades['pnl_percent'].sum(),
                'win_rate': (len(winning_trades) / len(symbol_trades)) * 100,
                'avg_pnl': symbol_trades['pnl_percent'].mean(),
                'contribution_percent': (symbol_trades['pnl_percent'].sum() / 
                                       trades_df['pnl_percent'].sum() * 100) 
                                       if trades_df['pnl_percent'].sum() != 0 else 0,
                'best_trade': symbol_trades['pnl_percent'].max(),
                'worst_trade': symbol_trades['pnl_percent'].min()
            }
        
        return attribution
    
    def _attribute_by_market_regime(self, trades_df: pd.DataFrame) -> Dict:
        """Attribute performance by market regime"""
        attribution = {}
        
        # If market_regime column doesn't exist, try to infer it
        if 'market_regime' not in trades_df.columns:
            trades_df['market_regime'] = 'unknown'
        
        for regime in self.regime_categories + ['unknown']:
            regime_trades = trades_df[trades_df['market_regime'] == regime]
            
            if len(regime_trades) > 0:
                winning_trades = regime_trades[regime_trades['pnl_percent'] > 0]
                
                attribution[regime] = {
                    'trade_count': len(regime_trades),
                    'total_pnl': regime_trades['pnl_percent'].sum(),
                    'win_rate': (len(winning_trades) / len(regime_trades)) * 100,
                    'avg_pnl': regime_trades['pnl_percent'].mean(),
                    'contribution_percent': (regime_trades['pnl_percent'].sum() / 
                                           trades_df['pnl_percent'].sum() * 100) 
                                           if trades_df['pnl_percent'].sum() != 0 else 0,
                    'regime_efficiency': self._calculate_regime_efficiency(regime_trades)
                }
        
        return attribution
    
    def _calculate_regime_efficiency(self, regime_trades: pd.DataFrame) -> float:
        """Calculate performance efficiency in specific market regime"""
        if len(regime_trades) < 2:
            return 0
        
        pnl = regime_trades['pnl_percent']
        efficiency = pnl.mean() / (pnl.std() + 1e-8)
        return efficiency
    
    def _attribute_by_timeframe(self, trades_df: pd.DataFrame) -> Dict:
        """Attribute performance by timeframe"""
        attribution = {}
        
        for timeframe in self.timeframe_categories:
            timeframe_trades = trades_df[trades_df['timeframe_category'] == timeframe]
            
            if len(timeframe_trades) > 0:
                winning_trades = timeframe_trades[timeframe_trades['pnl_percent'] > 0]
                
                attribution[timeframe] = {
                    'trade_count': len(timeframe_trades),
                    'total_pnl': timeframe_trades['pnl_percent'].sum(),
                    'win_rate': (len(winning_trades) / len(timeframe_trades)) * 100,
                    'avg_pnl': timeframe_trades['pnl_percent'].mean(),
                    'contribution_percent': (timeframe_trades['pnl_percent'].sum() / 
                                           trades_df['pnl_percent'].sum() * 100) 
                                           if trades_df['pnl_percent'].sum() != 0 else 0,
                    'holding_efficiency': self._calculate_timeframe_efficiency(timeframe_trades)
                }
        
        return attribution
    
    def _calculate_timeframe_efficiency(self, timeframe_trades: pd.DataFrame) -> float:
        """Calculate efficiency for specific timeframe"""
        if len(timeframe_trades) < 2:
            return 0
        
        # Efficiency metric for timeframe (higher = better)
        pnl = timeframe_trades['pnl_percent']
        efficiency = pnl.mean() / (pnl.std() + 1e-8)
        return efficiency
    
    def _create_detailed_breakdown(self, trades_df: pd.DataFrame) -> List[Dict]:
        """Create detailed attribution breakdown for individual trades"""
        breakdown = []
        
        for idx, trade in trades_df.iterrows():
            trade_attribution = {
                'symbol': trade.get('symbol', 'Unknown'),
                'timestamp': trade.get('timestamp'),
                'pnl_percent': trade.get('pnl_percent', 0),
                'dominant_strategy': trade.get('dominant_strategy', 'unknown'),
                'market_regime': trade.get('market_regime', 'unknown'),
                'timeframe_category': trade.get('timeframe_category', 'unknown'),
                'confidence': trade.get('confidence', 0),
                'composite_score': trade.get('composite_score', 0),
                'risk_adjusted_return': trade.get('risk_adjusted_return', 0)
            }
            breakdown.append(trade_attribution)
        
        return breakdown
    
    def _calculate_advanced_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate advanced performance metrics"""
        if trades_df.empty:
            return {}
        
        metrics = {}
        
        # 1. Consistency metrics
        metrics['consistency_score'] = self._calculate_consistency_score(trades_df)
        
        # 2. Strategy effectiveness
        metrics['strategy_effectiveness'] = self._calculate_strategy_effectiveness(trades_df)
        
        # 3. Regime adaptability
        metrics['regime_adaptability'] = self._calculate_regime_adaptability(trades_df)
        
        # 4. Risk-adjusted performance
        metrics['risk_adjusted_metrics'] = self._calculate_risk_adjusted_metrics(trades_df)
        
        # 5. Drawdown analysis
        metrics['drawdown_analysis'] = self._calculate_drawdown_analysis(trades_df)
        
        # 6. Performance persistence
        metrics['performance_persistence'] = self._calculate_performance_persistence(trades_df)
        
        return metrics
    
    def _calculate_consistency_score(self, trades_df: pd.DataFrame) -> float:
        """Calculate trading consistency score"""
        if len(trades_df) < 5:
            return 0.5
        
        # Calculate win streak analysis
        winning_streaks = []
        current_streak = 0
        
        for pnl in trades_df['pnl_percent']:
            if pnl > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    winning_streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            winning_streaks.append(current_streak)
        
        if winning_streaks:
            streak_consistency = np.mean(winning_streaks) / max(winning_streaks)
        else:
            streak_consistency = 0
        
        # PnL consistency (lower variance = better)
        pnl_std = trades_df['pnl_percent'].std()
        pnl_mean = abs(trades_df['pnl_percent'].mean())
        if pnl_mean > 0:
            pnl_consistency = 1 - min(1, pnl_std / pnl_mean)
        else:
            pnl_consistency = 0
        
        consistency_score = (streak_consistency + pnl_consistency) / 2
        return consistency_score
    
    def _calculate_strategy_effectiveness(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate effectiveness of different strategies"""
        effectiveness = {}
        
        for strategy in self.strategy_categories:
            strategy_trades = trades_df[trades_df['dominant_strategy'] == strategy]
            
            if len(strategy_trades) > 2:
                pnl = strategy_trades['pnl_percent']
                effectiveness[strategy] = {
                    'efficiency': pnl.mean() / (pnl.std() + 1e-8),
                    'reliability': (len(strategy_trades[strategy_trades['pnl_percent'] > 0]) / 
                                   len(strategy_trades)),
                    'avg_confidence': strategy_trades['confidence'].mean() if 'confidence' in strategy_trades else 50
                }
            else:
                effectiveness[strategy] = {
                    'efficiency': 0,
                    'reliability': 0,
                    'avg_confidence': 0
                }
        
        return effectiveness
    
    def _calculate_regime_adaptability(self, trades_df: pd.DataFrame) -> float:
        """Calculate how well the system adapts to different market regimes"""
        if 'market_regime' not in trades_df.columns:
            return 0.5
        
        regime_performance = {}
        
        for regime in trades_df['market_regime'].unique():
            regime_trades = trades_df[trades_df['market_regime'] == regime]
            if len(regime_trades) > 2:
                regime_performance[regime] = regime_trades['pnl_percent'].mean()
        
        if len(regime_performance) < 2:
            return 0.5
        
        # Adaptability score: lower performance variance across regimes = better adaptability
        performance_values = list(regime_performance.values())
        adaptability = 1 - (np.std(performance_values) / (np.mean(np.abs(performance_values)) + 1e-8))
        
        return max(0, adaptability)
    
    def _calculate_risk_adjusted_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk-adjusted performance metrics"""
        if len(trades_df) < 2:
            return {}
        
        pnl = trades_df['pnl_percent']
        
        metrics = {
            'sharpe_ratio': pnl.mean() / (pnl.std() + 1e-8),
            'sortino_ratio': self._calculate_sortino_ratio(trades_df),
            'calmar_ratio': self._calculate_calmar_ratio(trades_df),
            'omega_ratio': self._calculate_omega_ratio(trades_df),
            'var_95': np.percentile(pnl, 5),  # 5% VaR
            'cvar_95': pnl[pnl <= np.percentile(pnl, 5)].mean() if len(pnl[pnl <= np.percentile(pnl, 5)]) > 0 else 0
        }
        
        return metrics
    
    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        pnl = trades_df['pnl_percent']
        downside_returns = pnl[pnl < 0]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_risk = downside_returns.std()
        if downside_risk == 0:
            return 0
        
        return pnl.mean() / downside_risk
    
    def _calculate_calmar_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Calmar ratio (return vs max drawdown)"""
        if 'drawdown' not in trades_df.columns:
            return 0
        
        max_drawdown = trades_df['drawdown'].max()
        if max_drawdown == 0:
            return 0
        
        return trades_df['pnl_percent'].mean() / max_drawdown
    
    def _calculate_omega_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Omega ratio (gain/loss ratio)"""
        pnl = trades_df['pnl_percent']
        threshold = 0  # Benchmark return
        
        gains = pnl[pnl > threshold].sum()
        losses = abs(pnl[pnl < threshold].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0
        
        return gains / losses
    
    def _calculate_drawdown_analysis(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate drawdown analysis"""
        if len(trades_df) < 2:
            return {}
        
        # Calculate running cumulative returns
        cumulative_returns = (1 + trades_df['pnl_percent'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        analysis = {
            'max_drawdown': drawdowns.min(),
            'avg_drawdown': drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0,
            'drawdown_duration_avg': self._calculate_avg_drawdown_duration(drawdowns),
            'recovery_time_avg': self._calculate_avg_recovery_time(drawdowns, cumulative_returns)
        }
        
        return analysis
    
    def _calculate_avg_drawdown_duration(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration"""
        in_drawdown = False
        drawdown_durations = []
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    drawdown_durations.append(current_duration)
                    in_drawdown = False
        
        if in_drawdown:
            drawdown_durations.append(current_duration)
        
        return np.mean(drawdown_durations) if drawdown_durations else 0
    
    def _calculate_avg_recovery_time(self, drawdowns: pd.Series, cumulative_returns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        # Simplified implementation
        recovery_times = []
        in_recovery = False
        recovery_start = 0
        
        for i, (dd, cum_ret) in enumerate(zip(drawdowns, cumulative_returns)):
            if dd == 0 and not in_recovery and i > 0 and drawdowns.iloc[i-1] < 0:
                in_recovery = True
                recovery_start = i
            elif in_recovery and cum_ret >= cumulative_returns.iloc[recovery_start-1]:
                recovery_times.append(i - recovery_start)
                in_recovery = False
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_performance_persistence(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance persistence metrics"""
        if len(trades_df) < 10:
            return {}
        
        # Split data into halves and compare performance
        split_point = len(trades_df) // 2
        first_half = trades_df.iloc[:split_point]
        second_half = trades_df.iloc[split_point:]
        
        persistence = {
            'first_half_performance': first_half['pnl_percent'].mean(),
            'second_half_performance': second_half['pnl_percent'].mean(),
            'performance_correlation': self._calculate_performance_correlation(first_half, second_half),
            'strategy_persistence': self._calculate_strategy_persistence(first_half, second_half)
        }
        
        return persistence
    
    def _calculate_performance_correlation(self, first_half: pd.DataFrame, second_half: pd.DataFrame) -> float:
        """Calculate performance correlation between periods"""
        # Align symbols and calculate correlation of performance
        common_symbols = set(first_half['symbol']).intersection(set(second_half['symbol']))
        
        if len(common_symbols) < 2:
            return 0
        
        first_perf = first_half.groupby('symbol')['pnl_percent'].mean()
        second_perf = second_half.groupby('symbol')['pnl_percent'].mean()
        
        # Align the series
        aligned_first = first_perf.reindex(common_symbols)
        aligned_second = second_perf.reindex(common_symbols)
        
        correlation = aligned_first.corr(aligned_second)
        return correlation if not np.isnan(correlation) else 0
    
    def _calculate_strategy_persistence(self, first_half: pd.DataFrame, second_half: pd.DataFrame) -> float:
        """Calculate strategy performance persistence"""
        strategy_persistence = {}
        
        for strategy in self.strategy_categories:
            first_strategy = first_half[first_half['dominant_strategy'] == strategy]
            second_strategy = second_half[second_half['dominant_strategy'] == strategy]
            
            if len(first_strategy) > 2 and len(second_strategy) > 2:
                first_perf = first_strategy['pnl_percent'].mean()
                second_perf = second_strategy['pnl_percent'].mean()
                
                # Persistence: similar performance in both periods
                if first_perf > 0 and second_perf > 0:
                    strategy_persistence[strategy] = 1
                elif first_perf < 0 and second_perf < 0:
                    strategy_persistence[strategy] = 1
                else:
                    strategy_persistence[strategy] = 0
        
        if strategy_persistence:
            return np.mean(list(strategy_persistence.values()))
        else:
            return 0
    
    def _generate_attribution_insights(self, attribution_results: Dict) -> List[Dict]:
        """Generate actionable insights from attribution analysis"""
        insights = []
        
        try:
            summary = attribution_results.get('summary', {})
            by_strategy = attribution_results.get('by_strategy', {})
            by_symbol = attribution_results.get('by_symbol', {})
            by_regime = attribution_results.get('by_regime', {})
            
            # Insight 1: Best performing strategy
            best_strategy = max(by_strategy.items(), 
                              key=lambda x: x[1].get('efficiency_ratio', 0), 
                              default=(None, {}))
            
            if best_strategy[0]:
                insights.append({
                    'type': 'BEST_STRATEGY',
                    'message': f"{best_strategy[0].replace('_', ' ').title()} is the most efficient strategy",
                    'strategy': best_strategy[0],
                    'efficiency': best_strategy[1].get('efficiency_ratio', 0),
                    'priority': 'HIGH'
                })
            
            # Insight 2: Worst performing strategy
            worst_strategy = min(by_strategy.items(), 
                               key=lambda x: x[1].get('efficiency_ratio', 1), 
                               default=(None, {}))
            
            if worst_strategy[0] and worst_strategy[1].get('efficiency_ratio', 1) < 0:
                insights.append({
                    'type': 'UNDERPERFORMING_STRATEGY',
                    'message': f"Consider reducing exposure to {worst_strategy[0].replace('_', ' ')}",
                    'strategy': worst_strategy[0],
                    'efficiency': worst_strategy[1].get('efficiency_ratio', 0),
                    'priority': 'MEDIUM'
                })
            
            # Insight 3: Best performing symbol
            best_symbol = max(by_symbol.items(), 
                            key=lambda x: x[1].get('contribution_percent', 0), 
                            default=(None, {}))
            
            if best_symbol[0] and best_symbol[1].get('contribution_percent', 0) > 20:
                insights.append({
                    'type': 'TOP_CONTRIBUTOR',
                    'message': f"{best_symbol[0]} contributes {best_symbol[1]['contribution_percent']:.1f}% of total PnL",
                    'symbol': best_symbol[0],
                    'contribution': best_symbol[1].get('contribution_percent', 0),
                    'priority': 'MEDIUM'
                })
            
            # Insight 4: Overall performance assessment
            win_rate = summary.get('win_rate', 0)
            profit_factor = summary.get('profit_factor', 0)
            
            if win_rate > 60 and profit_factor > 1.5:
                insights.append({
                    'type': 'STRONG_PERFORMANCE',
                    'message': f"Excellent performance: {win_rate:.1f}% win rate, {profit_factor:.2f} profit factor",
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'priority': 'INFO'
                })
            elif win_rate < 40 or profit_factor < 1.0:
                insights.append({
                    'type': 'PERFORMANCE_CONCERN',
                    'message': f"Performance concerns: {win_rate:.1f}% win rate, {profit_factor:.2f} profit factor",
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'priority': 'HIGH'
                })
            
            # Insight 5: Regime performance insights
            for regime, data in by_regime.items():
                if data.get('trade_count', 0) > 5 and data.get('avg_pnl', 0) < -0.5:
                    insights.append({
                        'type': 'REGIME_WEAKNESS',
                        'message': f"Poor performance in {regime} regime: {data['avg_pnl']:.2f}% average PnL",
                        'regime': regime,
                        'performance': data['avg_pnl'],
                        'priority': 'MEDIUM'
                    })
        
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _store_attribution_results(self, attribution_results: Dict):
        """Store attribution results in database"""
        try:
            if self.database:
                self.database.store_system_event(
                    "PERFORMANCE_ATTRIBUTION",
                    {
                        'summary': attribution_results.get('summary', {}),
                        'timestamp': datetime.now(),
                        'insights_count': len(attribution_results.get('attribution_insights', [])),
                        'total_trades': attribution_results.get('summary', {}).get('total_trades', 0)
                    },
                    "INFO",
                    "Performance Analysis"
                )
        except Exception as e:
            self.logger.error(f"Error storing attribution results: {e}")
    
    def generate_attribution_report(self, days: int = 30) -> Dict:
        """Generate a comprehensive attribution report"""
        attribution = self.perform_comprehensive_attribution(days)
        
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'executive_summary': self._generate_executive_summary(attribution),
            'detailed_analysis': attribution,
            'recommendations': self._generate_recommendations(attribution)
        }
        
        return report
    
    def _generate_executive_summary(self, attribution: Dict) -> Dict:
        """Generate executive summary of attribution results"""
        summary = attribution.get('summary', {})
        
        executive_summary = {
            'overall_performance': 'STRONG' if summary.get('profit_factor', 0) > 1.5 else 'MODERATE' if summary.get('profit_factor', 0) > 1.0 else 'WEAK',
            'key_strengths': [],
            'key_weaknesses': [],
            'top_contributors': [],
            'main_insights': []
        }
        
        # Add strengths and weaknesses based on insights
        insights = attribution.get('attribution_insights', [])
        for insight in insights:
            if insight.get('priority') == 'HIGH' and 'UNDERPERFORMING' in insight.get('type', ''):
                executive_summary['key_weaknesses'].append(insight.get('message', ''))
            elif insight.get('priority') == 'HIGH':
                executive_summary['key_strengths'].append(insight.get('message', ''))
        
        return executive_summary
    
    def _generate_recommendations(self, attribution: Dict) -> List[Dict]:
        """Generate actionable recommendations based on attribution analysis"""
        recommendations = []
        
        by_strategy = attribution.get('by_strategy', {})
        insights = attribution.get('attribution_insights', [])
        
        # Strategy allocation recommendations
        strategy_efficiencies = {}
        for strategy, data in by_strategy.items():
            if data.get('trade_count', 0) > 5:
                strategy_efficiencies[strategy] = data.get('efficiency_ratio', 0)
        
        if strategy_efficiencies:
            best_strategy = max(strategy_efficiencies, key=strategy_efficiencies.get)
            worst_strategy = min(strategy_efficiencies, key=strategy_efficiencies.get)
            
            if strategy_efficiencies[best_strategy] > strategy_efficiencies[worst_strategy] * 2:
                recommendations.append({
                    'type': 'STRATEGY_ALLOCATION',
                    'message': f"Consider increasing weight for {best_strategy} and decreasing for {worst_strategy}",
                    'action': 'ADJUST_STRATEGY_WEIGHTS',
                    'confidence': 'HIGH',
                    'impact': 'MEDIUM'
                })
        
        # Risk management recommendations
        risk_metrics = attribution.get('performance_metrics', {}).get('risk_adjusted_metrics', {})
        if risk_metrics.get('var_95', 0) < -5:  # 5% VaR worse than -5%
            recommendations.append({
                'type': 'RISK_MANAGEMENT',
                'message': "High loss potential detected, consider reducing position sizes",
                'action': 'REDUCE_POSITION_SIZES',
                'confidence': 'MEDIUM',
                'impact': 'HIGH'
            })
        
        return recommendations