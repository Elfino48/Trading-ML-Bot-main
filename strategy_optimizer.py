import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from trading_database import TradingDatabase

class StrategyOptimizer:
    """
    Dynamic strategy optimization based on market conditions and performance
    Adjusts strategy weights and parameters in real-time
    """
    
    def __init__(self, trading_database: TradingDatabase):
        self.database = trading_database
        self.optimization_history = []
        self.current_weights = {}
        self.market_regime = "neutral"
        self.volatility_regime = "normal"
        
        self.logger = logging.getLogger('StrategyOptimizer')
        
        # Base strategy weights for different market regimes
        self.regime_weights = {
            "bull_trend": {
                'trend_following': 0.45,
                'mean_reversion': 0.25,
                'breakout': 0.20,
                'ml_prediction': 0.10
            },
            "bear_trend": {
                'trend_following': 0.35,
                'mean_reversion': 0.35,
                'breakout': 0.20,
                'ml_prediction': 0.10
            },
            "ranging": {
                'trend_following': 0.20,
                'mean_reversion': 0.50,
                'breakout': 0.20,
                'ml_prediction': 0.10
            },
            "high_volatility": {
                'trend_following': 0.30,
                'mean_reversion': 0.40,
                'breakout': 0.20,
                'ml_prediction': 0.10
            },
            "low_volatility": {
                'trend_following': 0.25,
                'mean_reversion': 0.35,
                'breakout': 0.30,
                'ml_prediction': 0.10
            }
        }
        
        print("🎯 Strategy Optimizer initialized")
    
    def optimize_weights(self, market_regime: str, volatility: float, 
                        recent_performance: Dict, aggressiveness: str) -> Dict[str, float]:
        """
        Optimize strategy weights based on current market conditions and performance
        
        Args:
            market_regime: Current market regime
            volatility: Current market volatility
            recent_performance: Recent trading performance
            aggressiveness: Trading aggressiveness level
            
        Returns:
            Optimized strategy weights
        """
        # Get base weights for current regime
        base_weights = self._get_base_weights(market_regime, volatility)
        
        # Adjust based on recent performance
        performance_weights = self._adjust_for_performance(base_weights, recent_performance)
        
        # Adjust for aggressiveness
        final_weights = self._adjust_for_aggressiveness(performance_weights, aggressiveness)
        
        # Store optimization record
        optimization_record = {
            'timestamp': datetime.now(),
            'market_regime': market_regime,
            'volatility': volatility,
            'aggressiveness': aggressiveness,
            'weights': final_weights.copy(),
            'performance_metrics': recent_performance
        }
        self.optimization_history.append(optimization_record)
        
        self.current_weights = final_weights
        self.logger.info(f"Strategy weights optimized: {final_weights}")
        
        return final_weights
    
    def _get_base_weights(self, market_regime: str, volatility: float) -> Dict[str, float]:
        """
        Get base weights for current market regime and volatility
        """
        # Determine volatility regime
        if volatility > 0.03:  # 3% volatility threshold
            volatility_regime = "high_volatility"
        elif volatility < 0.01:  # 1% volatility threshold
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal"
        
        # Get weights for primary regime
        if market_regime in self.regime_weights:
            base_weights = self.regime_weights[market_regime].copy()
        else:
            base_weights = self.regime_weights["bull_trend"].copy()
        
        # Blend with volatility regime weights if different
        if volatility_regime != "normal" and volatility_regime in self.regime_weights:
            volatility_weights = self.regime_weights[volatility_regime]
            # 70% primary regime, 30% volatility regime
            for strategy in base_weights:
                base_weights[strategy] = (
                    base_weights[strategy] * 0.7 + 
                    volatility_weights.get(strategy, base_weights[strategy]) * 0.3
                )
        
        return base_weights
    
    def _adjust_for_performance(self, base_weights: Dict[str, float], 
                               recent_performance: Dict) -> Dict[str, float]:
        """
        Adjust weights based on recent strategy performance
        """
        if not recent_performance:
            return base_weights
        
        adjusted_weights = base_weights.copy()
        
        # Get strategy performance from recent trades
        strategy_performance = self._analyze_strategy_performance()
        
        for strategy, performance in strategy_performance.items():
            if strategy in adjusted_weights:
                # Increase weight for better performing strategies
                performance_factor = self._calculate_performance_factor(performance)
                adjusted_weights[strategy] *= performance_factor
        
        # Normalize weights to sum to 1
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _adjust_for_aggressiveness(self, weights: Dict[str, float], 
                                  aggressiveness: str) -> Dict[str, float]:
        """
        Adjust weights based on trading aggressiveness
        """
        adjusted_weights = weights.copy()
        
        aggressiveness_factors = {
            "conservative": {
                'trend_following': 1.1,    # More trend following
                'mean_reversion': 0.9,     # Less mean reversion
                'breakout': 0.8,           # Less breakout
                'ml_prediction': 1.0       # Neutral on ML
            },
            "moderate": {
                'trend_following': 1.0,
                'mean_reversion': 1.0,
                'breakout': 1.0,
                'ml_prediction': 1.0
            },
            "aggressive": {
                'trend_following': 0.9,    # Less trend following
                'mean_reversion': 1.1,     # More mean reversion
                'breakout': 1.2,           # More breakout
                'ml_prediction': 0.9       # Slightly less ML
            },
            "high": {
                'trend_following': 0.8,    # Much less trend following
                'mean_reversion': 1.2,     # Much more mean reversion
                'breakout': 1.3,           # Much more breakout
                'ml_prediction': 0.7       # Less ML
            }
        }
        
        factors = aggressiveness_factors.get(aggressiveness, aggressiveness_factors["moderate"])
        
        for strategy, factor in factors.items():
            if strategy in adjusted_weights:
                adjusted_weights[strategy] *= factor
        
        # Normalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _analyze_strategy_performance(self, days: int = 7) -> Dict[str, Dict]:
        """
        Analyze recent performance of each strategy
        """
        try:
            trades_df = self.database.get_historical_trades(days=days)
            
            if trades_df.empty:
                return {}
            
            strategy_performance = {}
            
            # Group trades by inferred strategy (simplified)
            for strategy in ['trend_following', 'mean_reversion', 'breakout', 'ml_prediction']:
                strategy_trades = self._filter_trades_by_strategy(trades_df, strategy)
                
                if len(strategy_trades) > 0:
                    performance = {
                        'trade_count': len(strategy_trades),
                        'win_rate': (strategy_trades['pnl_percent'] > 0).mean() * 100,
                        'avg_pnl': strategy_trades['pnl_percent'].mean(),
                        'total_pnl': strategy_trades['pnl_percent'].sum(),
                        'success_rate': strategy_trades['success'].mean() * 100
                    }
                    strategy_performance[strategy] = performance
            
            return strategy_performance
            
        except Exception as e:
            self.logger.error(f"Failed to analyze strategy performance: {e}")
            return {}
    
    def _filter_trades_by_strategy(self, trades_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Filter trades by strategy type (simplified heuristic)
        """
        if trades_df.empty:
            return pd.DataFrame()
        
        # Simplified strategy identification based on trade characteristics
        if strategy == 'trend_following':
            # Trades with high composite score and strong trend signals
            return trades_df[
                (trades_df['composite_score'] > 20) & 
                (trades_df['confidence'] > 60)
            ]
        elif strategy == 'mean_reversion':
            # Trades with moderate scores and mean reversion characteristics
            return trades_df[
                (trades_df['composite_score'].between(10, 30)) &
                (trades_df['confidence'] > 40)
            ]
        elif strategy == 'breakout':
            # Trades with high volatility and breakout characteristics
            return trades_df[
                (trades_df['composite_score'] > 15) &
                (trades_df['risk_reward_ratio'] > 2.0)
            ]
        elif strategy == 'ml_prediction':
            # Trades where ML confidence was high
            return trades_df[
                trades_df['confidence'] > 50  # Simplified proxy for ML influence
            ]
        else:
            return pd.DataFrame()
    
    def _calculate_performance_factor(self, performance: Dict) -> float:
        """
        Calculate performance adjustment factor for a strategy
        """
        if performance['trade_count'] < 3:
            return 1.0  # Not enough data
        
        win_rate = performance['win_rate']
        avg_pnl = performance['avg_pnl']
        success_rate = performance['success_rate']
        
        # Calculate composite performance score
        performance_score = (
            (win_rate / 100) * 0.4 +          # 40% weight to win rate
            (max(0, avg_pnl) / 5) * 0.4 +     # 40% weight to average PnL (capped at 5%)
            (success_rate / 100) * 0.2        # 20% weight to success rate
        )
        
        # Convert to adjustment factor (0.8 to 1.2 range)
        adjustment_factor = 0.8 + (performance_score * 0.4)
        
        return max(0.5, min(2.0, adjustment_factor))  # Clamp between 0.5 and 2.0
    
    def analyze_market_regimes(self, symbol: str, historical_data: pd.DataFrame) -> Dict:
        """
        Analyze current market regime based on price action and indicators
        """
        if len(historical_data) < 50:
            return {'regime': 'neutral', 'confidence': 0.5}
        
        try:
            close_prices = historical_data['close'].astype(float)
            high_prices = historical_data['high'].astype(float)
            low_prices = historical_data['low'].astype(float)
            
            regime_scores = {
                'bull_trend': 0,
                'bear_trend': 0,
                'ranging': 0
            }
            
            # 1. Trend analysis using moving averages
            sma_20 = close_prices.rolling(20).mean()
            sma_50 = close_prices.rolling(50).mean()
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                current_sma_20 = sma_20.iloc[-1]
                current_sma_50 = sma_50.iloc[-1]
                
                if current_sma_20 > current_sma_50:
                    regime_scores['bull_trend'] += 0.3
                else:
                    regime_scores['bear_trend'] += 0.3
            
            # 2. Price momentum
            returns_5 = close_prices.pct_change(5)
            returns_20 = close_prices.pct_change(20)
            
            if len(returns_5) > 0 and len(returns_20) > 0:
                mom_5 = returns_5.iloc[-1]
                mom_20 = returns_20.iloc[-1]
                
                if mom_5 > 0.02 and mom_20 > 0.05:  # Strong upward momentum
                    regime_scores['bull_trend'] += 0.4
                elif mom_5 < -0.02 and mom_20 < -0.05:  # Strong downward momentum
                    regime_scores['bear_trend'] += 0.4
                elif abs(mom_5) < 0.01 and abs(mom_20) < 0.02:  # Low momentum
                    regime_scores['ranging'] += 0.4
            
            # 3. Volatility analysis
            volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1] if len(close_prices) > 20 else 0
            atr = (high_prices - low_prices).rolling(14).mean().iloc[-1] if len(high_prices) > 14 else 0
            
            if volatility_20 > 0.03:  # High volatility
                # In high volatility, trends are more significant
                regime_scores['bull_trend'] *= 1.2
                regime_scores['bear_trend'] *= 1.2
            elif volatility_20 < 0.01:  # Low volatility
                regime_scores['ranging'] *= 1.3
            
            # Determine dominant regime
            best_regime = max(regime_scores, key=regime_scores.get)
            best_score = regime_scores[best_regime]
            
            # Calculate confidence
            total_score = sum(regime_scores.values())
            confidence = best_score / total_score if total_score > 0 else 0.5
            
            result = {
                'regime': best_regime,
                'confidence': confidence,
                'scores': regime_scores,
                'volatility': volatility_20,
                'timestamp': datetime.now()
            }
            
            self.market_regime = best_regime
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze market regime: {e}")
            return {'regime': 'neutral', 'confidence': 0.5}
    
    def calculate_volatility(self, historical_data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate current market volatility
        """
        if len(historical_data) < period:
            return 0.02  # Default moderate volatility
        
        try:
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.rolling(period).std().iloc[-1]
            
            # Annualize if needed (assuming daily data)
            annualized_volatility = volatility * np.sqrt(365)
            
            return annualized_volatility if not np.isnan(annualized_volatility) else 0.02
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volatility: {e}")
            return 0.02
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary of current optimization state
        """
        recent_optimizations = self.optimization_history[-5:] if self.optimization_history else []
        
        return {
            'current_weights': self.current_weights,
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'recent_optimizations': recent_optimizations,
            'optimization_count': len(self.optimization_history)
        }
    
    def optimize_parameters(self, symbol: str, historical_data: pd.DataFrame) -> Dict:
        """
        Optimize trading parameters for specific symbol
        """
        if len(historical_data) < 100:
            return {}
        
        try:
            # Analyze symbol-specific characteristics
            volatility = self.calculate_volatility(historical_data)
            regime_analysis = self.analyze_market_regimes(symbol, historical_data)
            
            # Optimize parameters based on characteristics
            optimized_params = {
                'confidence_threshold': self._optimize_confidence_threshold(volatility),
                'position_size_multiplier': self._optimize_position_size(volatility),
                'stop_loss_multiplier': self._optimize_stop_loss(volatility),
                'take_profit_multiplier': self._optimize_take_profit(volatility, regime_analysis['regime'])
            }
            
            self.logger.info(f"Optimized parameters for {symbol}: {optimized_params}")
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Failed to optimize parameters for {symbol}: {e}")
            return {}
    
    def _optimize_confidence_threshold(self, volatility: float) -> float:
        """Optimize confidence threshold based on volatility"""
        # Higher volatility -> higher confidence required
        base_threshold = 25.0
        volatility_adjustment = max(0, (volatility - 0.02) * 500)  # Adjust for volatility
        return base_threshold + volatility_adjustment
    
    def _optimize_position_size(self, volatility: float) -> float:
        """Optimize position size multiplier based on volatility"""
        # Higher volatility -> smaller positions
        if volatility > 0.04:
            return 0.7  # 30% reduction
        elif volatility > 0.03:
            return 0.8  # 20% reduction
        elif volatility < 0.01:
            return 1.2  # 20% increase for low volatility
        else:
            return 1.0  # No change
    
    def _optimize_stop_loss(self, volatility: float) -> float:
        """Optimize stop loss multiplier based on volatility"""
        # Higher volatility -> wider stops
        if volatility > 0.04:
            return 1.3  # 30% wider stops
        elif volatility > 0.03:
            return 1.2  # 20% wider stops
        elif volatility < 0.01:
            return 0.8  # 20% tighter stops for low volatility
        else:
            return 1.0  # No change
    
    def _optimize_take_profit(self, volatility: float, regime: str) -> float:
        """Optimize take profit multiplier based on volatility and regime"""
        base_multiplier = 1.0
        
        # Adjust for volatility
        if volatility > 0.04:
            base_multiplier *= 1.3  # Wider targets in high volatility
        elif volatility < 0.01:
            base_multiplier *= 0.8  # Tighter targets in low volatility
        
        # Adjust for regime
        if regime == 'bull_trend':
            base_multiplier *= 1.2  # Let profits run in bull trends
        elif regime == 'bear_trend':
            base_multiplier *= 0.9  # Take profits quicker in bear trends
        
        return base_multiplier
    
    def get_performance_analytics(self, days: int = 30) -> Dict:
        """
        Get comprehensive performance analytics
        """
        try:
            trades_df = self.database.get_historical_trades(days=days)
            
            if trades_df.empty:
                return {'error': 'No trade data available'}
            
            analytics = {
                'summary': self._calculate_performance_summary(trades_df),
                'strategy_performance': self._analyze_strategy_performance(days),
                'time_based_analysis': self._analyze_time_performance(trades_df),
                'symbol_analysis': self._analyze_symbol_performance(trades_df)
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance analytics: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_summary(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate overall performance summary"""
        if trades_df.empty:
            return {}
        
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] < 0]
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades_df)) * 100,
            'avg_win': winning_trades['pnl_percent'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl_percent'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl_percent'].sum() / losing_trades['pnl_percent'].sum()) 
                            if len(losing_trades) > 0 and losing_trades['pnl_percent'].sum() != 0 else 0,
            'total_pnl_percent': trades_df['pnl_percent'].sum(),
            'best_trade': trades_df['pnl_percent'].max(),
            'worst_trade': trades_df['pnl_percent'].min()
        }
    
    def _analyze_time_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by time of day and day of week"""
        if trades_df.empty:
            return {}
        
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        trades_df['day_of_week'] = pd.to_datetime(trades_df['timestamp']).dt.day_name()
        
        hour_performance = trades_df.groupby('hour')['pnl_percent'].agg(['mean', 'count']).to_dict()
        day_performance = trades_df.groupby('day_of_week')['pnl_percent'].agg(['mean', 'count']).to_dict()
        
        return {
            'by_hour': hour_performance,
            'by_day': day_performance
        }
    
    def _analyze_symbol_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by symbol"""
        if trades_df.empty:
            return {}
        
        symbol_stats = {}
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            winning = symbol_trades[symbol_trades['pnl_percent'] > 0]
            
            symbol_stats[symbol] = {
                'trade_count': len(symbol_trades),
                'win_rate': (len(winning) / len(symbol_trades)) * 100,
                'avg_pnl': symbol_trades['pnl_percent'].mean(),
                'total_pnl': symbol_trades['pnl_percent'].sum(),
                'best_trade': symbol_trades['pnl_percent'].max(),
                'worst_trade': symbol_trades['pnl_percent'].min()
            }
        
        return symbol_stats


# Example usage and testing
if __name__ == "__main__":
    # Mock database for testing
    class MockDatabase:
        def get_historical_trades(self, days):
            # Return mock trade data
            return pd.DataFrame({
                'symbol': ['BTCUSDT', 'ETHUSDT', 'BTCUSDT', 'ETHUSDT'],
                'timestamp': [datetime.now() - timedelta(hours=i) for i in range(4)],
                'action': ['BUY', 'SELL', 'BUY', 'SELL'],
                'pnl_percent': [2.5, -1.2, 3.1, 0.8],
                'composite_score': [25, 18, 30, 22],
                'confidence': [65, 55, 70, 60],
                'success': [True, True, True, True],
                'risk_reward_ratio': [2.1, 1.8, 2.5, 2.0]
            })
    
    db = MockDatabase()
    optimizer = StrategyOptimizer(db)
    
    # Test optimization
    weights = optimizer.optimize_weights(
        market_regime="bull_trend",
        volatility=0.025,
        recent_performance={'win_rate': 65, 'avg_pnl': 1.5},
        aggressiveness="moderate"
    )
    
    print(f"Optimized weights: {weights}")
    
    # Test market regime analysis
    mock_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(100) * 0.01) + 100,
        'high': np.cumsum(np.random.randn(100) * 0.01) + 101,
        'low': np.cumsum(np.random.randn(100) * 0.01) + 99
    })
    
    regime = optimizer.analyze_market_regimes("BTCUSDT", mock_data)
    print(f"Market regime: {regime}")
    
    # Test performance analytics
    analytics = optimizer.get_performance_analytics(days=7)
    print(f"Performance analytics: {analytics.keys()}")