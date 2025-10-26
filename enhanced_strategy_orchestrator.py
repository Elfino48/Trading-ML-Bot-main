from enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
from ml_predictor import MLPredictor
from advanced_risk_manager import AdvancedRiskManager
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize

class EnhancedStrategyOrchestrator:
    def __init__(self, bybit_client, aggressiveness: str = "conservative", error_handler=None, database=None):
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.ml_predictor = MLPredictor(error_handler, database)
        self.risk_manager = AdvancedRiskManager(bybit_client, aggressiveness)
        self.aggressiveness = aggressiveness
        self.error_handler = error_handler
        self.database = database
        self.regime_specific_weights = {}
        self.sentiment_indicators = {}
        self.portfolio_signals = {}
        self.correlation_matrix = None
        self.asset_volatilities = {}
        
        self._set_aggressiveness_weights()
        
        print(f"🎯 Strategy Orchestrator set to: {self.aggressiveness.upper()} mode")
    
    def set_error_handler(self, error_handler):
        self.error_handler = error_handler
        self.ml_predictor.set_error_handler(error_handler)
        
    def set_database(self, database):
        self.database = database
        self.ml_predictor.set_database(database)
    
    def _set_aggressiveness_weights(self):
        if self.aggressiveness == "conservative":
            self.strategy_weights = {
                'trend_following': 0.40,
                'mean_reversion': 0.30,
                'breakout': 0.20,
                'ml_prediction': 0.10
            }
            self.buy_threshold = 20
            self.sell_threshold = -20
            self.strong_threshold = 40
            
        elif self.aggressiveness == "moderate":
            self.strategy_weights = {
                'trend_following': 0.25,
                'mean_reversion': 0.35,
                'breakout': 0.20,
                'ml_prediction': 0.20
            }
            self.buy_threshold = 15
            self.sell_threshold = -15
            self.strong_threshold = 30
            
        elif self.aggressiveness == "aggressive":
            self.strategy_weights = {
                'trend_following': 0.20,
                'mean_reversion': 0.40,
                'breakout': 0.25,
                'ml_prediction': 0.15
            }
            self.buy_threshold = 10
            self.sell_threshold = -10
            self.strong_threshold = 25
            
        elif self.aggressiveness == "high":
            self.strategy_weights = {
                'trend_following': 0.15,
                'mean_reversion': 0.45,
                'breakout': 0.30,
                'ml_prediction': 0.10
            }
            self.buy_threshold = 5
            self.sell_threshold = -5
            self.strong_threshold = 20
            
        else:
            self.strategy_weights = {
                'trend_following': 0.40,
                'mean_reversion': 0.30,
                'breakout': 0.20,
                'ml_prediction': 0.10
            }
            self.buy_threshold = 20
            self.sell_threshold = -20
            self.strong_threshold = 40
        
        print(f"   • Strategy Weights: {self.strategy_weights}")
        print(f"   • Buy Threshold: {self.buy_threshold}, Sell Threshold: {self.sell_threshold}")

    def analyze_portfolio_signals(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        try:
            portfolio_scores = {}
            total_score = 0
            signal_count = 0
            
            for symbol, data in symbol_data.items():
                if len(data) < 100:
                    continue
                    
                indicators = self.technical_analyzer.calculate_regime_indicators(data)
                signals = self.technical_analyzer.generate_enhanced_signals(indicators)
                ml_result = self.ml_predictor.predict(symbol, data)
                mtf_signals = self._multi_timeframe_analysis(data)
                
                composite_score = self._calculate_aggressive_composite_score(
                    signals, ml_result, mtf_signals, self.aggressiveness
                )
                
                portfolio_scores[symbol] = composite_score
                total_score += composite_score
                signal_count += 1
            
            if signal_count > 0:
                portfolio_avg = total_score / signal_count
                portfolio_scores['PORTFOLIO_AVG'] = portfolio_avg
                
                strong_buy_count = sum(1 for score in portfolio_scores.values() if score > self.strong_threshold)
                strong_sell_count = sum(1 for score in portfolio_scores.values() if score < -self.strong_threshold)
                
                portfolio_scores['STRONG_BUY_RATIO'] = strong_buy_count / signal_count
                portfolio_scores['STRONG_SELL_RATIO'] = strong_sell_count / signal_count
                portfolio_scores['MARKET_SENTIMENT'] = 'BULLISH' if portfolio_avg > 0 else 'BEARISH'
            
            self.portfolio_signals = portfolio_scores
            return portfolio_scores
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "portfolio_analysis")
            return {}

    def calculate_risk_parity_weights(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame], 
                                    portfolio_value: float) -> Dict[str, Dict]:
        try:
            volatilities = self._calculate_asset_volatilities(symbols, historical_data)
            correlations = self._calculate_correlation_matrix(symbols, historical_data)
            
            target_risk_contributions = self._calculate_target_risk_contributions(symbols, volatilities)
            
            initial_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
            
            def risk_parity_objective(weights_array):
                weights_dict = {symbol: weights_array[i] for i, symbol in enumerate(symbols)}
                portfolio_vol = self._calculate_portfolio_volatility(weights_dict, volatilities, correlations)
                risk_contributions = self._calculate_risk_contributions(weights_dict, volatilities, correlations, portfolio_vol)
                
                target_risk_array = np.array([target_risk_contributions[symbol] for symbol in symbols])
                actual_risk_array = np.array([risk_contributions[symbol] for symbol in symbols])
                
                return np.sum((actual_risk_array - target_risk_array) ** 2)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            bounds = [(0.01, 0.5) for _ in symbols]
            
            result = minimize(risk_parity_objective, 
                           list(initial_weights.values()), 
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=constraints)
            
            optimal_weights = {symbol: result.x[i] for i, symbol in enumerate(symbols)}
            
            position_sizes = self._calculate_risk_parity_positions(optimal_weights, portfolio_value, symbols, historical_data)
            
            if self.database:
                self.database.store_system_event(
                    "RISK_PARITY_OPTIMIZATION",
                    {
                        'symbols': symbols,
                        'volatilities': volatilities,
                        'optimal_weights': optimal_weights,
                        'position_sizes': position_sizes,
                        'portfolio_value': portfolio_value
                    },
                    "INFO",
                    "Risk Management"
                )
            
            return {
                'weights': optimal_weights,
                'positions': position_sizes,
                'volatilities': volatilities,
                'correlation_matrix': correlations
            }
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "risk_parity")
            return {}

    def _calculate_asset_volatilities(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        volatilities = {}
        for symbol in symbols:
            if symbol in historical_data and len(historical_data[symbol]) >= 20:
                returns = historical_data[symbol]['close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1]
                volatilities[symbol] = volatility if not np.isnan(volatility) else 0.02
            else:
                volatilities[symbol] = 0.02
        
        self.asset_volatilities = volatilities
        return volatilities

    def _calculate_correlation_matrix(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        try:
            returns_data = {}
            min_length = 20
            
            for symbol in symbols:
                if symbol in historical_data and len(historical_data[symbol]) >= min_length:
                    returns = historical_data[symbol]['close'].pct_change().dropna()
                    if len(returns) >= min_length:
                        returns_data[symbol] = returns.tail(min_length)
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "correlation_analysis")
            return pd.DataFrame()

    def _calculate_target_risk_contributions(self, symbols: List[str], volatilities: Dict[str, float]) -> Dict[str, float]:
        total_vol_inverse = sum(1.0 / max(vol, 0.001) for vol in volatilities.values())
        target_contributions = {}
        
        for symbol in symbols:
            vol = volatilities.get(symbol, 0.02)
            target_contributions[symbol] = (1.0 / max(vol, 0.001)) / total_vol_inverse
        
        return target_contributions

    def _calculate_portfolio_volatility(self, weights: Dict[str, float], volatilities: Dict[str, float], 
                                      correlations: pd.DataFrame) -> float:
        try:
            symbols = list(weights.keys())
            if len(symbols) < 2:
                return volatilities.get(symbols[0], 0.02) if symbols else 0.02
            
            variance = 0
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i == j:
                        variance += weights[sym1] ** 2 * volatilities[sym1] ** 2
                    else:
                        corr = correlations.loc[sym1, sym2] if not correlations.empty else 0.0
                        variance += weights[sym1] * weights[sym2] * volatilities[sym1] * volatilities[sym2] * corr
            
            return np.sqrt(max(variance, 0))
            
        except:
            return 0.02

    def _calculate_risk_contributions(self, weights: Dict[str, float], volatilities: Dict[str, float], 
                                    correlations: pd.DataFrame, portfolio_vol: float) -> Dict[str, float]:
        risk_contributions = {}
        symbols = list(weights.keys())
        
        for symbol in symbols:
            marginal_risk = 0
            for other_symbol in symbols:
                if symbol == other_symbol:
                    marginal_risk += weights[symbol] * volatilities[symbol] ** 2
                else:
                    corr = correlations.loc[symbol, other_symbol] if not correlations.empty else 0.0
                    marginal_risk += weights[other_symbol] * volatilities[symbol] * volatilities[other_symbol] * corr
            
            risk_contributions[symbol] = weights[symbol] * marginal_risk / max(portfolio_vol, 0.001)
        
        return risk_contributions

    def _calculate_risk_parity_positions(self, weights: Dict[str, float], portfolio_value: float, 
                                       symbols: List[str], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        positions = {}
        max_position_value = portfolio_value * 0.2
        
        for symbol in symbols:
            weight = weights.get(symbol, 0)
            position_value = portfolio_value * weight
            
            if position_value > max_position_value:
                position_value = max_position_value
            
            if symbol in historical_data and not historical_data[symbol].empty:
                current_price = historical_data[symbol]['close'].iloc[-1]
                quantity = position_value / current_price
                
                positions[symbol] = {
                    'size_usdt': position_value,
                    'quantity': quantity,
                    'weight': weight,
                    'current_price': current_price
                }
        
        return positions

    def analyze_cross_asset_correlations(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]) -> Dict:
        try:
            if len(symbols) < 2:
                return {}
            
            correlation_matrix = self._calculate_correlation_matrix(symbols, historical_data)
            
            if correlation_matrix.empty:
                return {}
            
            correlation_analysis = {
                'correlation_matrix': correlation_matrix,
                'highly_correlated_pairs': [],
                'diversification_opportunities': [],
                'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                'correlation_clusters': self._identify_correlation_clusters(correlation_matrix, symbols)
            }
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i < j:
                        corr = correlation_matrix.loc[sym1, sym2]
                        if abs(corr) > 0.7:
                            correlation_analysis['highly_correlated_pairs'].append({
                                'pair': f"{sym1}-{sym2}",
                                'correlation': corr,
                                'type': 'POSITIVE' if corr > 0 else 'NEGATIVE'
                            })
                        elif abs(corr) < 0.2:
                            correlation_analysis['diversification_opportunities'].append({
                                'pair': f"{sym1}-{sym2}",
                                'correlation': corr
                            })
            
            return correlation_analysis
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "correlation_analysis")
            return {}

    def _identify_correlation_clusters(self, correlation_matrix: pd.DataFrame, symbols: List[str]) -> List[Dict]:
        try:
            clusters = []
            visited = set()
            
            for symbol in symbols:
                if symbol not in visited:
                    cluster = [symbol]
                    visited.add(symbol)
                    
                    for other_symbol in symbols:
                        if other_symbol not in visited:
                            corr = correlation_matrix.loc[symbol, other_symbol]
                            if abs(corr) > 0.6:
                                cluster.append(other_symbol)
                                visited.add(other_symbol)
                    
                    if len(cluster) > 1:
                        cluster_correlations = []
                        for i, sym1 in enumerate(cluster):
                            for j, sym2 in enumerate(cluster):
                                if i < j:
                                    cluster_correlations.append(correlation_matrix.loc[sym1, sym2])
                        
                        clusters.append({
                            'symbols': cluster,
                            'average_correlation': np.mean(cluster_correlations) if cluster_correlations else 0,
                            'size': len(cluster)
                        })
            
            return clusters
            
        except:
            return []

    def optimize_portfolio_allocation(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame],
                                    individual_signals: Dict[str, Dict], portfolio_value: float) -> Dict[str, Dict]:
        try:
            risk_parity_result = self.calculate_risk_parity_weights(symbols, historical_data, portfolio_value)
            portfolio_signals = self.analyze_portfolio_signals(historical_data)
            correlation_analysis = self.analyze_cross_asset_correlations(symbols, historical_data)
            
            optimized_allocations = {}
            
            for symbol in symbols:
                if symbol in risk_parity_result.get('positions', {}):
                    base_allocation = risk_parity_result['positions'][symbol]
                    signal_strength = individual_signals.get(symbol, {}).get('composite_score', 0)
                    signal_confidence = individual_signals.get(symbol, {}).get('confidence', 0)
                    
                    signal_adjustment = 1.0 + (signal_strength / 100.0) * (signal_confidence / 100.0)
                    
                    adjusted_size = base_allocation['size_usdt'] * signal_adjustment
                    max_position_value = portfolio_value * 0.25
                    
                    if adjusted_size > max_position_value:
                        adjusted_size = max_position_value
                    
                    optimized_allocations[symbol] = {
                        'symbol': symbol,
                        'original_weight': base_allocation['weight'],
                        'adjusted_weight': adjusted_size / portfolio_value,
                        'position_size_usdt': adjusted_size,
                        'quantity': adjusted_size / historical_data[symbol]['close'].iloc[-1],
                        'signal_strength': signal_strength,
                        'signal_confidence': signal_confidence,
                        'adjustment_factor': signal_adjustment
                    }
            
            portfolio_optimization = {
                'allocations': optimized_allocations,
                'total_portfolio_value': portfolio_value,
                'risk_parity_weights': risk_parity_result.get('weights', {}),
                'portfolio_signals': portfolio_signals,
                'correlation_analysis': correlation_analysis,
                'diversification_score': self._calculate_diversification_score(optimized_allocations, correlation_analysis)
            }
            
            if self.database:
                self.database.store_system_event(
                    "PORTFOLIO_OPTIMIZATION",
                    portfolio_optimization,
                    "INFO",
                    "Portfolio Management"
                )
            
            return portfolio_optimization
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "portfolio_optimization")
            return {}

    def _calculate_diversification_score(self, allocations: Dict[str, Dict], correlation_analysis: Dict) -> float:
        try:
            if not allocations:
                return 0.0
            
            weight_concentration = sum(allocation['adjusted_weight'] ** 2 for allocation in allocations.values())
            herfindahl_score = 1.0 - weight_concentration
            
            avg_correlation = correlation_analysis.get('average_correlation', 0.5)
            correlation_score = 1.0 - abs(avg_correlation)
            
            cluster_penalty = 0.0
            clusters = correlation_analysis.get('correlation_clusters', [])
            for cluster in clusters:
                if cluster['size'] > 2:
                    cluster_penalty += 0.1 * cluster['size']
            
            diversification_score = (herfindahl_score * 0.6 + correlation_score * 0.4) * (1.0 - min(cluster_penalty, 0.3))
            
            return max(0.0, min(1.0, diversification_score))
            
        except:
            return 0.5

    def analyze_market_regime_advanced(self, historical_data: pd.DataFrame) -> Dict:
        try:
            indicators = self.technical_analyzer.calculate_regime_indicators(historical_data)
            advanced_features = self.technical_analyzer.calculate_advanced_features(historical_data)
            
            regime_analysis = {
                'basic_regime': indicators.get('market_regime', 'neutral'),
                'advanced_regime': advanced_features.get('composite_regime', 'neutral'),
                'regime_confidence': advanced_features.get('regime_confidence', 0.0),
                'volatility_regime': self._classify_volatility_regime(historical_data),
                'momentum_regime': self._classify_momentum_regime(historical_data),
                'regime_stability': self._assess_regime_stability(historical_data)
            }
            
            return regime_analysis
            
        except Exception as e:
            return {'basic_regime': 'neutral', 'regime_confidence': 0.0}

    def _classify_volatility_regime(self, df: pd.DataFrame) -> str:
        try:
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            if volatility > 0.03:
                return 'high_volatility'
            elif volatility < 0.01:
                return 'low_volatility'
            else:
                return 'normal_volatility'
        except:
            return 'normal_volatility'

    def _classify_momentum_regime(self, df: pd.DataFrame) -> str:
        try:
            price_trend = df['close'].iloc[-1] / df['close'].iloc[-20] - 1
            if price_trend > 0.05:
                return 'strong_bull'
            elif price_trend < -0.05:
                return 'strong_bear'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def _assess_regime_stability(self, df: pd.DataFrame, window: int = 50) -> float:
        try:
            if len(df) < window:
                return 0.0
                
            recent_data = df.iloc[-window:]
            regimes = []
            
            for i in range(10, len(recent_data), 10):
                chunk = recent_data.iloc[:i]
                regime = self.technical_analyzer._detect_market_regime(
                    chunk['close'], chunk['high'], chunk['low']
                )
                regimes.append(regime)
            
            if len(regimes) > 1:
                changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
                stability = 1 - (changes / (len(regimes) - 1))
                return stability
            else:
                return 0.5
                
        except:
            return 0.5
    
    def analyze_symbol(self, symbol: str, historical_data: pd.DataFrame, portfolio_value: float) -> Dict:
        try:
            return self.analyze_symbol_aggressive(symbol, historical_data, portfolio_value, self.aggressiveness)
        except Exception as e:
            error_msg = f"Error analyzing symbol {symbol}: {e}"
            print(f"❌ {error_msg}")
            
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "analysis")
            
            return {
                'symbol': symbol,
                'current_price': historical_data['close'].iloc[-1] if not historical_data.empty else 0,
                'action': 'HOLD',
                'confidence': 0,
                'composite_score': 0,
                'position_size': 0,
                'quantity': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_reward_ratio': 0,
                'signals': {},
                'ml_prediction': {'prediction': 0, 'confidence': 0},
                'market_regime': 'neutral',
                'volatility_regime': 'normal',
                'aggressiveness': self.aggressiveness,
                'trade_quality': {'quality_score': 0, 'quality_rating': 'POOR'},
                'market_context': {},
                'timestamp': pd.Timestamp.now(),
                'analysis_error': str(e)
            }
    
    def analyze_symbol_aggressive(self, symbol: str, historical_data: pd.DataFrame, 
                                    portfolio_value: float, aggressiveness: str = None) -> Dict:
            
            if aggressiveness is None:
                aggressiveness = self.aggressiveness
                
            try:
                if historical_data is None or len(historical_data) < 100:
                    raise ValueError(f"Insufficient historical data for {symbol}")
                
                indicators = self.technical_analyzer.calculate_regime_indicators(historical_data)
                signals = self.technical_analyzer.generate_enhanced_signals(indicators)
                
                ml_result = self.ml_predictor.predict(symbol, historical_data)
                
                current_price = historical_data['close'].iloc[-1]
                atr = indicators.get('atr', current_price * 0.02)
                
                mtf_signals = self._multi_timeframe_analysis(historical_data)
                
                # Get composite score AND individual scores
                score_results = self._calculate_aggressive_composite_score(signals, ml_result, mtf_signals, aggressiveness)
                composite_score = score_results['composite_score']
                
                action, confidence = self._determine_aggressive_action(composite_score, signals, ml_result, aggressiveness)
                
                position_info = self.risk_manager.calculate_aggressive_position_size(
                    symbol, confidence, current_price, atr, portfolio_value, aggressiveness
                )
                
                volatility_regime = 'high' if indicators.get('atr_percent', 0) > 3 else 'low' if indicators.get('atr_percent', 0) < 1 else 'normal'
                sl_tp_levels = self.risk_manager.calculate_aggressive_stop_loss(
                    symbol, action, current_price, atr, aggressiveness
                )
                
                trade_quality = self._assess_trade_quality(indicators, signals, ml_result, action)
                market_context = self._analyze_market_context(indicators, historical_data)
                
                decision = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'action': action,
                    'confidence': confidence,
                    'composite_score': composite_score,
                    # --- ADD INDIVIDUAL SCORES ---
                    'trend_score': score_results['trend_score'],
                    'mr_score': score_results['mr_score'],
                    'breakout_score': score_results['breakout_score'],
                    'ml_score': score_results['ml_score'],
                    'mtf_score': score_results['mtf_score'],
                    # --- END ADD ---
                    'position_size': position_info['size_usdt'],
                    'quantity': position_info['quantity'],
                    'stop_loss': sl_tp_levels['stop_loss'],
                    'take_profit': sl_tp_levels['take_profit'],
                    'risk_reward_ratio': sl_tp_levels['risk_reward_ratio'],
                    'signals': signals,
                    'ml_prediction': ml_result,
                    'market_regime': indicators.get('market_regime', 'neutral'),
                    'volatility_regime': volatility_regime,
                    'aggressiveness': aggressiveness,
                    'trade_quality': trade_quality,
                    'market_context': market_context,
                    'timestamp': pd.Timestamp.now()
                }
                
                if self.database and action != 'HOLD':
                    # Prepare data for DB event (excluding potentially large objects like signals/ml_prediction)
                    db_event_data = {k: v for k, v in decision.items() if k not in ['signals', 'ml_prediction', 'market_context', 'trade_quality']}
                    self.database.store_system_event(
                        "TRADING_DECISION",
                        db_event_data,
                        "INFO",
                        "Strategy Analysis"
                    )
                
                return decision
                
            except Exception as e:
                error_msg = f"Error in aggressive analysis for {symbol}: {e}"
                print(f"❌ {error_msg}")
                
                if self.error_handler:
                    self.error_handler.handle_trading_error(e, symbol, "aggressive_analysis")
                
                # Ensure the error return structure matches the success structure for consistency
                return {
                    'symbol': symbol,
                    'current_price': historical_data['close'].iloc[-1] if historical_data is not None and not historical_data.empty else 0,
                    'action': 'HOLD',
                    'confidence': 0,
                    'composite_score': 0,
                    'trend_score': 0, 'mr_score': 0, 'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0, # Add defaults
                    'position_size': 0,
                    'quantity': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'risk_reward_ratio': 0,
                    'signals': {},
                    'ml_prediction': {'prediction': 0, 'confidence': 0, 'raw_prediction': 0}, # Add default raw_prediction
                    'market_regime': 'neutral',
                    'volatility_regime': 'normal',
                    'aggressiveness': aggressiveness or self.aggressiveness,
                    'trade_quality': {'quality_score': 0, 'quality_rating': 'POOR'},
                    'market_context': {},
                    'timestamp': pd.Timestamp.now(),
                    'analysis_error': str(e)
                }
    
    def calculate_dynamic_weights(self, market_regime: str, volatility: float, recent_performance: dict, aggressiveness: str) -> Dict[str, float]:
        
        base_weights = self._get_base_weights(market_regime, volatility)
        
        performance_weights = self._adjust_for_performance(base_weights, recent_performance)
        
        final_weights = self._adjust_for_aggressiveness(performance_weights, aggressiveness)
        
        if self.database:
            self.database.store_system_event(
                "STRATEGY_WEIGHT_OPTIMIZATION",
                {
                    'market_regime': market_regime,
                    'volatility': volatility,
                    'aggressiveness': aggressiveness,
                    'old_weights': self.strategy_weights,
                    'new_weights': final_weights,
                    'recent_performance': recent_performance
                },
                "INFO",
                "Strategy Optimization"
            )
        
        return final_weights
    
    def _get_base_weights(self, market_regime: str, volatility: float) -> Dict[str, float]:
        if volatility > 0.03:
            volatility_regime = "high_volatility"
        elif volatility < 0.01:
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal"
        
        regime_weights = {
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
        
        if market_regime in regime_weights:
            base_weights = regime_weights[market_regime].copy()
        else:
            base_weights = regime_weights["bull_trend"].copy()
        
        if volatility_regime != "normal" and volatility_regime in regime_weights:
            volatility_weights = regime_weights[volatility_regime]
            for strategy in base_weights:
                base_weights[strategy] = (
                    base_weights[strategy] * 0.7 + 
                    volatility_weights.get(strategy, base_weights[strategy]) * 0.3
                )
        
        return base_weights
    
    def _adjust_for_performance(self, base_weights: Dict[str, float], recent_performance: Dict) -> Dict[str, float]:
        if not recent_performance:
            return base_weights
        
        adjusted_weights = base_weights.copy()
        
        if recent_performance.get('win_rate', 0) > 60:
            adjusted_weights['trend_following'] *= 1.1
            adjusted_weights['ml_prediction'] *= 1.1
        elif recent_performance.get('win_rate', 0) < 40:
            adjusted_weights['trend_following'] *= 0.9
            adjusted_weights['breakout'] *= 0.9
        
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _adjust_for_aggressiveness(self, weights: Dict[str, float], aggressiveness: str) -> Dict[str, float]:
        adjusted_weights = weights.copy()
        
        aggressiveness_factors = {
            "conservative": {
                'trend_following': 1.1,
                'mean_reversion': 0.9,
                'breakout': 0.8,
                'ml_prediction': 1.0
            },
            "moderate": {
                'trend_following': 1.0,
                'mean_reversion': 1.0,
                'breakout': 1.0,
                'ml_prediction': 1.0
            },
            "aggressive": {
                'trend_following': 0.9,
                'mean_reversion': 1.1,
                'breakout': 1.2,
                'ml_prediction': 0.9
            },
            "high": {
                'trend_following': 0.8,
                'mean_reversion': 1.2,
                'breakout': 1.3,
                'ml_prediction': 0.7
            }
        }
        
        factors = aggressiveness_factors.get(aggressiveness, aggressiveness_factors["moderate"])
        
        for strategy, factor in factors.items():
            if strategy in adjusted_weights:
                adjusted_weights[strategy] *= factor
        
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights

    def _multi_timeframe_analysis(self, df: pd.DataFrame) -> Dict:
        try:
            signals = {}
            
            if len(df) >= 200:
                long_term_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-100] else -1
                signals['long_term_trend'] = long_term_trend
                
                medium_term_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-50] else -1
                signals['medium_term_trend'] = medium_term_trend
                
                short_term_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-20] else -1
                signals['short_term_trend'] = short_term_trend
                
                trends = [long_term_trend, medium_term_trend, short_term_trend]
                bullish_count = sum(1 for t in trends if t == 1)
                bearish_count = sum(1 for t in trends if t == -1)
                
                if bullish_count >= 2:
                    signals['timeframe_alignment'] = 1
                    signals['alignment_strength'] = bullish_count / 3.0
                elif bearish_count >= 2:
                    signals['timeframe_alignment'] = -1
                    signals['alignment_strength'] = bearish_count / 3.0
                else:
                    signals['timeframe_alignment'] = 0
                    signals['alignment_strength'] = 0
                    
            elif len(df) >= 100:
                medium_term_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-50] else -1
                short_term_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-20] else -1
                
                signals['medium_term_trend'] = medium_term_trend
                signals['short_term_trend'] = short_term_trend
                signals['timeframe_alignment'] = 1 if medium_term_trend == short_term_trend else 0
                signals['alignment_strength'] = 0.5 if medium_term_trend == short_term_trend else 0
                
            else:
                signals['timeframe_alignment'] = 0
                signals['alignment_strength'] = 0
                
            return signals
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "multi_timeframe_analysis")
            return {'timeframe_alignment': 0, 'alignment_strength': 0}
    
    def _calculate_aggressive_composite_score(self, signals: Dict, ml_result: Dict, mtf_signals: Dict, aggressiveness: str) -> Dict[str, float]:
            """Calculates composite score and returns individual strategy scores."""
            try:
                # Determine weights based on aggressiveness (copied logic from original)
                if aggressiveness == "conservative":
                    weights = {
                        'trend_following': 0.40, 'mean_reversion': 0.30, 
                        'breakout': 0.20, 'ml_prediction': 0.10, 'mtf': 0.10 # Added MTF weight
                    }
                elif aggressiveness == "moderate":
                    weights = {
                        'trend_following': 0.25, 'mean_reversion': 0.35, 
                        'breakout': 0.20, 'ml_prediction': 0.20, 'mtf': 0.10
                    }
                elif aggressiveness == "aggressive":
                    weights = {
                        'trend_following': 0.20, 'mean_reversion': 0.40, 
                        'breakout': 0.25, 'ml_prediction': 0.15, 'mtf': 0.10
                    }
                elif aggressiveness == "high":
                    weights = {
                        'trend_following': 0.15, 'mean_reversion': 0.45, 
                        'breakout': 0.30, 'ml_prediction': 0.10, 'mtf': 0.10
                    }
                else: # Default to conservative
                    weights = {
                        'trend_following': 0.40, 'mean_reversion': 0.30, 
                        'breakout': 0.20, 'ml_prediction': 0.10, 'mtf': 0.10
                    }

                # Calculate individual strategy scores
                trend_score = self._trend_following_strategy(signals)
                mean_reversion_score = self._mean_reversion_strategy(signals)
                breakout_score = self._breakout_strategy(signals, mtf_signals)
                
                # Use raw_prediction for ML score to keep directionality (-2 to +2 range -> -100 to +100)
                ml_raw_pred = ml_result.get('raw_prediction', 0) 
                ml_score = ml_raw_pred * 50 # Scale to roughly -100 to +100
                
                # Multi-timeframe score
                mtf_alignment = mtf_signals.get('timeframe_alignment', 0)
                mtf_strength = mtf_signals.get('alignment_strength', 0.5) # Default to medium strength if aligned
                mtf_score = mtf_alignment * 50 * mtf_strength # Scale alignment (-1 to 1) * 50 * strength (0 to 1)

                # Normalize weights just in case they don't sum to 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    normalized_weights = {k: v / total_weight for k, v in weights.items()}
                else:
                    normalized_weights = weights # Avoid division by zero

                # Calculate composite score using normalized weights
                composite = (
                    trend_score * normalized_weights.get('trend_following', 0) +
                    mean_reversion_score * normalized_weights.get('mean_reversion', 0) +
                    breakout_score * normalized_weights.get('breakout', 0) +
                    ml_score * normalized_weights.get('ml_prediction', 0) +
                    mtf_score * normalized_weights.get('mtf', 0) # Include MTF score
                )

                # Return both composite and individual scores
                return {
                    'composite_score': composite,
                    'trend_score': trend_score,
                    'mr_score': mean_reversion_score, # Use shorter name for DB
                    'breakout_score': breakout_score,
                    'ml_score': ml_score,
                    'mtf_score': mtf_score
                }

            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_trading_error(e, "ALL", "composite_score_calculation")
                # Return neutral scores on error
                return {
                    'composite_score': 0, 'trend_score': 0, 'mr_score': 0, 
                    'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0
                }

    def _trend_following_strategy(self, indicators: Dict) -> float:
        try:
            score = 0
            
            ema_bullish = (
                indicators.get('ema_8', 0) > indicators.get('ema_21', 0) and
                indicators.get('ema_21', 0) > indicators.get('ema_55', 0) and
                indicators.get('ema_55', 0) > indicators.get('ema_89', 0)
            )
            
            ema_bearish = (
                indicators.get('ema_8', 0) < indicators.get('ema_21', 0) and
                indicators.get('ema_21', 0) < indicators.get('ema_55', 0) and
                indicators.get('ema_55', 0) < indicators.get('ema_89', 0)
            )
            
            if ema_bullish:
                score += 40
            elif ema_bearish:
                score -= 40
                
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                score += 20
            else:
                score -= 20
                
            current_price = indicators.get('bb_middle', 0)
            if current_price > indicators.get('sma_50', 1):
                score += 20
            else:
                score -= 20
                
            if indicators.get('hma', 0) < current_price:
                score += 10
            else:
                score -= 10
                
            return score
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "trend_following_strategy")
            return 0
    
    def _mean_reversion_strategy(self, indicators: Dict) -> float:
        try:
            score = 0
            
            rsi_14 = indicators.get('rsi_14', 50)
            bb_position = indicators.get('bb_position', 0.5)
            
            if rsi_14 < 30:
                score += 30
                if rsi_14 < 20:
                    score += 10
            elif rsi_14 > 70:
                score -= 30
                if rsi_14 > 80:
                    score -= 10
                    
            if bb_position < 0.2:
                score += 25
                if bb_position < 0.1:
                    score += 15
            elif bb_position > 0.8:
                score -= 25
                if bb_position > 0.9:
                    score -= 15
                    
            williams_r = indicators.get('williams_r', -50)
            if williams_r < -80:
                score += 15
            elif williams_r > -20:
                score -= 15
                
            return score
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "mean_reversion_strategy")
            return 0
    
    def _breakout_strategy(self, indicators: Dict, mtf_signals: Dict) -> float:
        try:
            score = 0
            
            resistance = indicators.get('resistance', 0)
            support = indicators.get('support', 0)
            current_price = indicators.get('bb_middle', 0)
            
            resistance_distance = (resistance - current_price) / current_price if current_price > 0 else 0
            if 0 < resistance_distance < 0.02 and indicators.get('rsi_14', 50) > 60:
                score += 25
                
            support_distance = (current_price - support) / current_price if current_price > 0 else 0
            if 0 < support_distance < 0.02 and indicators.get('rsi_14', 50) < 40:
                score -= 25
                
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                score += 20
                
            alignment_strength = mtf_signals.get('alignment_strength', 0)
            if alignment_strength > 0.6:
                score += 15
                
            atr_percent = indicators.get('atr_percent', 0)
            if atr_percent < 1.0:
                score += 10
                
            return score
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "breakout_strategy")
            return 0

    def _determine_aggressive_action(self, composite_score: float, signals: Dict, ml_result: Dict, aggressiveness: str) -> tuple:
        try:
            if aggressiveness == "conservative":
                buy_threshold = 20
                sell_threshold = -20
                strong_threshold = 40
                confidence_boost_factor = 0.5
                strong_signal_boost = 1.2
            elif aggressiveness == "moderate":
                buy_threshold = 15
                sell_threshold = -15
                strong_threshold = 30
                confidence_boost_factor = 0.6
                strong_signal_boost = 1.3
            elif aggressiveness == "aggressive":
                buy_threshold = 10
                sell_threshold = -10
                strong_threshold = 25
                confidence_boost_factor = 0.7
                strong_signal_boost = 1.4
            elif aggressiveness == "high":
                buy_threshold = 5
                sell_threshold = -5
                strong_threshold = 20
                confidence_boost_factor = 0.8
                strong_signal_boost = 1.5
            else:
                buy_threshold = 20
                sell_threshold = -20
                strong_threshold = 40
                confidence_boost_factor = 0.5
                strong_signal_boost = 1.2
            
            ml_confidence = ml_result.get('confidence', 0)
            confidence_boost = ml_confidence * confidence_boost_factor
            
            if composite_score >= buy_threshold:
                action = 'BUY'
                base_confidence = composite_score
            elif composite_score <= sell_threshold:
                action = 'SELL' 
                base_confidence = abs(composite_score)
            else:
                action = 'HOLD'
                base_confidence = 0
            
            confidence = min(100, base_confidence + confidence_boost)
            
            if abs(composite_score) >= strong_threshold:
                confidence = min(100, confidence * strong_signal_boost)
                
            if aggressiveness in ["aggressive", "high"] and action != 'HOLD':
                if 20 <= abs(composite_score) <= 40:
                    confidence = min(100, confidence * 1.1)
            
            return action, confidence
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "determine_action")
            return 'HOLD', 0

    def _assess_trade_quality(self, indicators: Dict, signals: Dict, ml_result: Dict, action: str) -> Dict:
        try:
            quality_score = 0
            strengths = []
            weaknesses = []
            
            if signals.get('trend_strength', 0) > 50:
                quality_score += 25
                strengths.append("Strong trend alignment")
            elif signals.get('trend_strength', 0) < -50:
                quality_score += 25
                strengths.append("Strong counter-trend signals")
            else:
                weaknesses.append("Weak trend confirmation")
                
            if signals.get('momentum_score', 0) > 60:
                quality_score += 20
                strengths.append("Strong momentum")
            elif signals.get('momentum_score', 0) < 40:
                quality_score += 20
                strengths.append("Favorable momentum for reversal")
            else:
                weaknesses.append("Mixed momentum signals")
                
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.2:
                quality_score += 15
                strengths.append("Above average volume")
            else:
                weaknesses.append("Below average volume")
                
            ml_confidence = ml_result.get('confidence', 0)
            if ml_confidence > 0.7:
                quality_score += 20
                strengths.append("High ML confidence")
            elif ml_confidence < 0.4:
                weaknesses.append("Low ML confidence")
                
            atr_percent = indicators.get('atr_percent', 0)
            if 1.0 < atr_percent < 3.0:
                quality_score += 10
                strengths.append("Ideal volatility levels")
            else:
                weaknesses.append("Suboptimal volatility")
                
            market_regime = indicators.get('market_regime', 'neutral')
            if market_regime in ['bull_trend', 'bear_trend']:
                quality_score += 10
                strengths.append("Clear market regime")
            else:
                weaknesses.append("Unclear market regime")
                
            return {
                'quality_score': min(100, quality_score),
                'quality_rating': self._get_quality_rating(quality_score),
                'strengths': strengths,
                'weaknesses': weaknesses
            }
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "trade_quality")
            return {
                'quality_score': 0,
                'quality_rating': 'POOR',
                'strengths': [],
                'weaknesses': ['Error assessing trade quality']
            }
    
    def _get_quality_rating(self, score: float) -> str:
        if score >= 80:
            return "EXCELLENT"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "FAIR"
        elif score >= 20:
            return "POOR"
        else:
            return "VERY POOR"
    
    def _analyze_market_context(self, indicators: Dict, historical_data: pd.DataFrame) -> Dict:
        try:
            context = {}
            
            atr_percent = indicators.get('atr_percent', 0)
            if atr_percent > 3.0:
                context['volatility_context'] = "HIGH_VOLATILITY"
            elif atr_percent < 1.0:
                context['volatility_context'] = "LOW_VOLATILITY"
            else:
                context['volatility_context'] = "NORMAL_VOLATILITY"
                
            market_regime = indicators.get('market_regime', 'neutral')
            context['market_regime'] = market_regime
            
            current_price = historical_data['close'].iloc[-1]
            resistance = indicators.get('resistance', current_price * 1.1)
            support = indicators.get('support', current_price * 0.9)
            
            resistance_distance = (resistance - current_price) / current_price
            support_distance = (current_price - support) / current_price
            
            if resistance_distance < 0.02:
                context['price_position'] = "NEAR_RESISTANCE"
            elif support_distance < 0.02:
                context['price_position'] = "NEAR_SUPPORT"
            else:
                context['price_position'] = "MID_RANGE"
                
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                context['volume_context'] = "HIGH_VOLUME"
            elif volume_ratio < 0.7:
                context['volume_context'] = "LOW_VOLUME"
            else:
                context['volume_context'] = "NORMAL_VOLUME"
                
            return context
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "market_context")
            return {}

    def get_strategy_summary(self) -> Dict:
        return {
            'aggressiveness': self.aggressiveness,
            'strategy_weights': self.strategy_weights,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'strong_threshold': self.strong_threshold,
            'description': self._get_strategy_description()
        }
    
    def _get_strategy_description(self) -> str:
        descriptions = {
            "conservative": "Focus on capital preservation with high-confidence trades only",
            "moderate": "Balanced approach with moderate risk and good trading frequency",
            "aggressive": "Higher trading frequency with larger positions and tighter stops",
            "high": "Maximum aggressiveness for experienced traders, high frequency and size"
        }
        return descriptions.get(self.aggressiveness, "Unknown strategy")
    
    def change_aggressiveness(self, new_aggressiveness: str):
        try:
            valid_levels = ["conservative", "moderate", "aggressive", "high"]
            if new_aggressiveness not in valid_levels:
                raise ValueError(f"Invalid aggressiveness level. Must be one of: {valid_levels}")
                
            old_level = self.aggressiveness
            self.aggressiveness = new_aggressiveness
            self._set_aggressiveness_weights()
            self.risk_manager.aggressiveness = new_aggressiveness
            self.risk_manager._set_aggressiveness_parameters()
            
            print(f"🔄 Strategy aggressiveness changed from {old_level.upper()} to {new_aggressiveness.upper()}")
            
            if self.database:
                self.database.store_system_event(
                    "AGGRESSIVENESS_CHANGED",
                    {
                        'old_level': old_level,
                        'new_level': new_aggressiveness,
                        'new_weights': self.strategy_weights
                    },
                    "INFO",
                    "Strategy Configuration"
                )
            
        except Exception as e:
            error_msg = f"Error changing aggressiveness: {e}"
            print(f"❌ {error_msg}")
            
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "change_aggressiveness")
            
            raise