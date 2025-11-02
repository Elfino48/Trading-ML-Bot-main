from datetime import datetime
import logging
import os

from scipy import stats
from config import RiskConfig
from enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
from ml_predictor import MLPredictor
from advanced_risk_manager import AdvancedRiskManager
from data_engine import DataEngine
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize

from performance_attribution import PerformanceAttribution
from strategy_optimizer import StrategyOptimizer

class EnhancedStrategyOrchestrator:
    def __init__(self, bybit_client, data_engine: DataEngine, aggressiveness: str = "conservative", error_handler=None, database=None, run_start_time_str: str = None, crypto_data_provider=None):
        self.ml_predictor = MLPredictor(error_handler, database)
        self.risk_manager = AdvancedRiskManager(bybit_client, aggressiveness)
        self.client = bybit_client 
        self.data_engine = data_engine 
        self.aggressiveness = aggressiveness
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.error_handler = error_handler
        self.database = database
        self.strategy_optimizer = StrategyOptimizer(database) if database else None
        self.crypto_data_provider = crypto_data_provider  # New: Crypto data provider
        
        # ADD THIS LINE for Phase 4
        self.performance_attribution = PerformanceAttribution(database) if database else None
        
        self.regime_specific_weights = {}
        self.sentiment_indicators = {}
        self.portfolio_signals = {}
        self.correlation_matrix = None
        self.asset_volatilities = {}
        self.run_start_time_str = run_start_time_str or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # NEW: Crypto-specific attributes
        self.on_chain_metrics = {}
        self.network_indicators = {}
        self.exchange_flows = {}
        self.crypto_regime_indicators = {}
        self.bitcoin_dominance = 0.0
        self.fear_greed_index = 50.0

        self.logger = logging.getLogger(__name__)

        self._setup_cycle_logger() 
        self._set_aggressiveness_weights()
        
        print(f"🎯 Strategy Orchestrator set to: {self.aggressiveness.upper()} mode")
        print(f"🔗 Crypto-specific indicators: ENABLED")
    
    def _setup_cycle_logger(self):
        """Sets up a logger for cycle details, creating a new file for each run."""
        try:
            # Use a unique name for the logger instance per run to avoid conflicts
            logger_name = f'CycleDetailsLogger_{self.run_start_time_str}'
            self.cycle_logger = logging.getLogger(logger_name)
            self.cycle_logger.setLevel(logging.INFO)

            # Prevent messages from propagating to the root logger
            self.cycle_logger.propagate = False

            # Only add handlers if they haven't been added for this specific logger instance
            if not self.cycle_logger.hasHandlers():
                # Ensure the 'logs' directory exists
                log_dir = 'logs'
                os.makedirs(log_dir, exist_ok=True)

                # Create the unique filename
                log_filename = os.path.join(log_dir, f'cycle_details_{self.run_start_time_str}.log')

                # Use a standard FileHandler (not rotating for run-specific files)
                file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.INFO)

                # Simple formatter for just the message content
                formatter = logging.Formatter('%(message)s')
                file_handler.setFormatter(formatter)

                self.cycle_logger.addHandler(file_handler)

                print(f"📝 Cycle detail logger initialized (Logs to: {log_filename})")
            else:
                # Find the existing handler's filename if needed for confirmation
                handler_filename = "Unknown"
                if self.cycle_logger.handlers and isinstance(self.cycle_logger.handlers[0], logging.FileHandler):
                    handler_filename = self.cycle_logger.handlers[0].baseFilename
                print(f"ℹ️ Cycle detail logger already configured for this run (Logging to: {handler_filename})")


        except Exception as e:
            print(f"❌ Failed to setup cycle detail logger: {e}")
            # Fallback: Create a logger that just prints to console if file setup fails
            self.cycle_logger = logging.getLogger(f'CycleDetailsLogger_Fallback_{self.run_start_time_str}')
            if not self.cycle_logger.hasHandlers():
                    self.cycle_logger.addHandler(logging.StreamHandler())
            self.cycle_logger.setLevel(logging.WARNING) # Log only warnings/errors if file fails
            self.cycle_logger.warning("File logging for cycle details failed. Using console fallback.")

    def _log_cycle_details(self, decision: Dict):
        try:
            log_entry = []
            log_entry.append(f"============================================================")
            log_entry.append(f"CYCLE DECISION LOG: {decision['symbol']}")
            log_entry.append(f"Timestamp: {decision.get('timestamp')}")
            log_entry.append(f"Aggressiveness: {decision.get('aggressiveness', 'N/A')}")
            
            # NEW: Crypto-specific logging
            if self._is_crypto_symbol(decision['symbol']):
                crypto_regime = decision.get('technical_indicators', {}).get('crypto_regime', 'N/A')
                on_chain = decision.get('technical_indicators', {}).get('on_chain_strength', 'N/A')
                fear_greed = decision.get('technical_indicators', {}).get('fear_greed_index', 'N/A')
                log_entry.append(f"Crypto Regime: {crypto_regime} | On-Chain: {on_chain} | F&G: {fear_greed}")
            
            log_entry.append(f"------------------------------------------------------------")
            log_entry.append(f"[DECISION]")
            log_entry.append(f"Action: {decision.get('action', 'N/A')}")
            log_entry.append(f"Final Confidence: {decision.get('confidence', 0):.2f}%")
            log_entry.append(f"Composite Score: {decision.get('composite_score', 0):.4f}")
            log_entry.append(f"Trade Quality: {decision.get('trade_quality', {}).get('quality_rating', 'N/A')} ({decision.get('trade_quality', {}).get('quality_score', 0)})")

            # --- NEW: Log execution check ---
            should_execute = decision.get('should_execute', False)
            execute_reason = decision.get('execute_reason', 'N/A')
            log_entry.append(f"\n[EXECUTION CHECK]")
            log_entry.append(f"  - Should Execute: {should_execute}")
            if not should_execute:
                log_entry.append(f"  - Reason: {execute_reason}")
            # --- END NEW ---

            log_entry.append(f"\n[SCORE BREAKDOWN]")
            log_entry.append(f"  - Trend Score:      {decision.get('trend_score', 0):.4f}")
            log_entry.append(f"  - Mean Rev Score:   {decision.get('mr_score', 0):.4f}")
            log_entry.append(f"  - Breakout Score:   {decision.get('breakout_score', 0):.4f}")
            log_entry.append(f"  - MTF Score:        {decision.get('mtf_score', 0):.4f}")
            log_entry.append(f"  - ML Score:         {decision.get('ml_score', 0):.4f}")

            ml_pred = decision.get('ml_prediction', {})
            log_entry.append(f"\n[ML PREDICTOR DETAILS]")
            log_entry.append(f"  - ML Raw Prediction: {ml_pred.get('raw_prediction', 0)} (This is scaled to ML Score)")
            log_entry.append(f"  - ML Model Confidence: {ml_pred.get('confidence', 0) * 100:.2f}% (This boosts Final Confidence)")
            log_entry.append(f"  - ML (RF) Pred: {ml_pred.get('rf_pred', 'N/A')} | Conf: {ml_pred.get('rf_confidence', 0) * 100:.2f}%")
            log_entry.append(f"  - ML (GB) Pred: {ml_pred.get('gb_pred', 'N/A')} | Conf: {ml_pred.get('gb_confidence', 0) * 100:.2f}%")

            # --- NEW SECTION FOR RAW INDICATORS ---
            indicators = decision.get('technical_indicators', {})
            if indicators:
                log_entry.append(f"\n[TECHNICAL INDICATORS (Raw Values)]")
                # Sort indicators alphabetically for consistent logging
                sorted_indicators = sorted(indicators.items())
                for key, value in sorted_indicators:
                    # Format floats nicely, leave others as is
                    if isinstance(value, (float, np.float64)):
                        log_entry.append(f"  - {key:<25}: {value:.4f}")
                    else:
                        log_entry.append(f"  - {key:<25}: {value}")
            # --- END NEW SECTION ---

            # NEW: Crypto indicators section
            indicators = decision.get('technical_indicators', {})
            crypto_indicators_present = any('crypto' in key.lower() or 
                                          'on_chain' in key.lower() or
                                          'fear_greed' in key.lower() 
                                          for key in indicators.keys())
            
            if crypto_indicators_present:
                log_entry.append(f"\n[CRYPTO-SPECIFIC INDICATORS]")
                crypto_keys = [k for k in indicators.keys() if any(term in k.lower() for term in 
                                ['crypto', 'on_chain', 'fear_greed', 'halving', 'miner', 'exchange', 'nvt', 'mvrv'])]
                for key in sorted(crypto_keys):
                    value = indicators[key]
                    if isinstance(value, (float, np.float64)):
                        log_entry.append(f"  - {key:<25}: {value:.4f}")
                    else:
                        log_entry.append(f"  - {key:<25}: {value}")

            log_entry.append(f"\n[CONTEXT & RISK]")
            log_entry.append(f"Market Regime: {decision.get('market_regime', 'N/A')}")
            log_entry.append(f"Volatility Regime: {decision.get('volatility_regime', 'N/A')}")
            log_entry.append(f"Analysis Price (Kline Close): {decision.get('analysis_price', 0):.4f}")
            log_entry.append(f"Latest Price (For Execution): {decision.get('current_price', 0):.4f}")
            log_entry.append(f"Proposed Size: ${decision.get('position_size', 0):.2f}")
            log_entry.append(f"Stop Loss: ${decision.get('stop_loss', 0):.4f}")
            log_entry.append(f"Take Profit: ${decision.get('take_profit', 0):.4f}")

            if decision.get('analysis_error'): # Use .get() for safety
                log_entry.append(f"\n[ANALYSIS ERROR]")
                log_entry.append(f"Error: {decision['analysis_error']}")

            log_entry.append(f"============================================================\n")

            self.cycle_logger.info("\n".join(log_entry))

        except Exception as e:
            # Use logger if available, otherwise print
            log_func = getattr(self, 'cycle_logger', None)
            if log_func and hasattr(log_func, 'error'):
                log_func.error(f"--- FAILED TO LOG CYCLE DETAILS --- \n{str(e)}\n")
            else:
                print(f"❌ Error in _log_cycle_details (logging failed): {e}")

            # Also print to console as a fallback
            print(f"❌ Error in _log_cycle_details: {e}")

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
            
        if self.aggressiveness == "aggressive":
            self.strategy_weights = {
                'trend_following': 0.30,  
                'mean_reversion': 0.30,  
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
                    
                # Enhanced analysis with crypto-specific indicators
                indicators = self.technical_analyzer.calculate_regime_indicators(data)
                
                # NEW: Add crypto regime analysis
                crypto_regime = self.analyze_crypto_market_regime(symbol, data)
                indicators['crypto_regime'] = crypto_regime
                
                signals = self.technical_analyzer.generate_enhanced_signals(indicators)
                ml_result = self.ml_predictor.predict(symbol, data)
                mtf_signals = self._multi_timeframe_analysis(data)
                
                # ENHANCED: Include crypto factors in composite score
                composite_score = self._calculate_crypto_enhanced_composite_score(
                    signals, ml_result, mtf_signals, indicators, crypto_regime
                )
                
                portfolio_scores[symbol] = composite_score
                total_score += composite_score
                signal_count += 1
            
            if signal_count > 0:
                portfolio_avg = total_score / signal_count
                portfolio_scores['PORTFOLIO_AVG'] = portfolio_avg
                
                # NEW: Crypto market health indicator
                crypto_health = self._calculate_crypto_market_health(portfolio_scores)
                portfolio_scores['CRYPTO_MARKET_HEALTH'] = crypto_health
                
                strong_buy_count = sum(1 for score in portfolio_scores.values() if score > self.strong_threshold)
                strong_sell_count = sum(1 for score in portfolio_scores.values() if score < -self.strong_threshold)
                
                portfolio_scores['STRONG_BUY_RATIO'] = strong_buy_count / signal_count
                portfolio_scores['STRONG_SELL_RATIO'] = strong_sell_count / signal_count
                portfolio_scores['MARKET_SENTIMENT'] = 'BULLISH' if portfolio_avg > 0 else 'BEARISH'
                
                # NEW: Crypto-specific sentiment
                portfolio_scores['CRYPTO_SENTIMENT'] = self._calculate_crypto_sentiment(portfolio_scores)
            
            self.portfolio_signals = portfolio_scores
            return portfolio_scores
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "portfolio_analysis")
            return {}

    def _calculate_crypto_market_health(self, portfolio_scores: Dict) -> str:
        """Calculate overall crypto market health"""
        try:
            strong_buy_ratio = portfolio_scores.get('STRONG_BUY_RATIO', 0)
            portfolio_avg = portfolio_scores.get('PORTFOLIO_AVG', 0)
            
            if strong_buy_ratio > 0.4 and portfolio_avg > 20:
                return "VERY_HEALTHY"
            elif strong_buy_ratio > 0.3 and portfolio_avg > 10:
                return "HEALTHY"
            elif strong_buy_ratio > 0.2 and portfolio_avg > 0:
                return "NEUTRAL"
            elif strong_buy_ratio < 0.1 and portfolio_avg < -10:
                return "WEAK"
            else:
                return "VERY_WEAK"
                
        except Exception as e:
            self.logger.error(f"Error calculating crypto market health: {e}")
            return "NEUTRAL"
        
    def _calculate_crypto_sentiment(self, portfolio_scores: Dict) -> str:
        """Calculate crypto-specific market sentiment"""
        try:
            # Combine technical signals with on-chain data
            technical_sentiment = portfolio_scores.get('MARKET_SENTIMENT', 'NEUTRAL')
            market_health = portfolio_scores.get('CRYPTO_MARKET_HEALTH', 'NEUTRAL')
            
            if technical_sentiment == 'BULLISH' and market_health in ['VERY_HEALTHY', 'HEALTHY']:
                return "STRONG_BULL"
            elif technical_sentiment == 'BULLISH':
                return "BULLISH"
            elif technical_sentiment == 'BEARISH' and market_health in ['WEAK', 'VERY_WEAK']:
                return "STRONG_BEAR"
            elif technical_sentiment == 'BEARISH':
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error calculating crypto sentiment: {e}")
            return "NEUTRAL"

    def _calculate_crypto_enhanced_composite_score(self, indicators: Dict, signals: Dict, 
                                                 ml_result: Dict, mtf_signals: Dict,
                                                 aggressiveness: str, crypto_regime: Dict) -> Dict[str, float]:
        """Calculate composite score with crypto enhancements"""
        try:
            # Get base composite score
            base_results = self._calculate_aggressive_composite_score(
                indicators, signals, ml_result, mtf_signals, aggressiveness
            )
            
            # Apply crypto enhancements if available
            if crypto_regime and 'crypto_regime' in crypto_regime:
                crypto_boost = self._calculate_crypto_boost_factor(crypto_regime, indicators)
                base_results['composite_score'] += crypto_boost
                
                # Log crypto enhancement for transparency
                if abs(crypto_boost) > 5:
                    self.logger.info(f"Crypto boost applied: {crypto_boost:.2f} points for {indicators.get('symbol', 'unknown')}")
            
            return base_results
            
        except Exception as e:
            self.logger.error(f"Error in crypto-enhanced composite score: {e}")
            return self._calculate_aggressive_composite_score(
                indicators, signals, ml_result, mtf_signals, aggressiveness
            )

    def _calculate_crypto_boost_factor(self, crypto_regime: Dict, indicators: Dict) -> float:
        """Calculate crypto-specific boost to composite score"""
        try:
            boost = 0.0
            
            # 1. Crypto regime boost
            regime_boost = {
                'crypto_bull': 15,
                'crypto_bear': -15,
                'crypto_neutral': 0,
                'crypto_transition': -5
            }
            boost += regime_boost.get(crypto_regime.get('crypto_regime', 'neutral'), 0)
            
            # 2. On-chain strength boost
            if crypto_regime.get('on_chain_strength') == 'healthy':
                boost += 10
            elif crypto_regime.get('on_chain_strength') == 'weak':
                boost -= 10
            
            # 3. Liquidity conditions
            if indicators.get('liquidity_signal') == 'high_liquidity':
                boost += 5
            
            # 4. Miner sentiment
            if indicators.get('miner_sentiment') == 'accumulating':
                boost += 8
            elif indicators.get('miner_sentiment') == 'distributing':
                boost -= 8
            
            # 5. Market sentiment extremes (contrarian)
            fear_greed = indicators.get('fear_greed_index', 50)
            if fear_greed > 75:  # Extreme greed - reduce confidence
                boost -= 10
            elif fear_greed < 25:  # Extreme fear - increase confidence
                boost += 10
            
            # 6. Halving cycle position
            cycle_position = indicators.get('halving_cycle_position', 0.5)
            if cycle_position < 0.3:  # Early in cycle
                boost += 12
            elif cycle_position > 0.7:  # Late in cycle
                boost -= 12
            
            # Apply confidence weighting
            confidence = crypto_regime.get('composite_confidence', 0.5)
            weighted_boost = boost * confidence
            
            # Cap the maximum influence
            return max(-25, min(25, weighted_boost))
            
        except Exception as e:
            self.logger.error(f"Error calculating crypto boost: {e}")
            return 0.0

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency"""
        crypto_pairs = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH', 'XRP', 'EOS', 'XTZ']
        return any(pair in symbol for pair in crypto_pairs)

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
            # --- FIX: Ensure df is not empty ---
            if df is None or df.empty or len(df) < window:
                # self.logger.debug(f"assess_regime_stability: Input DataFrame too short ({len(df) if df is not None else 0} < {window})")
                return 0.0 # Return low stability if not enough data

            # Use tail for recent data if df is long enough
            recent_data = df.iloc[-window:] if len(df) >= window else df

            regimes = []

            # --- FIX: Ensure loop range is valid and chunks are sufficient ---
            # Start loop earlier and ensure chunk size increases reasonably
            min_chunk_size = 20 # Need enough data for regime detection
            step = max(5, window // 10) # Adjust step based on window

            for i in range(min_chunk_size, len(recent_data) + 1, step):
                 # Ensure 'i' doesn't exceed length if step calculation is odd
                 current_idx = min(i, len(recent_data))
                 chunk = recent_data.iloc[:current_idx]

                 if len(chunk) >= min_chunk_size: # Only calculate if chunk is large enough
                     # Check if required columns exist
                     if not {'close', 'high', 'low'}.issubset(chunk.columns):
                          self.logger.warning(f"assess_regime_stability: Chunk missing required columns (close, high, low) at index {current_idx}")
                          continue # Skip this chunk

                     # --- FIX: Call internal _detect_market_regime method ---
                     # Ensure this method exists and handles potential errors gracefully
                     try:
                        # Assuming _detect_market_regime is defined elsewhere in the class
                        regime = self._detect_market_regime(
                            chunk['close'], chunk['high'], chunk['low']
                        )
                        regimes.append(regime)
                     except Exception as detect_e:
                        self.logger.warning(f"assess_regime_stability: Error detecting regime in chunk ending {current_idx}: {detect_e}")
                        # Append a neutral or error state? Let's skip appending on error.
                        pass
                 # else: # Chunk too small, skip
                 #    self.logger.debug(f"assess_regime_stability: Skipping chunk of size {len(chunk)} at index {current_idx}")


            # --- FIX: Handle case with less than 2 valid regimes ---
            if len(regimes) < 2:
                # self.logger.debug(f"assess_regime_stability: Not enough valid regimes calculated ({len(regimes)})")
                return 0.5 # Return neutral stability if few regimes

            changes = 0
            for i in range(1, len(regimes)):
                if regimes[i] != regimes[i-1]:
                    changes += 1

            # len(regimes) is guaranteed >= 2 here, so len(regimes) - 1 >= 1
            stability = 1.0 - (changes / (len(regimes) - 1))
            # self.logger.debug(f"assess_regime_stability: Calculated stability={stability:.3f} from {len(regimes)} regimes with {changes} changes.")
            return stability

        except Exception as e:
            # --- FIX: Improved Error Logging ---
            self.logger.error(f"Error calculating regime stability: {type(e).__name__} - {e}", exc_info=True)
            return 0.5 # Return neutral stability on unexpected error

    # --- ADD Helper method _detect_market_regime if it doesn't exist ---
    # This is a simplified version based on what _detect_market_regime_enhanced does
    def _detect_market_regime(self, close, high, low, lookback=20):
        """Simplified regime detection for stability check."""
        if len(close) < lookback:
            return 'neutral'
        try:
            returns = close.pct_change().dropna()
            volatility = returns.rolling(lookback).std().iloc[-1]
            y = close.iloc[-lookback:].values
            x = np.arange(len(y))
            slope, _, r_value, _, _ = stats.linregress(x, y)
            r_squared = r_value ** 2

            if r_squared > 0.5:
                return "bull_trend" if slope > 0 else "bear_trend"
            elif volatility < returns.quantile(0.25): # Lower quartile of volatility
                return "ranging"
            else:
                return "neutral"
        except Exception:
            return "neutral" # Fallback on error
        
    def _calculate_regime_series_stability(self, regime_series_df: pd.DataFrame, current_regime: str) -> float:
        """Calculate stability of the current regime based on a series of regime classifications."""
        try:
            # --- FIX: Proper input validation ---
            if not isinstance(regime_series_df, pd.DataFrame) or regime_series_df.empty:
                self.logger.warning("_calculate_regime_series_stability: Received empty or invalid DataFrame.")
                return 0.5

            # Use the primary regime series (first column)
            if regime_series_df.columns.empty:
                self.logger.warning("_calculate_regime_series_stability: DataFrame has no columns.")
                return 0.5
            
            primary_series = regime_series_df.iloc[:, 0]

            # Ensure we have a valid Series
            if not isinstance(primary_series, pd.Series) or primary_series.empty:
                self.logger.warning("_calculate_regime_series_stability: Primary series is empty or invalid.")
                return 0.5

            # Get recent part of the series
            window_size = min(20, len(primary_series))  # Look at max last 20 periods
            recent_series = primary_series.tail(window_size)

            # Convert to list for easier iteration
            recent_list = recent_series.tolist()
            
            # --- FIX: Handle the case where current_regime is None or empty ---
            if not current_regime:
                current_regime = 'neutral'
                
            consecutive_count = 0

            # Iterate backwards from the last element
            for i in range(len(recent_list) - 1, -1, -1):
                self.logger.debug(f"  Comparing: recent='{recent_list[i]}' vs current='{current_regime}'")
                
                # --- FIX: Normalize regime names for comparison ---
                recent_regime = self._normalize_regime_name(recent_list[i])
                normalized_current = self._normalize_regime_name(current_regime)
                
                if recent_regime == normalized_current:
                    consecutive_count += 1
                else:
                    break

            # Normalize stability based on a scale (e.g., 10 periods = full stability)
            stability_scale = 10.0
            stability = min(1.0, consecutive_count / stability_scale)

            self.logger.debug(f"Regime stability: {stability:.3f} (consecutive: {consecutive_count}, window: {window_size})")
            return stability

        except Exception as e:
            # --- FIX: Better error logging with context ---
            self.logger.error(f"Error calculating regime series stability: {type(e).__name__} - {e}. "
                            f"Input type: {type(regime_series_df)}, shape: {getattr(regime_series_df, 'shape', 'No shape')}")
            return 0.5

    def _normalize_regime_name(self, regime: str) -> str:
        """Normalize regime names to handle variations and ensure consistent comparison."""
        if not regime:
            return 'neutral'
        
        regime = str(regime).lower().strip()
        
        # Map similar regimes to standard names
        regime_mapping = {
            'ranging': 'neutral',  # Treat ranging as neutral for stability calculation
            'consolidation': 'neutral',
            'sideways': 'neutral',
            'choppy': 'neutral',
            'trending': 'bull_trend',  # Generic trending maps to bull trend
            'uptrend': 'bull_trend',
            'downtrend': 'bear_trend',
            'high_vol': 'high_volatility',
            'low_vol': 'low_volatility'
        }
        
        return regime_mapping.get(regime, regime)

    def analyze_symbol(self, symbol: str, historical_data: pd.DataFrame, portfolio_value: float) -> Dict:
            try:
                return self.analyze_symbol_aggressive(symbol, historical_data, portfolio_value, self.aggressiveness)
            except Exception as e:
                error_msg = f"Error analyzing symbol {symbol}: {e}"
                print(f"❌ {error_msg}")

                if self.error_handler:
                    self.error_handler.handle_trading_error(e, symbol, "analysis")

                analysis_price = historical_data['close'].iloc[-1] if historical_data is not None and not historical_data.empty else 0
                
                fallback_decision = {
                    'symbol': symbol,
                    'current_price': analysis_price, 
                    'action': 'HOLD',
                    'confidence': 0,
                    'composite_score': 0,
                    'trend_score': 0, 'mr_score': 0, 'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0,
                    'position_size': 0,
                    'quantity': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'risk_reward_ratio': 0,
                    'signals': {},
                    'ml_prediction': {'prediction': 0, 'confidence': 0, 'raw_prediction': 0},
                    'market_regime': 'neutral',
                    'volatility_regime': 'normal',
                    'aggressiveness': self.aggressiveness,
                    'trade_quality': {'quality_score': 0, 'quality_rating': 'POOR'},
                    'market_context': {},
                    'timestamp': pd.Timestamp.now(),
                    'analysis_error': str(e)
                }
                self._log_cycle_details(fallback_decision)
                
                return fallback_decision
    
    def _check_signal_alignment(self, scores: Dict) -> Tuple[bool, str, float]:
        """Check if signals are aligned across strategies with strength measurement"""
        trend_score = scores.get('trend_score', 0)
        mr_score = scores.get('mr_score', 0) 
        breakout_score = scores.get('breakout_score', 0)
        ml_score = scores.get('ml_score', 0)
        mtf_score = scores.get('mtf_score', 0)
        
        bullish_signals = 0
        bearish_signals = 0
        total_strength = 0
        
        # Trend signal
        if trend_score > 20: 
            bullish_signals += 1
            total_strength += abs(trend_score)
        elif trend_score < -20: 
            bearish_signals += 1
            total_strength += abs(trend_score)
            
        # Mean reversion signal  
        if mr_score > 20: 
            bullish_signals += 1
            total_strength += abs(mr_score)
        elif mr_score < -20: 
            bearish_signals += 1
            total_strength += abs(mr_score)
            
        # Breakout signal
        if breakout_score > 20: 
            bullish_signals += 1
            total_strength += abs(breakout_score)
        elif breakout_score < -20: 
            bearish_signals += 1
            total_strength += abs(breakout_score)
            
        # ML signal
        if ml_score > 20: 
            bullish_signals += 1
            total_strength += abs(ml_score)
        elif ml_score < -20: 
            bearish_signals += 1
            total_strength += abs(ml_score)
            
        # Multi-timeframe signal
        if mtf_score > 20: 
            bullish_signals += 1
            total_strength += abs(mtf_score)
        elif mtf_score < -20: 
            bearish_signals += 1
            total_strength += abs(mtf_score)
        
        total_signals = bullish_signals + bearish_signals
        alignment_strength = total_strength / max(total_signals, 1) if total_signals > 0 else 0
        
        # Require at least 2 aligned signals and no strong opposing signals
        if bullish_signals >= 2 and bearish_signals <= 1:
            return True, 'BULLISH', alignment_strength
        elif bearish_signals >= 2 and bullish_signals <= 1:
            return True, 'BEARISH', alignment_strength
        else:
            return False, 'MIXED', alignment_strength  

    def _create_fallback_decision(self, symbol: str, reason: str, current_price: float = None) -> Dict:
        """Create a fallback decision when analysis fails"""
        if current_price is None:
            current_price = self.data_engine.get_current_price(symbol)
            if current_price <= 0:
                current_price = 0  # Prevent invalid prices
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'action': 'HOLD',
            'confidence': 0,
            'composite_score': 0,
            'trend_score': 0, 'mr_score': 0, 'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0,
            'position_size': 0,
            'quantity': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward_ratio': 0,
            'signals': {},
            'ml_prediction': {'prediction': 0, 'confidence': 0, 'raw_prediction': 0},
            'market_regime': 'neutral',
            'volatility_regime': 'normal',
            'aggressiveness': self.aggressiveness,
            'trade_quality': {'quality_score': 0, 'quality_rating': 'POOR'},
            'market_context': {},
            'timestamp': pd.Timestamp.now(),
            'analysis_error': reason,
            'analysis_price': current_price  # Use current price as fallback
        }
    
    def _check_data_quality(self, symbol: str, historical_data: pd.DataFrame) -> Dict:
        """Comprehensive data quality check"""
        quality_report = {
            'symbol': symbol,
            'total_rows': len(historical_data),
            'valid_prices': 0,
            'data_issues': [],
            'is_usable': False
        }
        
        if historical_data is None or historical_data.empty:
            quality_report['data_issues'].append('No data')
            return quality_report
            
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in historical_data.columns]
        if missing_cols:
            quality_report['data_issues'].append(f'Missing columns: {missing_cols}')
            return quality_report
        
        # Check close prices
        close_prices = historical_data['close'].astype(float)
        valid_closes = close_prices[close_prices > 0].dropna()
        quality_report['valid_prices'] = len(valid_closes)
        
        if len(valid_closes) < 50:
            quality_report['data_issues'].append(f'Only {len(valid_closes)} valid prices (need 50)')
        else:
            quality_report['is_usable'] = True
            
        # Check for NaN values
        nan_count = historical_data[required_cols].isnull().sum().sum()
        if nan_count > 0:
            quality_report['data_issues'].append(f'{nan_count} NaN values')
            
        # Check for zero volumes
        zero_volumes = (historical_data['volume'] == 0).sum()
        if zero_volumes > len(historical_data) * 0.5:  # More than 50% zero volume
            quality_report['data_issues'].append(f'{zero_volumes} zero volume bars')
        
        return quality_report
    
    def analyze_symbol_aggressive(self, symbol: str, historical_data: pd.DataFrame,
                            portfolio_value: float, aggressiveness: str = None) -> Dict:

        if aggressiveness is None:
            aggressiveness = self.aggressiveness

        # Define the base structure for fallback *before* the main try block
        # This helps ensure all keys exist, even if values are defaults
        base_fallback = {
            'symbol': symbol,
            'current_price': 0.0, # Will be updated later
            'action': 'HOLD', 'confidence': 0, 'composite_score': 0,
            'trend_score': 0, 'mr_score': 0, 'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0,
            'position_size': 0, 'quantity': 0, 'stop_loss': 0, 'take_profit': 0, 'risk_reward_ratio': 0,
            'signals': {}, 'ml_prediction': {'prediction': 0, 'confidence': 0, 'raw_prediction': 0},
            'market_regime': 'neutral', 'volatility_regime': 'normal',
            'aggressiveness': aggressiveness or self.aggressiveness,
            'trade_quality': {'quality_score': 0, 'quality_rating': 'POOR'},
            'market_context': {}, 'timestamp': pd.Timestamp.now(),
            'technical_indicators': {},
            'analysis_error': None,
            'analysis_price': 0.0 # Will be updated
        }

        quality_report = self._check_data_quality(symbol, historical_data)
        if not quality_report['is_usable']:
            self.logger.error(f"Data quality issues for {symbol}: {quality_report['data_issues']}")
            error_fallback = base_fallback.copy() # Start with base structure
            error_fallback['analysis_error'] = f"Data quality: {quality_report['data_issues']}"
            error_fallback['timestamp'] = pd.Timestamp.now()
            # Try to get current price for logging context
            try: error_fallback['current_price'] = self.data_engine.get_current_price(symbol)
            except: pass
            self._log_cycle_details(error_fallback)
            return error_fallback

        indicators = {} # Define indicators dict outside try block for scope in except/finally
        analysis_price = 0.0 # Define analysis_price outside try block

        try:
            if historical_data is None or historical_data.empty:
                raise ValueError(f"Insufficient historical data (empty/None) for {symbol}")

            close_prices = historical_data['close'].astype(float)
            valid_closes = close_prices[close_prices > 0].dropna()
            if valid_closes.empty:
                raise ValueError(f"No valid close prices found for {symbol} in historical data")

            analysis_price = valid_closes.iloc[-1]

            # NEW: Fetch and integrate crypto-specific data
            crypto_data = {}
            if self._is_crypto_symbol(symbol):
                crypto_data = self.fetch_crypto_on_chain_data(symbol)
                self.update_crypto_market_data()  # Update global crypto metrics
                
            indicators = self.technical_analyzer.calculate_regime_indicators(historical_data)
            if not indicators:
                raise ValueError(f"Technical indicator calculation failed for {symbol}")

            # ENHANCED: Add crypto-specific indicators to the main indicators dict
            if crypto_data:
                crypto_indicators = self.technical_analyzer.calculate_crypto_specific_indicators(
                    symbol, historical_data, crypto_data, self._get_market_context()
                )
                indicators.update(crypto_indicators)
                
                # NEW: Crypto regime analysis
                crypto_regime = self.analyze_crypto_market_regime(symbol, historical_data)
                indicators['crypto_regime'] = crypto_regime.get('crypto_regime', 'neutral')
                indicators['crypto_regime_confidence'] = crypto_regime.get('composite_confidence', 0.5)
                indicators['on_chain_strength'] = crypto_regime.get('on_chain_strength', 'neutral')

            indicators['symbol'] = symbol
            indicators['analysis_price'] = analysis_price
            indicators['close'] = analysis_price

            # ========== VOLUME FILTER MODIFICATION START ==========
            # Check volume but don't return early - just note the warning
            volume_ok, volume_reason = self._should_trade_with_volume(indicators)
            volume_warning = None
            if not volume_ok:
                self.logger.warning(f"[{symbol}] Volume warning: {volume_reason}")
                volume_warning = volume_reason
                # We continue with analysis but will block execution later
            # ========== VOLUME FILTER MODIFICATION END ==========

            signals = self.technical_analyzer.generate_enhanced_signals(indicators)
            ml_result = self.ml_predictor.predict(symbol, historical_data)
            atr = indicators.get('atr', analysis_price * 0.02)
            mtf_signals = self._multi_timeframe_analysis(historical_data)

            score_results = self._calculate_crypto_enhanced_composite_score(
                indicators, signals, ml_result, mtf_signals, aggressiveness,
                crypto_regime if crypto_data else {}
            )
            composite_score = score_results['composite_score']

            alignment_ok, alignment_direction, alignment_strength = self._check_signal_alignment(score_results)
            if not alignment_ok:
                confidence_reduction = 0.6 if alignment_strength < 0.3 else 0.8
            else:
                confidence_reduction = 1.0

            action, confidence = self._determine_aggressive_action(
                composite_score, indicators, ml_result, aggressiveness
            )

            confidence = confidence * confidence_reduction

            latest_price = self.data_engine.get_current_price(symbol)
            if latest_price <= 0:
                self.logger.warning(f"Could not fetch latest price for {symbol}, falling back to analysis price: {analysis_price}")
                if self.error_handler:
                    self.error_handler.handle_data_error(Exception("Failed to get latest price"), "analyze_symbol_risk", symbol)
                latest_price = analysis_price

            position_info = self.risk_manager.calculate_aggressive_position_size(
                symbol, confidence, latest_price, atr, portfolio_value, aggressiveness
            )

            volatility_regime = 'high' if indicators.get('atr_percent', 0) > 3 else 'low' if indicators.get('atr_percent', 0) < 1 else 'normal'

            sl_tp_levels = {}
            if action != 'HOLD':
                sl_tp_levels = self.risk_manager.calculate_aggressive_stop_loss(
                    symbol, action, latest_price, atr, aggressiveness
                )
            else:
                sl_tp_levels = {'stop_loss': 0, 'take_profit': 0, 'risk_reward_ratio': 0}

            market_context = self._analyze_market_context(indicators, historical_data)
            trade_quality = self._assess_signal_quality_enhanced(indicators, signals, ml_result, action, composite_score)

            decision = {
                'symbol': symbol,
                'current_price': latest_price,
                'action': action,
                'confidence': confidence,
                'composite_score': composite_score,
                'trend_score': score_results['trend_score'],
                'mr_score': score_results['mr_score'],
                'breakout_score': score_results['breakout_score'],
                'ml_score': score_results['ml_score'],
                'mtf_score': score_results['mtf_score'],
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
                'timestamp': pd.Timestamp.now(),
                'analysis_price': analysis_price,
                'technical_indicators': indicators,
                'signal_alignment': {
                    'aligned': alignment_ok,
                    'direction': alignment_direction,
                    'strength': alignment_strength
                }
            }

            decision = self._apply_regime_aware_filters(decision, indicators)

            # ========== VOLUME FILTER EXECUTION BLOCK START ==========
            # Apply volume filter to block execution while still showing the analysis
            if volume_warning and decision['action'] != 'HOLD':
                original_action = decision['action']
                decision['action'] = 'HOLD'
                decision['execute_reason'] = f"Volume filter: {volume_warning} (would have been {original_action})"
                decision['should_execute'] = False
                # Reduce confidence for low volume scenarios
                decision['confidence'] = decision['confidence'] * 0.3
                if 'weaknesses' not in decision['trade_quality']:
                    decision['trade_quality']['weaknesses'] = []
                decision['trade_quality']['weaknesses'].append(f"Low volume: {volume_warning}")
            else:
                # Normal execution logic for volume-ok scenarios
                should_execute, execute_reason = self._should_execute_trade(decision)
                decision['should_execute'] = should_execute
                decision['execute_reason'] = execute_reason
            # ========== VOLUME FILTER EXECUTION BLOCK END ==========

            self._log_cycle_details(decision)

            if self.database and action != 'HOLD':
                db_event_data = {k: v for k, v in decision.items() if k not in ['signals', 'ml_prediction', 'market_context', 'trade_quality', 'technical_indicators']}
                db_event_data['portfolio_value_at_decision'] = portfolio_value
                self.database.store_system_event("TRADING_DECISION", db_event_data, "INFO", "Strategy Analysis")

            return decision

        except Exception as e:
            error_msg = f"Error in aggressive analysis for {symbol}: {e}"
            self.logger.error(error_msg, exc_info=True)

            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "aggressive_analysis")

            # Use the base fallback structure and populate known values
            error_fallback = base_fallback.copy()
            error_fallback['analysis_error'] = str(e)
            error_fallback['technical_indicators'] = indicators # Include indicators if calculated before error
            error_fallback['timestamp'] = pd.Timestamp.now()
            # Try to get current price for logging context
            try: error_fallback['current_price'] = self.data_engine.get_current_price(symbol)
            except: pass
            error_fallback['analysis_price'] = analysis_price if analysis_price > 0 else 0.0 # Use calculated analysis_price if valid

            self._log_cycle_details(error_fallback)

            return error_fallback
    
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
    
    def _normalize_score(self, score: float, strategy_type: str) -> float:
        """
        Normalizes a raw strategy score to a standard -100 to 100 scale.
        """
        # Define the *actual* max absolute score for each strategy based on its code
        strategy_max_abs_score = {
            'trend': 80.0,      # Clamped at 80 in _trend_following_strategy
            'mr': 115.0,        # Theoretical max (60+40+15)
            'breakout': 70.0,   # Clamped at 70 in _breakout_strategy
            'ml': 80.0,         # Max is 80 * 1.0 confidence
            'mtf': 50.0         # Max is 1 * 50 * 1.0 strength
        }
        
        max_score = strategy_max_abs_score.get(strategy_type, 100.0)
        
        if max_score == 0:
            return 0.0
            
        # Clamp the score to the defined range first
        clamped_score = max(-max_score, min(max_score, score))
        
        # Normalize to a -100 to 100 range
        return (clamped_score / max_score) * 100.0

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

    def _integrate_strategy_optimizer(self, symbol: str, indicators: Dict, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Integrate StrategyOptimizer for dynamic weight adjustment"""
        try:
            if self.strategy_optimizer is None:
                return self.strategy_weights
                
            # Get market regime and volatility for optimization
            market_regime = indicators.get('market_regime', 'neutral')
            volatility = indicators.get('atr_percent', 2) / 100  # Convert to decimal
            
            # Get recent performance from database
            recent_performance = self._get_recent_performance(symbol, days=7)
            
            # Optimize weights using StrategyOptimizer
            optimized_weights = self.strategy_optimizer.optimize_weights(
                market_regime=market_regime,
                volatility=volatility,
                recent_performance=recent_performance,
                aggressiveness=self.aggressiveness
            )
            
            # Blend with current weights for stability
            blend_factor = 0.3  # 30% towards optimized weights
            final_weights = {}
            for strategy in self.strategy_weights:
                current = self.strategy_weights.get(strategy, 0)
                optimized = optimized_weights.get(strategy, current)
                final_weights[strategy] = current * (1 - blend_factor) + optimized * blend_factor
            
            # Normalize
            total = sum(final_weights.values())
            if total > 0:
                final_weights = {k: v/total for k, v in final_weights.items()}
            
            self.logger.info(f"Optimized weights for {symbol}: {final_weights}")
            return final_weights
            
        except Exception as e:
            self.logger.error(f"Strategy optimization failed for {symbol}: {e}")
            return self.strategy_weights

    def _get_recent_performance(self, symbol: str, days: int = 7) -> Dict:
        """Get recent trading performance for optimization"""
        try:
            if self.database is None:
                return {}
                
            # Get recent trades from database
            recent_trades = self.database.get_historical_trades(symbol=symbol, days=days)
            
            if recent_trades.empty:
                return {}
                
            winning_trades = recent_trades[recent_trades['pnl_percent'] > 0]
            
            return {
                'win_rate': (len(winning_trades) / len(recent_trades)) * 100,
                'avg_pnl': recent_trades['pnl_percent'].mean(),
                'total_pnl': recent_trades['pnl_percent'].sum(),
                'trade_count': len(recent_trades),
                'sharpe_ratio': self._calculate_sharpe_ratio(recent_trades['pnl_percent']),
                'max_drawdown': self._calculate_max_drawdown(recent_trades['pnl_percent'])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recent performance for {symbol}: {e}")
            return {}

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(365)  # Annualized

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100  # Return as percentage

    # ENHANCED: Update volume filter with crypto-specific considerations
    def _should_trade_with_volume(self, indicators: Dict) -> Tuple[bool, str]:
        """Enhanced volume filter with crypto-specific adjustments"""
        try:
            # Base volume checks
            volume_ratio = indicators.get('volume_ratio', 1)
            market_regime = indicators.get('market_regime', 'neutral')
            
            # Crypto-specific volume thresholds
            if self._is_crypto_symbol(indicators.get('symbol', '')):
                # Crypto markets can have lower volume thresholds during certain conditions
                if indicators.get('crypto_regime') == 'crypto_bull':
                    min_volume_ratio = 0.25  # Lower threshold in bull markets
                elif indicators.get('on_chain_strength') == 'healthy':
                    min_volume_ratio = 0.3   # Healthy on-chain can compensate for lower volume
                else:
                    min_volume_ratio = 0.4
            else:
                # Traditional thresholds for non-crypto
                min_volume_ratio = 0.4

            if volume_ratio < min_volume_ratio:
                return False, f"Low volume ratio: {volume_ratio:.3f} < {min_volume_ratio}"

            # Additional crypto-specific volume checks
            if self._is_crypto_symbol(indicators.get('symbol', '')):
                # Check exchange flows
                net_flow = indicators.get('exchange_net_flow', 0)
                if net_flow > 1000000:  # Large inflows to exchanges (potential selling pressure)
                    return False, f"High exchange inflows: {net_flow:,.0f}"
                
                # Check funding rates for extremes
                funding_rate = indicators.get('funding_rate', 0)
                if abs(funding_rate) > 0.001:  # Extreme funding rates
                    return False, f"Extreme funding rate: {funding_rate:.4f}"

            return True, f"Volume conditions acceptable (ratio: {volume_ratio:.3f})"

        except Exception as e:
            self.logger.error(f"Error in crypto volume filter: {e}")
            return True, "Volume check error - proceeding with caution"

    def _calculate_aggressive_composite_score(self, indicators: Dict, signals: Dict, 
                                            ml_result: Dict, mtf_signals: Dict, 
                                            aggressiveness: str) -> Dict[str, float]:
        """Calculates composite score using individual strategies and returns breakdowns."""
        try:
            # 1. Define base weights that sum to 1.0
            # We prioritize optimized weights if they exist, otherwise use aggressiveness defaults.
            if self.strategy_optimizer:
                optimized_weights = self._integrate_strategy_optimizer(
                    indicators.get('symbol', 'UNKNOWN'), indicators, None
                )
            else:
                optimized_weights = None

            if optimized_weights and any(optimized_weights.values()):
                weights = optimized_weights
                # Ensure MTF is included if optimizer doesn't provide it
                if 'mtf' not in weights:
                    weights['mtf'] = 0.10
                    # Re-normalize if we had to add MTF
                    total_w = sum(weights.values())
                    if total_w > 0:
                        weights = {k: v / total_w for k, v in weights.items()}
            else:
                # Fallback to aggressiveness-based weights (now summing to 1.0)
                if aggressiveness == "conservative":
                    weights = {
                        'trend_following': 0.35, 'mean_reversion': 0.25,
                        'breakout': 0.15, 'ml_prediction': 0.15, 'mtf': 0.10
                    }
                elif aggressiveness == "moderate":
                    weights = {
                        'trend_following': 0.25, 'mean_reversion': 0.30,
                        'breakout': 0.20, 'ml_prediction': 0.15, 'mtf': 0.10
                    }
                elif aggressiveness == "aggressive":
                    weights = {
                        'trend_following': 0.20, 'mean_reversion': 0.30,
                        'breakout': 0.25, 'ml_prediction': 0.15, 'mtf': 0.10
                    }
                elif aggressiveness == "high": 
                    weights = {
                        'trend_following': 0.15, 'mean_reversion': 0.35,
                        'breakout': 0.25, 'ml_prediction': 0.15, 'mtf': 0.10
                    }
                else:
                    weights = {
                        'trend_following': 0.35, 'mean_reversion': 0.25,
                        'breakout': 0.15, 'ml_prediction': 0.15, 'mtf': 0.10
                    }

            # 2. Adjust ML weight based on confidence
            ml_confidence = ml_result.get('confidence', 0)
            ml_weight_multiplier = 0.5 if ml_confidence < 0.7 else 1.0

            original_ml_weight = weights.get('ml_prediction', 0)
            reduced_ml_weight = original_ml_weight * ml_weight_multiplier
            weights['ml_prediction'] = reduced_ml_weight
            weight_difference = original_ml_weight - reduced_ml_weight

            # Distribute the reduced weight proportionally among other strategies
            other_weights_total = sum(w for k, w in weights.items() if k != 'ml_prediction')
            if other_weights_total > 0 and weight_difference > 0:
                for k in weights:
                    if k != 'ml_prediction':
                        proportion = weights[k] / other_weights_total
                        weights[k] += weight_difference * proportion
            
            # Ensure weights still sum to 1 after adjustment
            total_final_weight = sum(weights.values())
            if total_final_weight > 0:
                normalized_weights = {k: v / total_final_weight for k, v in weights.items()}
            else:
                normalized_weights = weights

            self.logger.debug(f"[{indicators.get('symbol', 'UNK')}] Final normalized weights: {normalized_weights}")

            # 3. Calculate individual raw scores
            trend_score_raw = self._trend_following_strategy(indicators)
            mean_reversion_score_raw = self._mean_reversion_strategy(indicators)
            breakout_score_raw = self._breakout_strategy(indicators, mtf_signals)
            
            ml_raw_pred = ml_result.get('raw_prediction', 0)
            ml_confidence_pred = ml_result.get('confidence', 0)
            
            if ml_raw_pred == 2:   # Strong buy
                ml_score_raw = 80 * ml_confidence_pred
            elif ml_raw_pred == 1: # Buy
                ml_score_raw = 40 * ml_confidence_pred
            elif ml_raw_pred == -1: # Sell
                ml_score_raw = -40 * ml_confidence_pred
            elif ml_raw_pred == -2: # Strong sell
                ml_score_raw = -80 * ml_confidence_pred
            else:                  # Hold
                ml_score_raw = 0.0 # HOLD prediction should be neutral, not a penalty

            mtf_alignment = mtf_signals.get('timeframe_alignment', 0)
            mtf_strength = mtf_signals.get('alignment_strength', 0.5)
            mtf_score_raw = mtf_alignment * 50 * mtf_strength

            # 4. Normalize all scores to a -100 to 100 scale
            trend_score_norm = self._normalize_score(trend_score_raw, 'trend')
            mr_score_norm = self._normalize_score(mean_reversion_score_raw, 'mr')
            breakout_score_norm = self._normalize_score(breakout_score_raw, 'breakout')
            ml_score_norm = self._normalize_score(ml_score_raw, 'ml')
            mtf_score_norm = self._normalize_score(mtf_score_raw, 'mtf')

            # 5. Calculate final composite score
            composite = (
                trend_score_norm * normalized_weights.get('trend_following', 0) +
                mr_score_norm * normalized_weights.get('mean_reversion', 0) +
                breakout_score_norm * normalized_weights.get('breakout', 0) +
                ml_score_norm * normalized_weights.get('ml_prediction', 0) +
                mtf_score_norm * normalized_weights.get('mtf', 0)
            )

            # Return raw scores for logging, but composite score is normalized
            return {
                'composite_score': composite,
                'trend_score': trend_score_raw,
                'mr_score': mean_reversion_score_raw,
                'breakout_score': breakout_score_raw,
                'ml_score': ml_score_raw,
                'mtf_score': mtf_score_raw
            }

        except Exception as e:
            if hasattr(self, 'logger'):
                symbol = indicators.get('symbol', 'ALL') if isinstance(indicators, dict) else 'ALL'
                self.logger.error(f"[{symbol}] Error calculating composite score: {e}", exc_info=True)

            if self.error_handler:
                symbol = indicators.get('symbol', 'ALL') if isinstance(indicators, dict) else 'ALL'
                self.error_handler.handle_trading_error(e, symbol, "composite_score_calculation")

            return {
                'composite_score': 0, 'trend_score': 0, 'mr_score': 0,
                'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0
            }

        except Exception as e:
            if hasattr(self, 'logger'):
                symbol = indicators.get('symbol', 'ALL') if isinstance(indicators, dict) else 'ALL'
                self.logger.error(f"[{symbol}] Error calculating composite score: {e}", exc_info=True)

            if self.error_handler:
                symbol = indicators.get('symbol', 'ALL') if isinstance(indicators, dict) else 'ALL'
                self.error_handler.handle_trading_error(e, symbol, "composite_score_calculation")

            return {
                'composite_score': 0, 'trend_score': 0, 'mr_score': 0,
                'breakout_score': 0, 'ml_score': 0, 'mtf_score': 0
            }

    def _get_regime_aware_thresholds(self, market_regime: str, volatility_regime: str, aggressiveness: str) -> Dict[str, float]:
        """Get dynamic thresholds based on market regime and volatility"""
        base_thresholds = {
            "conservative": {'buy': 20, 'sell': -20, 'strong': 40, 'position_cap': 0.15},
            "moderate": {'buy': 15, 'sell': -15, 'strong': 30, 'position_cap': 0.20},
            "aggressive": {'buy': 10, 'sell': -10, 'strong': 25, 'position_cap': 0.25},
            "high": {'buy': 5, 'sell': -5, 'strong': 20, 'position_cap': 0.30}
        }
        
        thresholds = base_thresholds.get(aggressiveness, base_thresholds["conservative"]).copy()
        
        # Adjust based on market regime
        regime_adjustments = {
            "bull_trend": {'buy': -3, 'sell': 2, 'strong': -5, 'position_cap': 0.05},
            "bear_trend": {'buy': 2, 'sell': -3, 'strong': 5, 'position_cap': -0.05},
            "ranging": {'buy': 5, 'sell': -5, 'strong': 10, 'position_cap': -0.08},
            "high_volatility": {'buy': 8, 'sell': -8, 'strong': 15, 'position_cap': -0.10},
            "low_volatility": {'buy': -2, 'sell': 2, 'strong': -3, 'position_cap': 0.03}
        }
        
        # Apply regime adjustments
        if market_regime in regime_adjustments:
            adj = regime_adjustments[market_regime]
            for key in thresholds:
                if key in adj:
                    thresholds[key] += adj[key]
        
        # Apply volatility adjustments
        if volatility_regime == "high":
            thresholds.update({'buy': thresholds['buy'] + 5, 'sell': thresholds['sell'] - 5, 'position_cap': thresholds['position_cap'] - 0.05})
        elif volatility_regime == "low":
            thresholds.update({'buy': thresholds['buy'] - 2, 'sell': thresholds['sell'] + 2, 'position_cap': thresholds['position_cap'] + 0.03})
        
        # Ensure minimum values
        thresholds['buy'] = max(0, thresholds['buy'])
        thresholds['sell'] = min(0, thresholds['sell'])
        thresholds['position_cap'] = max(0.05, min(0.5, thresholds['position_cap']))
        
        return thresholds

    def _apply_regime_aware_filters(self, decision: Dict, indicators: Dict) -> Dict:
        try:
            market_regime = indicators.get('market_regime', 'neutral')
            crypto_regime = indicators.get('crypto_regime', 'neutral')
            
            # NEW: Comprehensive Overbought/Oversold Filter
            rsi_14 = indicators.get('rsi_14', 50)
            stoch_k = indicators.get('stoch_k', 50)
            bb_position = indicators.get('bb_position', 0.5)
            price_vs_high_20 = indicators.get('price_vs_high_20', 1.0)
            
            # Overbought conditions for BUY signals
            if decision['action'] == 'BUY':
                overbought_signals = 0
                if rsi_14 > 70: overbought_signals += 2  # Weight RSI more heavily
                if stoch_k > 80: overbought_signals += 1
                if bb_position > 0.8: overbought_signals += 1
                if price_vs_high_20 > 0.98: overbought_signals += 1
                
                # Trigger with just 1 strong signal or 2 weak signals
                if overbought_signals >= 2:
                    # Much stronger penalty
                    confidence_reduction = max(0.3, 1.0 - (overbought_signals * 0.25))
                    decision['confidence'] = decision['confidence'] * confidence_reduction
                    
                    # Consider changing action to HOLD in extreme cases
                    if overbought_signals >= 3 and decision['confidence'] < 40:
                        decision['action'] = 'HOLD'
                        decision['execute_reason'] = 'Extreme overbought conditions'
            
            # Oversold conditions for SELL signals  
            elif decision['action'] == 'SELL':
                oversold_signals = 0
                if rsi_14 < 30: oversold_signals += 1
                if stoch_k < 20: oversold_signals += 1
                if bb_position < 0.2: oversold_signals += 1
                if price_vs_high_20 < 0.90: oversold_signals += 1  # Well below 20-period high
                
                if oversold_signals >= 2:
                    confidence_reduction = max(0.4, 1.0 - (oversold_signals * 0.2))
                    decision['confidence'] = decision['confidence'] * confidence_reduction
                    
                    weakness_msg = f"Multiple oversold signals ({oversold_signals}/4)"
                    if 'weaknesses' not in decision['trade_quality']:
                        decision['trade_quality']['weaknesses'] = []
                    decision['trade_quality']['weaknesses'].append(weakness_msg)

            # Existing crypto-specific regime adjustments (keep all original code below)
            if crypto_regime == 'crypto_bull':
                if decision['action'] == 'BUY':
                    decision['confidence'] = min(95, decision['confidence'] * 1.1)
                elif decision['action'] == 'SELL':
                    decision['confidence'] = decision['confidence'] * 0.9
                    
            elif crypto_regime == 'crypto_bear':
                if decision['action'] == 'SELL':
                    decision['confidence'] = min(95, decision['confidence'] * 1.1)
                elif decision['action'] == 'BUY':
                    decision['confidence'] = decision['confidence'] * 0.9

            # On-chain strength adjustments
            on_chain_strength = indicators.get('on_chain_strength', 'neutral')
            if on_chain_strength == 'healthy' and decision['action'] == 'BUY':
                decision['confidence'] = min(95, decision['confidence'] * 1.05)
            elif on_chain_strength == 'weak' and decision['action'] == 'BUY':
                decision['confidence'] = decision['confidence'] * 0.95

            # Fear & Greed adjustments (contrarian)
            fear_greed = indicators.get('fear_greed_index', 50)
            if fear_greed < 30:  # Extreme fear - boost buy confidence
                if decision['action'] == 'BUY':
                    decision['confidence'] = min(95, decision['confidence'] * 1.08)
            elif fear_greed > 70:  # Extreme greed - boost sell confidence
                if decision['action'] == 'SELL':
                    decision['confidence'] = min(95, decision['confidence'] * 1.08)

            # Halving cycle adjustments
            cycle_phase = indicators.get('cycle_phase', 'neutral')
            if cycle_phase in ['accumulation_phase', 'early_bull']:
                if decision['action'] == 'BUY':
                    decision['position_size'] = min(
                        decision.get('portfolio_value', 10000) * 0.3,
                        decision['position_size'] * 1.2
                    )

            return decision

        except Exception as e:
            self.logger.error(f"Error applying crypto regime filters: {e}")
            return decision

    # ENHANCED: Update signal quality assessment with crypto metrics
    def _assess_signal_quality_enhanced(self, indicators: Dict, signals: Dict, 
                                    ml_result: Dict, action: str, composite_score: float) -> Dict:
        """Enhanced signal quality assessment with crypto-specific factors"""
        quality_score = 0
        strengths = []
        weaknesses = []
        filters = []
        
        # 1. Volume Quality (15 points) - reduced weight for crypto
        volume_ratio = indicators.get('volume_ratio', 1)
        volume_ok, volume_reason = self._should_trade_with_volume(indicators)
        if volume_ok:
            quality_score += 15
            strengths.append("Good volume confirmation")
        else:
            weaknesses.append(f"Volume issue: {volume_reason}")
            filters.append("VOLUME_FILTER")
        
        # 2. Crypto Regime Alignment (25 points) - NEW
        crypto_regime = indicators.get('crypto_regime', 'neutral')
        crypto_confidence = indicators.get('crypto_regime_confidence', 0.5)
        
        if crypto_confidence > 0.7:
            quality_score += 25
            strengths.append(f"Strong {crypto_regime} crypto regime")
        elif crypto_confidence > 0.5:
            quality_score += 15
            strengths.append(f"Moderate {crypto_regime} crypto regime")
        else:
            quality_score += 5
            weaknesses.append("Unclear crypto regime")
        
        # 3. On-Chain Strength (20 points) - NEW
        on_chain_strength = indicators.get('on_chain_strength', 'neutral')
        if on_chain_strength == 'healthy':
            quality_score += 20
            strengths.append("Healthy on-chain metrics")
        elif on_chain_strength == 'weak':
            quality_score += 5
            weaknesses.append("Weak on-chain metrics")
        else:
            quality_score += 10
        
        # 4. Signal Consistency (20 points) - reduced weight
        alignment_ok, alignment_direction, alignment_strength = self._check_signal_alignment({
            'trend_score': signals.get('trend_strength', 0),
            'mr_score': signals.get('mr_strength', 0),
            'breakout_score': signals.get('breakout_strength', 0),
            'ml_score': ml_result.get('confidence', 0) * 100,
            'mtf_score': indicators.get('mtf_alignment', 0)
        })
        
        if alignment_ok and alignment_strength > 0.6:
            quality_score += 20
            strengths.append(f"Strong signal alignment ({alignment_direction})")
        elif alignment_ok:
            quality_score += 15
            strengths.append(f"Moderate signal alignment ({alignment_direction})")
        else:
            quality_score += 5
            weaknesses.append("Mixed signal alignment")
            filters.append("ALIGNMENT_FILTER")
        
        # 5. Market Sentiment (10 points) - NEW
        fear_greed = indicators.get('fear_greed_index', 50)
        if 25 <= fear_greed <= 75:  # Neutral range
            quality_score += 10
            strengths.append("Neutral market sentiment")
        elif fear_greed < 25 and action == 'BUY':  # Extreme fear - good for contrarian buys
            quality_score += 15
            strengths.append("Extreme fear - contrarian opportunity")
        elif fear_greed > 75 and action == 'SELL':  # Extreme greed - good for contrarian sells
            quality_score += 15
            strengths.append("Extreme greed - contrarian opportunity")
        else:
            quality_score += 5
            weaknesses.append("Challenging sentiment conditions")
        
        # 6. Volatility Context (10 points)
        atr_percent = indicators.get('atr_percent', 0)
        if 1.0 < atr_percent < 3.0:
            quality_score += 10
            strengths.append("Ideal volatility for trading")
        elif atr_percent <= 1.0:
            quality_score += 5
            weaknesses.append("Low volatility - reduced edge")
        else:
            quality_score += 3
            weaknesses.append("High volatility - increased risk")
            filters.append("VOLATILITY_FILTER")
        
        # Cap at 100
        quality_score = min(100, quality_score)
        
        return {
            'quality_score': quality_score,
            'quality_rating': self._get_quality_rating(quality_score),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'active_filters': filters,
            'execution_confidence': self._calculate_execution_confidence(quality_score, composite_score, action)
        }

    def _calculate_execution_confidence(self, quality_score: float, composite_score: float, action: str) -> float:
        """Calculate execution confidence based on quality and score"""
        if action == 'HOLD':
            return 0
        
        base_confidence = quality_score / 100.0
        
        # Adjust based on composite score strength
        score_strength = min(1.0, abs(composite_score) / 50.0)  # Normalize to 0-1
        execution_confidence = base_confidence * (0.7 + 0.3 * score_strength)
        
        return min(0.95, execution_confidence)

    def _should_execute_trade(self, decision: Dict) -> Tuple[bool, str]:
        """Dynamic trend-aware execution criteria"""
        
        action = decision['action']
        market_regime = decision.get('market_regime', 'neutral')
        trend_score = decision.get('trend_score', 0)
        ml_confidence = decision['ml_prediction'].get('confidence', 0)
        ml_prediction = decision['ml_prediction'].get('raw_prediction', 0)
        composite_score = decision.get('composite_score', 0)
        
        # DYNAMIC TREND THRESHOLDS BASED ON MARKET REGIME
        trend_thresholds = {
            'bull_trend': {'buy': 15, 'sell': -30},
            'bear_trend': {'buy': 30, 'sell': -15},
            'neutral': {'buy': 20, 'sell': -20},
            'high_volatility': {'buy': 25, 'sell': -25}
        }
        
        regime = market_regime if market_regime in trend_thresholds else 'neutral'
        buy_threshold = trend_thresholds[regime]['buy']
        sell_threshold = trend_thresholds[regime]['sell']
        
        # TREND SCORE VALIDATION
        if action == 'BUY' and trend_score < buy_threshold:
            return False, f"Trend score {trend_score} below {regime} buy threshold {buy_threshold}"
        
        if action == 'SELL' and trend_score > sell_threshold:
            return False, f"Trend score {trend_score} above {regime} sell threshold {sell_threshold}"
        
        if ml_confidence > 0.75:
            if (action == 'BUY' and ml_prediction <= 0) or (action == 'SELL' and ml_prediction >= 0):
                return False, f"High-confidence ML veto (conf: {ml_confidence:.1%}, pred: {ml_prediction})"
        
        # Add signal alignment requirement
        if not decision.get('signal_alignment', {}).get('aligned', False):
            return False, "Signals not aligned across strategies"
        
        return True, "Execution approved"

    def _trend_following_strategy(self, indicators: Dict) -> float:
            try:
                score = 0

                # --- FIX 3 Check: Ensure indicators are valid numbers ---
                ema_8 = float(indicators.get('ema_8', 0))
                ema_21 = float(indicators.get('ema_21', 0))
                ema_55 = float(indicators.get('ema_55', 0))
                ema_89 = float(indicators.get('ema_89', 0))
                macd = float(indicators.get('macd', 0))
                macd_signal = float(indicators.get('macd_signal', 0))
                sma_50 = float(indicators.get('sma_50', 0)) # Uses the newly calculated SMA 50
                hma = float(indicators.get('hma', 0))
                current_price = float(indicators.get('bb_middle', indicators.get('ema_8', 0))) # Use bb_middle or ema_8 as fallback

                # --- FIX 3 Logic Check: EMA Alignment ---
                # Award points based on the *degree* of alignment, not just all-or-nothing
                ema_bullish_points = 0
                if ema_8 > ema_21: ema_bullish_points += 10
                if ema_21 > ema_55: ema_bullish_points += 10
                if ema_55 > ema_89: ema_bullish_points += 10

                ema_bearish_points = 0
                if ema_8 < ema_21: ema_bearish_points -= 10
                if ema_21 < ema_55: ema_bearish_points -= 10
                if ema_55 < ema_89: ema_bearish_points -= 10

                # Assign score based on strongest alignment (max 30 points)
                if ema_bullish_points >= 20: # Strong bullish alignment
                    score += 30
                elif ema_bearish_points <= -20: # Strong bearish alignment
                    score -= 30
                #else: score remains 0 for mixed EMAs

                # --- FIX 3 Logic Check: MACD ---
                # Use a threshold to avoid noise near zero crossover
                macd_diff = macd - macd_signal
                if macd_diff > abs(macd_signal * 0.01): # Bullish cross with threshold
                    score += 20
                elif macd_diff < -abs(macd_signal * 0.01): # Bearish cross with threshold
                    score -= 20

                # --- FIX 1 Usage: SMA 50 ---
                # Ensure sma_50 is valid before comparison
                if sma_50 > 0:
                    if current_price > sma_50:
                        score += 20
                    else:
                        score -= 20
                #else: No points if SMA 50 calculation failed

                # --- FIX 3 Logic Check: HMA ---
                # Compare HMA trend direction relative to price (HMA rising/falling could be added)
                if hma < current_price: # Price above HMA suggests uptrend
                    score += 10
                elif hma > current_price: # Price below HMA suggests downtrend
                    score -= 10

                # --- FIX 3 Clamp Score: Ensure score is within -100 to +100 ---
                score = max(-80, min(80, score)) # Limit range based on components (30+20+20+10 = 80)

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
            
            # Proper cascading logic
            if rsi_14 > 75:
                score -= 60  # Extreme overbought
            elif rsi_14 > 70:
                score -= 40  # Overbought
            elif rsi_14 > 65:
                score -= 15  # Warning zone
            elif rsi_14 < 25:
                score += 40  # Extreme oversold  
            elif rsi_14 < 30:
                score += 30  # Oversold
                    
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
                symbol = indicators.get('symbol', 'UNKNOWN')

                resistance = float(indicators.get('resistance', 0))
                support = float(indicators.get('support', 0))
                current_price = float(indicators.get('close', indicators.get('bb_middle', indicators.get('ema_8', 0)))) # Use actual close if available
                rsi_14 = float(indicators.get('rsi_14', 50))
                volume_ratio = float(indicators.get('volume_ratio', 1))
                atr_percent = float(indicators.get('atr_percent', 2)) # Default to 2% if missing
                alignment_strength = float(mtf_signals.get('alignment_strength', 0))

                self.logger.debug(f"[{symbol}] Breakout Start: Initial score = {score}, Price={current_price:.4f}, Res={resistance:.4f}, Sup={support:.4f}, RSI={rsi_14:.2f}, ATR%={atr_percent:.2f}, VolRatio={volume_ratio:.2f}, MTFStr={alignment_strength:.2f}") # Log Start with key inputs

                # --- S/R Check with Detailed Logging ---
                resistance_distance = (resistance - current_price) / current_price if current_price > 0 else 0
                support_distance = (current_price - support) / current_price if current_price > 0 else 0
                sr_score_change = 0

                # Get additional indicators for overbought detection
                stoch_k = float(indicators.get('stoch_k', 50))

                # ENHANCED: Resistance Check with Overbought Awareness
                res_cond1 = resistance > 0
                res_cond2 = 0 < resistance_distance < 0.015

                self.logger.debug(f"[{symbol}] Res Check Values: res_gt_0={res_cond1}, dist_ok={res_cond2} (dist={resistance_distance:.6f}), rsi={rsi_14:.2f}, stoch_k={stoch_k:.2f}")

                if res_cond1 and res_cond2:
                    if (rsi_14 > 65 and stoch_k > 80) or rsi_14 > 75:
                        sr_score_change = -40  # Strong penalty
                        self.logger.debug(f"[{symbol}] Breakout S/R Check: STRONG PENALTY {sr_score_change} (Severely Overbought Near Resistance)")
                    elif rsi_14 > 60 or stoch_k > 75:
                        self.logger.debug(f"[{symbol}] Breakout S/R Check: PENALIZED {sr_score_change} (Overbought Near Resistance)")
                        sr_score_change = -25  # Moderate penalty  
                    elif rsi_14 > 55:
                        self.logger.debug(f"[{symbol}] Breakout S/R Check: CAUTIOUS +{sr_score_change} (Near Resistance)")
                        sr_score_change = -10  # Mild penalty
                    else:
                        sr_score_change = 25   # Only reward if NOT overbought
                        self.logger.debug(f"[{symbol}] Breakout S/R Check: PASSED +{sr_score_change} (Near Resistance)")

                else:
                    # Support Check Branch (only if resistance check failed)
                    sup_cond1 = support > 0
                    sup_cond2 = 0 < support_distance < 0.015
                    sup_cond3 = rsi_14 < 45
                    self.logger.debug(f"[{symbol}] Sup Check Values: sup_gt_0={sup_cond1}, dist_ok={sup_cond2} (dist={support_distance:.6f}), rsi_ok={sup_cond3} (rsi={rsi_14:.2f})")
                    if sup_cond1 and sup_cond2 and sup_cond3:
                        sr_score_change = -25
                        self.logger.debug(f"[{symbol}] Breakout S/R Check: PASSED {sr_score_change} (Near Support)")
                    else:
                        self.logger.debug(f"[{symbol}] Breakout S/R Check: FAILED +0 (Conditions not met)")

                score += sr_score_change
                self.logger.debug(f"[{symbol}] Breakout Score after S/R = {score}") # Log after S/R

                # --- Volume Check ---
                vol_score_change = 0
                vol_cond = volume_ratio > 1.5
                self.logger.debug(f"[{symbol}] Vol Check Values: vol_ratio={volume_ratio:.2f}, condition_met={vol_cond}")
                if vol_cond:
                    vol_score_change = 20
                    self.logger.debug(f"[{symbol}] Breakout Volume Check: PASSED +{vol_score_change}")
                else:
                    self.logger.debug(f"[{symbol}] Breakout Volume Check: FAILED +0")
                score += vol_score_change
                self.logger.debug(f"[{symbol}] Breakout Score after Volume = {score}") # Log after Volume

                # --- MTF Alignment Check ---
                mtf_score_change = 0
                mtf_cond = alignment_strength >= 0.7
                self.logger.debug(f"[{symbol}] MTF Check Values: align_strength={alignment_strength:.2f}, condition_met={mtf_cond}")
                if mtf_cond: # Require stronger alignment
                    mtf_alignment_direction = int(mtf_signals.get('timeframe_alignment', 0))
                    mtf_score_change = (15 * mtf_alignment_direction)
                    self.logger.debug(f"[{symbol}] Breakout MTF Check: PASSED +{mtf_score_change} (Direction={mtf_alignment_direction})")
                else:
                    self.logger.debug(f"[{symbol}] Breakout MTF Check: FAILED +0")
                score += mtf_score_change
                self.logger.debug(f"[{symbol}] Breakout Score after MTF = {score}") # Log after MTF

                # --- ATR/Volatility Check with Detailed Logging ---
                atr_score_change = 0
                atr_cond = atr_percent < 1.0
                self.logger.debug(f"[{symbol}] ATR Check Values: atr_percent={atr_percent:.4f}, condition_met={atr_cond}")
                if atr_cond:
                    atr_score_change = 10
                    self.logger.debug(f"[{symbol}] Breakout ATR Check: PASSED +{atr_score_change}")
                else:
                    self.logger.debug(f"[{symbol}] Breakout ATR Check: FAILED +0")
                score += atr_score_change
                self.logger.debug(f"[{symbol}] Breakout Score after ATR = {score}") # Log after ATR

                # --- Clamp Score ---
                final_score = max(-70, min(70, score))
                if final_score != score:
                    self.logger.debug(f"[{symbol}] Breakout Score Clamped: {score} -> {final_score}") # Log clamping
                else:
                    self.logger.debug(f"[{symbol}] Breakout Final Score: {final_score} (No clamping needed)") # Log final score

                return final_score

            except Exception as e:
                # Log the error using the class logger *before* passing to error handler
                # Try getting symbol even in exception for context
                symbol_in_ex = indicators.get('symbol', 'UNKNOWN') if isinstance(indicators, dict) else 'UNKNOWN'
                self.logger.error(f"[{symbol_in_ex}] Error in breakout strategy calculation: {e}", exc_info=True)
                if self.error_handler:
                    # Pass symbol if available
                    self.error_handler.handle_trading_error(e, symbol_in_ex, "breakout_strategy")
                # Fallback to neutral score
                return 0

    def _determine_aggressive_action(self, composite_score: float, signals: Dict, ml_result: Dict, aggressiveness: str) -> tuple:
        try:
            market_regime = signals.get('market_regime', 'neutral')
            
            # --- FIX: Define volatility_regime before use ---
            volatility_regime = 'normal' # Default
            atr_percent = signals.get('atr_percent', 0) # Get ATR percent from signals/indicators
            if atr_percent > 3:
                volatility_regime = 'high'
            elif atr_percent < 1:
                volatility_regime = 'low'
            # --- END FIX ---

            thresholds = self._get_regime_aware_thresholds(
                market_regime, volatility_regime, aggressiveness
            )

            buy_threshold = thresholds['buy']
            sell_threshold = thresholds['sell']
            strong_threshold = thresholds['strong']

            if aggressiveness == "conservative":
                confidence_boost_factor = 0.5
                strong_signal_boost = 1.2
            elif aggressiveness == "moderate":
                confidence_boost_factor = 0.6
                strong_signal_boost = 1.3
            elif aggressiveness == "aggressive":
                confidence_boost_factor = 0.7
                strong_signal_boost = 1.4
            elif aggressiveness == "high":
                confidence_boost_factor = 0.8
                strong_signal_boost = 1.5
            else:
                confidence_boost_factor = 0.5
                strong_signal_boost = 1.2

            ml_confidence = ml_result.get('confidence', 0)

            confidence_boost = min(ml_confidence * 25 * confidence_boost_factor, 20)

            if composite_score >= self.buy_threshold:
                action = 'BUY'
                base_confidence = min(composite_score, 70)
            elif composite_score <= self.sell_threshold:
                action = 'SELL' 
                base_confidence = min(abs(composite_score), 70)
            else:
                action = 'HOLD'
                base_confidence = 0

            # ML confidence adjustment - FIXED
            ml_confidence = ml_result.get('confidence', 0)
            ml_prediction = ml_result.get('raw_prediction', 0)
            
            # Only boost if ML agrees with action
            if (action == 'BUY' and ml_prediction in [1, 2]) or \
            (action == 'SELL' and ml_prediction in [-1, -2]):
                confidence_boost = min(ml_confidence * 25 * confidence_boost_factor, 20)
            else:
                # Reduce confidence if ML disagrees
                confidence_boost = -min(ml_confidence * 15 * confidence_boost_factor, 15)
            
            confidence = max(0, min(85, base_confidence + confidence_boost))
            
            return action, confidence

        except Exception as e:
            # --- FIX: Pass 'indicators' dict symbol if available ---
            symbol_for_error = signals.get('symbol', "ALL") if isinstance(signals, dict) else "ALL"
            # --- END FIX ---
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol_for_error, "determine_action")
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
                print(f"❌ Invalid aggressiveness level. Must be one of: {valid_levels}")
                return False
                
            old_level = self.aggressiveness
            self.aggressiveness = new_aggressiveness
            self.risk_manager.aggressiveness = new_aggressiveness
            self.risk_manager.config = RiskConfig.get_config(new_aggressiveness)
            self.risk_manager._set_parameters_from_config()
            
            print(f"🔄 Strategy aggressiveness changed from {old_level.upper()} to {new_aggressiveness.upper()}")
            
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

    def analyze_portfolio_correlation_advanced(self, symbols: List[str], 
                                            historical_data: Dict[str, pd.DataFrame],
                                            lookback_days: int = 30) -> Dict:
        """
        Advanced portfolio correlation analysis with regime-aware adjustments
        and dynamic correlation clustering.
        """
        try:
            if len(symbols) < 2:
                return {}
            
            # Calculate rolling correlations with multiple timeframes
            correlation_analysis = {
                'timeframe_correlations': {},
                'regime_aware_correlations': {},
                'correlation_clusters': {},
                'diversification_metrics': {},
                'correlation_regime': 'normal'
            }
            
            # Multi-timeframe correlation analysis
            timeframes = [5, 10, 20]  # days
            for tf in timeframes:
                tf_correlations = self._calculate_rolling_correlations(
                    symbols, historical_data, window=tf
                )
                correlation_analysis['timeframe_correlations'][f'{tf}day'] = tf_correlations
            
            # Regime-aware correlation adjustment
            regime_correlations = self._calculate_regime_aware_correlations(
                symbols, historical_data
            )
            correlation_analysis['regime_aware_correlations'] = regime_correlations
            
            # Dynamic correlation clustering
            clusters = self._perform_dynamic_correlation_clustering(
                symbols, historical_data, n_clusters=3
            )
            correlation_analysis['correlation_clusters'] = clusters
            
            # Diversification metrics
            diversification_metrics = self._calculate_advanced_diversification_metrics(
                symbols, historical_data, clusters
            )
            correlation_analysis['diversification_metrics'] = diversification_metrics
            
            # Correlation regime detection
            correlation_regime = self._detect_correlation_regime(
                correlation_analysis['timeframe_correlations']
            )
            correlation_analysis['correlation_regime'] = correlation_regime
            
            # Portfolio concentration risk
            concentration_risk = self._assess_portfolio_concentration_risk(
                symbols, historical_data, clusters
            )
            correlation_analysis['concentration_risk'] = concentration_risk
            
            if self.database:
                self.database.store_system_event(
                    "PORTFOLIO_CORRELATION_ANALYSIS",
                    {
                        'symbols': symbols,
                        'correlation_regime': correlation_regime,
                        'clusters': clusters,
                        'diversification_score': diversification_metrics.get('composite_score', 0),
                        'concentration_risk': concentration_risk
                    },
                    "INFO",
                    "Portfolio Analysis"
                )
            
            return correlation_analysis
            
        except Exception as e:
            self.logger.error(f"Error in advanced correlation analysis: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "correlation_analysis")
            return {}

    def _calculate_rolling_correlations(self, symbols: List[str], 
                                    historical_data: Dict[str, pd.DataFrame],
                                    window: int) -> pd.DataFrame:
        """Calculate rolling correlations with regime adjustments"""
        try:
            returns_data = {}
            
            for symbol in symbols:
                if symbol in historical_data and len(historical_data[symbol]) >= window:
                    returns = historical_data[symbol]['close'].pct_change().dropna()
                    if len(returns) >= window:
                        # Apply volatility weighting
                        volatility = returns.rolling(window).std()
                        weighted_returns = returns / (volatility + 1e-8)  # Avoid division by zero
                        returns_data[symbol] = weighted_returns.tail(window)
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            returns_df = pd.DataFrame(returns_data)
            rolling_corr = returns_df.rolling(window//2).corr().dropna()
            
            # Return the most recent correlation matrix
            if not rolling_corr.empty:
                # Get the latest complete correlation matrix
                latest_corr = rolling_corr.groupby(level=0).last()
                return latest_corr
            else:
                return returns_df.corr()
                
        except Exception as e:
            self.logger.error(f"Error calculating rolling correlations: {e}")
            return pd.DataFrame()

    def _calculate_average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
            """Calculates the average absolute correlation from the off-diagonal elements."""
            try:
                if correlation_matrix.empty or len(correlation_matrix) < 2:
                    return 0.0

                # Get the upper triangle values (excluding the diagonal k=1)
                upper_tri_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]

                if len(upper_tri_values) == 0:
                    return 0.0

                # Calculate the mean of the absolute values
                avg_corr = np.mean(np.abs(upper_tri_values))
                return avg_corr if not np.isnan(avg_corr) else 0.0

            except Exception as e:
                self.logger.error(f"Error calculating average correlation: {e}", exc_info=True)
                return 0.0 # Return neutral value on error

    def _calculate_regime_aware_correlations(self, symbols: List[str],
                                            historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate correlations adjusted for market regimes"""
        try:
            regime_correlations = {}

            # Define correlation regimes
            regimes = ['high_volatility', 'low_volatility', 'trending', 'ranging']

            for regime in regimes:
                regime_data = self._filter_data_by_regime(symbols, historical_data, regime)
                # Check if regime_data is usable (contains data for at least 2 symbols)
                valid_symbols_in_regime = [s for s in symbols if s in regime_data and isinstance(regime_data[s], pd.DataFrame) and not regime_data[s].empty and len(regime_data[s]) >= 20]

                if len(valid_symbols_in_regime) >= 2:
                    # Pass only the valid data for this regime
                    regime_data_filtered = {s: regime_data[s] for s in valid_symbols_in_regime}
                    corr_matrix = self._calculate_correlation_matrix(valid_symbols_in_regime, regime_data_filtered)

                    # Use a representative symbol for stability calculation
                    first_symbol = valid_symbols_in_regime[0]
                    stability_data = regime_data_filtered[first_symbol]

                    regime_correlations[regime] = {
                        'correlation_matrix': corr_matrix,
                        'average_correlation': self._calculate_average_correlation(corr_matrix),
                        # --- FIX: Ensure we pass window as integer ---
                        'regime_stability': self._calculate_regime_stability(stability_data, window=50)  # Explicit integer
                        # --- END FIX ---
                    }
                else:
                    self.logger.warning(f"Skipping regime '{regime}': Insufficient data for correlation ({len(valid_symbols_in_regime)} valid symbols)")
                    regime_correlations[regime] = {
                        'correlation_matrix': pd.DataFrame(),
                        'average_correlation': 0.0,
                        'regime_stability': 0.0
                    }

            return regime_correlations

        except Exception as e:
            self.logger.error(f"Error in regime-aware correlations: {e}", exc_info=True)
            return {}

    def _filter_data_by_regime(self, symbols: List[str], 
                            historical_data: Dict[str, pd.DataFrame],
                            regime: str) -> Dict[str, pd.DataFrame]:
        """Filter historical data based on market regime - FIXED VERSION"""
        filtered_data = {}
        
        for symbol in symbols:
            if symbol not in historical_data:
                continue
                
            df = historical_data[symbol].copy()
            
            # Ensure we have a valid DataFrame with required columns
            if df.empty or not {'close', 'high', 'low', 'volume'}.issubset(df.columns):
                continue
            
            # Simple regime classification based on price action
            try:
                if regime == 'high_volatility':
                    volatility = df['close'].pct_change().rolling(20).std()
                    mask = volatility > volatility.quantile(0.7)
                elif regime == 'low_volatility':
                    volatility = df['close'].pct_change().rolling(20).std()
                    mask = volatility < volatility.quantile(0.3)
                elif regime == 'trending':
                    returns = df['close'].pct_change(20)
                    mask = abs(returns) > returns.std()
                elif regime == 'ranging':
                    returns = df['close'].pct_change(20)
                    mask = abs(returns) < returns.std() * 0.5
                else:
                    mask = pd.Series(True, index=df.index)
                
                filtered_df = df[mask]
                
                # Only include if we have sufficient data after filtering
                if len(filtered_df) >= 20:
                    filtered_data[symbol] = filtered_df
                    
            except Exception as e:
                self.logger.warning(f"Error filtering data for {symbol} in regime {regime}: {e}")
                continue
        
        return filtered_data

    def _perform_dynamic_correlation_clustering(self, symbols: List[str],
                                            historical_data: Dict[str, pd.DataFrame],
                                            n_clusters: int = 3) -> Dict:
        """Perform dynamic correlation clustering using PCA and clustering algorithms"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import DBSCAN
            import scipy.cluster.hierarchy as sch
            
            # Calculate correlation matrix
            corr_matrix = self._calculate_correlation_matrix(symbols, historical_data)
            if corr_matrix.empty:
                return {}
            
            # Convert correlation matrix to distance matrix
            distance_matrix = 1 - corr_matrix.abs()
            
            # Hierarchical clustering
            linkage = sch.linkage(sch.distance.pdist(distance_matrix), method='ward')
            clusters_hierarchical = sch.fcluster(linkage, n_clusters, criterion='maxclust')
            
            # DBSCAN clustering for outlier detection
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
            clusters_dbscan = dbscan.fit_predict(distance_matrix)
            
            # Organize clusters
            clusters = {}
            for i, symbol in enumerate(symbols):
                cluster_info = {
                    'symbol': symbol,
                    'hierarchical_cluster': int(clusters_hierarchical[i]),
                    'dbscan_cluster': int(clusters_dbscan[i]),
                    'is_outlier': clusters_dbscan[i] == -1
                }
                
                cluster_key = f"cluster_{clusters_hierarchical[i]}"
                if cluster_key not in clusters:
                    clusters[cluster_key] = {
                        'symbols': [],
                        'average_intra_correlation': 0,
                        'cluster_volatility': 0,
                        'size': 0
                    }
                
                clusters[cluster_key]['symbols'].append(symbol)
            
            # Calculate cluster statistics
            for cluster_key, cluster_data in clusters.items():
                cluster_symbols = cluster_data['symbols']
                if len(cluster_symbols) > 1:
                    # Intra-cluster correlation
                    intra_correlations = []
                    for i, sym1 in enumerate(cluster_symbols):
                        for j, sym2 in enumerate(cluster_symbols):
                            if i < j:
                                intra_correlations.append(corr_matrix.loc[sym1, sym2])
                    
                    cluster_data['average_intra_correlation'] = np.mean(intra_correlations) if intra_correlations else 0
                    cluster_data['size'] = len(cluster_symbols)
                    
                    # Cluster volatility (average of member volatilities)
                    volatilities = []
                    for symbol in cluster_symbols:
                        if symbol in historical_data:
                            returns = historical_data[symbol]['close'].pct_change().dropna()
                            volatilities.append(returns.std())
                    cluster_data['cluster_volatility'] = np.mean(volatilities) if volatilities else 0
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in dynamic correlation clustering: {e}")
            return {}

    def _calculate_advanced_diversification_metrics(self, symbols: List[str],
                                                historical_data: Dict[str, pd.DataFrame],
                                                clusters: Dict) -> Dict:
        """Calculate advanced diversification metrics"""
        try:
            metrics = {}
            
            # Correlation matrix
            corr_matrix = self._calculate_correlation_matrix(symbols, historical_data)
            if corr_matrix.empty:
                return {}
            
            # 1. Effective Number of Correlated Assets (ENCA)
            eigenvals = np.linalg.eigvals(corr_matrix)
            effective_assets = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
            metrics['effective_assets'] = effective_assets
            
            # 2. Diversification Ratio
            volatilities = self._calculate_asset_volatilities(symbols, historical_data)
            avg_volatility = np.mean(list(volatilities.values())) if volatilities else 0.02
            portfolio_vol = self._calculate_portfolio_volatility(
                {s: 1/len(symbols) for s in symbols}, volatilities, corr_matrix
            )
            diversification_ratio = avg_volatility / portfolio_vol if portfolio_vol > 0 else 1
            metrics['diversification_ratio'] = diversification_ratio
            
            # 3. Cluster-based diversification score
            cluster_diversification = 1.0
            for cluster_key, cluster_data in clusters.items():
                cluster_weight = cluster_data['size'] / len(symbols)
                # Penalize large clusters
                if cluster_data['size'] > len(symbols) * 0.4:  # If cluster has >40% of symbols
                    cluster_diversification *= 0.7
                elif cluster_data['average_intra_correlation'] > 0.7:
                    cluster_diversification *= 0.8
            
            metrics['cluster_diversification'] = cluster_diversification
            
            # 4. Correlation stability score
            stability_score = self._calculate_correlation_stability(symbols, historical_data)
            metrics['correlation_stability'] = stability_score
            
            # 5. Composite diversification score
            composite_score = (
                metrics['effective_assets'] / len(symbols) * 0.3 +
                min(metrics['diversification_ratio'], 3) / 3 * 0.3 +
                metrics['cluster_diversification'] * 0.2 +
                metrics['correlation_stability'] * 0.2
            )
            metrics['composite_score'] = composite_score
            
            # 6. Diversification rating
            if composite_score > 0.8:
                metrics['rating'] = 'EXCELLENT'
            elif composite_score > 0.6:
                metrics['rating'] = 'GOOD'
            elif composite_score > 0.4:
                metrics['rating'] = 'FAIR'
            else:
                metrics['rating'] = 'POOR'
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification metrics: {e}")
            return {}

    def _calculate_correlation_stability(self, symbols: List[str],
                                    historical_data: Dict[str, pd.DataFrame],
                                    window: int = 20) -> float:
        """Calculate correlation stability over time"""
        try:
            if len(symbols) < 2:
                return 1.0
            
            rolling_correlations = []
            
            # Calculate rolling correlations
            for i in range(window, len(historical_data[symbols[0]]) - window, window//2):
                window_data = {}
                for symbol in symbols:
                    if symbol in historical_data:
                        window_data[symbol] = historical_data[symbol].iloc[i-window:i]
                
                if len(window_data) >= 2:
                    corr_matrix = self._calculate_correlation_matrix(symbols, window_data)
                    if not corr_matrix.empty:
                        rolling_correlations.append(corr_matrix.values)
            
            if len(rolling_correlations) < 2:
                return 1.0
            
            # Calculate correlation matrix changes
            changes = []
            for i in range(1, len(rolling_correlations)):
                change = np.mean(np.abs(rolling_correlations[i] - rolling_correlations[i-1]))
                changes.append(change)
            
            stability = 1.0 - min(np.mean(changes), 0.5)  # Normalize to 0-1
            return max(0.0, stability)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation stability: {e}")
            return 0.5

    def _detect_correlation_regime(self, timeframe_correlations: Dict) -> str:
        """Detect the current correlation regime"""
        try:
            avg_correlations = []
            
            for timeframe, corr_matrix in timeframe_correlations.items():
                if not corr_matrix.empty:
                    # Get upper triangle of correlation matrix (excluding diagonal)
                    upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                    avg_correlations.append(np.mean(np.abs(upper_tri)))
            
            if not avg_correlations:
                return 'normal'
            
            avg_correlation = np.mean(avg_correlations)
            
            if avg_correlation > 0.7:
                return 'high_correlation'
            elif avg_correlation < 0.3:
                return 'low_correlation'
            elif any(corr > 0.8 for corr in avg_correlations):
                return 'mixed_high'
            else:
                return 'normal'
                
        except Exception as e:
            self.logger.error(f"Error detecting correlation regime: {e}")
            return 'normal'

    def _assess_portfolio_concentration_risk(self, symbols: List[str],
                                        historical_data: Dict[str, pd.DataFrame],
                                        clusters: Dict) -> Dict:
        """Assess portfolio concentration risk"""
        try:
            risk_metrics = {}
            
            # 1. Cluster concentration risk
            cluster_sizes = [cluster_data['size'] for cluster_data in clusters.values()]
            if cluster_sizes:
                largest_cluster_ratio = max(cluster_sizes) / len(symbols)
                risk_metrics['largest_cluster_ratio'] = largest_cluster_ratio
            else:
                risk_metrics['largest_cluster_ratio'] = 0
            
            # 2. Volatility concentration
            volatilities = self._calculate_asset_volatilities(symbols, historical_data)
            if volatilities:
                vol_herfindahl = sum((vol / sum(volatilities.values())) ** 2 for vol in volatilities.values())
                risk_metrics['volatility_concentration'] = vol_herfindahl
            else:
                risk_metrics['volatility_concentration'] = 0
            
            # 3. Composite concentration risk
            concentration_risk = (
                risk_metrics['largest_cluster_ratio'] * 0.6 +
                risk_metrics['volatility_concentration'] * 0.4
            )
            risk_metrics['composite_concentration_risk'] = concentration_risk
            
            # 4. Risk rating
            if concentration_risk > 0.7:
                risk_metrics['rating'] = 'HIGH'
            elif concentration_risk > 0.5:
                risk_metrics['rating'] = 'MEDIUM_HIGH'
            elif concentration_risk > 0.3:
                risk_metrics['rating'] = 'MEDIUM'
            else:
                risk_metrics['rating'] = 'LOW'
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing concentration risk: {e}")
            return {}

    def detect_market_regime_transitions(self, symbol: str,
                                    historical_data: pd.DataFrame,
                                    lookback_periods: List[int] = [20, 50, 100]) -> Dict:
        """
        Detect market regime transitions and calculate transition probabilities.
        """
        try:
            # --- FIX: Better data validation ---
            if historical_data is None or len(historical_data) < 100:
                self.logger.warning(f"Insufficient data for regime transition analysis on {symbol}")
                return {
                    'current_regime': 'neutral',
                    'transition_probabilities': {},
                    'regime_stability': 0.5,
                    'expected_duration': 5,
                    'transition_alerts': []
                }

            transition_analysis = {
                'current_regime': 'neutral',
                'transition_probabilities': {},
                'regime_stability': 0.5,
                'expected_duration': 5,
                'transition_alerts': []
            }

            # Analyze regime history across multiple timeframes
            regime_series_df = self._build_regime_time_series(symbol, historical_data, lookback_periods)

            # --- FIX: Handle empty or invalid regime series ---
            if regime_series_df is None or regime_series_df.empty:
                self.logger.warning(f"Could not build regime time series for {symbol}")
                return transition_analysis

            # Calculate transition probabilities
            transition_matrix = self._calculate_regime_transition_matrix(regime_series_df)
            transition_analysis['transition_probabilities'] = transition_matrix

            # Current regime stability
            current_regime = self._get_current_regime(symbol, historical_data)
            transition_analysis['current_regime'] = current_regime

            # Use the fixed stability function
            transition_analysis['regime_stability'] = self._calculate_regime_series_stability(
                regime_series_df, current_regime
            )

            # Expected regime duration
            expected_duration = self._calculate_expected_regime_duration(transition_matrix, current_regime)
            transition_analysis['expected_duration'] = expected_duration

            # Transition alerts
            alerts = self._generate_regime_transition_alerts(
                regime_series_df, transition_matrix, current_regime
            )
            transition_analysis['transition_alerts'] = alerts

            # Regime momentum
            momentum = self._calculate_regime_momentum(regime_series_df)
            transition_analysis['regime_momentum'] = momentum

            if self.database:
                self.database.store_system_event(
                    "REGIME_TRANSITION_ANALYSIS",
                    {
                        'symbol': symbol,
                        'current_regime': current_regime,
                        'stability': transition_analysis['regime_stability'],
                        'expected_duration': expected_duration,
                        'alerts': len(alerts),
                        'momentum': momentum
                    },
                    "INFO",
                    "Market Analysis"
                )

            return transition_analysis

        except Exception as e:
            self.logger.error(f"Error detecting regime transitions for {symbol}: {type(e).__name__} - {e}", exc_info=True)
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "regime_transition_detection")
            return {
                'current_regime': 'neutral',
                'transition_probabilities': {},
                'regime_stability': 0.5,
                'expected_duration': 5,
                'transition_alerts': []
            }
    
    def _build_regime_time_series(self, symbol: str, historical_data: pd.DataFrame,
                                lookback_periods: List[int]) -> pd.DataFrame:
        """Build a time series of market regimes across multiple timeframes"""
        try:
            regime_data = {}
            
            for period in lookback_periods:
                regime_key = f'regime_{period}'
                regime_data[regime_key] = []
                
                # Ensure we have enough data for this period
                if len(historical_data) < period:
                    self.logger.warning(f"Insufficient data for {period}-period regime analysis on {symbol}")
                    continue
                    
                for i in range(period, len(historical_data)):
                    window_data = historical_data.iloc[i-period:i]
                    regime = self._classify_single_period_regime(window_data)
                    regime_data[regime_key].append(regime)
            
            # If no regime data was collected, return empty DataFrame
            if not regime_data:
                return pd.DataFrame()
            
            # Align the series (they have different lengths due to different lookbacks)
            min_length = min(len(series) for series in regime_data.values())
            
            # Ensure we have at least some data
            if min_length == 0:
                self.logger.warning(f"No valid regime data collected for {symbol}")
                return pd.DataFrame()
                
            for key in regime_data:
                regime_data[key] = regime_data[key][-min_length:]
            
            return pd.DataFrame(regime_data)
            
        except Exception as e:
            self.logger.error(f"Error building regime time series for {symbol}: {e}")
            return pd.DataFrame()

    def _classify_single_period_regime(self, data: pd.DataFrame) -> str:
        """Classify market regime for a single period - Standardized Output"""
        # --- Standard Regime Set: 'bull_trend', 'bear_trend', 'ranging', 'high_volatility', 'neutral' ---
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame) or data.empty or not {'close', 'high', 'low', 'volume'}.issubset(data.columns):
                self.logger.warning("_classify_single_period_regime: Invalid or incomplete input data.")
                return 'neutral' # Fallback on bad input

            close = data['close'].astype(float)
            if len(close) < 20: # Need enough data for calculations
                return 'neutral'

            # Calculate indicators needed for classification
            returns = close.pct_change().dropna()
            if returns.empty: 
                return 'neutral' # Need returns for volatility

            volatility = returns.std() # Use std dev of returns in the window
            trend_strength = self._calculate_trend_strength_single(close) # R-squared based strength (0-1)
            slope = self._calculate_slope_single(close) # Simple slope for direction

            # Define thresholds
            trend_r2_threshold = 0.5
            vol_high_threshold = 0.03
            vol_low_threshold = 0.01

            # --- Standardized Regime Classification Logic ---
            if trend_strength >= trend_r2_threshold:
                # Significant trend based on R-squared
                regime = "bull_trend" if slope > 0 else "bear_trend"
            elif volatility >= vol_high_threshold:
                # High volatility takes precedence over weak trend/ranging
                regime = "high_volatility"
            elif volatility <= vol_low_threshold and trend_strength < 0.3:
                # Low volatility AND low trend strength -> Ranging
                regime = "ranging"
            else:
                # Otherwise, classify as neutral (weak trend, moderate volatility)
                regime = "neutral"

            self.logger.debug(f"_classify_single_period: TStr={trend_strength:.3f}, Slope={slope:.4f}, Vol={volatility:.4f} -> Regime='{regime}'")
            return regime

        except Exception as e:
            # Log with traceback
            self.logger.error(f"Error classifying single period regime: {type(e).__name__} - {e}", exc_info=True)
            return 'neutral' # Fallback remains 'neutral'

    # --- Ensure these helpers exist ---
    def _calculate_trend_strength_single(self, prices: pd.Series) -> float:
        """Calculate trend strength for a single period"""
        if len(prices) < 10: return 0.0 # Changed threshold to 0.0
        try:
            x = np.arange(len(prices))
            # Use .values for direct numpy array, handle potential NaNs in prices
            valid_prices = prices.dropna()
            if len(valid_prices) < 5: return 0.0 # Need points for regression
            x_valid = np.arange(len(valid_prices))
            slope, _, r_value, _, _ = stats.linregress(x_valid, valid_prices.values)
            price_std = valid_prices.std()
            if price_std is None or price_std == 0: return 0.0 # Avoid division by zero

            # Normalize slope and combine with R²
            normalized_slope = abs(slope) / (price_std + 1e-8)
            trend_strength = min(1.0, normalized_slope * (r_value ** 2))
            return trend_strength if not np.isnan(trend_strength) else 0.0 # Handle potential NaN result
        except Exception as e:
            self.logger.warning(f"Error in _calculate_trend_strength_single: {e}")
            return 0.0

    def _calculate_slope_single(self, prices: pd.Series) -> float:
        """Calculate simple slope for direction."""
        if len(prices) < 5: return 0.0
        try:
            valid_prices = prices.dropna()
            if len(valid_prices) < 3: return 0.0
            x_valid = np.arange(len(valid_prices))
            slope, _, _, _, _ = stats.linregress(x_valid, valid_prices.values)
            return slope if not np.isnan(slope) else 0.0
        except Exception as e:
            self.logger.warning(f"Error in _calculate_slope_single: {e}")
            return 0.0

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend strength"""
        if len(volume) < 10:
            return 0
        
        # Volume momentum
        volume_momentum = volume.iloc[-1] / volume.mean() - 1
        return max(-1.0, min(1.0, volume_momentum))

    def _calculate_regime_transition_matrix(self, regime_series: pd.DataFrame) -> Dict:
        """Calculate Markov transition probabilities between regimes"""
        try:
            # Use the shortest lookback for transition analysis
            primary_series = regime_series.iloc[:, 0]
            
            regimes = ['strong_trend', 'trending', 'high_volatility', 'ranging', 'neutral']
            transition_counts = {from_regime: {to_regime: 0 for to_regime in regimes} for from_regime in regimes}
            
            # Count transitions
            for i in range(1, len(primary_series)):
                from_regime = primary_series.iloc[i-1]
                to_regime = primary_series.iloc[i]
                
                if from_regime in transition_counts and to_regime in transition_counts[from_regime]:
                    transition_counts[from_regime][to_regime] += 1
            
            # Calculate probabilities
            transition_matrix = {}
            for from_regime in regimes:
                total_transitions = sum(transition_counts[from_regime].values())
                transition_matrix[from_regime] = {}
                
                for to_regime in regimes:
                    if total_transitions > 0:
                        prob = transition_counts[from_regime][to_regime] / total_transitions
                    else:
                        prob = 0.2  # Uniform probability if no data
                    transition_matrix[from_regime][to_regime] = prob
            
            return transition_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating transition matrix: {e}")
            return {}

    def _get_current_regime(self, symbol: str, historical_data: pd.DataFrame) -> str:
        """Get current market regime with consistent naming"""
        regime_analysis = self.analyze_market_regime_advanced(historical_data)
        current_regime = regime_analysis.get('advanced_regime', 'neutral')
        
        # Ensure the regime name is normalized
        return self._normalize_regime_name(current_regime)

    def _calculate_regime_stability(self, df: pd.DataFrame, window: int = 50) -> float:
        """Calculate regime stability with proper error handling"""
        try:
            # --- FIX: Better debugging to identify the source of the issue ---
            original_window = window
            if not isinstance(window, int):
                try:
                    window = int(window)
                    self.logger.debug(f"Converted window from '{original_window}' to {window}")
                except (ValueError, TypeError) as conversion_error:
                    # If conversion fails, use default but log the actual problematic value
                    window = 50
                    self.logger.warning(f"Cannot convert window value '{original_window}' to integer, using default: {window}. Error: {conversion_error}")
            
            # Rest of the method remains the same...
            if not isinstance(df, pd.DataFrame) or df.empty:
                self.logger.warning(f"_calculate_regime_stability: Invalid input - type: {type(df)}")
                return 0.5

            if len(df) < window:
                return 0.0

            recent_data = df.iloc[-window:] if len(df) >= window else df

            regimes = []

            min_chunk_size = 20
            step = max(5, window // 10)

            for i in range(min_chunk_size, len(recent_data) + 1, step):
                current_idx = min(i, len(recent_data))
                chunk = recent_data.iloc[:current_idx]

                if len(chunk) >= min_chunk_size:
                    if not {'close', 'high', 'low'}.issubset(chunk.columns):
                        continue

                    try:
                        regime = self._detect_market_regime(
                            chunk['close'], chunk['high'], chunk['low']
                        )
                        regimes.append(regime)
                    except Exception as detect_e:
                        continue

            if len(regimes) < 2:
                return 0.5

            changes = 0
            for i in range(1, len(regimes)):
                if regimes[i] != regimes[i-1]:
                    changes += 1

            stability = 1.0 - (changes / (len(regimes) - 1))
            return stability

        except Exception as e:
            self.logger.error(f"Error calculating regime stability: {e}. Input type: {type(df)}, window: {window}, window type: {type(window)}", exc_info=True)
            return 0.5

    def _calculate_expected_regime_duration(self, transition_matrix: Dict, current_regime: str) -> int:
        """Calculate expected duration of current regime in periods"""
        try:
            if current_regime not in transition_matrix:
                return 5  # Default
            
            # Self-transition probability
            p_stay = transition_matrix[current_regime].get(current_regime, 0)
            
            # Expected duration = 1 / (1 - p_stay)
            if p_stay < 1:
                expected_duration = 1 / (1 - p_stay)
            else:
                expected_duration = 20  # Large number if probability is 1
            
            return int(expected_duration)
            
        except Exception as e:
            self.logger.error(f"Error calculating expected duration: {e}")
            return 5

    def _generate_regime_transition_alerts(self, regime_series_df: pd.DataFrame, # Renamed arg for clarity
                                        transition_matrix: Dict, current_regime: str) -> List[Dict]:
        """Generate alerts for potential regime transitions"""
        alerts = []

        try:
            # Check for regime instability using the CORRECT function
            # --- FIX: Call _calculate_regime_SERIES_stability ---
            stability = self._calculate_regime_series_stability(regime_series_df, current_regime)
            # --- END FIX ---

            if stability < 0.3:
                alerts.append({
                    'type': 'LOW_STABILITY',
                    'message': f'Current regime {current_regime} shows low stability ({stability:.2f})', # Added score
                    'stability_score': stability,
                    'priority': 'MEDIUM'
                })

            # Check for high-probability transitions
            if current_regime in transition_matrix:
                for target_regime, prob in transition_matrix[current_regime].items():
                    if target_regime != current_regime and prob > 0.3:
                        alerts.append({
                            'type': 'HIGH_TRANSITION_PROB',
                            'message': f'High probability ({prob:.1%}) of transition from {current_regime} to {target_regime}', # Clarified message
                            'from_regime': current_regime,
                            'to_regime': target_regime,
                            'probability': prob,
                            'priority': 'HIGH' if prob > 0.5 else 'MEDIUM'
                        })

            # Check for regime conflicts across timeframes
            # --- FIX: Pass the DataFrame ---
            conflict_alerts = self._check_regime_conflicts(regime_series_df)
            # --- END FIX ---
            alerts.extend(conflict_alerts)

        except Exception as e:
            # Log with traceback
            self.logger.error(f"Error generating regime alerts: {type(e).__name__} - {e}", exc_info=True)

        return alerts

    def _check_regime_conflicts(self, regime_series: pd.DataFrame) -> List[Dict]:
        """Check for conflicts between regime classifications across timeframes"""
        conflicts = []
        
        try:
            # Compare the most recent regime classifications across timeframes
            recent_regimes = regime_series.iloc[-1]
            unique_regimes = recent_regimes.unique()
            
            if len(unique_regimes) > 2:  # More than 2 different regimes
                conflicts.append({
                    'type': 'REGIME_CONFLICT',
                    'message': f'Multiple regime classifications: {", ".join(unique_regimes)}',
                    'regimes': unique_regimes.tolist(),
                    'priority': 'MEDIUM'
                })
            
            # Check for significant regime changes in short timeframes
            for i, col in enumerate(regime_series.columns):
                if 'regime_20' in col:  # Short-term regime
                    short_term = regime_series[col]
                    if len(short_term) >= 3:
                        recent_changes = short_term.iloc[-3:].nunique()
                        if recent_changes == 3:  # All different
                            conflicts.append({
                                'type': 'VOLATILE_SHORT_TERM',
                                'message': 'High regime volatility in short-term analysis',
                                'timeframe': col,
                                'priority': 'HIGH'
                            })
            
        except Exception as e:
            self.logger.error(f"Error checking regime conflicts: {e}")
        
        return conflicts

    def _calculate_regime_momentum(self, regime_series: pd.DataFrame) -> str:
        """Calculate the momentum direction of regime changes"""
        try:
            # Use the primary series
            primary_series = regime_series.iloc[:, 0]
            
            if len(primary_series) < 3:
                return 'neutral'
            
            # Get recent regimes
            recent = primary_series.tail(3).values
            
            # Define regime strength order
            regime_strength = {
                'strong_trend': 4,
                'trending': 3,
                'high_volatility': 2,
                'neutral': 1,
                'ranging': 0
            }
            
            # Calculate momentum
            current_strength = regime_strength.get(recent[-1], 1)
            previous_strength = regime_strength.get(recent[-2], 1)
            
            if current_strength > previous_strength:
                return 'strengthening'
            elif current_strength < previous_strength:
                return 'weakening'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Error calculating regime momentum: {e}")
            return 'neutral'

    def calculate_advanced_risk_parity(self, symbols: List[str],
                                    historical_data: Dict[str, pd.DataFrame],
                                    portfolio_value: float,
                                    risk_budget: Dict[str, float] = None) -> Dict:
        """
        Advanced risk parity allocation with regime adjustments and correlation clustering.
        """
        try:
            if len(symbols) < 2:
                return {}
            
            advanced_allocation = {
                'base_weights': {},
                'regime_adjusted_weights': {},
                'cluster_adjusted_weights': {},
                'final_weights': {},
                'risk_contributions': {},
                'allocation_metrics': {}
            }
            
            # 1. Base risk parity weights
            base_parity = self.calculate_risk_parity_weights(symbols, historical_data, portfolio_value)
            base_weights = base_parity.get('weights', {})
            advanced_allocation['base_weights'] = base_weights
            
            # 2. Regime-adjusted weights
            regime_weights = self._calculate_regime_adjusted_risk_parity(
                symbols, historical_data, base_weights
            )
            advanced_allocation['regime_adjusted_weights'] = regime_weights
            
            # 3. Cluster-adjusted weights
            cluster_weights = self._calculate_cluster_adjusted_risk_parity(
                symbols, historical_data, regime_weights
            )
            advanced_allocation['cluster_adjusted_weights'] = cluster_weights
            
            # 4. Apply risk budget if provided
            if risk_budget:
                final_weights = self._apply_risk_budget(cluster_weights, risk_budget)
            else:
                final_weights = cluster_weights
            
            advanced_allocation['final_weights'] = final_weights
            
            # 5. Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions_advanced(
                symbols, historical_data, final_weights
            )
            advanced_allocation['risk_contributions'] = risk_contributions
            
            # 6. Allocation metrics
            allocation_metrics = self._calculate_allocation_metrics(
                symbols, historical_data, final_weights, risk_contributions
            )
            advanced_allocation['allocation_metrics'] = allocation_metrics
            
            # 7. Position sizes
            position_sizes = self._calculate_risk_parity_positions(
                final_weights, portfolio_value, symbols, historical_data
            )
            advanced_allocation['positions'] = position_sizes
            
            if self.database:
                self.database.store_system_event(
                    "ADVANCED_RISK_PARITY",
                    {
                        'symbols': symbols,
                        'portfolio_value': portfolio_value,
                        'final_weights': final_weights,
                        'allocation_score': allocation_metrics.get('allocation_score', 0),
                        'risk_budget_used': risk_budget is not None
                    },
                    "INFO",
                    "Risk Management"
                )
            
            return advanced_allocation
            
        except Exception as e:
            self.logger.error(f"Error in advanced risk parity: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "PORTFOLIO", "advanced_risk_parity")
            return {}

    def _calculate_regime_adjusted_risk_parity(self, symbols: List[str],
                                            historical_data: Dict[str, pd.DataFrame],
                                            base_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust risk parity weights based on market regimes"""
        try:
            regime_weights = base_weights.copy()
            
            # Get current market regime
            regime_analysis = self.analyze_market_regime_advanced(
                next(iter(historical_data.values()))
            )
            current_regime = regime_analysis.get('advanced_regime', 'neutral')
            regime_confidence = regime_analysis.get('regime_confidence', 0.5)
            
            # Regime-specific adjustments
            regime_adjustments = {
                'high_volatility': {
                    'adjustment': 0.8,  # Reduce position sizes
                    'min_weight': 0.02,
                    'max_weight': 0.25
                },
                'low_volatility': {
                    'adjustment': 1.2,  # Increase position sizes
                    'min_weight': 0.03,
                    'max_weight': 0.35
                },
                'trending': {
                    'adjustment': 1.1,  # Slightly increase
                    'min_weight': 0.025,
                    'max_weight': 0.3
                },
                'ranging': {
                    'adjustment': 0.9,  # Slightly decrease
                    'min_weight': 0.02,
                    'max_weight': 0.25
                }
            }
            
            regime_params = regime_adjustments.get(current_regime, {})
            adjustment = regime_params.get('adjustment', 1.0)
            min_weight = regime_params.get('min_weight', 0.01)
            max_weight = regime_params.get('max_weight', 0.5)
            
            # Apply regime adjustment
            for symbol in regime_weights:
                regime_weights[symbol] *= adjustment
            
            # Apply weight constraints
            total_weight = sum(regime_weights.values())
            if total_weight > 0:
                regime_weights = {k: v/total_weight for k, v in regime_weights.items()}
            
            # Enforce min/max weights
            for symbol in regime_weights:
                regime_weights[symbol] = max(min_weight, min(max_weight, regime_weights[symbol]))
            
            # Renormalize
            total_weight = sum(regime_weights.values())
            if total_weight > 0:
                regime_weights = {k: v/total_weight for k, v in regime_weights.items()}
            
            return regime_weights
            
        except Exception as e:
            self.logger.error(f"Error in regime-adjusted risk parity: {e}")
            return base_weights

    def _calculate_cluster_adjusted_risk_parity(self, symbols: List[str],
                                            historical_data: Dict[str, pd.DataFrame],
                                            regime_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on correlation clusters"""
        try:
            cluster_weights = regime_weights.copy()
            
            # Get correlation clusters
            correlation_analysis = self.analyze_cross_asset_correlations(symbols, historical_data)
            clusters = correlation_analysis.get('correlation_clusters', [])
            
            if not clusters:
                return cluster_weights
            
            # Calculate cluster-level risk budgets
            cluster_budgets = {}
            total_cluster_risk = 0
            
            for cluster in clusters:
                cluster_symbols = cluster['symbols']
                cluster_risk = cluster.get('average_correlation', 0.5) * len(cluster_symbols)
                cluster_budgets[tuple(cluster_symbols)] = cluster_risk
                total_cluster_risk += cluster_risk
            
            # Adjust weights within clusters
            for cluster_symbols, cluster_risk in cluster_budgets.items():
                if total_cluster_risk > 0:
                    cluster_weight_budget = cluster_risk / total_cluster_risk
                    
                    # Current weight in this cluster
                    current_cluster_weight = sum(cluster_weights.get(s, 0) for s in cluster_symbols)
                    
                    if current_cluster_weight > 0:
                        # Adjust weights proportionally
                        adjustment_factor = cluster_weight_budget / current_cluster_weight
                        for symbol in cluster_symbols:
                            if symbol in cluster_weights:
                                cluster_weights[symbol] *= adjustment_factor
            
            # Renormalize
            total_weight = sum(cluster_weights.values())
            if total_weight > 0:
                cluster_weights = {k: v/total_weight for k, v in cluster_weights.items()}
            
            return cluster_weights
            
        except Exception as e:
            self.logger.error(f"Error in cluster-adjusted risk parity: {e}")
            return regime_weights

    def _apply_risk_budget(self, weights: Dict[str, float], risk_budget: Dict[str, float]) -> Dict[str, float]:
        """Apply explicit risk budget constraints"""
        try:
            budgeted_weights = weights.copy()
            
            # Normalize risk budget to sum to 1
            total_budget = sum(risk_budget.values())
            if total_budget > 0:
                normalized_budget = {k: v/total_budget for k, v in risk_budget.items()}
            else:
                return budgeted_weights
            
            # Blend current weights with risk budget (70% current, 30% budget)
            for symbol in budgeted_weights:
                if symbol in normalized_budget:
                    budgeted_weights[symbol] = (
                        budgeted_weights[symbol] * 0.7 + 
                        normalized_budget[symbol] * 0.3
                    )
            
            # Renormalize
            total_weight = sum(budgeted_weights.values())
            if total_weight > 0:
                budgeted_weights = {k: v/total_weight for k, v in budgeted_weights.items()}
            
            return budgeted_weights
            
        except Exception as e:
            self.logger.error(f"Error applying risk budget: {e}")
            return weights

    def _calculate_risk_contributions_advanced(self, symbols: List[str],
                                            historical_data: Dict[str, pd.DataFrame],
                                            weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate advanced risk contributions with regime adjustments"""
        try:
            volatilities = self._calculate_asset_volatilities(symbols, historical_data)
            correlations = self._calculate_correlation_matrix(symbols, historical_data)
            
            if not volatilities or correlations.empty:
                return {}
            
            portfolio_vol = self._calculate_portfolio_volatility(weights, volatilities, correlations)
            risk_contributions = {}
            
            for symbol in symbols:
                marginal_risk = 0
                for other_symbol in symbols:
                    if symbol == other_symbol:
                        marginal_risk += weights[symbol] * volatilities[symbol] ** 2
                    else:
                        corr = correlations.loc[symbol, other_symbol] if not correlations.empty else 0.0
                        marginal_risk += weights[other_symbol] * volatilities[symbol] * volatilities[other_symbol] * corr
                
                risk_contributions[symbol] = (weights[symbol] * marginal_risk / max(portfolio_vol, 0.001))
            
            return risk_contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced risk contributions: {e}")
            return {}

    def _calculate_allocation_metrics(self, symbols: List[str],
                                    historical_data: Dict[str, pd.DataFrame],
                                    weights: Dict[str, float],
                                    risk_contributions: Dict[str, float]) -> Dict:
        """Calculate metrics for allocation quality"""
        try:
            metrics = {}
            
            # 1. Risk parity efficiency (how equal are risk contributions)
            if risk_contributions:
                risk_values = list(risk_contributions.values())
                risk_equality = 1 - (np.std(risk_values) / (np.mean(risk_values) + 1e-8))
                metrics['risk_parity_efficiency'] = max(0, risk_equality)
            else:
                metrics['risk_parity_efficiency'] = 0
            
            # 2. Diversification score
            diversification = self._calculate_diversification_score_advanced(
                symbols, historical_data, weights
            )
            metrics['diversification_score'] = diversification
            
            # 3. Concentration metrics
            weight_values = list(weights.values())
            herfindahl = sum(w ** 2 for w in weight_values)
            metrics['herfindahl_index'] = herfindahl
            metrics['effective_n'] = 1 / herfindahl if herfindahl > 0 else len(symbols)
            
            # 4. Regime alignment
            regime_alignment = self._calculate_regime_alignment(symbols, historical_data, weights)
            metrics['regime_alignment'] = regime_alignment
            
            # 5. Composite allocation score
            composite_score = (
                metrics['risk_parity_efficiency'] * 0.4 +
                metrics['diversification_score'] * 0.3 +
                min(metrics['effective_n'] / len(symbols), 1) * 0.2 +
                metrics['regime_alignment'] * 0.1
            )
            metrics['allocation_score'] = composite_score
            
            # 6. Allocation rating
            if composite_score > 0.8:
                metrics['rating'] = 'EXCELLENT'
            elif composite_score > 0.6:
                metrics['rating'] = 'GOOD'
            elif composite_score > 0.4:
                metrics['rating'] = 'FAIR'
            else:
                metrics['rating'] = 'POOR'
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating allocation metrics: {e}")
            return {}

    def _calculate_diversification_score_advanced(self, symbols: List[str],
                                            historical_data: Dict[str, pd.DataFrame],
                                            weights: Dict[str, float]) -> float:
        """Calculate advanced diversification score"""
        try:
            # Correlation matrix
            corr_matrix = self._calculate_correlation_matrix(symbols, historical_data)
            if corr_matrix.empty:
                return 0.5
            
            # Weighted average correlation
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.5
            
            weighted_corr = 0
            weight_pairs = 0
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i < j:
                        corr = corr_matrix.loc[sym1, sym2]
                        weight_product = weights.get(sym1, 0) * weights.get(sym2, 0)
                        weighted_corr += abs(corr) * weight_product
                        weight_pairs += weight_product
            
            avg_correlation = weighted_corr / weight_pairs if weight_pairs > 0 else 0.5
            
            # Diversification score (lower correlation = better)
            diversification = 1 - avg_correlation
            return max(0, diversification)
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced diversification: {e}")
            return 0.5

    def _calculate_regime_alignment(self, symbols: List[str],
                                historical_data: Dict[str, pd.DataFrame],
                                weights: Dict[str, float]) -> float:
        """Calculate how well the allocation aligns with current market regime"""
        try:
            # Get current regime
            sample_data = next(iter(historical_data.values()))
            regime_analysis = self.analyze_market_regime_advanced(sample_data)
            current_regime = regime_analysis.get('advanced_regime', 'neutral')
            
            # Define regime-appropriate allocations
            regime_preferences = {
                'high_volatility': {
                    'preference': 'defensive',  # Lower weights on high-vol assets
                    'max_single_weight': 0.2
                },
                'low_volatility': {
                    'preference': 'concentrated',  # Can have higher concentrations
                    'max_single_weight': 0.35
                },
                'trending': {
                    'preference': 'momentum',  # Higher weights on trending assets
                    'max_single_weight': 0.3
                },
                'ranging': {
                    'preference': 'equal_weight',  # More equal distribution
                    'max_single_weight': 0.25
                }
            }
            
            regime_params = regime_preferences.get(current_regime, {})
            max_single_weight = regime_params.get('max_single_weight', 0.3)
            
            # Check if any weight exceeds regime-appropriate maximum
            max_weight = max(weights.values()) if weights else 0
            if max_weight <= max_single_weight:
                alignment = 1.0
            else:
                # Penalize for exceeding maximum
                excess = max_weight - max_single_weight
                alignment = 1.0 - min(1.0, excess / max_single_weight)
            
            return alignment
            
        except Exception as e:
            self.logger.error(f"Error calculating regime alignment: {e}")
            return 0.5

    def analyze_portfolio_performance(self, days: int = 30) -> Dict:
        """Comprehensive portfolio performance analysis"""
        if self.performance_attribution:
            return self.performance_attribution.perform_comprehensive_attribution(days)
        return {}

    def generate_performance_report(self, days: int = 30) -> Dict:
        """Generate performance attribution report"""
        if self.performance_attribution:
            return self.performance_attribution.generate_attribution_report(days)
        return {}
    
    def fetch_crypto_on_chain_data(self, symbol: str) -> Dict:
        """Fetch on-chain metrics for crypto assets"""
        try:
            if self.crypto_data_provider is None:
                return {}
            
            # Convert symbol to base asset (e.g., BTCUSDT -> BTC)
            base_asset = symbol.replace('USDT', '')
            
            on_chain_data = {}
            
            # Network Value Metrics
            on_chain_data.update(self.crypto_data_provider.get_network_value(base_asset))
            
            # Exchange Flows
            on_chain_data.update(self.crypto_data_provider.get_exchange_flows(base_asset))
            
            # Miner Activity
            on_chain_data.update(self.crypto_data_provider.get_miner_activity(base_asset))
            
            # Market Metrics
            on_chain_data.update(self.crypto_data_provider.get_market_metrics(base_asset))
            
            self.on_chain_metrics[symbol] = on_chain_data
            return on_chain_data
            
        except Exception as e:
            self.logger.error(f"Error fetching on-chain data for {symbol}: {e}")
            return {}
    
    def analyze_crypto_market_regime(self, symbol: str, historical_data: pd.DataFrame) -> Dict:
        """Advanced crypto market regime analysis"""
        try:
            # Fetch on-chain data
            on_chain_data = self.fetch_crypto_on_chain_data(symbol)
            
            # Get crypto-specific indicators
            crypto_indicators = self.technical_analyzer.calculate_crypto_specific_indicators(
                symbol, historical_data, on_chain_data, self._get_market_context()
            )
            
            # Combine with technical analysis
            technical_regime = self.analyze_market_regime_advanced(historical_data)
            
            # Crypto-enhanced regime classification
            crypto_regime = {
                'technical_regime': technical_regime.get('composite_regime', 'neutral'),
                'crypto_regime': self._classify_crypto_regime(crypto_indicators),
                'on_chain_strength': crypto_indicators.get('market_health', 'neutral'),
                'liquidity_conditions': crypto_indicators.get('liquidity_signal', 'neutral'),
                'miner_sentiment': crypto_indicators.get('miner_sentiment', 'neutral'),
                'cycle_phase': crypto_indicators.get('cycle_phase', 'neutral'),
                'composite_confidence': self._calculate_crypto_regime_confidence(
                    technical_regime, crypto_indicators
                )
            }
            
            # Store for portfolio optimization
            self.crypto_regime_indicators[symbol] = crypto_regime
            
            return crypto_regime
            
        except Exception as e:
            self.logger.error(f"Error analyzing crypto market regime for {symbol}: {e}")
            return {'crypto_regime': 'neutral', 'composite_confidence': 0.5}
    
    def _classify_crypto_regime(self, crypto_indicators: Dict) -> str:
        """Classify crypto-specific market regime"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # On-chain signals
            if crypto_indicators.get('market_health') == 'healthy':
                bullish_signals += 1
            if crypto_indicators.get('flow_sentiment') == 'bullish':
                bullish_signals += 1
            if crypto_indicators.get('miner_sentiment') == 'accumulating':
                bullish_signals += 1
            if crypto_indicators.get('sopr_signal') == 'neutral':
                bullish_signals += 0.5
            
            # Market structure signals
            if crypto_indicators.get('liquidity_signal') == 'high_liquidity':
                bullish_signals += 1
            if crypto_indicators.get('market_rotation') == 'altcoin_season':
                bullish_signals += 1
            if crypto_indicators.get('cycle_phase') in ['accumulation_phase', 'early_bull']:
                bullish_signals += 1
            
            # Bearish signals
            if crypto_indicators.get('nvt_signal') == 'overvalued':
                bearish_signals += 1
            if crypto_indicators.get('flow_sentiment') == 'bearish':
                bearish_signals += 1
            if crypto_indicators.get('sopr_signal') == 'profit_taking':
                bearish_signals += 1
            if crypto_indicators.get('funding_signal') == 'long_squeeze_risk':
                bearish_signals += 1
            
            # Determine regime
            if bullish_signals - bearish_signals >= 3:
                return "crypto_bull"
            elif bearish_signals - bullish_signals >= 3:
                return "crypto_bear"
            elif abs(bullish_signals - bearish_signals) <= 1:
                return "crypto_neutral"
            else:
                return "crypto_transition"
                
        except Exception as e:
            self.logger.error(f"Error classifying crypto regime: {e}")
            return "crypto_neutral"
    
    def _calculate_crypto_regime_confidence(self, technical_regime: Dict, crypto_indicators: Dict) -> float:
        """Calculate confidence score combining technical and on-chain analysis"""
        try:
            technical_confidence = technical_regime.get('regime_confidence', 0.5)
            
            # On-chain confidence factors
            on_chain_confidence = 0.5
            
            # Strong on-chain signals increase confidence
            strong_bull_signals = [
                crypto_indicators.get('market_health') == 'healthy',
                crypto_indicators.get('flow_sentiment') == 'bullish',
                crypto_indicators.get('miner_sentiment') == 'accumulating'
            ]
            
            strong_bear_signals = [
                crypto_indicators.get('nvt_signal') == 'overvalued',
                crypto_indicators.get('flow_sentiment') == 'bearish',
                crypto_indicators.get('sopr_signal') == 'profit_taking'
            ]
            
            strong_signals = sum(strong_bull_signals) + sum(strong_bear_signals)
            if strong_signals >= 2:
                on_chain_confidence = 0.8
            elif strong_signals >= 1:
                on_chain_confidence = 0.65
            else:
                on_chain_confidence = 0.5
            
            # Combine technical and on-chain confidence
            composite_confidence = (technical_confidence * 0.6 + on_chain_confidence * 0.4)
            
            # Adjust for market sentiment
            fear_greed = crypto_indicators.get('fear_greed_index', 50)
            if 25 <= fear_greed <= 75:  # Neutral range
                sentiment_boost = 1.0
            else:  # Extreme sentiment reduces confidence
                sentiment_boost = 0.8
            
            return min(composite_confidence * sentiment_boost, 0.95)
            
        except Exception as e:
            self.logger.error(f"Error calculating crypto regime confidence: {e}")
            return 0.5
    
    def _get_market_context(self) -> Dict:
        """Get broader crypto market context"""
        return {
            'bitcoin_dominance': self.bitcoin_dominance,
            'fear_greed_index': self.fear_greed_index,
            'total_crypto_mcap': getattr(self, 'total_market_cap', 0),
            'stablecoin_supply': getattr(self, 'stablecoin_supply', 0)
        }
    
    def set_crypto_data_provider(self, provider):
        """Set the crypto data provider for on-chain metrics"""
        self.crypto_data_provider = provider
        self.logger.info("Crypto data provider configured")
    
    def update_crypto_market_data(self):
        """Update global crypto market data"""
        try:
            if self.crypto_data_provider:
                # Update Bitcoin dominance
                self.bitcoin_dominance = self.crypto_data_provider.get_bitcoin_dominance()
                
                # Update Fear & Greed Index
                self.fear_greed_index = self.crypto_data_provider.get_fear_greed_index()
                
                # Update total market cap
                self.total_market_cap = self.crypto_data_provider.get_total_market_cap()
                
                # Update stablecoin supply
                self.stablecoin_supply = self.crypto_data_provider.get_stablecoin_supply()
                
                self.logger.info(f"Updated crypto market data: BTC Dominance={self.bitcoin_dominance:.1f}%, F&G={self.fear_greed_index}")
                
        except Exception as e:
            self.logger.error(f"Error updating crypto market data: {e}")