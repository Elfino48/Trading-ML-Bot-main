import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm
from typing import Dict, List
import time
import psutil
from config import RiskConfig, RISK_MULTIPLIER

class AdvancedRiskManager:
    def __init__(self, bybit_client, aggressiveness: str = "conservative", error_handler=None, database=None, symbol_to_sector=None):
        self.client = bybit_client
        self.risk_multiplier = RISK_MULTIPLIER
        self.positions = {}
        self.portfolio = {}
        self.correlation_matrix = {}
        self.aggressiveness = aggressiveness
        self.daily_pnl = 0
        self.trades_today = 0
        self.daily_start_balance = 0
        self.consecutive_losses = 0
        self.circuit_breaker = False
        self.max_consecutive_losses = 3
        self.error_handler = error_handler
        self.database = database
        self.symbol_to_sector = symbol_to_sector or {}
        self.sector_exposure = {}
        
        self.config = RiskConfig.get_config(aggressiveness)
        self._set_parameters_from_config()
        self._initialize_daily_balance()
        
        self.trailing_stop_activation = 0.02
        self.max_portfolio_correlation = 0.7
        self.volatility_adjustment_period = 20
        self.max_sector_exposure = 0.25
        
        self.dynamic_correlations = {}
        self.stress_test_results = {}
        self.regime_multipliers = {}
        self.volatility_regime = "normal"
        
        print(f"🎯 Risk Manager set to: {self.aggressiveness.upper()} mode")
        print(f"    • Max Position: ${self.max_position_size_usdt}")
        print(f"    • Max Daily Loss: {self.max_daily_loss_percent}%")
        print(f"    • Min Confidence: {self.min_confidence}%")
        print(f"    • Circuit Breaker: {self.max_consecutive_losses} consecutive losses")

    def set_error_handler(self, error_handler):
        self.error_handler = error_handler
        
    def set_database(self, database):
        self.database = database
    
    def _set_parameters_from_config(self):
        config = self.config
        
        self.max_position_size_usdt = config["max_position_size_usdt"]
        self.max_daily_loss_percent = config["max_daily_loss_percent"]
        self.global_stop_loss_percent = config["global_stop_loss_percent"]
        self.base_size_percent = config["base_size_percent"]
        self.kelly_multiplier = config["kelly_multiplier"]
        self.min_confidence = config["min_confidence"]
        self.sl_atr_multiple = config["sl_atr_multiple"]
        self.tp_atr_multiple = config["tp_atr_multiple"]
        self.max_sl_percent = config["max_sl_percent"]
    
    def _initialize_daily_balance(self, initial_balance_fallback=10000):
        try:
            balance = self.client.get_wallet_balance()
            if balance and balance.get('retCode') == 0:
                self.daily_start_balance = float(balance['result']['list'][0]['totalEquity'])
                print(f"💰 Daily starting balance: ${self.daily_start_balance:.2f}")
                
                if self.database:
                    self.database.store_system_event(
                        "DAILY_BALANCE_INIT",
                        {"daily_start_balance": self.daily_start_balance},
                        "INFO",
                        "Risk Management"
                    )
        except Exception as e:
            print(f"Error initializing daily balance: {e}")
            if self.error_handler:
                self.error_handler.handle_api_error(e, "initialize_daily_balance")
            self.daily_start_balance = initial_balance_fallback

    def calculate_var(self, positions: Dict, confidence_level: float = 0.95, horizon: int = 1) -> float:
        try:
            portfolio_value = sum(pos.get('value', 0) for pos in positions.values()) if positions else 0
            portfolio_volatility = self._calculate_portfolio_volatility(positions)
            
            z_score = {0.95: 1.645, 0.99: 2.326}.get(confidence_level, 1.645)
            var = portfolio_value * z_score * portfolio_volatility * np.sqrt(horizon)
            
            if self.database:
                self.database.store_system_event(
                    "VAR_CALCULATION",
                    {
                        'portfolio_value': portfolio_value,
                        'var': var,
                        'confidence_level': confidence_level,
                        'horizon': horizon
                    },
                    "INFO",
                    "Risk Management"
                )
            
            return max(0, var)
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "var_calculation")
            portfolio_value = sum(pos.get('value', 0) for pos in positions.values()) if positions else 0
            return portfolio_value * 0.02
    
    def _calculate_portfolio_volatility(self, positions: Dict) -> float:
        if not positions or len(positions) == 0:
            return 0.02
            
        try:
            weights = []
            volatilities = []
            
            for symbol, position in positions.items():
                position_value = position.get('value', 0)
                total_value = sum(pos.get('value', 0) for pos in positions.values())
                weight = position_value / total_value if total_value > 0 else 0
                
                volatility = 0.02
                
                weights.append(weight)
                volatilities.append(volatility)
                
            avg_correlation = 0.3
            portfolio_variance = 0
            
            for i, w_i in enumerate(weights):
                for j, w_j in enumerate(weights):
                    if i == j:
                        portfolio_variance += w_i * w_j * volatilities[i] * volatilities[j]
                    else:
                        portfolio_variance += w_i * w_j * volatilities[i] * volatilities[j] * avg_correlation
                        
            return np.sqrt(max(0, portfolio_variance))
            
        except Exception as e:
            print(f"Error calculating portfolio volatility: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "volatility_calculation")
            return 0.02
    
    def calculate_position_size(self, confidence: float, symbol: str, current_price: float = None, market_data: Dict = None) -> float:
        try:
            if current_price is None:
                current_price = 1000
                
            portfolio_value = self.daily_start_balance
            atr = current_price * 0.02
            
            position_info = self.calculate_correlation_aware_position_size(
                symbol, confidence, current_price, atr, portfolio_value, market_data
            )
            
            return position_info['size_usdt']
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "position_sizing")
            return 100

    def calculate_correlation_aware_position_size(self, symbol: str, confidence: float, current_price: float, 
                                                 atr: float, portfolio_value: float, market_data: Dict = None) -> Dict:
        
        try:
            base_info = self.calculate_aggressive_position_size(
                symbol, confidence, current_price, atr, portfolio_value, self.aggressiveness
            )
            
            correlation_adjustment = 1.0
            volatility_adjustment = 1.0
            sector_adjustment = 1.0
            stress_test_adjustment = 1.0
            
            if market_data and 'correlation_matrix' in market_data:
                correlation_adjustment = self._get_correlation_adjustment(symbol, market_data['correlation_matrix'])
            
            if market_data and 'volatility_regime' in market_data:
                volatility_adjustment = self._get_volatility_adjustment(market_data['volatility_regime'])
            
            sector_adjustment = self._get_sector_adjustment(symbol, base_info['size_usdt'])
            
            if market_data and 'stress_test_results' in market_data:
                stress_test_adjustment = self._get_stress_test_adjustment(market_data['stress_test_results'])
            
            total_adjustment = correlation_adjustment * volatility_adjustment * sector_adjustment * stress_test_adjustment
            
            adjusted_size = base_info['size_usdt'] * total_adjustment
            adjusted_size = max(portfolio_value * 0.005, min(adjusted_size, self.max_position_size_usdt))
            
            adjusted_quantity = adjusted_size / current_price if current_price > 0 else 0
            
            result = {
                **base_info,
                'size_usdt': adjusted_size,
                'quantity': adjusted_quantity,
                'correlation_adjustment': correlation_adjustment,
                'volatility_adjustment': volatility_adjustment,
                'sector_adjustment': sector_adjustment,
                'stress_test_adjustment': stress_test_adjustment,
                'total_adjustment': total_adjustment
            }
            
            if self.database:
                self.database.store_system_event(
                    "CORRELATION_AWARE_SIZING",
                    {
                        'symbol': symbol,
                        'confidence': confidence,
                        'base_size': base_info['size_usdt'],
                        'adjusted_size': adjusted_size,
                        'adjustment_factors': {
                            'correlation': correlation_adjustment,
                            'volatility': volatility_adjustment,
                            'sector': sector_adjustment,
                            'stress_test': stress_test_adjustment
                        }
                    },
                    "INFO",
                    "Risk Management"
                )
            
            return result
            
        except Exception as e:
            print(f"Error in correlation-aware position sizing: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "correlation_aware_sizing")
            
            return self.calculate_aggressive_position_size(
                symbol, confidence, current_price, atr, portfolio_value, self.aggressiveness
            )

    def _get_correlation_adjustment(self, symbol: str, correlation_matrix: Dict) -> float:
        try:
            current_symbols = self._get_current_position_symbols()
            if not current_symbols:
                return 1.0
            
            avg_correlation = 0.0
            count = 0
            
            for existing_symbol in current_symbols:
                if symbol in correlation_matrix and existing_symbol in correlation_matrix[symbol]:
                    corr = abs(correlation_matrix[symbol][existing_symbol])
                    avg_correlation += corr
                    count += 1
                elif existing_symbol in correlation_matrix and symbol in correlation_matrix[existing_symbol]:
                    corr = abs(correlation_matrix[existing_symbol][symbol])
                    avg_correlation += corr
                    count += 1
            
            if count == 0:
                return 1.0
            
            avg_correlation /= count
            
            if avg_correlation > 0.7:
                return 0.5
            elif avg_correlation > 0.5:
                return 0.7
            elif avg_correlation > 0.3:
                return 0.9
            else:
                return 1.1
                
        except Exception as e:
            print(f"Error calculating correlation adjustment: {e}")
            return 1.0

    def _get_volatility_adjustment(self, volatility_regime: str) -> float:
        volatility_multipliers = {
            'very_high': 0.5,
            'high': 0.7,
            'normal': 1.0,
            'low': 1.2,
            'very_low': 1.5
        }
        return volatility_multipliers.get(volatility_regime, 1.0)

    def _get_sector_adjustment(self, symbol: str, proposed_size: float) -> float:
        try:
            sector = self.symbol_to_sector.get(symbol, 'unknown')
            if sector == 'unknown':
                return 1.0
            
            self._update_sector_exposure()
            
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            if portfolio_value <= 0:
                return 1.0
            
            current_sector_exposure = self.sector_exposure.get(sector, 0)
            proposed_sector_exposure = current_sector_exposure + proposed_size
            sector_exposure_ratio = proposed_sector_exposure / portfolio_value
            
            if sector_exposure_ratio > self.max_sector_exposure:
                excess_ratio = sector_exposure_ratio - self.max_sector_exposure
                reduction_factor = 1.0 - (excess_ratio / sector_exposure_ratio)
                return max(0.1, reduction_factor)
            
            return 1.0
            
        except Exception as e:
            print(f"Error calculating sector adjustment: {e}")
            return 1.0

    def _get_stress_test_adjustment(self, stress_test_results: Dict) -> float:
        try:
            max_drawdown = 0.0
            for scenario, result in stress_test_results.items():
                drawdown = abs(result.get('drawdown_percent', 0))
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            if max_drawdown > 20:
                return 0.5
            elif max_drawdown > 15:
                return 0.7
            elif max_drawdown > 10:
                return 0.9
            else:
                return 1.0
                
        except Exception as e:
            print(f"Error calculating stress test adjustment: {e}")
            return 1.0

    def _update_sector_exposure(self):
        try:
            self.sector_exposure = {}
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            
            if portfolio_value <= 0:
                return
            
            current_symbols = self._get_current_position_symbols()
            for symbol in current_symbols:
                sector = self.symbol_to_sector.get(symbol, 'unknown')
                try:
                    position_response = self.client.get_position_info(symbol)
                    if position_response and position_response.get('retCode') == 0:
                        positions = position_response['result']['list']
                        position = next((p for p in positions if p['symbol'] == symbol), None)
                        if position and float(position['size']) > 0:
                            position_value = float(position.get('positionValue', 0))
                            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + abs(position_value)
                except:
                    continue
                    
        except Exception as e:
            print(f"Error updating sector exposure: {e}")

    def calculate_aggressive_position_size(self, symbol: str, confidence: float, current_price: float, 
                                          atr: float, portfolio_value: float, aggressiveness: str = None) -> Dict:
        
        if aggressiveness is None:
            aggressiveness = self.aggressiveness
            
        config = RiskConfig.get_config(aggressiveness)
        
        try:
            base_size = portfolio_value * config["base_size_percent"]
            
            confidence_adj = max(confidence / 100, config["min_confidence"] / 100)
            
            atr_ratio = atr / current_price if current_price > 0 else 0.02
            volatility_adj = max(0.1, min(1.0, 0.02 / atr_ratio)) if atr_ratio > 0 else 1.0
            
            kelly_fraction = confidence_adj * config["kelly_multiplier"]
            
            position_size = portfolio_value * kelly_fraction * volatility_adj
            
            min_size = portfolio_value * 0.005
            max_size = config["max_position_size_usdt"]
            
            final_size = max(min_size, min(position_size, max_size))
            
            final_size = final_size * self.risk_multiplier
            
            quantity = final_size / current_price if current_price > 0 else 0
            
            position_info = {
                'size_usdt': final_size,
                'quantity': quantity,
                'confidence_multiplier': confidence_adj,
                'volatility_multiplier': volatility_adj,
                'kelly_fraction': kelly_fraction,
                'aggressiveness': aggressiveness
            }
            
            if self.database:
                self.database.store_system_event(
                    "POSITION_SIZING",
                    {
                        'symbol': symbol,
                        'confidence': confidence,
                        'position_size': final_size,
                        'portfolio_value': portfolio_value,
                        'aggressiveness': aggressiveness,
                        'calculated_metrics': position_info
                    },
                    "INFO",
                    "Risk Management"
                )
            
            return position_info
            
        except Exception as e:
            print(f"Error in aggressive position sizing: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "aggressive_position_sizing")
            
            return {
                'size_usdt': 100,
                'quantity': 100 / current_price if current_price > 0 else 0,
                'confidence_multiplier': 0.5,
                'volatility_multiplier': 1.0,
                'kelly_fraction': 0.1,
                'aggressiveness': aggressiveness
            }
    
    def calculate_dynamic_stop_loss(self, symbol: str, action: str, current_price: float, 
                                        atr: float, volatility_regime: str = "normal") -> Dict:
        try:
            return self.calculate_aggressive_stop_loss(symbol, action, current_price, atr, self.aggressiveness)
        except Exception as e:
            print(f"Error calculating dynamic stop loss: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "stop_loss_calculation")
            
            if action == 'BUY':
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.02
            else:
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.98
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': 1.0,
                'distance_percent': 2.0,
                'aggressiveness': self.aggressiveness
            }
    
    def calculate_trailing_stop_loss(self, symbol: str, action: str, entry_price: float, 
                                            current_price: float, atr: float, highest_profit: float = None) -> Dict:
        try:
            if action == 'BUY':
                if highest_profit is None:
                    highest_profit = current_price
                
                highest_profit = max(highest_profit, current_price)
                
                profit_pct = (highest_profit - entry_price) / entry_price
                
                if profit_pct >= self.trailing_stop_activation:
                    trail_distance = atr * 1.5
                    trailing_stop = highest_profit - trail_distance
                    
                    trailing_stop = max(trailing_stop, entry_price)
                    
                    return {
                        'trailing_stop': trailing_stop,
                        'highest_profit': highest_profit,
                        'trail_distance': trail_distance,
                        'activated': True
                    }
                else:
                    regular_stop = entry_price - (atr * self.sl_atr_multiple)
                    return {
                        'trailing_stop': regular_stop,
                        'highest_profit': current_price,
                        'trail_distance': atr * self.sl_atr_multiple,
                        'activated': False
                    }
                    
            else:
                if highest_profit is None:
                    highest_profit = current_price
                
                highest_profit = min(highest_profit, current_price)
                
                profit_pct = (entry_price - highest_profit) / entry_price
                
                if profit_pct >= self.trailing_stop_activation:
                    trail_distance = atr * 1.5
                    trailing_stop = highest_profit + trail_distance
                    
                    trailing_stop = min(trailing_stop, entry_price)
                    
                    return {
                        'trailing_stop': trailing_stop,
                        'highest_profit': highest_profit,
                        'trail_distance': trail_distance,
                        'activated': True
                    }
                else:
                    regular_stop = entry_price + (atr * self.sl_atr_multiple)
                    return {
                        'trailing_stop': regular_stop,
                        'highest_profit': current_price,
                        'trail_distance': atr * self.sl_atr_multiple,
                        'activated': False
                    }
                    
        except Exception as e:
            print(f"❌ Error calculating trailing stop: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "trailing_stop")
            
            if action == 'BUY':
                stop_loss = current_price - (atr * self.sl_atr_multiple)
            else:
                stop_loss = current_price + (atr * self.sl_atr_multiple)
                
            return {
                'trailing_stop': stop_loss,
                'highest_profit': current_price,
                'activated': False
            }

    def partial_position_close(self, symbol: str, position_size: float, close_percentage: float, 
                                    reason: str = "profit_target") -> Dict:
        try:
            if close_percentage <= 0 or close_percentage > 1:
                return {'success': False, 'message': 'Invalid close percentage'}
            
            close_quantity = position_size * close_percentage
            
            position_response = self.client.get_position_info(symbol)
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False, 'message': 'Failed to get position info'}
            
            positions = position_response['result']['list']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or float(position['size']) == 0:
                return {'success': False, 'message': 'No position found'}
            
            current_side = position['side']
            close_side = 'Buy' if current_side == 'Sell' else 'Sell'
            
            print(f"🔄 Partial close for {symbol}: {close_percentage*100}% ({close_quantity} units)")
            
            order_response = self.client.place_order(
                symbol=symbol,
                side=close_side,
                order_type='Market',
                qty=close_quantity,
                price=None
            )
            
            if order_response and order_response.get('retCode') == 0:
                order_id = order_response['result']['orderId']
                
                if self.database:
                    self.database.store_system_event(
                        "PARTIAL_POSITION_CLOSE",
                        {
                            'symbol': symbol,
                            'close_percentage': close_percentage,
                            'close_quantity': close_quantity,
                            'remaining_size': position_size - close_quantity,
                            'reason': reason,
                            'order_id': order_id
                        },
                        "INFO",
                        "Risk Management"
                    )
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'closed_quantity': close_quantity,
                    'remaining_quantity': position_size - close_quantity,
                    'reason': reason
                }
            else:
                error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                return {'success': False, 'message': f'Partial close failed: {error_msg}'}
                
        except Exception as e:
            error_msg = f"Error in partial position close: {e}"
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "partial_close")
            return {'success': False, 'message': error_msg}

    def volatility_adjusted_sizing(self, symbol: str, base_size: float, 
                                        historical_data: pd.DataFrame = None) -> Dict:
        try:
            if historical_data is None or len(historical_data) < self.volatility_adjustment_period:
                return {'adjusted_size': base_size, 'volatility_factor': 1.0, 'reason': 'Insufficient data'}
            
            returns = historical_data['close'].pct_change().dropna()
            recent_volatility = returns.rolling(self.volatility_adjustment_period).std().iloc[-1]
            
            normalized_vol = recent_volatility / 0.02
            
            if normalized_vol > 2.0:
                adjustment_factor = 0.5
                reason = "Very high volatility - 50% size reduction"
            elif normalized_vol > 1.5:
                adjustment_factor = 0.7
                reason = "High volatility - 30% size reduction"
            elif normalized_vol < 0.5:
                adjustment_factor = 1.3
                reason = "Very low volatility - 30% size increase"
            elif normalized_vol < 0.8:
                adjustment_factor = 1.15
                reason = "Low volatility - 15% size increase"
            else:
                adjustment_factor = 1.0
                reason = "Normal volatility - no adjustment"
            
            adjusted_size = base_size * adjustment_factor
            
            adjusted_size = min(adjusted_size, self.max_position_size_usdt)
            
            result = {
                'adjusted_size': adjusted_size,
                'volatility_factor': adjustment_factor,
                'recent_volatility': recent_volatility,
                'normalized_volatility': normalized_vol,
                'reason': reason
            }
            
            if self.database:
                self.database.store_system_event(
                    "VOLATILITY_ADJUSTED_SIZING",
                    result,
                    "INFO",
                    "Risk Management"
                )
            
            return result
            
        except Exception as e:
            print(f"❌ Error in volatility adjusted sizing: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "volatility_adjustment")
            
            return {'adjusted_size': base_size, 'volatility_factor': 1.0, 'reason': f'Error: {e}'}

    def portfolio_correlation_limits(self, new_symbol: str, new_position_size: float, 
                                        correlation_data: Dict = None) -> Dict:
        try:
            current_exposure = self.get_current_exposure()
            total_planned_exposure = current_exposure + new_position_size
            
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            
            if portfolio_value <= 0:
                return {'approved': True, 'reason': 'No portfolio value data'}
            
            scaled_max_exposure_pct = 40 * self.risk_multiplier
            planned_exposure_pct = (total_planned_exposure / portfolio_value) * 100
            
            if planned_exposure_pct > scaled_max_exposure_pct:
                allowed_exposure = max(0, (portfolio_value * scaled_max_exposure_pct / 100) - current_exposure)
                return {
                    'approved': False,
                    'reason': f'Portfolio exposure limit exceeded: {planned_exposure_pct:.1f}% > {scaled_max_exposure_pct:.1f}% (scaled x{self.risk_multiplier})',
                    'suggested_size': allowed_exposure
                }

            scaled_max_portfolio_correlation = self.max_portfolio_correlation * self.risk_multiplier

            if correlation_data and 'correlation_matrix' in correlation_data:
                avg_correlation = self._calculate_average_correlation(new_symbol, correlation_data)
                
                if avg_correlation > scaled_max_portfolio_correlation:
                    reduction_factor = scaled_max_portfolio_correlation / avg_correlation
                    suggested_size = new_position_size * reduction_factor
                    
                    return {
                        'approved': False,
                        'reason': f'High portfolio correlation: {avg_correlation:.2f} > {scaled_max_portfolio_correlation:.2f} (scaled x{self.risk_multiplier})',
                        'suggested_size': suggested_size
                    }
            
            symbol_concentration = self._calculate_symbol_concentration(new_symbol, new_position_size)
            
            scaled_max_symbol_concentration = 0.15 * self.risk_multiplier
            
            if symbol_concentration > scaled_max_symbol_concentration * 100:
                suggested_size = (portfolio_value * scaled_max_symbol_concentration) - current_exposure
                suggested_size = max(0, suggested_size)
                return {
                    'approved': False,
                    'reason': f'Symbol concentration limit: {symbol_concentration:.1f}% > {scaled_max_symbol_concentration*100:.1f}% (scaled x{self.risk_multiplier})',
                    'suggested_size': suggested_size
                }
            
            return {'approved': True, 'reason': 'All correlation and concentration checks passed'}
            
        except Exception as e:
            print(f"❌ Error in portfolio correlation limits: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, new_symbol, "correlation_limits")
            
            return {'approved': True, 'reason': f'Error in correlation check: {e}'}

    def _calculate_average_correlation(self, new_symbol: str, correlation_data: Dict) -> float:
        try:
            correlation_matrix = correlation_data.get('correlation_matrix', {})
            if new_symbol not in correlation_matrix:
                return 0.5
            
            current_symbols = self._get_current_position_symbols()
            if not current_symbols:
                return 0.0
            
            correlations = []
            for symbol in current_symbols:
                corr = correlation_matrix.get(new_symbol, {}).get(symbol) or \
                        correlation_matrix.get(symbol, {}).get(new_symbol)
                        
                if corr is not None:
                    correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.5
            
        except Exception as e:
            print(f"Error calculating average correlation: {e}")
            return 0.5

    def _calculate_symbol_concentration(self, new_symbol: str, new_position_size: float) -> float:
        try:
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            if portfolio_value <= 0:
                return 0.0
            
            current_symbol_exposure = 0
            try:
                position_response = self.client.get_position_info(new_symbol)
                if position_response and position_response.get('retCode') == 0:
                    positions = position_response['result']['list']
                    position = next((p for p in positions if p['symbol'] == new_symbol), None)
                    if position and float(position['size']) > 0:
                        current_symbol_exposure = float(position.get('positionValue', 0))
            except:
                pass
            
            total_symbol_exposure = current_symbol_exposure + new_position_size
            concentration = (total_symbol_exposure / portfolio_value) * 100
            
            return concentration
            
        except Exception as e:
            print(f"Error calculating symbol concentration: {e}")
            return 0.0

    def calculate_dynamic_correlation_matrix(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        try:
            returns_data = {}
            
            for symbol, data in symbols_data.items():
                if len(data) > 20:
                    returns_data[symbol] = data['close'].pct_change().dropna()
            
            correlation_matrix = {}
            all_symbols = list(returns_data.keys())
            
            for i, sym1 in enumerate(all_symbols):
                correlation_matrix[sym1] = {}
                for j, sym2 in enumerate(all_symbols):
                    if i == j:
                        correlation_matrix[sym1][sym2] = 1.0
                    else:
                        common_dates = returns_data[sym1].index.intersection(returns_data[sym2].index)
                        if len(common_dates) > 10:
                            corr = returns_data[sym1].loc[common_dates].corr(returns_data[sym2].loc[common_dates])
                            correlation_matrix[sym1][sym2] = corr if not np.isnan(corr) else 0.0
                        else:
                            correlation_matrix[sym1][sym2] = 0.0
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            return {}

    def calculate_historical_var(self, portfolio_returns: pd.Series, confidence_level: float = 0.95) -> float:
        try:
            if len(portfolio_returns) < 100:
                return 0.0
                
            return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        except:
            return 0.0

    def calculate_monte_carlo_var(self, positions: Dict, num_simulations: int = 10000, horizon: int = 1) -> float:
        try:
            portfolio_value = sum(pos.get('value', 0) for pos in positions.values())
            if portfolio_value == 0:
                return 0.0
                
            simulations = []
            
            for _ in range(num_simulations):
                pnl = 0
                for symbol, position in positions.items():
                    returns = np.random.normal(0, 0.02, horizon)
                    position_pnl = position['value'] * np.prod(1 + returns) - position['value']
                    pnl += position_pnl
                simulations.append(pnl)
                
            var_95 = np.percentile(simulations, 5)
            return abs(var_95)
            
        except Exception as e:
            return portfolio_value * 0.02

    def stress_test_portfolio(self, positions: Dict, scenarios: List[Dict]) -> Dict:
        try:
            results = {}
            base_portfolio_value = sum(pos.get('value', 0) for pos in positions.values())
            
            for scenario in scenarios:
                scenario_pnl = 0
                scenario_name = scenario.get('name', 'unknown')
                
                for symbol, position in positions.items():
                    shock = scenario.get('shock', 0.1)
                    if scenario.get('direction') == 'down':
                        scenario_pnl += position['value'] * -shock
                    else:
                        scenario_pnl += position['value'] * shock
                        
                results[scenario_name] = {
                    'pnl_impact': scenario_pnl,
                    'new_portfolio_value': base_portfolio_value + scenario_pnl,
                    'drawdown_percent': (scenario_pnl / base_portfolio_value) * 100
                }
                
            self.stress_test_results = results
            return results
            
        except Exception as e:
            return {}

    def regime_aware_position_sizing(self, symbol: str, confidence: float, current_price: float, 
                                        atr: float, portfolio_value: float, market_regime: str) -> Dict:
        try:
            base_info = self.calculate_aggressive_position_size(
                symbol, confidence, current_price, atr, portfolio_value, self.aggressiveness
            )
            
            regime_multipliers = {
                'high_volatility': 0.7,
                'low_volatility': 1.2,
                'trending': 1.1,
                'ranging': 0.9,
                'crisis': 0.5
            }
            
            multiplier = regime_multipliers.get(market_regime, 1.0)
            
            adjusted_size = base_info['size_usdt'] * multiplier
            adjusted_quantity = adjusted_size / current_price if current_price > 0 else 0
            
            return {
                **base_info,
                'size_usdt': adjusted_size,
                'quantity': adjusted_quantity,
                'regime_multiplier': multiplier,
                'market_regime': market_regime
            }
            
        except Exception as e:
            return self.calculate_aggressive_position_size(
                symbol, confidence, current_price, atr, portfolio_value, self.aggressiveness
            )

    def _get_current_position_symbols(self) -> List[str]:
        try:
            position_response = self.client.get_position_info(category="linear", settleCoin="USDT")
            if not position_response or position_response.get('retCode') != 0:
                return []
            
            symbols = []
            for position in position_response['result']['list']:
                if float(position.get('size', 0)) > 0:
                    symbols.append(position['symbol'])
            
            return symbols
            
        except Exception as e:
            print(f"Error getting current position symbols: {e}")
            return []

    def calculate_aggressive_stop_loss(self, symbol: str, action: str, current_price: float, 
                                          atr: float, aggressiveness: str = None) -> Dict:
        
        if aggressiveness is None:
            aggressiveness = self.aggressiveness
            
        config = RiskConfig.get_config(aggressiveness)
        
        try:
            if action == 'BUY':
                base_stop_loss = current_price - (atr * config["sl_atr_multiple"])
                base_take_profit = current_price + (atr * config["tp_atr_multiple"])
            else:
                base_stop_loss = current_price + (atr * config["sl_atr_multiple"])
                base_take_profit = current_price - (atr * config["tp_atr_multiple"])
            
            if action == 'BUY':
                base_stop_loss = max(0.01, base_stop_loss)
                base_take_profit = max(base_stop_loss + 0.01, base_take_profit)
            else:
                base_take_profit = max(0.01, base_take_profit)
                base_stop_loss = max(base_take_profit + 0.01, base_stop_loss)
            
            sl_percent = abs(current_price - base_stop_loss) / current_price * 100
            if sl_percent > config["max_sl_percent"]:
                if action == 'BUY':
                    stop_loss = current_price * (1 - config["max_sl_percent"] / 100)
                    take_profit = current_price * (1 + (config["max_sl_percent"] * config["tp_atr_multiple"] / config["sl_atr_multiple"]) / 100)
                else:
                    stop_loss = current_price * (1 + config["max_sl_percent"] / 100)
                    take_profit = current_price * (1 - (config["max_sl_percent"] * config["tp_atr_multiple"] / config["sl_atr_multiple"]) / 100)
            else:
                stop_loss = base_stop_loss
                take_profit = base_take_profit
            
            if action == 'BUY':
                if stop_loss >= current_price:
                    stop_loss = current_price * 0.99
                if take_profit <= current_price:
                    take_profit = current_price * 1.01
            else:
                if stop_loss <= current_price:
                    stop_loss = current_price * 1.01
                if take_profit >= current_price:
                    take_profit = current_price * 0.99
            
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            result = {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'distance_percent': (abs(current_price - stop_loss) / current_price) * 100,
                'aggressiveness': aggressiveness
            }
            
            if self.database:
                self.database.store_system_event(
                    "STOP_LOSS_CALCULATION",
                    {
                        'symbol': symbol,
                        'action': action,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': risk_reward_ratio,
                        'aggressiveness': aggressiveness
                    },
                    "INFO",
                    "Risk Management"
                )
            
            return result
            
        except Exception as e:
            print(f"Error in aggressive stop loss calculation: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "aggressive_stop_loss")
            
            if action == 'BUY':
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.02
            else:
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.98
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': 1.0,
                'distance_percent': 2.0,
                'aggressiveness': aggressiveness
            }

    def can_trade(self, symbol: str, size_usdt: float, market_data: Dict = None) -> Dict:
        MIN_ORDER_VALUE_USDT = 5.0
        try:
            approval = {
                'approved': False,
                'reason': '',
                'adjusted_size': 0
            }

            if self.circuit_breaker:
                approval['reason'] = f"Circuit breaker active after {self.consecutive_losses} consecutive losses"
                if self.database:
                    self.database.store_system_event("CIRCUIT_BREAKER_BLOCK", {'symbol': symbol, 'consecutive_losses': self.consecutive_losses, 'size_requested': size_usdt}, "WARNING", "Risk Management")
                return approval

            self.update_daily_pnl()

            if self.daily_pnl <= -self.max_daily_loss_percent:
                approval['reason'] = f"Daily loss limit reached: {self.daily_pnl:.2f}% (max: {self.max_daily_loss_percent}%)"
                if self.database:
                    self.database.store_system_event("DAILY_LOSS_LIMIT_BLOCK", {'symbol': symbol, 'daily_pnl': self.daily_pnl, 'max_daily_loss': self.max_daily_loss_percent, 'size_requested': size_usdt}, "WARNING", "Risk Management")
                return approval

            adjusted_size = size_usdt
            
            scaled_max_position_size = self.max_position_size_usdt * self.risk_multiplier
            
            try:
                wallet_balance = self.client.get_wallet_balance()
                if wallet_balance and wallet_balance.get('retCode') == 0:
                    total_equity = float(wallet_balance['result']['list'][0]['totalEquity'])

                    # 1. Check Max Risk Per Trade (based on equity)
                    max_risk_percent = self.config["base_size_percent"] * 2
                    max_risk_per_trade = (total_equity * max_risk_percent) * self.risk_multiplier
                    
                    if adjusted_size > max_risk_per_trade:
                        adjusted_size = max_risk_per_trade
                        approval['reason'] = f"Size capped by max risk per trade (x{self.risk_multiplier}): ${max_risk_per_trade:.2f}"

                    # 2. Check Max Position Size (hard limit)
                    if adjusted_size > scaled_max_position_size:
                        adjusted_size = scaled_max_position_size
                        approval['reason'] = f"Size capped by max_position_size (x{self.risk_multiplier}): ${scaled_max_position_size:.2f}"
                    
                    # 3. Check Total Portfolio Exposure (based on current open positions)
                    current_exposure = self.get_current_exposure()
                    scaled_max_exposure_percent = 0.3 * self.risk_multiplier
                    max_total_exposure = total_equity * scaled_max_exposure_percent
                    
                    if current_exposure + adjusted_size > max_total_exposure:
                        available_exposure = max(0, max_total_exposure - current_exposure)
                        adjusted_size = available_exposure
                        approval['reason'] = f"Size capped by portfolio exposure limit (x{self.risk_multiplier}): ${available_exposure:.2f} available"
                    
                    # If reason hasn't been set by a cap, set it to default
                    if not approval['reason']:
                        approval['reason'] = "Initial size within limits"

                else:
                    # Fallback if wallet balance fails
                    if adjusted_size > scaled_max_position_size:
                        adjusted_size = scaled_max_position_size
                        approval['reason'] = f"Size capped by max_position_size (x{self.risk_multiplier}): ${scaled_max_position_size:.2f} (Wallet fetch failed)"
                    else:
                        approval['reason'] = "Initial size within limits (Wallet fetch failed)"

            except Exception as e:
                self.logger.warning(f"Error checking wallet balance/exposure in can_trade: {e}")
                # Fallback on exception
                if adjusted_size > scaled_max_position_size:
                    adjusted_size = scaled_max_position_size
                    approval['reason'] = f"Size capped by max_position_size (x{self.risk_multiplier}): ${scaled_max_position_size:.2f} (Exception)"
                else:
                    approval['reason'] = "Initial size within limits (Exception)"


            correlation_data = self.correlation_matrix
            if market_data and 'correlation_matrix' in market_data:
                correlation_data = market_data['correlation_matrix']

            correlation_check = self.portfolio_correlation_limits(symbol, adjusted_size, {'correlation_matrix': correlation_data})
            if not correlation_check['approved']:
                suggested_size_corr = correlation_check.get('suggested_size', 0)
                if suggested_size_corr < adjusted_size:
                     adjusted_size = suggested_size_corr
                     approval['reason'] = correlation_check['reason'] + f", adjusted size to ${adjusted_size:.2f}"
                else:
                     approval['reason'] = correlation_check['reason']
                     approval['adjusted_size'] = 0
                     return approval


            sector_check = self._check_sector_exposure(symbol, adjusted_size)
            if not sector_check['approved']:
                 suggested_size_sector = sector_check.get('suggested_size', 0)
                 if suggested_size_sector < adjusted_size:
                      adjusted_size = suggested_size_sector
                      approval['reason'] = sector_check['reason'] + f", adjusted size to ${adjusted_size:.2f}"
                 else:
                      approval['reason'] = sector_check['reason']
                      approval['adjusted_size'] = 0
                      return approval


            if adjusted_size < MIN_ORDER_VALUE_USDT:
                approval['approved'] = False
                approval['reason'] = f"Final adjusted size ${adjusted_size:.2f} is below minimum order value ${MIN_ORDER_VALUE_USDT:.2f}"
                approval['adjusted_size'] = 0
                if self.database:
                    self.database.store_system_event(
                        "TRADE_REJECTED_MIN_VALUE",
                        {'symbol': symbol, 'requested_size': size_usdt, 'adjusted_size': adjusted_size, 'min_value': MIN_ORDER_VALUE_USDT},
                        "INFO",
                        "Risk Management"
                    )
                return approval


            approval['approved'] = True
            approval['adjusted_size'] = adjusted_size

            if self.database:
                self.database.store_system_event(
                    "TRADE_APPROVED",
                    {
                        'symbol': symbol,
                        'original_size': size_usdt,
                        'approved_size': approval['adjusted_size'],
                        'reason': approval['reason'] or "Passed all checks"
                    },
                    "INFO",
                    "Risk Management"
                )

            return approval

        except Exception as e:
            self.logger.error(f"Critical error in can_trade check for {symbol}: {e}", exc_info=True)
            if self.error_handler:
                self.error_handler.handle_trading_error(e, symbol, "trade_approval_critical")

            return {
                'approved': False,
                'reason': f'Critical error in risk check: {str(e)}',
                'adjusted_size': 0
            }

    def _check_sector_exposure(self, symbol: str, proposed_size: float) -> Dict:
        try:
            sector = self.symbol_to_sector.get(symbol, 'unknown')
            if sector == 'unknown':
                return {'approved': True, 'reason': 'Unknown sector'}
            
            self._update_sector_exposure()
            
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            if portfolio_value <= 0:
                return {'approved': True, 'reason': 'No portfolio value'}
            
            current_sector_exposure = self.sector_exposure.get(sector, 0)
            proposed_sector_exposure = current_sector_exposure + proposed_size
            sector_exposure_ratio = proposed_sector_exposure / portfolio_value
            
            if sector_exposure_ratio > self.max_sector_exposure:
                max_allowed_exposure = portfolio_value * self.max_sector_exposure
                suggested_size = max(0, max_allowed_exposure - current_sector_exposure)
                return {
                    'approved': False,
                    'reason': f'Sector exposure limit exceeded: {sector_exposure_ratio*100:.1f}% > {self.max_sector_exposure*100}%',
                    'suggested_size': suggested_size
                }
            
            return {'approved': True, 'reason': 'Sector exposure within limits'}
            
        except Exception as e:
            print(f"Error checking sector exposure: {e}")
            return {'approved': True, 'reason': f'Error in sector check: {e}'}
    
    def get_current_exposure(self) -> float:
        try:
            positions_response = self.client.get_position_info(category="linear", settleCoin="USDT")
            if positions_response and positions_response.get('retCode') == 0:
                positions = positions_response['result']['list']
                total_exposure = 0
                for position in positions:
                    if float(position.get('size', 0)) > 0:
                        position_value = float(position.get('positionValue', 0))
                        total_exposure += abs(position_value)
                return total_exposure
            return 0
        except Exception as e:
            print(f"Error calculating current exposure: {e}")
            if self.error_handler:
                self.error_handler.handle_api_error(e, "get_current_exposure")
            return 0
    
    def update_daily_pnl(self):
        try:
            wallet_balance = self.client.get_wallet_balance()
            if wallet_balance and wallet_balance.get('retCode') == 0:
                total_equity = float(wallet_balance['result']['list'][0]['totalEquity'])
                self.daily_pnl = ((total_equity - self.daily_start_balance) / self.daily_start_balance) * 100
                
                if self.database and abs(self.daily_pnl) > 2.0:
                    self.database.store_system_event(
                        "DAILY_PNL_UPDATE",
                        {
                            'daily_pnl': self.daily_pnl,
                            'portfolio_value': total_equity,
                            'daily_start_balance': self.daily_start_balance
                        },
                        "INFO",
                        "Risk Management"
                    )
        except Exception as e:
            print(f"Error updating daily PNL: {e}")
            if self.error_handler:
                self.error_handler.handle_api_error(e, "update_daily_pnl")
    
    def record_trade_outcome(self, successful: bool, pnl: float = 0):
        try:
            self.trades_today += 1
            
            if not successful or pnl < 0:
                self.consecutive_losses += 1
                print(f"📉 Consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
            else:
                self.consecutive_losses = 0
                print("📈 Trade successful - reset consecutive losses")
                
            if self.database:
                self.database.store_system_event(
                    "TRADE_OUTCOME_RECORDED",
                    {
                        'successful': successful,
                        'pnl': pnl,
                        'consecutive_losses': self.consecutive_losses,
                        'trades_today': self.trades_today
                    },
                    "INFO",
                    "Risk Management"
                )
            
        except Exception as e:
            print(f"Error recording trade outcome: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "record_trade_outcome")
    
    def reset_circuit_breaker(self):
        try:
            self.circuit_breaker = False
            self.consecutive_losses = 0
            print("🔄 Circuit breaker reset")
            
            if self.database:
                self.database.store_system_event(
                    "CIRCUIT_BREAKER_RESET",
                    {},
                    "INFO",
                    "Risk Management"
                )
                
        except Exception as e:
            print(f"Error resetting circuit breaker: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "reset_circuit_breaker")
    
    def get_risk_summary(self) -> Dict:
        try:
            current_exposure = self.get_current_exposure()
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            exposure_percent = (current_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
            
            risk_summary = {
                'daily_pnl_percent': self.daily_pnl,
                'daily_start_balance': self.daily_start_balance,
                'current_portfolio_value': portfolio_value,
                'max_daily_loss': self.max_daily_loss_percent,
                'max_position_size': self.max_position_size_usdt,
                'trades_today': self.trades_today,
                'aggressiveness': self.aggressiveness,
                'consecutive_losses': self.consecutive_losses,
                'circuit_breaker': self.circuit_breaker,
                'current_exposure': current_exposure,
                'exposure_percent': exposure_percent,
                'current_var': self.calculate_var(self.positions),
                'max_consecutive_losses': self.max_consecutive_losses
            }
            
            return risk_summary
            
        except Exception as e:
            print(f"Error getting risk summary: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "risk_summary")
            
            return {
                'daily_pnl_percent': 0,
                'aggressiveness': self.aggressiveness,
                'circuit_breaker': self.circuit_breaker,
                'consecutive_losses': self.consecutive_losses
            }

    def reset_daily_stats(self):
        try:
            balance = self.client.get_wallet_balance()
            if balance and balance.get('retCode') == 0:
                self.daily_start_balance = float(balance['result']['list'][0]['totalEquity'])
                self.daily_pnl = 0
                self.trades_today = 0
                self.reset_circuit_breaker()
                print(f"🔄 Daily stats reset. New starting balance: ${self.daily_start_balance:.2f}")
                
                if self.database:
                    self.database.store_system_event(
                        "DAILY_STATS_RESET",
                        {
                            'new_starting_balance': self.daily_start_balance
                        },
                        "INFO",
                        "Risk Management"
                    )
        except Exception as e:
            print(f"Error resetting daily stats: {e}")
            if self.error_handler:
                self.error_handler.handle_api_error(e, "reset_daily_stats")

    def get_portfolio_correlation(self, symbols: List[str]) -> float:
        if len(symbols) < 2:
            return 0.0
            
        try:
            positions = [self.positions.get(symbol, {}).get('size', 0) for symbol in symbols]
            total_exposure = sum(abs(pos) for pos in positions)
            
            if total_exposure == 0:
                return 0.0
                
            avg_correlation = 0.3
            return avg_correlation
            
        except Exception as e:
            print(f"Error calculating portfolio correlation: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "portfolio_correlation")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
            
        try:
            peak = equity_curve[0]
            max_dd = 0.0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    
            return max_dd * 100
            
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "max_drawdown")
            return 0.0

    def get_position_concentration(self) -> Dict:
        try:
            total_equity = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            if total_equity <= 0:
                return {}
                
            current_exposure = self.get_current_exposure()
            
            if current_exposure == 0:
                return {}
                
            try:
                positions_response = self.client.get_position_info(category="linear", settleCoin="USDT")
                position_sizes = []
                if positions_response and positions_response.get('retCode') == 0:
                    positions = positions_response['result']['list']
                    for position in positions:
                        size = float(position.get('size', 0))
                        if size > 0:
                            position_value = float(position.get('positionValue', 0))
                            position_sizes.append(abs(position_value))
            except:
                position_sizes = [abs(pos.get('size', 0)) for pos in self.positions.values()]
            
            if not position_sizes:
                return {}
                
            largest_position = max(position_sizes) if position_sizes else 0
            concentration_ratio = largest_position / current_exposure if current_exposure > 0 else 0
            
            herfindahl = sum((size / current_exposure) ** 2 for size in position_sizes) if current_exposure > 0 else 0
            
            concentration_data = {
                'total_exposure': current_exposure,
                'exposure_percent': (current_exposure / total_equity) * 100,
                'largest_position': largest_position,
                'concentration_ratio': concentration_ratio,
                'herfindahl_index': herfindahl,
                'position_count': len(position_sizes),
                'diversification_score': max(0, 1 - herfindahl) * 100
            }
            
            if self.database and len(position_sizes) > 0:
                self.database.store_system_event(
                    "POSITION_CONCENTRATION_ANALYSIS",
                    concentration_data,
                    "INFO",
                    "Risk Management"
                )
            
            return concentration_data
            
        except Exception as e:
            print(f"Error calculating position concentration: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "position_concentration")
            return {}

    def get_risk_assessment(self) -> Dict:
        try:
            risk_summary = self.get_risk_summary()
            concentration = self.get_position_concentration()
            health_checks = self._perform_health_checks()
            
            risk_assessment = {
                'summary': risk_summary,
                'concentration': concentration,
                'health_checks': health_checks,
                'overall_risk_level': self._calculate_overall_risk_level(risk_summary, concentration, health_checks)
            }
            
            if self.database:
                self.database.store_system_event(
                    "RISK_ASSESSMENT",
                    risk_assessment,
                    "INFO",
                    "Risk Management"
                )
            
            return risk_assessment
            
        except Exception as e:
            print(f"Error performing risk assessment: {e}")
            if self.error_handler:
                self.error_handler.handle_trading_error(e, "ALL", "risk_assessment")
            
            return {
                'summary': {},
                'concentration': {},
                'health_checks': {},
                'overall_risk_level': 'UNKNOWN'
            }
    
    def _perform_health_checks(self) -> Dict:
        checks = {}
        
        try:
            checks['daily_pnl_ok'] = self.daily_pnl > -self.max_daily_loss_percent
            
            checks['circuit_breaker_ok'] = not self.circuit_breaker
            
            checks['consecutive_losses_ok'] = self.consecutive_losses < self.max_consecutive_losses
            
            portfolio_value = self.daily_start_balance + (self.daily_start_balance * self.daily_pnl / 100)
            current_exposure = self.get_current_exposure()
            exposure_ratio = current_exposure / portfolio_value if portfolio_value > 0 else 0
            checks['exposure_ok'] = exposure_ratio <= 0.5
            
            concentration = self.get_position_concentration()
            checks['concentration_ok'] = concentration.get('herfindahl_index', 0) <= 0.3
            
            checks['all_ok'] = all(checks.values())
            
        except Exception as e:
            print(f"Error performing health checks: {e}")
            checks['all_ok'] = False
        
        return checks
    
    def _calculate_overall_risk_level(self, risk_summary: Dict, concentration: Dict, health_checks: Dict) -> str:
        if not health_checks.get('all_ok', False):
            return "HIGH"
        
        try:
            risk_score = 0
            
            pnl_risk = min(40, max(0, -risk_summary.get('daily_pnl_percent', 0) / self.max_daily_loss_percent * 40))
            risk_score += pnl_risk
            
            exposure_risk = min(30, risk_summary.get('exposure_percent', 0) / 50 * 30)
            risk_score += exposure_risk
            
            concentration_risk = min(20, concentration.get('herfindahl_index', 0) * 20 / 0.3)
            risk_score += concentration_risk
            
            loss_risk = min(10, risk_summary.get('consecutive_losses', 0) / self.max_consecutive_losses * 10)
            risk_score += loss_risk
            
            if risk_score >= 60:
                return "HIGH"
            elif risk_score >= 30:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            print(f"Error calculating overall risk level: {e}")
            return "UNKNOWN"

    def should_reduce_risk(self) -> bool:
        try:
            risk_assessment = self.get_risk_assessment()
            
            conditions = [
                risk_assessment.get('overall_risk_level') == "HIGH",
                self.daily_pnl <= -self.max_daily_loss_percent * 0.8,
                self.consecutive_losses >= self.max_consecutive_losses - 1,
                risk_assessment['summary'].get('exposure_percent', 0) > 40
            ]
            
            return any(conditions)
            
        except Exception as e:
            print(f"Error determining risk reduction: {e}")
            return False

    def get_risk_reduction_suggestions(self) -> List[str]:
        try:
            suggestions = []
            risk_assessment = self.get_risk_assessment()
            summary = risk_assessment['summary']
            concentration = risk_assessment['concentration']
            
            if summary.get('daily_pnl_percent', 0) < -self.max_daily_loss_percent * 0.5:
                suggestions.append("Reduce position sizes - daily PnL is negative")
            
            if summary.get('consecutive_losses', 0) >= 2:
                suggestions.append(f"Consider reducing trading frequency - {summary['consecutive_losses']} consecutive losses")
            
            if concentration.get('exposure_percent', 0) > 30:
                suggestions.append(f"Reduce portfolio exposure - currently {concentration['exposure_percent']:.1f}%")
            
            if concentration.get('herfindahl_index', 0) > 0.2:
                suggestions.append("Diversify positions - portfolio is too concentrated")
            
            if self.circuit_breaker:
                suggestions.append("Circuit breaker active - trading suspended until conditions improve")
            
            if not suggestions:
                suggestions.append("Risk levels are acceptable - continue current strategy")
            
            return suggestions
            
        except Exception as e:
            print(f"Error getting risk reduction suggestions: {e}")
            return ["Unable to assess risk - check system logs"]