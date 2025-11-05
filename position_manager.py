import time
from datetime import datetime, timedelta
import pandas as pd
from config import TIMEFRAME

class PositionManager:
    
    def __init__(self, execution_engine, strategy_orchestrator, data_engine, risk_manager, database, error_handler, telegram_bot=None):
        self.execution_engine = execution_engine
        self.strategy_orchestrator = strategy_orchestrator
        self.data_engine = data_engine
        self.risk_manager = risk_manager
        self.database = database
        self.error_handler = error_handler
        self.telegram_bot = telegram_bot
        self.logger = self.strategy_orchestrator.logger

        # Initialize with validation
        self._initialize_with_validation()

    def _initialize_with_validation(self):
        """Initialize and validate configuration parameters"""
        self.time_stop_candles = 50
        self.partial_tp_percent = 2.0
        self.partial_tp_size = 0.5
        
        # Validate critical parameters
        assert 0 < self.partial_tp_size < 1, f"Partial TP size ({self.partial_tp_size}) must be between 0 and 1"
        assert self.time_stop_candles > 0, f"Time stop candles ({self.time_stop_candles}) must be positive"
        assert self.partial_tp_percent > 0, f"Partial TP percent ({self.partial_tp_percent}) must be positive"
        
        self.logger.info("PositionManager configuration validated successfully")

    def manage_open_positions(self):
        self.logger.info("--- 🛡️ Starting Active Position Management ---")
        try:
            open_positions = self.execution_engine.get_live_positions_from_cache()
            if not open_positions:
                self.logger.info("No open positions to manage.")
                return

            for symbol, position in open_positions.items():
                if position.get('size', 0) <= 0:
                    continue
                
                self.logger.info(f"Managing open position: {symbol}")
                self._manage_single_position(symbol, position)

        except Exception as e:
            self.logger.error(f"Error during position management cycle: {e}", exc_info=True)
            self.error_handler.handle_trading_error(e, "ALL", "position_management_cycle")
        finally:
            self.logger.info("--- 🛡️ Active Position Management Finished ---")

    def _manage_single_position(self, symbol: str, position: dict):
        try:
            original_trade = self.database.get_open_trade_for_symbol(symbol)
            if not original_trade:
                self.logger.warning(f"No open trade found in DB for live position {symbol}. Cannot manage.")
                return

            market_data = self.data_engine.get_market_data_for_analysis(symbol)
            if market_data is None or market_data.empty:
                self.logger.warning(f"No market data for {symbol}. Cannot manage position.")
                return

            current_price = market_data['close'].iloc[-1]
            entry_price = original_trade.get('entry_price', 0)
            position_side = position.get('side', 'Buy')
            
            if entry_price == 0:
                self.logger.warning(f"Trade {original_trade['id']} for {symbol} has entry_price 0. Skipping.")
                return

            if position_side == 'Buy':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_percent = ((entry_price - current_price) / entry_price) * 100

            trade_id = original_trade['id']
            trade_flags = self.database.get_trade_flags(trade_id)

            self.logger.info(f"Managing {symbol}: Side={position_side}, Size={position.get('size', 0)}, "
                            f"PnL={pnl_percent:.2f}%, Price={current_price:.4f}")

            if self._check_time_stop(original_trade, pnl_percent):
                return True
            
            if self._manage_partial_take_profit(symbol, pnl_percent, trade_id, trade_flags):
                return True
            
            if self._manage_trailing_stop(symbol, position, original_trade, current_price, pnl_percent, market_data):
                return True

            if self._check_signal_invalidation(symbol, position_side, market_data):
                return True

        except Exception as e:
            self.logger.error(f"Failed to manage position for {symbol}: {e}", exc_info=True)
            self.error_handler.handle_trading_error(e, symbol, "manage_single_position")

    def _calculate_recent_volatility(self, market_data: pd.DataFrame, period: int = 10) -> float:
        """Calculate recent price volatility as percentage"""
        try:
            returns = market_data['close'].pct_change().dropna()
            recent_returns = returns.tail(period)
            volatility = recent_returns.std() * 100  # As percentage
            return volatility if not pd.isna(volatility) else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _check_time_stop(self, original_trade: dict, pnl_percent: float) -> bool:
        try:
            trade_timestamp = pd.to_datetime(original_trade['timestamp'])
            candles_open = (datetime.now() - trade_timestamp).total_seconds() / (int(TIMEFRAME) * 60)
            
            if candles_open > self.time_stop_candles and pnl_percent < 0.25:
                self.logger.info(f"TIME STOP: Closing {original_trade['symbol']} after {candles_open:.0f} candles with {pnl_percent:.2f}% PnL.")
                close_result = self.execution_engine.close_position(original_trade['symbol'], "TimeStop")
                if close_result.get('success') and self.telegram_bot:
                    self.telegram_bot.log_time_stop_close(original_trade['symbol'], pnl_percent, int(candles_open))
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in _check_time_stop: {e}", exc_info=True)
            return False

    def _manage_partial_take_profit(self, symbol: str, pnl_percent: float, trade_id: int, flags: dict) -> bool:
        if flags.get('partial_tp_taken', False):
            return False
            
        try:
            if pnl_percent >= self.partial_tp_percent:
                self.logger.info(f"PARTIAL TP: {symbol} at {pnl_percent:.2f}%. Closing {self.partial_tp_size*100}% of position.")
                result = self.execution_engine.execute_partial_close(symbol, self.partial_tp_size)
                
                if result.get('success'):
                    self.database.set_trade_flag(trade_id, 'partial_tp_taken', True)
                    self.logger.info(f"PARTIAL TP: Successfully closed {self.partial_tp_size*100}% of {symbol}.")
                    if self.telegram_bot:
                        self.telegram_bot.log_partial_take_profit(
                            symbol,
                            pnl_percent,
                            result.get('quantity_closed', 0),
                            self.partial_tp_size
                        )
                    return True
                else:
                    self.logger.warning(f"PARTIAL TP: Failed to close {symbol}: {result.get('message')}")
                    return False
            return False
        except Exception as e:
            self.logger.error(f"Error in _manage_partial_take_profit: {e}", exc_info=True)
            return False

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR) correctly"""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0

    def _manage_trailing_stop(self, symbol: str, position: dict, original_trade: dict, current_price: float, pnl_percent: float, market_data: pd.DataFrame) -> bool:
        try:
            atr = self._calculate_atr(market_data)
            if atr == 0:
                self.logger.warning(f"ATR calculation failed for {symbol}, skipping trailing stop")
                return False
            
            position_side = position.get('side', 'Buy')
            current_sl_price = float(position.get('stopLoss', 0))
            current_tp_price = float(position.get('takeProfit', 0))

            target_sl_price = self.risk_manager.calculate_trailing_stop(
                original_trade,
                current_price,
                pnl_percent,
                atr
            )
            
            if target_sl_price == 0:
                return False

            sl_price_to_use = current_sl_price
            needs_sl_update = False

            if position_side == 'Buy':
                if target_sl_price > current_sl_price:
                    needs_sl_update = True
            elif position_side == 'Sell':
                if target_sl_price < current_sl_price and target_sl_price > 0:
                    needs_sl_update = True

            if needs_sl_update:
                self.logger.info(f"TRAIL: Moving {symbol} {position_side} SL from {current_sl_price:.4f} to {target_sl_price:.4f}")
                
                tp_to_send = str(current_tp_price) if current_tp_price > 0 else None
                
                sl_tp_response = self.execution_engine.client.set_trading_stop(
                    symbol=symbol,
                    stop_loss=str(target_sl_price),
                    take_profit=tp_to_send
                )

                if sl_tp_response and sl_tp_response.get('retCode') == 0:
                    self.logger.info(f"TRAIL: Successfully updated SL for {symbol} on exchange.")
                    sl_price_to_use = target_sl_price
                    if self.telegram_bot:
                        self.telegram_bot.log_trailing_stop_update(
                            symbol,
                            position_side,
                            current_sl_price,
                            target_sl_price,
                            pnl_percent
                        )
                else:
                    err = sl_tp_response.get('retMsg', 'Failed') if sl_tp_response else 'No Response'
                    self.logger.warning(f"TRAIL: Failed to update SL for {symbol}: {err}")
            
            if sl_price_to_use > 0:
                if position_side == 'Buy' and current_price < sl_price_to_use:
                    self.logger.info(f"TRAILING STOP HIT: Closing {symbol} BUY. Price {current_price} hit TSL {sl_price_to_use}")
                    close_result = self.execution_engine.close_position(symbol, "TrailingStopLoss")
                    if close_result.get('success') and self.telegram_bot:
                        self.telegram_bot.log_trailing_stop_close(symbol, pnl_percent, sl_price_to_use, "Buy")
                    return True
                elif position_side == 'Sell' and current_price > sl_price_to_use:
                    self.logger.info(f"TRAILING STOP HIT: Closing {symbol} SELL. Price {current_price} hit TSL {sl_price_to_use}")
                    close_result = self.execution_engine.close_position(symbol, "TrailingStopLoss")
                    if close_result.get('success') and self.telegram_bot:
                        self.telegram_bot.log_trailing_stop_close(symbol, pnl_percent, sl_price_to_use, "Sell")
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error in _manage_trailing_stop: {e}", exc_info=True)
            return False

    def _check_signal_invalidation(self, symbol: str, position_side: str, market_data: pd.DataFrame) -> bool:
        try:
            portfolio_value = self.strategy_orchestrator.risk_manager.daily_start_balance
            new_decision = self.strategy_orchestrator.analyze_symbol(symbol, market_data, portfolio_value)
            
            new_action = new_decision.get('action')
            new_confidence = new_decision.get('confidence', 0)
            min_confidence = self.risk_manager.min_confidence
            
            is_invalid = False
            if position_side == 'Buy' and new_action == 'SELL':
                is_invalid = True
            elif position_side == 'Sell' and new_action == 'BUY':
                is_invalid = True
                
            if is_invalid and new_confidence >= min_confidence:
                self.logger.info(f"SIGNAL INVALIDATION: Closing {symbol} {position_side} position. New signal is {new_action} (Confidence: {new_confidence:.1f}% >= {min_confidence}%)")
                close_result = self.execution_engine.close_position(symbol, "Signal Invalidation")
                if close_result.get('success') and self.telegram_bot:
                    self.telegram_bot.log_signal_invalidation_close(symbol, position_side, new_action)
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Error in _check_signal_invalidation: {e}", exc_info=True)
            return False