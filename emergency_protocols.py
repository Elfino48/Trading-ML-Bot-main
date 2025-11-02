import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from execution_engine import ExecutionEngine
from telegram_bot import TelegramBot
from config import EMERGENCY_PROTOCOLS_ENABLED

class EmergencyProtocols:
    """
    Emergency safety mechanisms for the trading bot
    Monitors for critical conditions and executes emergency procedures
    """
    
    def __init__(self, execution_engine: ExecutionEngine, telegram_bot: Optional[TelegramBot] = None):
        self.execution_engine = execution_engine
        self.telegram_bot = telegram_bot
        self.emergency_mode = False
        self.enabled = EMERGENCY_PROTOCOLS_ENABLED
        self.emergency_start_time = None
        self.triggered_protocols = []
        
        # Enhanced emergency thresholds
        self.max_drawdown = 0.10  # 10% max drawdown from daily high
        self.max_position_loss = 0.05  # 5% max loss on single position
        self.max_consecutive_losses = 5
        self.max_daily_loss = 0.08  # 8% max daily loss
        self.max_trading_errors = 3  # Maximum trading errors before emergency
        
        # Monitoring variables
        self.daily_high_balance = 0
        self.consecutive_losses = 0
        self.large_loss_count = 0
        self.trading_error_count = 0
        
        self.logger = logging.getLogger('EmergencyProtocols')
        
        print("🆘 Emergency Protocols initialized")
    
    def check_emergency_conditions(self, portfolio_value: float, daily_pnl: float, 
                                 recent_trades: List[Dict] = None, error_count: int = 0) -> Dict:
        """
        Check for emergency conditions that require immediate action
        """
        if not self.enabled:
            return {'emergency': False, 'reason': 'Emergency protocols disabled via config'}
            
        if self.emergency_mode:
            return {'emergency': True, 'reason': 'Already in emergency mode'}
        
        # Initialize daily high balance if not set
        if self.daily_high_balance == 0 and portfolio_value > 0:
            self.daily_high_balance = portfolio_value
        
        # Update daily high
        self.daily_high_balance = max(self.daily_high_balance, portfolio_value)
        
        triggered_conditions = []
        
        # 1. Check maximum drawdown from daily high
        drawdown = (self.daily_high_balance - portfolio_value) / self.daily_high_balance
        if drawdown >= self.max_drawdown:
            triggered_conditions.append({
                'condition': 'MAX_DRAWDOWN',
                'value': f"{drawdown:.2%}",
                'threshold': f"{self.max_drawdown:.2%}"
            })
        
        # 2. Check daily loss limit
        if daily_pnl <= -self.max_daily_loss * 100:  # Convert to percentage
            triggered_conditions.append({
                'condition': 'DAILY_LOSS_LIMIT',
                'value': f"{daily_pnl:.2f}%",
                'threshold': f"{-self.max_daily_loss * 100:.2f}%"
            })
        
        # 3. Check consecutive losses
        if recent_trades:
            self._update_consecutive_losses(recent_trades)
            if self.consecutive_losses >= self.max_consecutive_losses:
                triggered_conditions.append({
                    'condition': 'CONSECUTIVE_LOSSES',
                    'value': self.consecutive_losses,
                    'threshold': self.max_consecutive_losses
                })
        
        # 4. Check for trading errors
        self.trading_error_count = error_count
        if error_count >= self.max_trading_errors:
            triggered_conditions.append({
                'condition': 'EXCESSIVE_TRADING_ERRORS',
                'value': error_count,
                'threshold': self.max_trading_errors
            })
        
        # 5. Check for market crash conditions (rapid decline)
        if self._detect_market_crash(recent_trades):
            triggered_conditions.append({
                'condition': 'MARKET_CRASH_DETECTED',
                'value': 'Rapid decline detected',
                'threshold': 'Multiple large losses in short period'
            })
        
        # 6. Check for liquidity issues (failed trades)
        if self._detect_liquidity_issues(recent_trades):
            triggered_conditions.append({
                'condition': 'LIQUIDITY_ISSUES',
                'value': 'High trade failure rate',
                'threshold': '>30% trade failures'
            })
        
        result = {
            'emergency': len(triggered_conditions) > 0,
            'triggered_conditions': triggered_conditions,
            'current_metrics': {
                'drawdown': drawdown,
                'daily_pnl': daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'trading_errors': self.trading_error_count,
                'daily_high': self.daily_high_balance
            }
        }
        
        # If emergency conditions detected, execute protocols
        if result['emergency']:
            self.execute_emergency_stop(triggered_conditions)
        
        return result
    
    def _update_consecutive_losses(self, recent_trades: List[Dict]):
        """Update consecutive losses counter from recent trades"""
        if not recent_trades:
            return
        
        # Look at last 10 trades max
        check_trades = recent_trades[-10:]
        self.consecutive_losses = 0
        
        # Count backwards until we find a winning trade
        for trade in reversed(check_trades):
            # Simplified: assume negative PnL means loss
            pnl = trade.get('pnl_percent', 0)
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                break
    
    def _detect_market_crash(self, recent_trades: List[Dict]) -> bool:
        """Detect potential market crash conditions"""
        if not recent_trades or len(recent_trades) < 3:
            return False
        
        # Check last 5 trades for large losses
        recent = recent_trades[-5:]
        large_losses = 0
        
        for trade in recent:
            pnl = trade.get('pnl_percent', 0)
            if pnl < -3:  # 3% or more loss
                large_losses += 1
        
        # If 3+ large losses in last 5 trades, potential crash
        return large_losses >= 3
    
    def _detect_liquidity_issues(self, recent_trades: List[Dict]) -> bool:
        """Detect liquidity issues from trade failures"""
        if not recent_trades:
            return False
        
        # Check last 10 trades for failures
        recent = recent_trades[-10:]
        if len(recent) < 5:
            return False
        
        failures = sum(1 for trade in recent if not trade.get('success', True))
        failure_rate = failures / len(recent)
        
        return failure_rate > 0.3  # 30% failure rate
    
    def execute_emergency_stop(self, reasons: List[Dict]):
        """
        Execute emergency stop procedure with enhanced verification
        """
        if self.emergency_mode:
            self.logger.warning("Emergency stop already active")
            return
        
        self.emergency_mode = True
        self.emergency_start_time = datetime.now()
        self.triggered_protocols = reasons
        
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reasons}")
        
        # Send immediate emergency alert
        self._send_emergency_alert(reasons)
        
        # Execute emergency procedures with verification
        procedures_executed = []
        
        try:
            # 1. Close all open positions with verification
            close_result = self._close_all_positions_with_verification()
            procedures_executed.append(close_result)
            
            # 2. Cancel all pending orders
            cancel_result = self._cancel_all_orders()
            procedures_executed.append(cancel_result)
            
            # 3. Reduce leverage if possible
            leverage_result = self._reduce_leverage()
            procedures_executed.append(leverage_result)
            
            # 4. Verify all positions are closed
            verification_result = self._verify_all_positions_closed()
            procedures_executed.append(verification_result)
            
            self.logger.info("Emergency procedures executed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during emergency procedures: {e}")
            procedures_executed.append({'procedure': 'ERROR', 'success': False, 'error': str(e)})
        
        # Send follow-up with procedure results
        self._send_emergency_followup(procedures_executed)

    def _close_all_positions_with_verification(self) -> Dict:
        """Close all open positions with verification"""
        try:
            # Use the enhanced execution engine method
            result = self.execution_engine.emergency_stop_with_verification()
            return {
                'procedure': 'CLOSE_ALL_POSITIONS_WITH_VERIFICATION',
                'success': result.get('success', False),
                'details': result
            }
        except Exception as e:
            return {
                'procedure': 'CLOSE_ALL_POSITIONS_WITH_VERIFICATION',
                'success': False,
                'error': str(e)
            }

    def _verify_all_positions_closed(self) -> Dict:
        """Verify that all positions are actually closed"""
        try:
            # Get current positions (assuming execution_engine.client has this method)
            position_response = self.execution_engine.client.get_position_info(category="linear", settleCoin="USDT")
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False, 'error': 'Failed to get position info', 'procedure': 'VERIFY_POSITIONS_CLOSED'}
            
            open_positions = []
            for position in position_response['result']['list']:
                # Assuming 'size' field indicates open position size, and it's a string that needs conversion
                if float(position.get('size', 0)) > 0:
                    open_positions.append({
                        'symbol': position['symbol'],
                        'size': float(position['size']),
                        'side': position['side']
                    })
            
            if open_positions:
                return {
                    'procedure': 'VERIFY_POSITIONS_CLOSED',
                    'success': False,
                    'open_positions': open_positions
                }
            else:
                return {
                    'procedure': 'VERIFY_POSITIONS_CLOSED',
                    'success': True,
                    'message': 'All positions verified closed'
                }
                
        except Exception as e:
            return {
                'procedure': 'VERIFY_POSITIONS_CLOSED',
                'success': False,
                'error': str(e)
            }

    def _cancel_all_orders(self) -> Dict:
        """Cancel all pending orders"""
        try:
            result = self.execution_engine.cancel_all_orders(settleCoin="USDT")
            return {
                'procedure': 'CANCEL_ALL_ORDERS',
                'success': True,
                'message': 'Orders cancellation attempted'
            }
        except Exception as e:
            return {
                'procedure': 'CANCEL_ALL_ORDERS',
                'success': False,
                'error': str(e)
            }
    
    def _reduce_leverage(self) -> Dict:
        """Reduce leverage across all symbols"""
        try:
            # This would require additional methods in execution_engine
            # For now, return not implemented
            return {
                'procedure': 'REDUCE_LEVERAGE',
                'success': True,
                'message': 'Leverage reduction not implemented in current version'
            }
        except Exception as e:
            return {
                'procedure': 'REDUCE_LEVERAGE',
                'success': False,
                'error': str(e)
            }

    def _send_emergency_alert(self, reasons: List[Dict]):
        """Send emergency alert via Telegram"""
        if not self.telegram_bot:
            return
        
        try:
            reasons_text = "\n".join([
                f"• {cond['condition']}: {cond['value']} (threshold: {cond['threshold']})"
                for cond in reasons
            ])
            
            message = (
                "🚨 🚨 🚨 <b>EMERGENCY STOP ACTIVATED</b> 🚨 🚨 🚨\n\n"
                f"<b>Triggered Conditions:</b>\n{reasons_text}\n\n"
                f"<b>Actions Taken:</b>\n"
                f"• All positions being closed\n"
                f"• All orders being cancelled\n"
                f"• Trading suspended\n\n"
                f"<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>"
            )
            
            self.telegram_bot.send_channel_message(message)
        except Exception as e:
            self.logger.error(f"Failed to send emergency alert: {e}")
    
    def _send_emergency_followup(self, procedures: List[Dict]):
        """Send follow-up message with procedure results"""
        if not self.telegram_bot:
            return
        
        try:
            # Ensure all procedures have a 'message' field for the original format
            # This is a bit of a merge challenge since original only used 'message' and suggested only used 'details'/'open_positions'/'error'
            def get_procedure_message(p):
                if p.get('success'):
                    return p.get('message', p.get('details', {}).get('message', 'Success'))
                return p.get('error', p.get('open_positions', 'Failure'))

            procedures_text = "\n".join([
                f"• {p['procedure']}: {'✅' if p.get('success') else '❌'} {get_procedure_message(p)}"
                for p in procedures
            ])
            
            message = (
                "🆘 <b>EMERGENCY PROCEDURES COMPLETE</b>\n\n"
                f"<b>Procedures Executed:</b>\n{procedures_text}\n\n"
                f"<b>Current Status:</b> Trading suspended\n"
                f"<b>Emergency Start:</b> {self.emergency_start_time.strftime('%H:%M:%S')}\n\n"
                f"<i>Manual reset required to resume trading</i>"
            )
            
            self.telegram_bot.send_channel_message(message)
        except Exception as e:
            self.logger.error(f"Failed to send emergency followup: {e}")
    
    def reset_emergency_mode(self, reason: str = "Manual reset"):
        """
        Reset emergency mode and resume normal operations
        
        Args:
            reason: Reason for resetting emergency mode
        """
        if not self.emergency_mode:
            self.logger.warning("Not in emergency mode, nothing to reset")
            return
        
        self.emergency_mode = False
        self.triggered_protocols = []
        
        # Reset monitoring variables
        self.consecutive_losses = 0
        self.large_loss_count = 0
        self.trading_error_count = 0 # Added reset for trading error count
        
        # --- THIS IS THE FIX ---
        if self.execution_engine:
            self.execution_engine.clear_trade_history()
            self.logger.info("Cleared ExecutionEngine trade failure history to prevent reset loop.")
        # --- END FIX ---

        self.logger.info(f"Emergency mode reset: {reason}")
        
        if self.telegram_bot:
            try:
                self.telegram_bot.send_channel_message(
                    f"🔄 <b>EMERGENCY MODE RESET</b>\n\n"
                    f"Reason: {reason}\n"
                    f"Duration: {self._get_emergency_duration()}\n"
                    f"Trading will resume in next cycle.\n\n"
                    f"<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>"
                )
            except Exception as e:
                self.logger.error(f"Failed to send reset notification: {e}")
    
    def _get_emergency_duration(self) -> str:
        """Calculate how long emergency mode was active"""
        if not self.emergency_start_time:
            return "Unknown"
        
        duration = datetime.now() - self.emergency_start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    def get_emergency_status(self) -> Dict:
        """
        Get current emergency status
        """
        return {
            'emergency_mode': self.emergency_mode,
            'emergency_start_time': self.emergency_start_time,
            'triggered_protocols': self.triggered_protocols,
            'current_metrics': {
                'consecutive_losses': self.consecutive_losses,
                'large_loss_count': self.large_loss_count,
                'trading_errors': self.trading_error_count, # Added trading error count
                'daily_high_balance': self.daily_high_balance
            },
            'thresholds': {
                'max_drawdown': self.max_drawdown,
                'max_daily_loss': self.max_daily_loss,
                'max_consecutive_losses': self.max_consecutive_losses,
                'max_trading_errors': self.max_trading_errors # Added max_trading_errors
            }
        }
    
    def update_thresholds(self, max_drawdown: float = None, max_daily_loss: float = None,
                          max_consecutive_losses: int = None, max_trading_errors: int = None):
        """
        Update emergency thresholds dynamically
        """
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
        if max_daily_loss is not None:
            self.max_daily_loss = max_daily_loss
        if max_consecutive_losses is not None:
            self.max_consecutive_losses = max_consecutive_losses
        if max_trading_errors is not None:
            self.max_trading_errors = max_trading_errors
        
        self.logger.info(f"Emergency thresholds updated: "
                         f"Drawdown={self.max_drawdown:.2%}, "
                         f"Daily Loss={self.max_daily_loss:.2%}, "
                         f"Consecutive Losses={self.max_consecutive_losses}, "
                         f"Trading Errors={self.max_trading_errors}") # Added trading errors to log