import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from telegram_bot import TelegramBot

class ErrorHandler:
    """
    Comprehensive error handling framework for the trading bot
    Implements retry logic, circuit breakers, and error tracking
    """
    
    def __init__(self, telegram_bot: Optional[TelegramBot] = None):
        self.telegram_bot = telegram_bot
        self.error_count = 0
        self.max_errors_before_stop = 10
        self.circuit_breaker = False
        self.error_history = []
        self.api_error_count = 0
        self.trading_error_count = 0
        self.ml_error_count = 0
        self.data_error_count = 0
        
        # Enhanced trading error tracking
        self.consecutive_trading_errors = 0
        self.max_consecutive_trading_errors = 3
        self.last_error_time = None
        self.error_time_window = 300  # 5 minutes
        
        # Setup logging
        self._setup_logging()
        
        print("🛡️ Error Handler initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_errors.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ErrorHandler')
    
    def handle_api_error(self, error: Exception, context: str, retry_count: int = 0) -> Dict:
        """
        Handle API errors with retry logic and circuit breakers
        
        Args:
            error: The exception that occurred
            context: Context where error happened (e.g., 'get_balance', 'place_order')
            retry_count: Current retry attempt
            
        Returns:
            Dict with handling results and retry instructions
        """

        is_websocket_error = context and context.startswith("ws_")

        if not is_websocket_error:
            self.error_count += 1
            self.api_error_count += 1
        else:
            print(f"ℹ️ Ignoring WebSocket error for emergency count: {error} in {context}") # Optional logging
        
        error_info = {
            'timestamp': datetime.now(),
            'type': 'API_ERROR',
            'context': context,
            'error': str(error),
            'retry_count': retry_count,
            'handled': False
        }
        
        # Log the error
        self.logger.error(f"API Error in {context}: {error} (Retry {retry_count})")
        
        # Check if we should enable circuit breaker
        if self.api_error_count >= 5:
            self._activate_circuit_breaker("Too many API errors")
            error_info['circuit_breaker_activated'] = True
        
        # Determine retry strategy based on error type
        if self._is_retryable_error(error):
            if retry_count < 3:
                wait_time = self._calculate_retry_delay(retry_count)
                error_info['retry_after'] = wait_time
                error_info['should_retry'] = True
                error_info['handled'] = True
                
                self.logger.info(f"Will retry API call in {wait_time} seconds")
            else:
                error_info['should_retry'] = False
                error_info['max_retries_exceeded'] = True
        else:
            error_info['should_retry'] = False
            error_info['non_retryable_error'] = True
        
        # Send Telegram alert for serious errors
        if retry_count >= 2 or not error_info['should_retry']:
            self._send_error_alert(f"API Error in {context}", str(error))
        
        self.error_history.append(error_info)
        return error_info
    
    def handle_trading_error(self, error: Exception, symbol: str, action: str) -> Dict:
        """
        Handle trading-specific errors with enhanced circuit breaking
        """
        error_str = str(error).lower()
        
        # --- ADD: Check for ignorable errors ---
        ignorable_errors = [
            'initial quantity', 'below minimum',  # Insufficient balance errors
            'stoploss', 'base_price', 'should lower than', 'should higher than'  # Stop loss placement errors
        ]
        
        if any(ignore_pattern in error_str for ignore_pattern in ignorable_errors):
            # Log as warning but don't count towards error limits
            self.logger.warning(f"Ignoring trading error for {symbol} {action}: {error}")
            return {
                'timestamp': datetime.now(),
                'type': 'TRADING_ERROR_IGNORED',
                'symbol': symbol,
                'action': action,
                'error': str(error),
                'handled': True,
                'ignored': True
            }
        # --- END ADD ---

        self.error_count += 1
        self.trading_error_count += 1
        self.consecutive_trading_errors += 1
        self.last_error_time = datetime.now()
        
        error_info = {
            'timestamp': datetime.now(),
            'type': 'TRADING_ERROR',
            'symbol': symbol,
            'action': action,
            'error': str(error),
            'handled': False,
            'consecutive_trading_errors': self.consecutive_trading_errors
        }
        
        self.logger.error(f"Trading Error for {symbol} {action}: {error}")
        
        # Check for excessive consecutive trading errors (but skip ignored ones)
        if self.consecutive_trading_errors >= self.max_consecutive_trading_errors:
            error_info['circuit_breaker_triggered'] = True
            self._activate_circuit_breaker(f"Too many consecutive trading errors: {self.consecutive_trading_errors}")
        
        # Check error rate in time window (excluding ignored errors)
        recent_errors = [e for e in self._get_recent_errors(self.error_time_window) 
                        if not e.get('ignored', False)]
        if len(recent_errors) >= 5:
            error_info['high_error_rate'] = True
            self._activate_circuit_breaker("High error rate detected in recent window")
        
        # Specific handling for common trading errors
        if 'insufficient balance' in error_str:
            error_info['handled'] = True
            error_info['suggestion'] = 'Check account balance and reduce position sizes'
            # This is a critical error that should stop trading
            self._activate_circuit_breaker("Insufficient balance - trading suspended")
        elif 'position size too small' in error_str:
            error_info['handled'] = True
            error_info['suggestion'] = 'Increase position size or skip trade'
        elif 'market closed' in error_str:
            error_info['handled'] = True
            error_info['suggestion'] = 'Market may be closed, skip trading'
        elif 'rate limit' in error_str:
            error_info['handled'] = True
            error_info['suggestion'] = 'API rate limit exceeded, implement backoff'
            
        self.error_history.append(error_info)
        return error_info

    def handle_ml_error(self, error: Exception, symbol: str, operation: str = 'prediction') -> Dict:
        """
        Handle ML training and prediction errors
        
        Args:
            error: ML error
            symbol: Trading symbol
            operation: ML operation (training/prediction)
            
        Returns:
            Dict with error handling results
        """
        self.error_count += 1
        self.ml_error_count += 1
        
        error_info = {
            'timestamp': datetime.now(),
            'type': 'ML_ERROR',
            'symbol': symbol,
            'operation': operation,
            'error': str(error),
            'handled': False
        }
        
        self.logger.error(f"ML Error for {symbol} during {operation}: {error}")
        
        # ML errors are usually non-critical, so we can continue without ML
        error_info['handled'] = True
        error_info['suggestion'] = 'Continue trading without ML predictions'
        
        # Only alert for training errors, not prediction errors
        if operation == 'training':
            self._send_error_alert(
                f"ML Training Error - {symbol}",
                f"Failed to train ML model: {error}"
            )
        
        self.error_history.append(error_info)
        return error_info
    
    def handle_data_error(self, error: Exception, context: str, symbol: str = None) -> Dict:
        """
        Handle data-related errors with enhanced monitoring
        """
        self.error_count += 1
        self.data_error_count += 1
        
        error_info = {
            'timestamp': datetime.now(),
            'type': 'DATA_ERROR',
            'context': context,
            'symbol': symbol,
            'error': str(error),
            'handled': False
        }
        
        self.logger.error(f"Data Error in {context} for {symbol}: {error}")
        
        # Data errors are critical - we can't trade without good data
        if context == 'market_data_validation':
            error_info['handled'] = True
            error_info['suggestion'] = 'Skip symbol until data quality improves'
            error_info['should_skip_symbol'] = True
            
            # If multiple data validation errors, consider circuit breaker
            recent_data_errors = len([e for e in self._get_recent_errors(300) 
                                     if e.get('type') == 'DATA_ERROR' and e.get('context') == 'market_data_validation'])
            if recent_data_errors >= 3:
                self._activate_circuit_breaker("Multiple data validation errors - possible data source issue")
        
        self.error_history.append(error_info)
        return error_info

    def _get_recent_errors(self, time_window_seconds: int) -> List[Dict]:
        """Get errors that occurred in the recent time window"""
        cutoff_time = datetime.now().timestamp() - time_window_seconds
        return [error for error in self.error_history 
                if error['timestamp'].timestamp() > cutoff_time]

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable
        """
        error_str = str(error).lower()
        
        retryable_errors = [
            'timeout', 'connection', 'network', 'rate limit', 
            'too many requests', 'temporary', 'busy'
        ]
        
        non_retryable_errors = [
            'invalid api key', 'authentication', 'insufficient balance',
            'invalid symbol', 'market closed', 'position size too small'
        ]
        
        # Check for non-retryable errors first
        for non_retryable in non_retryable_errors:
            if non_retryable in error_str:
                return False
        
        # Check for retryable errors
        for retryable in retryable_errors:
            if retryable in error_str:
                return True
        
        # Default: assume retryable for transient issues
        return True
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay for retries
        """
        base_delay = 2  # seconds
        max_delay = 30  # seconds
        delay = min(base_delay * (2 ** retry_count), max_delay)
        return delay + (retry_count * 0.5)  # Add jitter
    
    def _activate_circuit_breaker(self, reason: str):
        """
        Activate circuit breaker to prevent further damage
        """
        self.circuit_breaker = True
        self.logger.critical(f"Circuit breaker activated: {reason}")
        
        if self.telegram_bot:
            self.telegram_bot.send_channel_message(
                f"🚨 <b>CIRCUIT BREAKER ACTIVATED</b>\n\n"
                f"Reason: {reason}\n"
                f"Total Errors: {self.error_count}\n"
                f"Trading Errors: {self.trading_error_count}\n"
                f"API Errors: {self.api_error_count}\n"
                f"Consecutive Trading Errors: {self.consecutive_trading_errors}\n"
                f"Trading will be suspended until manually reset."
            )
    
    def reset_circuit_breaker(self):
        """
        Reset circuit breaker after issues are resolved
        """
        self.circuit_breaker = False
        self.error_count = 0
        self.api_error_count = 0
        self.trading_error_count = 0
        self.consecutive_trading_errors = 0
        self.last_error_time = None
        self.logger.info("Circuit breaker reset")
        
        if self.telegram_bot:
            self.telegram_bot.send_channel_message("🔄 Circuit breaker reset - trading resumed")
    
    def _send_error_alert(self, title: str, message: str):
        """
        Send error alert via Telegram
        """
        if self.telegram_bot:
            try:
                self.telegram_bot.send_channel_message(
                    f"⚠️ <b>{title}</b>\n\n{message}\n\n"
                    f"<i>Total errors today: {self.error_count}</i>"
                )
            except Exception as e:
                self.logger.error(f"Failed to send Telegram alert: {e}")
    
    def get_error_summary(self) -> Dict:
        """
        Get summary of current error state
        """
        recent_errors = self.error_history[-10:] if self.error_history else []
        ignored_errors = [e for e in recent_errors if e.get('ignored', False)]
        real_errors = [e for e in recent_errors if not e.get('ignored', False)]
        
        return {
            'total_errors': self.error_count,
            'api_errors': self.api_error_count,
            'trading_errors': self.trading_error_count,
            'ml_errors': self.ml_error_count,
            'data_errors': self.data_error_count,
            'ignored_errors_count': len(ignored_errors),
            'circuit_breaker_active': self.circuit_breaker,
            'consecutive_trading_errors': self.consecutive_trading_errors,
            'recent_errors': real_errors,  # Only show real errors
            'recent_ignored_errors': ignored_errors,  # Show ignored separately
            'health_status': self.get_health_status()
        }
    
    def get_health_status(self) -> str:
        """
        Get overall health status based on error patterns
        """
        if self.circuit_breaker:
            return "CRITICAL"
        elif self.consecutive_trading_errors >= self.max_consecutive_trading_errors:
            return "CRITICAL"
        elif self.error_count >= 8:
            return "POOR"
        elif self.error_count >= 5:
            return "DEGRADED"
        elif self.error_count >= 2:
            return "FAIR"
        else:
            return "HEALTHY"
    
    def should_continue_trading(self) -> bool:
        """
        Determine if trading should continue based on error state
        """
        if self.circuit_breaker:
            return False
        
        # Count only non-ignored errors
        non_ignored_errors = [e for e in self.error_history 
                            if not e.get('ignored', False)]
        
        # Too many errors in short period (excluding ignored ones)
        if len(non_ignored_errors) >= self.max_errors_before_stop:
            self._activate_circuit_breaker("Maximum error threshold exceeded")
            return False
        
        # Check consecutive trading errors (excluding ignored ones)
        recent_non_ignored = [e for e in self._get_recent_errors(300) 
                            if not e.get('ignored', False)]
        if len(recent_non_ignored) >= 5:
            self._activate_circuit_breaker("High error rate detected")
            return False
        
        return True
    
    def record_successful_trade(self):
        """Reset consecutive trading errors on successful trade"""
        self.consecutive_trading_errors = 0
        self.logger.debug("Reset consecutive trading errors counter")

    def cleanup_old_errors(self, max_age_hours: int = 24):
        """
        Clean up old error records to prevent memory bloat
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        self.error_history = [
            error for error in self.error_history 
            if error['timestamp'].timestamp() > cutoff_time
        ]
    
    def log_successful_recovery(self, context: str):
        """
        Log successful error recovery for monitoring
        """
        self.logger.info(f"Successful error recovery in {context}")
        
        recovery_info = {
            'timestamp': datetime.now(),
            'type': 'RECOVERY_SUCCESS',
            'context': context
        }
        self.error_history.append(recovery_info)


# Example usage and testing
if __name__ == "__main__":
    # Test the error handler
    handler = ErrorHandler()
    
    # Simulate various errors
    test_errors = [
        {"type": "api", "error": Exception("API Rate Limit Exceeded"), "context": "get_balance"},
        {"type": "trading", "error": Exception("Insufficient Balance"), "context": "BTCUSDT", "action": "BUY"},
        {"type": "ml", "error": Exception("Model training failed"), "context": "ETHUSDT", "operation": "training"}
    ]
    
    for test in test_errors:
        if test["type"] == "api":
            result = handler.handle_api_error(test["error"], test["context"])
        elif test["type"] == "trading":
            result = handler.handle_trading_error(test["error"], test["context"], test["action"])
        elif test["type"] == "ml":
            result = handler.handle_ml_error(test["error"], test["context"], test.get("operation", "prediction"))
        
        print(f"Handled {test['type']} error: {result}")
    
    print(f"Error summary: {handler.get_error_summary()}")