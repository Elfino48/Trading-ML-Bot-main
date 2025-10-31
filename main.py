import time
from typing import Dict, List
import pandas as pd
import threading
from bybit_client import BybitClient
from data_engine import DataEngine
from enhanced_strategy_orchestrator import EnhancedStrategyOrchestrator
from advanced_risk_manager import AdvancedRiskManager
from execution_engine import ExecutionEngine
from telegram_bot import TelegramBot
from config import SYMBOLS, TIMEFRAME, TELEGRAM_CONFIG, DEBUG_MODE, RiskConfig
from error_handler import ErrorHandler
from trading_database import TradingDatabase
from emergency_protocols import EmergencyProtocols
from strategy_optimizer import StrategyOptimizer
from advanced_backtester import AdvancedBacktester
import logging
import logging.config
import logging.handlers
import os
import psutil
import gc
from datetime import datetime

class AdvancedTradingBot:

    def __init__(self, aggressiveness: str = "moderate"):
        self.start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🚀 Initializing Advanced Trading Bot (Run ID: {self.start_time_str})...")
        
        self._setup_comprehensive_logging()
        self._ws_unhealthy_start_time = None

        try:
            self.telegram_bot = None
            if TELEGRAM_CONFIG["ENABLED"]:
                self.telegram_bot = TelegramBot(
                    TELEGRAM_CONFIG["BOT_TOKEN"],
                    TELEGRAM_CONFIG["CHANNEL_ID"],
                    TELEGRAM_CONFIG.get("ALLOWED_USER_IDS", [])
                )
            
            self.client = BybitClient()
            self.data_engine = DataEngine(self.client)
            self.risk_manager = AdvancedRiskManager(self.client, aggressiveness)
            self.strategy_orchestrator = EnhancedStrategyOrchestrator(
                self.client,
                self.data_engine,
                aggressiveness,
                run_start_time_str=self.start_time_str
            )
            self.execution_engine = ExecutionEngine(self.client, self.risk_manager)
            
            self.error_handler = ErrorHandler(self.telegram_bot)
            self.database = TradingDatabase()
            self.emergency_protocols = EmergencyProtocols(self.execution_engine, self.telegram_bot)
            self.strategy_optimizer = StrategyOptimizer(self.database)
            self.backtester = AdvancedBacktester(self.strategy_orchestrator)
            
            self.client.set_error_handler(self.error_handler)
            self.data_engine.set_error_handler(self.error_handler)
            self.risk_manager.set_error_handler(self.error_handler)
            self.strategy_orchestrator.set_error_handler(self.error_handler)
            
            # --- Set Database and Central WS Callback ---
            self.risk_manager.set_database(self.database)
            self.strategy_orchestrator.set_database(self.database)
            self.execution_engine.set_emergency_protocols(self.emergency_protocols)
            
            # --- set_main_bot removed ---
            
            # Set the new central router as the main WS callback
            self.client.set_ws_callback(self._on_ws_message)
            # --- End New Block ---
            self.strategy_orchestrator.set_database(self.database)
            self.execution_engine.set_emergency_protocols(self.emergency_protocols)
            
            self.aggressiveness = aggressiveness
            
            if self.telegram_bot:
                self.telegram_bot.set_trading_bot(self)
            
            self._validate_configuration()
            
            self._initialize_ml_models()
            
            portfolio_value = self._test_connection()
            
            self._initialize_leverage()
            
            self._initialize_database()
            
            # --- Startup reconciliation logic removed ---
            # (It will now run in the first trading cycle)
            
            if self.telegram_bot:
                self.telegram_bot.send_channel_message(
                    f"🤖 <b>TRADING BOT STARTED</b>\n\n"
                    f"💰 <b>Portfolio:</b> ${portfolio_value:,.2f}\n"
                    f"📊 <b>Symbols:</b> {', '.join(SYMBOLS)}\n"
                    f"🎯 <b>Aggressiveness:</b> {self.aggressiveness.upper()}\n"
                    f"🛡️ <b>Error Handler:</b> Active\n"
                    f"💾 <b>Database:</b> Active\n"
                    f"🆘 <b>Emergency Protocols:</b> Active\n"
                    f"🕒 <b>Time:</b> {pd.Timestamp.now().strftime('%Y%m%d %H:%M:%S')}\n\n"
                    f"<i>Bot is now running. Use /help for commands.</i>"
                )

            print("✅ Bot initialization completed successfully!")
            print(f"💰 Starting portfolio: ${portfolio_value:.2f}")
            print(f"🎯 Aggressiveness: {self.aggressiveness.upper()}")
            print("🛡️ Error Handler: Active")
            print("💾 Database: Active")
            print("🆘 Emergency Protocols: Active")
            
        except Exception as e:
            error_msg = f"Bot initialization failed: {e}"
            print(f"❌ {error_msg}")
            if self.telegram_bot:
                self.telegram_bot.send_channel_message(f"🚨 <b>BOT STARTUP FAILED</b>\n\n{error_msg}")
            raise

    def _setup_comprehensive_logging(self):
        try:
            import os
            os.makedirs('logs', exist_ok=True)
            
            logging.config.dictConfig({
                'version': 1,
                'formatters': {
                    'detailed': {
                        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
                    },
                    'json': {
                         'format': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "file": "%(filename)s", "line": %(lineno)d}'
                    }
                },
                'handlers': {
                    'file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'filename': 'logs/trading_bot.log',
                        'maxBytes': 10485760,
                        'backupCount': 5,
                        'formatter': 'detailed',
                        'encoding': 'utf-8'
                    },
                    'json_file': {
                         'class': 'logging.handlers.RotatingFileHandler',
                         'filename': 'logs/trading_bot_json.log',
                         'maxBytes': 10485760,
                         'backupCount': 5,
                         'formatter': 'json',
                         'encoding': 'utf-8'
                    },
                    'console': {
                        'class': 'logging.StreamHandler',
                        'formatter': 'detailed',
                        'stream': 'ext://sys.stdout'
                    }
                },
                'loggers': {
                    '': {
                        'handlers': ['file', 'json_file', 'console'],
                        'level': 'INFO'
                    },
                    'AdvancedTradingBot': {
                        'level': 'DEBUG',
                        'handlers': ['file', 'json_file', 'console'],
                        'propagate': False
                    },
                     'MLPredictor': {
                         'level': 'DEBUG',
                         'handlers': ['file', 'json_file', 'console'],
                         'propagate': False
                     },
                     'ExecutionEngine': {
                         'level': 'DEBUG',
                         'handlers': ['file', 'json_file', 'console'],
                         'propagate': False
                     },
                }
            })
            
            self.logger = logging.getLogger('AdvancedTradingBot')
            self.logger.info("Comprehensive logging system initialized")
            
        except Exception as e:
            print(f"❌ Failed to setup comprehensive logging: {e}")

    def _log_trading_cycle_metrics(self, cycle_decisions, cycle_duration: float):
        try:
            confidence_values = [d.get('confidence', 0) for d in cycle_decisions]
            composite_scores = [d.get('composite_score', 0) for d in cycle_decisions]

            metrics = {
                'cycle_duration': cycle_duration,
                'total_decisions': len(cycle_decisions),
                'buy_decisions': len([d for d in cycle_decisions if d['action'] == 'BUY']),
                'sell_decisions': len([d for d in cycle_decisions if d['action'] == 'SELL']),
                'hold_decisions': len([d for d in cycle_decisions if d['action'] == 'HOLD']),
                'avg_confidence': sum(confidence_values) / len(confidence_values) if confidence_values else 0,
                'avg_composite_score': sum(composite_scores) / len(composite_scores) if composite_scores else 0,
                'high_confidence_trades': len([d for d in cycle_decisions if d.get('confidence', 0) > 70]),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Trading cycle metrics", extra={'metrics': metrics})
            
            if self.database:
                self.database.store_system_event(
                    "TRADING_CYCLE_METRICS",
                    metrics,
                    "INFO",
                    "Performance Monitoring"
                )
                
        except Exception as e:
            self.logger.error(f"Error logging trading cycle metrics: {e}")

    def _monitor_system_health(self):
        try:
            import psutil
            import gc
            
            health_metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_threads': threading.active_count(),
                'python_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'gc_objects': len(gc.get_objects())
            }
            
            if health_metrics['memory_percent'] > 85:
                self.logger.warning("High memory usage detected", extra=health_metrics)
                
            if health_metrics['cpu_percent'] > 90:
                self.logger.warning("High CPU usage detected", extra=health_metrics)
            
            self.logger.debug("System health check", extra=health_metrics)
            
            if self.database:
                self.database.store_system_event(
                    "SYSTEM_HEALTH_CHECK",
                    health_metrics,
                    "INFO",
                    "System Monitoring"
                )
                
        except Exception as e:
            self.logger.error(f"Error monitoring system health: {e}")

    def _validate_configuration(self):
        print("🔍 Validating configuration...")
        
        if not self.client.api_key or not self.client.api_secret:
            raise ValueError("API keys not configured properly")
        
        if not SYMBOLS:
            raise ValueError("No trading symbols configured")
        
        if not TIMEFRAME or not TIMEFRAME.isdigit():
            raise ValueError("Invalid timeframe configured")
        
        print("✅ Configuration validation passed")

    def _initialize_ml_models(self):
        """Loads existing models and trains any missing models at startup."""
        print("\n🤖 Initializing ML models...")
        try:
            # 1. Load all existing models from disk
            self.strategy_orchestrator.ml_predictor.load_models()
            models_loaded_count = len(self.strategy_orchestrator.ml_predictor.models)
            if models_loaded_count > 0:
                print(f"✅ Loaded {models_loaded_count} pre-trained models from disk")
            else:
                print("ℹ️ No pre-trained models found on disk.")
            
            # 2. Check for missing models
            loaded_symbols = set(self.strategy_orchestrator.ml_predictor.models.keys())
            required_symbols = set(SYMBOLS)
            missing_symbols = list(required_symbols - loaded_symbols)
            
            if not missing_symbols and models_loaded_count > 0:
                print(f"✅ All {len(required_symbols)} required models are loaded.")
                if self.telegram_bot:
                    self.telegram_bot.log_important_event(
                        "MODELS LOADED",
                        f"Successfully loaded all {len(required_symbols)} pre-trained ML models"
                    )
            elif not missing_symbols and models_loaded_count == 0:
                # No models loaded and none are missing? (SYMBOLS list is empty)
                print("⚠️ No symbols configured, no models to load or train.")
            else:
                # 3. Train missing models
                print(f"📚 Found {len(missing_symbols)} missing models. Training new models for: {missing_symbols}")
                if self.telegram_bot:
                    self.telegram_bot.log_important_event(
                        "INITIAL MODELS TRAINING",
                        f"Starting training for {len(missing_symbols)} new symbols: {', '.join(missing_symbols)}"
                    )
                
                trained_count = self._retrain_all_models(symbols_to_train=missing_symbols)
                
                if trained_count > 0:
                    self.strategy_orchestrator.ml_predictor.save_models()
                    print(f"💾 Saved {trained_count} new models to disk")
                    if self.telegram_bot:
                        self.telegram_bot.log_important_event(
                            "NEW MODELS TRAINED",
                            f"Successfully trained and saved {trained_count} new ML models"
                        )
                else:
                    error_msg = f"Initial model training failed for {missing_symbols}"
                    self.error_handler.handle_ml_error(Exception(error_msg), "ALL", "initial_training")

        except Exception as e:
            self.error_handler.handle_ml_error(e, "ALL", "initialization")
            print("⚠️ Continuing without ML models or using potentially incomplete set")

    def _retrain_all_models(self, force_all: bool = False, symbols_to_train: List[str] = None) -> int:
        """Enhanced model retraining with performance-based selection"""
        trained_count = 0
        self.logger.info("[MODEL RETRAINING] Starting enhanced retraining process...")
        
        # Get symbols that actually need retraining
        if symbols_to_train:
            symbols_to_retrain = symbols_to_train
            self.logger.info(f"[MODEL RETRAINING] Training specific list of {len(symbols_to_retrain)} symbols: {symbols_to_retrain}")
        elif force_all:
            symbols_to_retrain = SYMBOLS
            self.logger.info(f"[MODEL RETRAINING] Forcing retraining for all {len(symbols_to_retrain)} symbols")
        else:
            symbols_to_retrain = self.strategy_orchestrator.ml_predictor.get_models_needing_retraining()
            self.logger.info(f"[MODEL RETRAINING] {len(symbols_to_retrain)} symbols need retraining: {symbols_to_retrain}")
        
        if not symbols_to_retrain:
            self.logger.info("[MODEL RETRAINING] No models need retraining at this time")
            return 0

        for symbol in symbols_to_retrain:
            try:
                self.logger.info(f"[MODEL RETRAINING] Retraining model for {symbol}...")
                
                historical_data = self.data_engine.get_historical_data(symbol, TIMEFRAME, limit=1500)

                if historical_data is not None and len(historical_data) >= 250:
                    if self.data_engine.validate_market_data(historical_data.reset_index()):
                        old_performance = self.strategy_orchestrator.ml_predictor.model_versions.get(symbol, {}).get('accuracy', 0)
                        
                        success = self.strategy_orchestrator.ml_predictor.train_model(symbol, historical_data)
                        
                        if success:
                            trained_count += 1
                            new_performance = self.strategy_orchestrator.ml_predictor.model_versions.get(symbol, {}).get('accuracy', 0)
                            
                            perf_change = new_performance - old_performance
                            self.logger.info(f"[MODEL RETRAINING] {symbol} retrained successfully. Accuracy: {old_performance:.3f} -> {new_performance:.3f} ({perf_change:+.3f})")
                            
                            if self.telegram_bot and abs(perf_change) > 0.05:
                                direction = "📈 Improved" if perf_change > 0 else "📉 Declined"
                                self.telegram_bot.log_ml_training(
                                    symbol, 
                                    new_performance * 100, 
                                    f"{direction} by {abs(perf_change)*100:.1f}%"
                                )
                        else:
                            self.logger.warning(f"[MODEL RETRAINING] Model training failed for {symbol}")
                            self.error_handler.handle_ml_error(
                                Exception("Model training failed"), symbol, "retraining"
                            )
                    else:
                        self.logger.warning(f"[MODEL RETRAINING] Data validation failed for {symbol}")
                else:
                    self.logger.warning(f"[MODEL RETRAINING] Insufficient data for {symbol}: {len(historical_data) if historical_data is not None else 0} rows")

                time.sleep(0.3)
                
            except Exception as e:
                error_msg = f"Error during retraining for {symbol}: {e}"
                self.logger.error(error_msg)
                self.error_handler.handle_ml_error(e, symbol, "retraining")

        self.logger.info(f"[MODEL RETRAINING] Retraining complete. {trained_count}/{len(symbols_to_retrain)} models updated")
        
        if trained_count > 0:
            save_success = self.strategy_orchestrator.ml_predictor.save_models()
            if save_success:
                self.logger.info(f"[MODEL RETRAINING] Saved {trained_count} updated models to disk")
            else:
                self.logger.error("[MODEL RETRAINING] Failed to save models to disk")
        
        return trained_count

    def _test_connection(self):
        print("\n🔗 Testing Bybit API connection...")
        try:
            balance = self.client.get_wallet_balance()
            if balance and balance.get('retCode') == 0:
                print("✅ API connection successful!")
                equity = float(balance['result']['list'][0]['totalEquity'])
                print(f"💰 Demo account equity: ${equity:.2f}")
                return equity
            else:
                error_msg = balance.get('retMsg', 'Unknown error') if balance else 'No response'
                print(f"❌ API connection failed: {error_msg}")
                self.error_handler.handle_api_error(Exception(error_msg), "connection_test")
                return 10000
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            self.error_handler.handle_api_error(e, "connection_test")
            return 10000

    def _initialize_leverage(self):
        print("\n⚙️ Checking leverage settings...")
        
        if "demo" in self.client.base_url.lower():
            print("⚠️  Demo environment detected - skipping leverage setting")
            print("💡 Note: Demo accounts may have fixed leverage restrictions")
            return
        
        leverage_map = {
            "conservative": 10,
            "moderate": 15,
            "aggressive": 20,
            "high": 20
        }
        leverage = leverage_map.get(self.aggressiveness, 10)
        
        for symbol in SYMBOLS:
            try:
                print(f"🔄 Setting leverage for {symbol}...")
                
                try:
                    position_response = self.client.get_position_info(symbol)
                    if position_response and position_response.get('retCode') == 0:
                        positions = position_response['result']['list']
                        if positions:
                            current_leverage = float(positions[0].get('leverage', 0))
                            if current_leverage == leverage:
                                print(f"✅ Leverage already set to {leverage}x for {symbol}")
                                continue
                except Exception as e:
                    print(f"⚠️ Could not check current leverage for {symbol}: {e}")
                
                result = self.client.set_leverage(symbol, leverage)
                
                if result and result.get('retCode') == 0:
                    print(f"✅ Leverage set to {leverage}x for {symbol}")
                else:
                    error_msg = result.get('retMsg', 'Unknown error') if result else 'No response'
                    
                    if "leverage not modified" in error_msg.lower():
                        print(f"ℹ️  Leverage modification restricted for {symbol} (common in demo)")
                    elif "not in range" in error_msg.lower():
                        print(f"⚠️  Leverage {leverage}x not available for {symbol}")
                    else:
                        print(f"⚠️  Could not set leverage for {symbol}: {error_msg}")
                        
            except Exception as e:
                print(f"⚠️  Error setting leverage for {symbol}: {e}")
                continue
            
            time.sleep(0.2)

    def _initialize_database(self):
        try:
            portfolio_value = self.get_portfolio_value()
            self.database.store_system_event(
                "BOT_STARTED",
                {
                    "portfolio_value": portfolio_value,
                    "aggressiveness": self.aggressiveness,
                    "symbols": SYMBOLS,
                    "timeframe": TIMEFRAME
                },
                "INFO",
                "System Startup"
            )
            print("💾 Database initialized with startup data")
        except Exception as e:
            print(f"⚠️ Failed to initialize database: {e}")

    def _on_ws_message(self, message: Dict):
        """
        Central WebSocket callback router.
        Forwards messages to the correct module based on topic.
        """
        topic = message.get("topic", "")

        try:
            # Route public market data (Kline, Tickers)
            if topic.startswith("kline.") or topic.startswith("tickers."):
                if self.data_engine:
                    self.data_engine._handle_ws_message(message)
            
            # Route private data (Orders, Positions)
            elif topic in ["order", "position"]:
                if self.execution_engine:
                    self.execution_engine.handle_private_ws_message(message)
            
            # Optional: Log other message types if needed
            # else:
            #     self.logger.debug(f"Received unhandled WS message topic: {topic}")

        except Exception as e:
            self.logger.error(f"Error processing WS message in central router (topic: {topic}): {e}", exc_info=True)
            if self.error_handler:
                self.error_handler.handle_api_error(e, f"ws_central_router_{topic}")

    def get_portfolio_value(self) -> float:
        try:
            balance = self.client.get_wallet_balance()
            if balance and balance.get('retCode') == 0:
                return float(balance['result']['list'][0]['totalEquity'])
            self.logger.warning("Failed to get portfolio value from API, returning cached or default.")
            return getattr(self, '_last_portfolio_value', 10000)
        except Exception as e:
            self.error_handler.handle_api_error(e, "get_balance")
            self.logger.error(f"Exception getting portfolio value: {e}, returning cached or default.")
            return getattr(self, '_last_portfolio_value', 10000)

    def run_trading_cycle(self):
        cycle_start = time.time()
        self.logger.info("Starting Enhanced Trading Cycle...")
        
        trading_decisions = []
        all_symbol_data_copies = {}
        account_info = {}
        
        try:
            print("\n" + "="*60)
            print("🔄 Starting Enhanced Trading Cycle...")
            print("="*60)
            
            if not self.error_handler.should_continue_trading():
                print("🚫 Trading suspended due to error conditions")
                if self.telegram_bot:
                    self.telegram_bot.send_channel_message(
                        "🚫 <b>TRADING SUSPENDED</b>\n\n"
                        "Too many errors detected. Check logs and reset error handler to resume."
                    )
                return []
            
            portfolio_value = self.get_portfolio_value()
            self._last_portfolio_value = portfolio_value
            if self.risk_manager.daily_start_balance == 0:
                self.risk_manager.daily_start_balance = portfolio_value
                print(f"💰 Initialized daily starting balance: ${portfolio_value:.2f}")

            self.risk_manager.update_daily_pnl()
            emergency_check = self.emergency_protocols.check_emergency_conditions(
                portfolio_value, 
                self.risk_manager.daily_pnl,
                self.execution_engine.get_trade_history(limit=10),
                self.error_handler.error_count
            )
            
            if emergency_check['emergency']:
                print("🆘 Emergency conditions detected - skipping cycle")
                return []

            ws_check_interval_cycles = 2
            force_restart_threshold_seconds = 180

            if hasattr(self, 'cycle_count') and self.cycle_count % ws_check_interval_cycles == 0:
             try:
                is_public_connected = self.client.ws_public_connected
                critical_stream_down = not is_public_connected

                if critical_stream_down:
                    current_time = time.time()
                    if self._ws_unhealthy_start_time is None:
                        self._ws_unhealthy_start_time = current_time
                        self.logger.warning("Critical WebSocket (Public) detected as DISCONNECTED. Starting monitoring timer.")
                    else:
                        disconnected_duration = current_time - self._ws_unhealthy_start_time
                        self.logger.debug(f"Critical WebSocket has been disconnected for {disconnected_duration:.1f} seconds.")

                        if disconnected_duration > force_restart_threshold_seconds:
                            self.logger.error(f"Critical WebSocket has been disconnected for over {force_restart_threshold_seconds}s. Forcing restart.")
                            if self.telegram_bot:
                                self.telegram_bot.log_important_event("WebSocket Restart", f"Forcing restart of critical WebSocket(s) due to prolonged disconnected state.")

                            self.client.restart_websocket('public')
                            
                            self._ws_unhealthy_start_time = None
                            self.logger.info("Pausing briefly after initiating WS restart...")
                            time.sleep(5)

                else:
                    if self._ws_unhealthy_start_time is not None:
                         self.logger.info("Critical WebSocket(s) appear CONNECTED again. Resetting unhealthy timer.")
                         self._ws_unhealthy_start_time = None

             except Exception as ws_health_e:
                  self.logger.error(f"Error during WebSocket health check logic: {ws_health_e}", exc_info=True)

            print("📊 Accessing latest market data...")
            data_access_start = time.time()
            valid_data_count = 0
            for symbol in SYMBOLS:
                data_copy = self.data_engine.get_market_data_for_analysis(symbol)
                if data_copy is not None and not data_copy.empty:
                    close_prices = data_copy['close'].astype(float)
                    valid_closes = close_prices[close_prices > 0].dropna()
                    
                    if len(valid_closes) >= 50:
                        all_symbol_data_copies[symbol] = data_copy
                        valid_data_count += 1
                    else:
                        self.logger.warning(f"Invalid data for {symbol}: only {len(valid_closes)} valid prices")
                else:
                    self.logger.warning(f"No data available for {symbol}")
            data_access_duration = time.time() - data_access_start
            print(f"   Accessed data for {valid_data_count}/{len(SYMBOLS)} symbols in {data_access_duration:.2f}s")
            
            if valid_data_count == 0:
                print("🚫 No valid market data available for any symbol. Skipping cycle.")
                return []

            print("💰 Fetching account balance and positions...")
            account_fetch_start = time.time()
            try:
                account_info['portfolio_value'] = portfolio_value 
                positions_response = self.client.get_position_info(category="linear", settleCoin="USDT")
                if positions_response and positions_response.get('retCode') == 0:
                    account_info['positions'] = {
                        p['symbol']: p for p in positions_response['result']['list'] if float(p.get('size', 0)) > 0
                    }
                else:
                    account_info['positions'] = {}
                    print("⚠️ Failed to fetch position info")
            except Exception as e:
                print(f"❌ Error fetching account data: {e}")
                self.error_handler.handle_api_error(e, "cycle_account_fetch")
            account_fetch_duration = time.time() - account_fetch_start
            print(f"   Account data fetch complete in {account_fetch_duration:.2f}s")

            # --- START FIX: Reconcile open positions before analysis ---
            print("🔍 Reconciling open positions with exchange...")
            recon_start = time.time()
            try:
                self._reconcile_open_positions(account_info['positions'])
                recon_duration = time.time() - recon_start
                print(f"   Reconciliation complete in {recon_duration:.2f}s")
            except Exception as recon_e:
                self.logger.error(f"Reconciliation failed: {recon_e}", exc_info=True)
                self.error_handler.handle_api_error(recon_e, "reconciliation_cycle")
            # --- END FIX ---

            if self.should_run_advanced_analysis():
                print("🔬 Running Phase 4: Advanced Portfolio Analysis...")
                try:
                    correlation_analysis = self.strategy_orchestrator.analyze_portfolio_correlation_advanced(
                        SYMBOLS, all_symbol_data_copies
                    )
                    print(f"   📊 Correlation analysis completed: {len(correlation_analysis.get('correlation_clusters', {}))} clusters found")
                    
                    regime_transitions = {}
                    for symbol in SYMBOLS:
                        if symbol in all_symbol_data_copies:
                            regime_transition = self.strategy_orchestrator.detect_market_regime_transitions(
                                symbol, all_symbol_data_copies[symbol]
                            )
                            regime_transitions[symbol] = regime_transition
                    
                    print(f"   🔄 Regime transition analysis completed for {len(regime_transitions)} symbols")
                    
                    risk_parity = self.strategy_orchestrator.calculate_advanced_risk_parity(
                        SYMBOLS, all_symbol_data_copies, portfolio_value
                    )
                    print(f"   ⚖️ Advanced risk parity allocation calculated")
                    
                    performance_report = self.strategy_orchestrator.generate_performance_report(days=7)
                    print(f"   📈 Performance attribution report generated")
                    
                    if self.database:
                        self.database.store_system_event(
                            "ADVANCED_ANALYSIS_COMPLETED",
                            {
                                'correlation_clusters': len(correlation_analysis.get('correlation_clusters', {})),
                                'regime_transitions_analyzed': len(regime_transitions),
                                'risk_parity_allocated': len(risk_parity.get('final_weights', {})),
                                'performance_insights': len(performance_report.get('attribution_insights', []))
                            },
                            "INFO",
                            "Advanced Analysis"
                        )
                        
                except Exception as e:
                    print(f"⚠️ Advanced analysis failed: {e}")
                    self.error_handler.handle_trading_error(e, "PORTFOLIO", "advanced_analysis")

            print("🧠 Analyzing symbols...")
            analysis_start = time.time()
            raw_decisions = [] 
            for symbol in SYMBOLS:
                if symbol not in all_symbol_data_copies:
                    continue 

                try:
                    print(f"\n   🔍 Analyzing {symbol}...")
                    historical_data_copy = all_symbol_data_copies[symbol] 
                    
                    decision = self.strategy_orchestrator.analyze_symbol(
                        symbol, historical_data_copy, account_info['portfolio_value'] 
                    )
                    raw_decisions.append(decision)
                    
                    action_emoji = "🎯" if decision['action'] != 'HOLD' else "⏸️"
                    print(f"      {action_emoji} Decision: {decision['action']} (Conf: {decision['confidence']:.1f}%)")

                except Exception as e:
                    error_msg = f"Error analyzing {symbol}: {e}"
                    print(f"      ❌ {error_msg}")
                    self.error_handler.handle_trading_error(e, symbol, "analysis")
                    if self.telegram_bot:
                        self.telegram_bot.log_error(error_msg, f"Analysis - {symbol}")
            
            analysis_duration = time.time() - analysis_start
            print(f"   Analysis complete in {analysis_duration:.2f}s")
            
            min_confidence = self.risk_manager.min_confidence

            # --- Step 3.5: Pre-Execution Risk Check (NEW) ---
            print("🛡️ Performing pre-execution risk check...")
            try:
                # Get current equity and exposure
                total_equity = account_info.get('portfolio_value', self.get_portfolio_value())
                current_exposure = self.risk_manager.get_current_exposure()
                
                # Get scaled portfolio limits
                scaled_max_exposure_percent = 0.3 * self.risk_manager.risk_multiplier
                max_total_exposure = total_equity * scaled_max_exposure_percent
                available_exposure = max(0, max_total_exposure - current_exposure)
                
                # Get intended trades
                intended_trades = [d for d in raw_decisions if d['action'] != 'HOLD' and d['confidence'] >= min_confidence]
                total_intended_size = sum(d['position_size'] for d in intended_trades)
                
                print(f"   Available Exposure: ${available_exposure:.2f} (Max: ${max_total_exposure:.2f})")
                print(f"   Total Intended Size: ${total_intended_size:.2f} across {len(intended_trades)} trades")

                # If total intended size exceeds available exposure, scale all trades down
                if total_intended_size > available_exposure and total_intended_size > 0:
                    scaling_factor = available_exposure / total_intended_size
                    print(f"   ⚠️ Intended size exceeds available exposure. Scaling all trades by {scaling_factor:.2f}")
                    
                    for decision in intended_trades:
                        original_size = decision['position_size']
                        decision['position_size'] = original_size * scaling_factor
                        decision['quantity'] = decision['quantity'] * scaling_factor
                        print(f"      {decision['symbol']} size reduced: ${original_size:.2f} -> ${decision['position_size']:.2f}")
                else:
                    print("   ✅ Total intended size is within portfolio exposure limits.")

            except Exception as pre_risk_e:
                self.logger.error(f"Error during pre-execution risk check: {pre_risk_e}", exc_info=True)
                print(f"   ⚠️ Error during pre-execution risk check. Proceeding without scaling...")

            # --- Step 4: Execute Trades ---
            print("🚀 Executing trades...")
            execution_start = time.time()
            actions_taken = 0
            strong_trades = 0
            moderate_trades = 0

            for decision in raw_decisions: 
                symbol = decision['symbol']
                action = decision['action']
                confidence = decision['confidence']
                signal_strength = decision.get('signals', {}).get('signal_strength', 'NEUTRAL')
                
                trading_decisions.append(decision.copy()) 

                print(f"\n   🚦 Evaluating {symbol}: {action} (Conf: {confidence:.1f}%)")

                # --- START FIX: POSITION MANAGEMENT LOGIC ---
                current_position = account_info['positions'].get(symbol)
                
                if current_position:
                    current_side = current_position.get('side') # 'Buy' or 'Sell'
                    
                    if (action == 'BUY' and current_side == 'Buy') or \
                       (action == 'SELL' and current_side == 'Sell'):
                        print(f"      ℹ️ Ignoring {action} signal: Position already open in the same direction.")
                        continue # Skip to the next decision

                    elif (action == 'BUY' and current_side == 'Sell') or \
                         (action == 'SELL' and current_side == 'Buy'):
                        # This is an exit signal for the current position
                        print(f"      ✨ New {action} signal conflicts with open {current_side} position. Closing position...")
                        try:
                            # Use the simple close_position method
                            close_result = self.execution_engine.close_position(symbol)
                            if close_result.get('success'):
                                print(f"      ✅ Successfully placed order to close {symbol}.")
                                if self.telegram_bot:
                                    self.telegram_bot.log_position_closure(symbol, f"Opposing signal ({action}) received.")
                                # After closing, we STOP. We don't open the new trade in the same cycle.
                                continue # Skip to the next decision
                            else:
                                print(f"      ❌ Failed to place close order for {symbol}: {close_result.get('message')}")
                                self.error_handler.handle_trading_error(Exception(close_result.get('message')), symbol, "exit_logic_close_failed")
                                continue # Skip to the next decision
                        except Exception as close_e:
                            print(f"      ❌ Exception while trying to close {symbol}: {close_e}")
                            self.error_handler.handle_trading_error(close_e, symbol, "exit_logic_exception")
                            continue # Skip to the next decision
                
                # --- END FIX ---

                if action != 'HOLD':
                    print(f"      📈 Signal Strength: {signal_strength}")
                    print(f"      🎯 Composite Score: {decision['composite_score']:.1f}")
                    print(f"      💰 Proposed Size: ${decision['position_size']:.2f}")
                    print(f"      🛡️ SL: ${decision['stop_loss']:.2f}, TP: ${decision['take_profit']:.2f}, R/R: {decision['risk_reward_ratio']:.2f}:1")
                    print(f"      🌡️ Regime: {decision['market_regime']}, Vol: {decision['volatility_regime']}")
                    quality_rating = decision.get('trade_quality', {}).get('quality_rating', 'UNKNOWN')
                    print(f"      🏆 Trade Quality: {quality_rating}")

                if action != 'HOLD' and confidence >= min_confidence:
                    try:
                        risk_check = self.risk_manager.can_trade(symbol, decision['position_size'], market_data=None) 

                        if risk_check['approved']:
                            adjusted_size = risk_check.get('adjusted_size', decision['position_size'])
                            if adjusted_size > 0 and adjusted_size != decision['position_size']:
                                print(f"      ℹ️ Adjusting size for {symbol}: ${adjusted_size:.2f} (Reason: {risk_check['reason']})")
                                decision['position_size'] = adjusted_size
                                latest_price = self.data_engine.get_current_price(symbol)
                                if latest_price <= 0:
                                    print(f"      ❌ Cannot calculate quantity for adjusted size, invalid latest price ({latest_price}). Skipping.")
                                    continue
                                decision['quantity'] = adjusted_size / latest_price
                            elif adjusted_size <= 0:
                                print(f"      ❌ Risk manager approved but adjusted size is <= 0 ({adjusted_size:.2f}). Skipping.")
                                continue
                            
                            decision['current_price'] = self.data_engine.get_current_price(symbol)
                            if decision['current_price'] <= 0:
                                print(f"      ❌ Cannot execute trade, invalid latest price ({decision['current_price']}). Skipping.")
                                continue
                                
                            if decision['quantity'] <= 0:
                                print(f"      ❌ Calculated quantity is zero or negative ({decision['quantity']:.4f}). Skipping.")
                                continue

                            execution_result = self.execution_engine.execute_enhanced_trade(decision) 
                            
                            if execution_result['success']:
                                print(f"      ✅ Trade executed successfully!")
                                actions_taken += 1

                                self.risk_manager.record_trade_outcome(True, pnl=0) 
                                
                                if signal_strength in ['STRONG_BUY', 'STRONG_SELL']:
                                    strong_trades += 1
                                else:
                                    moderate_trades += 1
                                
                                if self.telegram_bot:
                                    self.telegram_bot.log_trade_execution(decision)
                                
                                if 'order_id' in execution_result:
                                    print(f"         📝 Order ID: {execution_result['order_id']}")
                            else:
                                error_msg = execution_result['message']
                                print(f"      ❌ Trade execution failed: {error_msg}")

                                # --- NEW: Check for ignorable "not enough balance" error ---
                                if "110007" in error_msg:
                                    self.logger.info(f"Ignoring ignorable error in main loop: {error_msg}")
                                    # We DO NOT store this in the database (engine already skipped)
                                    # We DO NOT record this as a loss
                                    # We DO NOT report this to the error handler
                                else:
                                    # This is a REAL failure
                                    # The ExecutionEngine ALREADY stored this trade and recorded the loss.
                                    # We just need to report the error to the ErrorHandler.
                                    if self.telegram_bot:
                                        self.telegram_bot.log_trade_error(symbol, action, error_msg)
                                    self.error_handler.handle_trading_error(
                                        Exception(error_msg), symbol, f"execution_{action}"
                                    )
                        else:
                            print(f"      ❌ Risk management rejected trade: {risk_check['reason']}")

                    except Exception as exec_e:
                        error_msg = f"Unhandled error during execution for {symbol}: {exec_e}"
                        print(f"      ❌ {error_msg}")
                        self.error_handler.handle_trading_error(exec_e, symbol, f"execution_wrapper_{action}")
                        if self.telegram_bot:
                            self.telegram_bot.log_error(error_msg, f"Execution - {symbol}")

                elif action != 'HOLD':
                    print(f"      ⚠️ Skipping trade - confidence {confidence:.1f}% below minimum {min_confidence}%")
            
            execution_duration = time.time() - execution_start
            print(f"   Execution attempts complete in {execution_duration:.2f}s")

            print(f"\n📈 Cycle Summary:")
            print(f"   • {actions_taken} trades executed out of {len(raw_decisions)} decisions") 
            print(f"   • {strong_trades} strong signals, {moderate_trades} moderate signals executed")
            print(f"   • Aggressiveness: {self.aggressiveness.upper()}")
            
            new_portfolio_value = self.get_portfolio_value() 
            self._last_portfolio_value = new_portfolio_value 
            
            self.risk_manager.update_daily_pnl() 
            pnl_percent = self.risk_manager.daily_pnl
            pnl = new_portfolio_value - self.risk_manager.daily_start_balance
            
            print(f"   💰 Portfolio: ${new_portfolio_value:.2f} ({pnl_percent:+.2f}%)")
            
            performance = self.execution_engine.get_performance_metrics()
            if performance:
                print(f"   📊 Win Rate (Session): {performance.get('win_rate', 0):.1f}%")
                print(f"   🎯 Avg Confidence (Session): {performance.get('avg_confidence', 0):.1f}%")
                
                self.database.store_performance_metrics({
                    'portfolio_value': new_portfolio_value,
                    'daily_pnl_percent': pnl_percent,
                    'total_trades': performance.get('total_trades', 0),
                    'winning_trades': int(performance.get('total_trades', 0) * performance.get('win_rate', 0) / 100),
                    'win_rate': performance.get('win_rate', 0),
                    'avg_confidence': performance.get('avg_confidence', 0),
                    'avg_risk_reward': performance.get('avg_risk_reward', 0),
                    'max_drawdown': 0
                })
            else:
                self.database.store_performance_metrics({
                    'portfolio_value': new_portfolio_value, 'daily_pnl_percent': pnl_percent,
                    'total_trades': 0, 'winning_trades': 0, 'win_rate': 0,
                    'avg_confidence': 0, 'avg_risk_reward': 0, 'max_drawdown': 0
                })
            
            if self.telegram_bot and (actions_taken > 0 or abs(pnl_percent) > 0.5):
                summary = {
                    'trades_executed': actions_taken, 'strong_signals': strong_trades,
                    'moderate_signals': moderate_trades, 'portfolio_value': new_portfolio_value,
                    'pnl_percent': pnl_percent, 'aggressiveness': self.aggressiveness
                }
                self.telegram_bot.log_cycle_summary(summary)

            self.database.store_system_event("TRADING_CYCLE_COMPLETE",
                { 'trades_executed': actions_taken, 'portfolio_value': new_portfolio_value,
                'pnl_percent': pnl_percent, 'cycle_pnl': pnl }, "INFO", "Trading Cycle")
            
            if hasattr(self, 'cycle_count'):
                if self.cycle_count % 5 == 0:
                    print("🔄 Optimizing strategy weights...")
                    self._optimize_strategy_weights() 
                
                if self.cycle_count % 30 == 0:
                    print("🔍 Checking for critically underperforming models...")
                    critical_retrain_thread = threading.Thread(target=self._check_and_retrain_low_performance_models, daemon=True)
                    critical_retrain_thread.start()
                
                if self.cycle_count % 60 == 0:
                    print("🔄 Retraining ML models with new data...")
                    if self.telegram_bot:
                        self.telegram_bot.log_important_event("MODEL RETRAINING", f"Starting periodic retraining\nCycle: #{self.cycle_count}")
                    retrain_thread = threading.Thread(target=self._retrain_all_models, daemon=True)
                    retrain_thread.start()

            cycle_duration = time.time() - cycle_start
            self._log_trading_cycle_metrics(trading_decisions, cycle_duration) 
            self._monitor_system_health()
            
            self.logger.info(f"Enhanced Trading Cycle completed in {cycle_duration:.2f} seconds.")
            
            return trading_decisions 
                
        except Exception as e:
            cycle_duration = time.time() - cycle_start
            self.logger.error("Trading cycle failed", extra={'error': str(e), 'cycle_duration': cycle_duration, 'timestamp': datetime.now().isoformat()})
            error_msg = f"Critical Error in trading cycle: {e}"
            print(f"❌ {error_msg}")
            self.error_handler.handle_trading_error(e, "ALL", "trading_cycle_main")
            if self.telegram_bot:
                self.telegram_bot.log_error(error_msg, "Trading Cycle Main Loop")
            return []

    def should_run_advanced_analysis(self) -> bool:
        if not hasattr(self, '_last_advanced_analysis'):
            self._last_advanced_analysis = datetime.now()
            return True
        
        time_since_last = (datetime.now() - self._last_advanced_analysis).total_seconds()
        if time_since_last > 4 * 3600:
            self._last_advanced_analysis = datetime.now()
            return True
        
        return False

    def _reconcile_open_positions(self, bybit_live_positions: Dict[str, Dict]):
        """
        Checks local DB for open trades that are no longer open on Bybit
        and updates them with the final PnL.
        """
        self.logger.info("Starting open position reconciliation...")
        reconciled_count = 0
        
        try:
            # 1. Get all trades from our DB that are still marked as "open"
            stale_trades_map = self.database.get_open_stale_trades_by_symbol()
            if not stale_trades_map:
                self.logger.info("No stale trades found in local DB. Reconciliation complete.")
                return

            self.logger.info(f"Found {len(stale_trades_map)} symbols with open trades in local DB.")

            # 2. Get Bybit's closed PnL history
            first_trade_ts_iso = self.database.get_first_trade_timestamp_iso()
            if not first_trade_ts_iso:
                self.logger.info("No trades found in DB. Skipping reconciliation.")
                return
            
            first_trade_dt = datetime.fromisoformat(first_trade_ts_iso)
            first_trade_ms = int(first_trade_dt.timestamp() * 1000)
            
            # Fetch PnL history from the bot's start time
            closed_pnl_resp = self.client.get_closed_pnl_history(
                category="linear",
                start_time_ms=first_trade_ms,
                limit=50 # Max 50, fetch most recent
            )

            bybit_pnl_map = {}
            if closed_pnl_resp and closed_pnl_resp.get('retCode') == 0:
                pnl_records = closed_pnl_resp['result'].get('list', [])
                pnl_records.reverse() # Process oldest first (FIFO)
                for record in pnl_records:
                    symbol = record.get('symbol')
                    if symbol not in bybit_pnl_map:
                        bybit_pnl_map[symbol] = []
                    bybit_pnl_map[symbol].append(record)
            else:
                err = closed_pnl_resp.get('retMsg', 'No response') if closed_pnl_resp else 'No response'
                self.logger.error(f"Failed to fetch closed PnL history for reconciliation: {err}")
                return # Can't reconcile without PnL data

            # 3. Find "ghost" positions
            for symbol, stale_trades in stale_trades_map.items():
                if symbol in bybit_live_positions:
                    self.logger.debug(f"Position {symbol} is still live. Skipping reconciliation for it.")
                    continue
                
                # This is a "ghost" position. It's in our DB but not live on Bybit.
                self.logger.warning(f"Found 'ghost' position: {symbol}. In local DB but not live on Bybit. Attempting to match PnL.")

                symbol_pnl_records = bybit_pnl_map.get(symbol, [])
                if not symbol_pnl_records:
                    self.logger.error(f"No closed PnL history found for ghost position {symbol}. Cannot reconcile.")
                    continue
                
                trade_queue = stale_trades.copy() # FIFO queue of our DB trades

                # 4. Match ghost trades to PnL records (FIFO)
                for pnl_record in symbol_pnl_records:
                    if not trade_queue:
                        break # All stale trades for this symbol are matched

                    pnl_record_qty = float(pnl_record.get('cumEntryValue', 0)) # Position value in USDT
                    pnl_record_pnl = float(pnl_record.get('closedPnl', 0))
                    pnl_record_exit_price = float(pnl_record.get('avgExitPrice', 0))
                    pnl_record_exit_time_ms = int(pnl_record.get('updatedTime', 0))
                    pnl_record_exit_time_iso = datetime.fromtimestamp(pnl_record_exit_time_ms / 1000.0).isoformat()
                    pnl_record_exit_type = f"Reconciled_{pnl_record.get('exitType', 'Unknown')}"

                    if pnl_record_qty == 0 or pnl_record_exit_time_ms == 0:
                        continue # Invalid record

                    qty_to_allocate = pnl_record_qty
                    
                    while qty_to_allocate > 1e-9 and trade_queue:
                        stale_trade = trade_queue[0] # Get the oldest stale trade
                        stale_trade_id = stale_trade['id']
                        stale_trade_qty_usdt = float(stale_trade.get('position_size_usdt', 0))
                        
                        # Check if this PnL record has already been processed for this trade
                        if stale_trade.get('outcome_updated', 0) == 1 or stale_trade.get('exit_price') is not None:
                             self.logger.debug(f"Trade {stale_trade_id} already closed, popping from queue.")
                             trade_queue.pop(0)
                             continue

                        self.logger.info(f"Matching PnL record (Time: {pnl_record_exit_time_iso}) to DB Trade ID {stale_trade_id} (Symbol: {symbol})")
                        
                        # Simple 1:1 match for now. Assumes one PnL record closes one DB trade.
                        # This is a simplification. A more complex system would handle partial closes.
                        
                        trade_pnl_usdt = pnl_record_pnl
                        trade_pnl_percent = 0.0
                        if stale_trade_qty_usdt > 0:
                            trade_pnl_percent = (trade_pnl_usdt / stale_trade_qty_usdt) * 100

                        update_success = self.database.update_trade_exit_by_id(
                            trade_id=stale_trade_id,
                            exit_price=pnl_record_exit_price,
                            pnl_usdt=trade_pnl_usdt,
                            pnl_percent=trade_pnl_percent,
                            exit_time_iso=pnl_record_exit_time_iso,
                            exit_reason=pnl_record_exit_type
                        )

                        if update_success:
                            self.logger.info(f"  -> SUCCESS: Reconciled trade {stale_trade_id} with PnL ${trade_pnl_usdt:.2f}")
                            reconciled_count += 1
                        else:
                            self.logger.error(f"  -> FAILED: Could not update trade {stale_trade_id} in DB.")
                        
                        # Whether update succeeded or failed, remove from queue to avoid re-processing
                        trade_queue.pop(0)
                        qty_to_allocate = 0 # Simple 1:1 match, so PnL record is "used"
            
            if reconciled_count > 0:
                self.logger.info(f"Reconciliation complete. Updated {reconciled_count} ghost trades.")
                if self.telegram_bot:
                    self.telegram_bot.log_important_event("RECONCILIATION", f"Successfully found and updated {reconciled_count} manually closed trades.")

        except Exception as e:
            self.logger.error(f"Critical error during position reconciliation: {e}", exc_info=True)
            if self.error_handler:
                self.error_handler.handle_api_error(e, "reconcile_open_positions")

    def _check_and_retrain_low_performance_models(self):
        try:
            low_performance_symbols = []
            
            for symbol in SYMBOLS:
                if symbol in self.strategy_orchestrator.ml_predictor.model_versions:
                    model_info = self.strategy_orchestrator.ml_predictor.model_versions[symbol]
                    accuracy = model_info.get('accuracy', 0)
                    
                    if accuracy < 0.50:
                        low_performance_symbols.append(symbol)
                        self.logger.warning(f"[CRITICAL] Model for {symbol} has critically low accuracy: {accuracy:.3f}")
            
            if low_performance_symbols:
                self.logger.info(f"[CRITICAL RETRAINING] Retraining {len(low_performance_symbols)} critically underperforming models: {low_performance_symbols}")
                
                if self.telegram_bot:
                    self.telegram_bot.send_channel_message(
                        f"🚨 <b>CRITICAL MODEL RETRAINING</b>\n\n"
                        f"Retraining {len(low_performance_symbols)} underperforming models:\n"
                        f"{', '.join(low_performance_symbols)}\n"
                        f"Accuracy below 50% threshold"
                    )
                
                for symbol in low_performance_symbols:
                    try:
                        historical_data = self.data_engine.get_historical_data(symbol, TIMEFRAME, limit=1500)
                        if historical_data is not None and len(historical_data) >= 250:
                            success = self.strategy_orchestrator.ml_predictor.train_model(symbol, historical_data)
                            if success:
                                self.logger.info(f"[CRITICAL RETRAINING] Successfully retrained {symbol}")
                    except Exception as e:
                        self.logger.error(f"[CRITICAL RETRAINING] Failed to retrain {symbol}: {e}")
                
                self.strategy_orchestrator.ml_predictor.save_models()
                        
        except Exception as e:
            self.logger.error(f"Error in critical retraining check: {e}")

    def _optimize_strategy_weights(self):
        try:
            recent_trades = self.database.get_historical_trades(days=7)
            recent_performance = {}
            
            market_data_for_opt = {}
            for symbol in SYMBOLS:
                data = self.data_engine.historical_data.get(symbol)
                if data is None or len(data) < 100:
                    data = self.data_engine.get_historical_data(symbol, TIMEFRAME)
                if data is not None and len(data) >= 100:
                    market_data_for_opt[symbol] = data
            
            if market_data_for_opt:
                ref_symbol = "BTCUSDT" if "BTCUSDT" in market_data_for_opt else list(market_data_for_opt.keys())[0]
                ref_data = market_data_for_opt[ref_symbol]

                regime_analysis = self.strategy_optimizer.analyze_market_regimes(
                    ref_symbol, ref_data
                )
                
                volatility = self.strategy_optimizer.calculate_volatility(
                    ref_data
                )
                
                optimized_weights = self.strategy_optimizer.optimize_weights(
                    market_regime=regime_analysis['regime'],
                    volatility=volatility,
                    recent_performance=recent_performance,
                    aggressiveness=self.aggressiveness
                )
                
                self.strategy_orchestrator.strategy_weights = optimized_weights
                
                print(f"🎯 Strategy weights optimized: {optimized_weights}")
                
                self.database.store_system_event(
                    "STRATEGY_WEIGHT_OPTIMIZATION",
                    {
                        'new_weights': optimized_weights,
                        'market_regime': regime_analysis['regime'],
                        'volatility': volatility
                    },
                    "INFO",
                    "Strategy Optimization"
                )
            else:
                print("⚠️ Strategy optimization skipped: Insufficient market data.")
                
        except Exception as e:
            print(f"⚠️ Strategy optimization failed: {e}")
            self.error_handler.handle_trading_error(e, "ALL", "strategy_optimization")

    def run_backtest_mode(self, cycles: int = 5):
        print(f"\n🧪 Running Backtest Mode for {cycles} cycles...")
        print("💡 Note: No real trades will be executed")
        print(f"🎯 Aggressiveness: {self.aggressiveness.upper()}")
        
        if self.telegram_bot:
            self.telegram_bot.log_important_event(
                "BACKTEST STARTED",
                f"Running {cycles} backtest cycles\nAggressiveness: {self.aggressiveness.upper()}\nMode: BACKTEST"
            )
        
        self.database.store_system_event(
            "BACKTEST_STARTED",
            {
                'cycles': cycles,
                'aggressiveness': self.aggressiveness,
                'symbols': SYMBOLS
            },
            "INFO",
            "Backtest"
        )
        
        total_pnl = 0
        total_trades = 0
        total_strong_signals = 0
        
        for cycle in range(cycles):
            print(f"\n🔬 Backtest Cycle {cycle + 1}/{cycles}")
            self.cycle_count = cycle + 1
            
            decisions = self.run_trading_cycle()
            
            profitable_decisions = [d for d in decisions if d.get('composite_score', 0) > 20]
            potential_trades = [d for d in decisions if d['action'] != 'HOLD']
            strong_signals = [d for d in decisions if d.get('signals', {}).get('signal_strength') in ['STRONG_BUY', 'STRONG_SELL']]
            
            print(f"   📊 {len(profitable_decisions)} profitable signals, {len(potential_trades)} potential trades")
            print(f"   💪 {len(strong_signals)} strong signals detected")
            
            cycle_pnl = sum(d.get('composite_score', 0) for d in profitable_decisions) / 100
            total_pnl += cycle_pnl
            total_trades += len(potential_trades)
            total_strong_signals += len(strong_signals)
            
            if cycle < cycles - 1:
                print(f"\n⏰ Waiting 1 minute for next backtest cycle...")
                time.sleep(60)
        
        avg_score = total_pnl / cycles if cycles > 0 else 0
        avg_trades = total_trades / cycles if cycles > 0 else 0
        
        print(f"\n🎯 Backtest Complete!")
        print(f"   • Total Cycles: {cycles}")
        print(f"   • Total Potential Trades: {total_trades}")
        print(f"   • Total Strong Signals: {total_strong_signals}")
        print(f"   • Average Score per Cycle: {avg_score:.2f}")
        print(f"   • Average Trades per Cycle: {avg_trades:.1f}")
        print(f"   • Aggressiveness: {self.aggressiveness.upper()}")
        
        self.database.store_system_event(
            "BACKTEST_COMPLETED",
            {
                'total_cycles': cycles,
                'total_trades': total_trades,
                'total_strong_signals': total_strong_signals,
                'avg_score': avg_score,
                'avg_trades': avg_trades,
                'aggressiveness': self.aggressiveness
            },
            "INFO",
            "Backtest"
        )
        
        if self.telegram_bot:
            self.telegram_bot.log_important_event(
                "BACKTEST COMPLETE",
                f"""Results for {self.aggressiveness.upper()} mode:
• {total_trades} potential trades over {cycles} cycles
• {total_strong_signals} strong signals detected
• Average Score: {avg_score:.2f}
• Average Trades/Cycle: {avg_trades:.1f}"""
            )
        
        return {
            'total_cycles': cycles,
            'total_trades': total_trades,
            'total_strong_signals': total_strong_signals,
            'avg_score': avg_score,
            'avg_trades': avg_trades,
            'aggressiveness': self.aggressiveness
        }

    def get_performance_summary(self):
        try:
            performance = self.execution_engine.get_performance_metrics()
            performance['aggressiveness'] = self.aggressiveness
            
            db_stats = self.database.get_trading_statistics(days=7)
            performance['recent_stats'] = db_stats
            
            return performance
        except Exception as e:
            self.error_handler.handle_trading_error(e, "ALL", "performance_summary")
            return {'aggressiveness': self.aggressiveness}
    
    def change_aggressiveness(self, new_aggressiveness: str):
        valid_levels = ["conservative", "moderate", "aggressive", "high"]
        if new_aggressiveness not in valid_levels:
            print(f"❌ Invalid aggressiveness level. Must be one of: {valid_levels}")
            return False
            
        old_level = self.aggressiveness
        self.aggressiveness = new_aggressiveness
        self.strategy_orchestrator.change_aggressiveness(new_aggressiveness)
        self.risk_manager.aggressiveness = new_aggressiveness
        self.risk_manager.config = RiskConfig.get_config(new_aggressiveness)
        self.risk_manager._set_parameters_from_config()
        
        print(f"🔄 Strategy aggressiveness changed from {old_level.upper()} to {new_aggressiveness.upper()}")
        
        self.database.store_system_event(
            "AGGRESSIVENESS_CHANGED",
            {
                'old_level': old_level,
                'new_level': new_aggressiveness
            },
            "INFO",
            "Configuration"
        )
        
        if self.telegram_bot:
            self.telegram_bot.log_important_event(
                "STRATEGY CHANGE",
                f"Changed aggressiveness from {old_level.upper()} to {new_aggressiveness.upper()}"
            )
        
        return True

    def start_telegram_bot(self):
        if self.telegram_bot:
            telegram_thread = threading.Thread(target=self.telegram_bot.start_polling, daemon=True)
            telegram_thread.start()
            print("✅ Telegram command bot started in background")
            return True
        return False

    def get_error_summary(self):
        return self.error_handler.get_error_summary()

    def reset_error_handler(self):
        self.error_handler.reset_circuit_breaker()
        print("🔄 Error handler circuit breaker reset")

    def get_emergency_status(self):
        return self.emergency_protocols.get_emergency_status()

    def reset_emergency_mode(self, reason: str = "Manual reset"):
        self.emergency_protocols.reset_emergency_mode(reason)

def main():
    print("🚀 Advanced Bybit Trading Bot Starting...")
    print("🔐 Using DEMO TRADING environment")
    print("🧠 Features: ML Prediction + Advanced TA + Risk Management")
    print("💾 Model Persistence: Enabled")
    print("📱 Telegram Logging & Commands: Enabled")
    print("🎯 Multiple Aggressiveness Levels Available")
    print("🛡️ Enhanced Error Handling: Active")
    print("💾 Database Logging: Active")
    print("🆘 Emergency Protocols: Active")

    selected_aggressiveness = "aggressive"
    mode_choice = "1"

    print(f"\n🎯 Selected: {selected_aggressiveness.upper()} mode")
    print("🎮 Mode: LIVE TRADING" if mode_choice == "1" else "🎮 Mode: BACKTEST")

    bot = None
    try:
        bot = AdvancedTradingBot(aggressiveness=selected_aggressiveness)
        bot.start_telegram_bot()
        
        if bot.telegram_bot:
            bot.telegram_bot.start_summary_updater(interval_seconds=600)
        
        if mode_choice == "1":
            cycle_count = 0
            print("\n🔔 Live trading started. Press Ctrl+C to stop.")
            print("💡 You can control the bot via Telegram commands!")
            print("💡 Use /help in Telegram to see available commands")

            cycle_interval_seconds = 60
            print(f"🔄 Running analysis cycle every {cycle_interval_seconds} seconds.")

            print("   🔌 Starting persistent WebSocket streams...")
            start_success = False
            try:
                bot.data_engine.start_streams()
                print("      Waiting for WS connections...")
                time.sleep(10)
                ws_status = bot.client.get_websocket_status()
                if ws_status['public']['connected']:
                     print("      ✅ Public WebSocket connected.")
                     start_success = True
                else:
                     bot.logger.error("Initial Public WS connection failed. Bot will attempt to recover.")
                     if bot.telegram_bot: bot.telegram_bot.log_error("Initial Public WS failed to connect", "Bot Startup")
            except Exception as start_e:
                bot.logger.error(f"Error starting WebSockets during init: {start_e}", exc_info=True)

            while True:
                if hasattr(bot, 'running') and not bot.running:
                     print("🛑 Stop command received. Exiting main loop.")
                     break

                cycle_start_time = time.time()
                try:
                    cycle_count += 1
                    bot.cycle_count = cycle_count

                    print(f"\n⏰ Preparing for cycle {cycle_count}...")
                    
                    print(f"\n{'='*70}")
                    print(f"📈 LIVE TRADING CYCLE #{cycle_count}")
                    print(f"🎯 Aggressiveness: {bot.aggressiveness.upper()}")
                    print(f"🛡️ Error Status: {bot.error_handler.get_health_status()}")
                    print(f"{'='*70}")

                    bot.run_trading_cycle()

                    if cycle_count % 10 == 0:
                        print("💾 Updating ML prediction outcomes from closed trades...")
                        ml_update_thread = threading.Thread(
                            target=bot.strategy_orchestrator.ml_predictor.update_ml_prediction_outcomes,
                            daemon=True
                        )
                        ml_update_thread.start()
                    
                    cycle_duration = time.time() - cycle_start_time
                    sleep_time = max(0, cycle_interval_seconds - cycle_duration)
                    print(f"   Cycle duration: {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s...")
                    time.sleep(sleep_time)

                except KeyboardInterrupt:
                    print("\n\n🛑 Bot stopping...")
                    if bot:
                        bot.running = False
                        portfolio_value = bot.get_portfolio_value()
                        performance = bot.get_performance_summary()
                        print("🔌 Stopping WebSocket streams...")
                        bot.data_engine.stop()
                        bot.database.store_system_event( "BOT_STOPPED", { 'final_portfolio_value': portfolio_value, 'total_cycles': cycle_count, 'performance': performance }, "INFO", "System Shutdown")
                        if bot.telegram_bot: bot.telegram_bot.log_bot_stop(portfolio_value, performance)
                        print("💾 Saving models before exit...")
                        bot.strategy_orchestrator.ml_predictor.save_models()
                        print("💾 Closing database connection...")
                        bot.database.close()
                    print("✅ Bot stopped gracefully.")
                    break

                except Exception as e:
                    error_msg = f"Critical error in main loop: {e}"
                    print(f"\n❌ {error_msg}")
                    if hasattr(bot, 'logger'): bot.logger.critical(error_msg, exc_info=True)
                    else: import traceback; traceback.print_exc()

                    if bot:
                        bot.error_handler.handle_trading_error(e, "ALL", "main_loop_critical")
                        if bot.telegram_bot: bot.telegram_bot.send_channel_message(f"🚨 <b>CRITICAL LOOP ERROR</b>\n\n{error_msg}\n\nCheck logs!")
                    print("🔄 Attempting to recover cycle...")
                    
                    print(f"   Sleeping for {cycle_interval_seconds} seconds after error...")
                    time.sleep(cycle_interval_seconds)
            
            print("Main loop exited.")
            if bot:
                print("🔌 Ensuring final cleanup...")
                bot.data_engine.stop()
                if bot.telegram_bot: bot.telegram_bot.stop_polling()
                bot.strategy_orchestrator.ml_predictor.save_models()
                bot.database.close()
            print("✅ Bot shut down completely.")

        elif mode_choice == "2":
             bot.run_backtest_mode(cycles=10)
             print("✅ Backtest finished. Exiting.")
             if bot:
                 try:
                      print("🔌 Stopping WebSocket streams (if running)...")
                      bot.data_engine.stop()
                      print("💾 Closing database connection...")
                      bot.database.close()
                 except Exception as cleanup_e:
                      print(f"⚠️ Error during backtest cleanup: {cleanup_e}")

    except Exception as e:
        error_msg = f"Fatal error during bot initialization or critical failure: {e}"
        print(f"❌ {error_msg}")
        if bot and hasattr(bot, 'logger'): bot.logger.critical(error_msg, exc_info=True)
        else: import traceback; traceback.print_exc()
        if bot and bot.telegram_bot: bot.telegram_bot.send_channel_message(f"🚨 <b>FATAL BOT ERROR</b>\n\n{error_msg}\n\nBot will exit.")
        print("Bot cannot continue. Exiting.")
        if bot:
            try:
                if hasattr(bot, 'data_engine'): bot.data_engine.stop()
                if hasattr(bot, 'strategy_orchestrator'): bot.strategy_orchestrator.ml_predictor.save_models()
                if hasattr(bot, 'database'): bot.database.close()
            except: pass

if __name__ == "__main__":
    main()