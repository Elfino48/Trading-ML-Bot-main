import time

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

        # --- Capture Start Time ---
        self.start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🚀 Initializing Advanced Trading Bot (Run ID: {self.start_time_str})...")
        # --- End Capture Start Time ---

        print("🚀 Initializing Advanced Trading Bot...")

        self._setup_comprehensive_logging()

        

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
                run_start_time_str=self.start_time_str # Pass the unique start time
            )

            self.execution_engine = ExecutionEngine(self.client, self.risk_manager)

            

            self.error_handler = ErrorHandler(self.telegram_bot)

            self.database = TradingDatabase()

            self.emergency_protocols = EmergencyProtocols(self.execution_engine, self.telegram_bot)

            self.strategy_optimizer = StrategyOptimizer(self.database)

            self.backtester = AdvancedBacktester(self.strategy_orchestrator)

            

            # --- Dependency Injection ---

            self.client.set_error_handler(self.error_handler)

            self.data_engine.set_error_handler(self.error_handler)

            self.risk_manager.set_error_handler(self.error_handler)

            self.risk_manager.set_database(self.database)

            self.strategy_orchestrator.set_error_handler(self.error_handler)

            self.strategy_orchestrator.set_database(self.database)

            self.execution_engine.set_emergency_protocols(self.emergency_protocols)

            # --- End Dependency Injection ---

            

            self.aggressiveness = aggressiveness

            

            if self.telegram_bot:

                self.telegram_bot.set_trading_bot(self)

            

            self._validate_configuration()

            

            # --- Initialize ML Models (Load or Train ONCE) ---

            self._initialize_ml_models()

            # --- End ML Init ---

            

            portfolio_value = self._test_connection()

            

            self._initialize_leverage()

            

            self._initialize_database()

            

            if self.telegram_bot:

                self.telegram_bot.send_channel_message(

                    f"🤖 <b>TRADING BOT STARTED</b>\n\n"

                    f"💰 <b>Portfolio:</b> ${portfolio_value:,.2f}\n"

                    f"📊 <b>Symbols:</b> {', '.join(SYMBOLS)}\n"

                    f"🎯 <b>Aggressiveness:</b> {self.aggressiveness.upper()}\n"

                    f"🛡️ <b>Error Handler:</b> Active\n"

                    f"💾 <b>Database:</b> Active\n"

                    f"🆘 <b>Emergency Protocols:</b> Active\n"

                    f"🕒 <b>Time:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

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
                        'maxBytes': 10485760, # 10MB
                        'backupCount': 5,
                        'formatter': 'detailed',
                        'encoding': 'utf-8'
                    },
                    'json_file': {
                         'class': 'logging.handlers.RotatingFileHandler',
                         'filename': 'logs/trading_bot_json.log',
                         'maxBytes': 10485760, # 10MB
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
                    # Root logger configuration
                    '': {
                        'handlers': ['file', 'json_file', 'console'],
                        'level': 'INFO' # <<< Root level is INFO
                    },
                    # Specific loggers set to DEBUG
                    'AdvancedTradingBot': {
                        'level': 'DEBUG',
                        'handlers': ['file', 'json_file', 'console'],
                        'propagate': False
                    },
                     # --- ADD THIS SECTION ---
                     # --- END ADDED SECTION ---
                     'MLPredictor': { # Optional: Add if you want MLPredictor debug logs too
                         'level': 'DEBUG',
                         'handlers': ['file', 'json_file', 'console'],
                         'propagate': False
                     },
                     'ExecutionEngine': { # Optional: Add for ExecutionEngine
                         'level': 'DEBUG',
                         'handlers': ['file', 'json_file', 'console'],
                         'propagate': False
                     },
                     # Add other specific loggers if needed...
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

        """Loads existing models or trains new ones ONLY at startup."""

        print("\n🤖 Initializing ML models...")

        try:

            models_loaded = self.strategy_orchestrator.ml_predictor.load_models()

            if not models_loaded:

                print("📚 No saved models found. Training initial models...")

                trained_count = self._retrain_all_models(force_all=True) # Use the retrain function for initial training too

                if trained_count > 0:

                    self.strategy_orchestrator.ml_predictor.save_models()

                    print(f"💾 Saved {trained_count} new models to disk")

                    if self.telegram_bot:

                        self.telegram_bot.log_important_event(

                            "INITIAL MODELS TRAINED",

                            f"Successfully trained and saved {trained_count} initial ML models"

                        )

                else:

                    error_msg = "Initial model training failed"

                    self.error_handler.handle_ml_error(Exception(error_msg), "ALL", "initial_training")

            else:

                print("✅ Using pre-trained models from disk")

                if self.telegram_bot:

                    self.telegram_bot.log_important_event(

                        "MODELS LOADED",

                        f"Successfully loaded {len(self.strategy_orchestrator.ml_predictor.models)} pre-trained ML models"

                    )

        except Exception as e:

            self.error_handler.handle_ml_error(e, "ALL", "initialization")

            print("⚠️ Continuing without ML models or using potentially incomplete set")



    def _retrain_all_models(self, force_all: bool = False) -> int:
        """Enhanced model retraining with performance-based selection"""
        trained_count = 0
        self.logger.info("[MODEL RETRAINING] Starting enhanced retraining process...")
        
        # Get symbols that actually need retraining
        if force_all:
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
                
                # Fetch fresh data for retraining (more data for better models)
                historical_data = self.data_engine.get_historical_data(symbol, TIMEFRAME, limit=1500)

                if historical_data is not None and len(historical_data) >= 250:
                    if self.data_engine.validate_market_data(historical_data.reset_index()):
                        # Store previous performance for comparison
                        old_performance = self.strategy_orchestrator.ml_predictor.model_versions.get(symbol, {}).get('accuracy', 0)
                        
                        # Retrain the model
                        success = self.strategy_orchestrator.ml_predictor.train_model(symbol, historical_data)
                        
                        if success:
                            trained_count += 1
                            new_performance = self.strategy_orchestrator.ml_predictor.model_versions.get(symbol, {}).get('accuracy', 0)
                            
                            # Log performance change
                            perf_change = new_performance - old_performance
                            self.logger.info(f"[MODEL RETRAINING] {symbol} retrained successfully. Accuracy: {old_performance:.3f} -> {new_performance:.3f} ({perf_change:+.3f})")
                            
                            # Send Telegram notification for significant improvements
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

                time.sleep(0.3)  # Rate limiting between symbols
                
            except Exception as e:
                error_msg = f"Error during retraining for {symbol}: {e}"
                self.logger.error(error_msg)
                self.error_handler.handle_ml_error(e, symbol, "retraining")

        self.logger.info(f"[MODEL RETRAINING] Retraining complete. {trained_count}/{len(symbols_to_retrain)} models updated")
        
        # Save all models after retraining session
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

                

                # Store startup event only after successful connection

                # self.database.store_system_event(

                #     "STARTUP",

                #     {"portfolio_value": equity, "aggressiveness": self.aggressiveness},

                #     "INFO",

                #     "Bot Startup"

                # )

                return equity

            else:

                error_msg = balance.get('retMsg', 'Unknown error') if balance else 'No response'

                print(f"❌ API connection failed: {error_msg}")

                self.error_handler.handle_api_error(Exception(error_msg), "connection_test")

                # Fallback value if connection fails

                return 10000

        except Exception as e:

            print(f"❌ Connection test failed: {e}")

            self.error_handler.handle_api_error(e, "connection_test")

            # Fallback value if connection fails

            return 10000



    def _initialize_leverage(self):

        print("\n⚙️ Checking leverage settings...")

        

        # Skip leverage setting in demo environment

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

                

                # First check current leverage

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

                

                # Try to set leverage

                result = self.client.set_leverage(symbol, leverage)

                

                if result and result.get('retCode') == 0:

                    print(f"✅ Leverage set to {leverage}x for {symbol}")

                else:

                    error_msg = result.get('retMsg', 'Unknown error') if result else 'No response'

                    

                    # Check for specific leverage modification errors

                    if "leverage not modified" in error_msg.lower():

                        print(f"ℹ️  Leverage modification restricted for {symbol} (common in demo)")

                    elif "not in range" in error_msg.lower():

                        print(f"⚠️  Leverage {leverage}x not available for {symbol}")

                    else:

                        print(f"⚠️  Could not set leverage for {symbol}: {error_msg}")

                        

            except Exception as e:

                print(f"⚠️  Error setting leverage for {symbol}: {e}")

                # Don't trigger circuit breaker for leverage errors

                continue

            

            time.sleep(0.2)  # Rate limiting



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



    



    def get_portfolio_value(self) -> float:

        try:

            balance = self.client.get_wallet_balance()

            if balance and balance.get('retCode') == 0:

                return float(balance['result']['list'][0]['totalEquity'])

            # Fallback if API fails during operation

            self.logger.warning("Failed to get portfolio value from API, returning cached or default.")

            return getattr(self, '_last_portfolio_value', 10000) # Use a cached value or default

        except Exception as e:

            self.error_handler.handle_api_error(e, "get_balance")

            # Fallback if API fails during operation

            self.logger.error(f"Exception getting portfolio value: {e}, returning cached or default.")

            return getattr(self, '_last_portfolio_value', 10000)

    def run_trading_cycle(self):
        cycle_start = time.time()
        self.logger.info("Starting Enhanced Trading Cycle...")
        
        trading_decisions = [] # Stores full decision dicts for logging/return
        all_symbol_data_copies = {} # Store copies for this cycle's analysis
        account_info = {}
        
        try:
            print("\n" + "="*60)
            print("🔄 Starting Enhanced Trading Cycle...")
            print("="*60)
            
            # --- Pre-Cycle Checks ---
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

            # --- Step 1: Get Current Market Data State (Updated by WS) ---
            print("📊 Accessing latest market data...")
            data_access_start = time.time()
            valid_data_count = 0
            for symbol in SYMBOLS:
                # Get a COPY of the latest data frame managed by DataEngine
                data_copy = self.data_engine.get_market_data_for_analysis(symbol)
                if data_copy is not None and not data_copy.empty:
                    # Check if we have valid close prices
                    close_prices = data_copy['close'].astype(float)
                    valid_closes = close_prices[close_prices > 0].dropna()
                    
                    if len(valid_closes) >= 50:  # Require at least 50 valid prices
                        all_symbol_data_copies[symbol] = data_copy
                        valid_data_count += 1
                    else:
                        self.logger.warning(f"Invalid data for {symbol}: only {len(valid_closes)} valid prices")
                else:
                    self.logger.warning(f"No data available for {symbol}")
            data_access_duration = time.time() - data_access_start
            print(f"   Accessed data for {valid_data_count}/{len(SYMBOLS)} symbols in {data_access_duration:.2f}s")
            
            # Exit cycle if no valid data is available for any symbol
            if valid_data_count == 0:
                print("🚫 No valid market data available for any symbol. Skipping cycle.")
                return []

            # --- Step 2: Fetch Account Data Once ---
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
                    #self.error_handler.handle_api_error(Exception("Failed to get positions"), "cycle_account_fetch")
            except Exception as e:
                print(f"❌ Error fetching account data: {e}")
                self.error_handler.handle_api_error(e, "cycle_account_fetch")
            account_fetch_duration = time.time() - account_fetch_start
            print(f"   Account data fetch complete in {account_fetch_duration:.2f}s")

            # --- PHASE 4: Advanced Analysis (Run periodically) ---
            if self.should_run_advanced_analysis():
                print("🔬 Running Phase 4: Advanced Portfolio Analysis...")
                try:
                    # Advanced correlation analysis
                    correlation_analysis = self.strategy_orchestrator.analyze_portfolio_correlation_advanced(
                        SYMBOLS, all_symbol_data_copies
                    )
                    print(f"   📊 Correlation analysis completed: {len(correlation_analysis.get('correlation_clusters', {}))} clusters found")
                    
                    # Regime transition detection for each symbol
                    regime_transitions = {}
                    for symbol in SYMBOLS:
                        if symbol in all_symbol_data_copies:
                            regime_transition = self.strategy_orchestrator.detect_market_regime_transitions(
                                symbol, all_symbol_data_copies[symbol]
                            )
                            regime_transitions[symbol] = regime_transition
                    
                    print(f"   🔄 Regime transition analysis completed for {len(regime_transitions)} symbols")
                    
                    # Advanced risk parity allocation
                    risk_parity = self.strategy_orchestrator.calculate_advanced_risk_parity(
                        SYMBOLS, all_symbol_data_copies, portfolio_value
                    )
                    print(f"   ⚖️ Advanced risk parity allocation calculated")
                    
                    # Performance attribution
                    performance_report = self.strategy_orchestrator.generate_performance_report(days=7)
                    print(f"   📈 Performance attribution report generated")
                    
                    # Store advanced analysis results
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

            # --- Step 3: Analyze Symbols (Using Data Copies) ---
            print("🧠 Analyzing symbols...")
            analysis_start = time.time()
            raw_decisions = [] 
            for symbol in SYMBOLS:
                if symbol not in all_symbol_data_copies:
                    continue 

                try:
                    print(f"\n   🔍 Analyzing {symbol}...")
                    # Use the data copy obtained in Step 1
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
            
            # --- Step 4: Execute Trades ---
            print("🚀 Executing trades...")
            execution_start = time.time()
            actions_taken = 0
            strong_trades = 0
            moderate_trades = 0
            min_confidence = self.risk_manager.min_confidence

            for decision in raw_decisions: 
                symbol = decision['symbol']
                action = decision['action']
                confidence = decision['confidence']
                signal_strength = decision.get('signals', {}).get('signal_strength', 'NEUTRAL')
                
                trading_decisions.append(decision.copy()) 

                print(f"\n   🚦 Evaluating {symbol}: {action} (Conf: {confidence:.1f}%)")
                
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
                        # Pass market_data=None initially, risk manager can fetch correlation if needed
                        risk_check = self.risk_manager.can_trade(symbol, decision['position_size'], market_data=None) 

                        if risk_check['approved']:
                            adjusted_size = risk_check.get('adjusted_size', decision['position_size'])
                            if adjusted_size > 0 and adjusted_size != decision['position_size']:
                                print(f"      ℹ️ Adjusting size for {symbol}: ${adjusted_size:.2f} (Reason: {risk_check['reason']})")
                                # Update decision dict before execution
                                decision['position_size'] = adjusted_size
                                # Use the latest price for quantity calculation during execution
                                latest_price = self.data_engine.get_current_price(symbol)
                                if latest_price <= 0:
                                    print(f"      ❌ Cannot calculate quantity for adjusted size, invalid latest price ({latest_price}). Skipping.")
                                    continue
                                decision['quantity'] = adjusted_size / latest_price
                            elif adjusted_size <= 0:
                                print(f"      ❌ Risk manager approved but adjusted size is <= 0 ({adjusted_size:.2f}). Skipping.")
                                continue
                            
                            # Use latest price for execution checks if needed, but decision SL/TP is based on analysis price
                            decision['current_price'] = self.data_engine.get_current_price(symbol)
                            if decision['current_price'] <= 0:
                                print(f"      ❌ Cannot execute trade, invalid latest price ({decision['current_price']}). Skipping.")
                                continue
                                
                            # Ensure quantity is positive
                            if decision['quantity'] <= 0:
                                print(f"      ❌ Calculated quantity is zero or negative ({decision['quantity']:.4f}). Skipping.")
                                continue

                            execution_result = self.execution_engine.execute_enhanced_trade(decision) 
                            
                            if execution_result['success']:
                                print(f"      ✅ Trade executed successfully!")
                                actions_taken += 1
                                
                                trade_record = decision.copy() 
                                trade_record.update({
                                    'success': True,
                                    'order_id': execution_result.get('order_id'),
                                    'timestamp': pd.Timestamp.now(),
                                    # Use actual executed price if available from result
                                    'entry_price': execution_result.get('price', decision.get('current_price')) 
                                })
                                
                                self.database.store_trade(trade_record) 
                                self.risk_manager.record_trade_outcome(True, pnl=0) 
                                
                                if signal_strength in ['STRONG_BUY', 'STRONG_SELL']:
                                    strong_trades += 1
                                else:
                                    moderate_trades += 1
                                
                                if self.telegram_bot:
                                    self.telegram_bot.log_trade_execution(decision) # Log original decision intent
                                
                                if 'order_id' in execution_result:
                                    print(f"         📝 Order ID: {execution_result['order_id']}")
                            else:
                                # Log failed execution attempts
                                error_msg = execution_result['message']
                                print(f"      ❌ Trade execution failed: {error_msg}")
                                trade_record = decision.copy()
                                trade_record.update({
                                    'success': False,
                                    'error_message': error_msg,
                                    'timestamp': pd.Timestamp.now() 
                                })
                                self.database.store_trade(trade_record)
                                self.risk_manager.record_trade_outcome(False, pnl=0)
                                if self.telegram_bot:
                                    self.telegram_bot.log_trade_error(symbol, action, error_msg)
                                self.error_handler.handle_trading_error(
                                    Exception(error_msg), symbol, f"execution_{action}"
                                )
                        else:
                            print(f"      ❌ Risk management rejected trade: {risk_check['reason']}")
                            # Optionally log rejected trade attempts
                            # ... (logging code as before) ...

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

            # --- Cycle Summary ---
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
            
            # Store performance metrics (logic remains the same)
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
                    'max_drawdown': 0 # Needs proper calculation
                })
            else:
                self.database.store_performance_metrics({
                    'portfolio_value': new_portfolio_value, 'daily_pnl_percent': pnl_percent,
                    'total_trades': 0, 'winning_trades': 0, 'win_rate': 0,
                    'avg_confidence': 0, 'avg_risk_reward': 0, 'max_drawdown': 0
                })
            
            # Send Telegram summary (logic remains the same)
            if self.telegram_bot and (actions_taken > 0 or abs(pnl_percent) > 0.5):
                # ... (telegram summary code) ...
                summary = {
                    'trades_executed': actions_taken, 'strong_signals': strong_trades,
                    'moderate_signals': moderate_trades, 'portfolio_value': new_portfolio_value,
                    'pnl_percent': pnl_percent, 'aggressiveness': self.aggressiveness
                }
                self.telegram_bot.log_cycle_summary(summary)

            # Store cycle complete event (logic remains the same)
            self.database.store_system_event("TRADING_CYCLE_COMPLETE", # ... (event data) ...
                { 'trades_executed': actions_taken, 'portfolio_value': new_portfolio_value,
                'pnl_percent': pnl_percent, 'cycle_pnl': pnl }, "INFO", "Trading Cycle")
            
            # --- Periodic Tasks ---
            if hasattr(self, 'cycle_count'):
                if self.cycle_count % 5 == 0: # Optimize weights every 5 cycles
                    print("🔄 Optimizing strategy weights...")
                    self._optimize_strategy_weights() 
                
                # Enhanced retraining logic
                if self.cycle_count % 30 == 0:  # Check for critical models every 30 cycles
                    print("🔍 Checking for critically underperforming models...")
                    critical_retrain_thread = threading.Thread(target=self._check_and_retrain_low_performance_models, daemon=True)
                    critical_retrain_thread.start()
                
                if self.cycle_count % 60 == 0:  # Full retraining every 60 cycles
                    print("🔄 Retraining ML models with new data...")
                    if self.telegram_bot:
                        self.telegram_bot.log_important_event("MODEL RETRAINING", f"Starting periodic retraining\nCycle: #{self.cycle_count}")
                    # Run retraining in a separate thread to avoid blocking main loop
                    retrain_thread = threading.Thread(target=self._retrain_all_models, daemon=True)
                    retrain_thread.start()

            cycle_duration = time.time() - cycle_start
            self._log_trading_cycle_metrics(trading_decisions, cycle_duration) 
            self._monitor_system_health()
            
            self.logger.info(f"Enhanced Trading Cycle completed in {cycle_duration:.2f} seconds.")
            
            return trading_decisions 
                
        except Exception as e:
            # Error handling remains the same
            # ... (error logging code) ...
            cycle_duration = time.time() - cycle_start
            self.logger.error("Trading cycle failed", extra={'error': str(e), 'cycle_duration': cycle_duration, 'timestamp': datetime.now().isoformat()})
            error_msg = f"Critical Error in trading cycle: {e}"
            print(f"❌ {error_msg}")
            self.error_handler.handle_trading_error(e, "ALL", "trading_cycle_main")
            if self.telegram_bot:
                self.telegram_bot.log_error(error_msg, "Trading Cycle Main Loop")
            return []

    def should_run_advanced_analysis(self) -> bool:
        """Determine if advanced analysis should run (e.g., every 4 hours)"""
        if not hasattr(self, '_last_advanced_analysis'):
            self._last_advanced_analysis = datetime.now()
            return True
        
        time_since_last = (datetime.now() - self._last_advanced_analysis).total_seconds()
        if time_since_last > 4 * 3600:  # 4 hours
            self._last_advanced_analysis = datetime.now()
            return True
        
        return False

    def _check_and_retrain_low_performance_models(self):
        """Check for poorly performing models and retrain them immediately"""
        try:
            low_performance_symbols = []
            
            for symbol in SYMBOLS:
                if symbol in self.strategy_orchestrator.ml_predictor.model_versions:
                    model_info = self.strategy_orchestrator.ml_predictor.model_versions[symbol]
                    accuracy = model_info.get('accuracy', 0)
                    
                    # Check if model performance is critically low
                    if accuracy < 0.50:  # Below 50% accuracy
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
                
                # Retrain only the underperforming models
                for symbol in low_performance_symbols:
                    try:
                        historical_data = self.data_engine.get_historical_data(symbol, TIMEFRAME, limit=1500)
                        if historical_data is not None and len(historical_data) >= 250:
                            success = self.strategy_orchestrator.ml_predictor.train_model(symbol, historical_data)
                            if success:
                                self.logger.info(f"[CRITICAL RETRAINING] Successfully retrained {symbol}")
                    except Exception as e:
                        self.logger.error(f"[CRITICAL RETRAINING] Failed to retrain {symbol}: {e}")
                
                # Save updated models
                self.strategy_orchestrator.ml_predictor.save_models()
                        
        except Exception as e:
            self.logger.error(f"Error in critical retraining check: {e}")


    def _optimize_strategy_weights(self):

        try:

            recent_trades = self.database.get_historical_trades(days=7)

            recent_performance = {} # Placeholder for performance metrics if needed

            

            # Fetch market data needed for regime and volatility

            market_data_for_opt = {}

            for symbol in SYMBOLS:

                data = self.data_engine.historical_data.get(symbol) # Use cached data first

                if data is None or len(data) < 100:

                    data = self.data_engine.get_historical_data(symbol, TIMEFRAME) # Fetch if needed

                if data is not None and len(data) >= 100:

                    market_data_for_opt[symbol] = data

            

            if market_data_for_opt:

                # Use a representative symbol (e.g., BTCUSDT or the first one) for overall regime/volatility

                ref_symbol = "BTCUSDT" if "BTCUSDT" in market_data_for_opt else list(market_data_for_opt.keys())[0]

                ref_data = market_data_for_opt[ref_symbol]



                regime_analysis = self.strategy_optimizer.analyze_market_regimes(

                    ref_symbol, ref_data

                )

                

                volatility = self.strategy_optimizer.calculate_volatility(

                    ref_data

                )

                

                # Pass necessary arguments to optimize_weights

                optimized_weights = self.strategy_optimizer.optimize_weights(

                    market_regime=regime_analysis['regime'],

                    volatility=volatility,

                    recent_performance=recent_performance, # Pass performance if calculated

                    aggressiveness=self.aggressiveness

                )

                

                # Update the orchestrator's weights

                self.strategy_orchestrator.strategy_weights = optimized_weights

                

                print(f"🎯 Strategy weights optimized: {optimized_weights}")

                

                # Log the optimization event

                self.database.store_system_event(

                    "STRATEGY_WEIGHT_OPTIMIZATION",

                    {

                        #'old_weights': self.strategy_orchestrator.strategy_weights, # Log previous weights if needed

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

        self.risk_manager.config = RiskConfig.get_config(new_aggressiveness) # Update config in risk manager

        self.risk_manager._set_parameters_from_config() # Apply new parameters

        

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

    selected_aggressiveness = "aggressive" # Default or get from user/config
    mode_choice = "1" # Default to live mode

    # --- Simplified Startup Selection (Example - keep commented out or remove for production) ---
    # print("\nSelect Mode:")
    # print("  1. Live Trading (Demo Account)")
    # print("  2. Backtest Mode (Simulated)")
    # mode_choice = input("Enter choice (1 or 2): ") or "1"
    # if mode_choice not in ["1", "2"]: mode_choice = "1"
    #
    # print("\nSelect Aggressiveness:")
    # print("  1. Conservative")
    # print("  2. Moderate (Default)")
    # print("  3. Aggressive")
    # print("  4. High")
    # agg_choice = input("Enter choice (1-4, default 2): ") or "2"
    # levels = { "1": "conservative", "2": "moderate", "3": "aggressive", "4": "high" }
    # selected_aggressiveness = levels.get(agg_choice, "moderate")
    # --- End Selection ---

    print(f"\n🎯 Selected: {selected_aggressiveness.upper()} mode")
    print("🎮 Mode: LIVE TRADING" if mode_choice == "1" else "🎮 Mode: BACKTEST")

    bot = None
    try:
        bot = AdvancedTradingBot(aggressiveness=selected_aggressiveness)

        bot.start_telegram_bot()

        if mode_choice == "1":
            cycle_count = 0
            print("\n🔔 Live trading started. Press Ctrl+C to stop.")
            print("💡 You can control the bot via Telegram commands!")
            print("💡 Use /help in Telegram to see available commands")

            # --- WebSocket Wait Logic ---
            kline_interval_minutes = int(TIMEFRAME) # Assuming TIMEFRAME is in minutes
            last_kline_timestamp = None
            first_cycle = True

            while True: # Main loop
                try:
                    cycle_count += 1
                    bot.cycle_count = cycle_count # Make cycle count available to bot methods

                    # --- MODIFIED: Kline Wait Logic with Debug Mode ---
                    print(f"\n⏰ Preparing for cycle {cycle_count}...")
                    primary_symbol = SYMBOLS[0]
                    latest_data = bot.data_engine.get_market_data_for_analysis(primary_symbol)
                    current_kline_ts = None
                    if latest_data is not None and not latest_data.empty and isinstance(latest_data.index, pd.DatetimeIndex):
                        current_kline_ts = latest_data.index[-1]

                    # Initialize last_kline_timestamp on first run or if it's None
                    if last_kline_timestamp is None and current_kline_ts is not None:
                        last_kline_timestamp = current_kline_ts
                        print(f"   Initial kline timestamp set: {last_kline_timestamp}")

                    # --- DEBUG MODE Check ---
                    if first_cycle and DEBUG_MODE:
                        print("   ⚠️ DEBUG MODE: Skipping initial kline wait.")
                        if current_kline_ts is not None:
                             last_kline_timestamp = current_kline_ts # Use current ts for this cycle
                             print(f"   Using current kline timestamp for debug cycle: {last_kline_timestamp}")
                        else:
                             print(f"   ⚠️ Could not get current kline timestamp for debug, proceeding anyway.")
                             # last_kline_timestamp remains None or its previous value
                        # Proceed directly without waiting loop
                    # --- END DEBUG MODE Check ---
                    else:
                        # --- Original Wait Logic ---
                        print(f"   Waiting for next {TIMEFRAME}m kline close after {last_kline_timestamp}...")
                        while True:
                            latest_data_wait = bot.data_engine.get_market_data_for_analysis(primary_symbol)
                            current_kline_ts_wait = None
                            if latest_data_wait is not None and not latest_data_wait.empty and isinstance(latest_data_wait.index, pd.DatetimeIndex):
                                current_kline_ts_wait = latest_data_wait.index[-1]

                            print(f"   Checking... Current TS: {current_kline_ts_wait} | Last Cycle TS: {last_kline_timestamp}", end='\r')

                            # Check if a new kline has arrived
                            if current_kline_ts_wait is not None and last_kline_timestamp is not None and current_kline_ts_wait > last_kline_timestamp:
                                print("\n" + " " * 80) # Clear the debug print line
                                print(f"   ✅ New kline detected: {current_kline_ts_wait} (Previous: {last_kline_timestamp})")
                                last_kline_timestamp = current_kline_ts_wait # Update timestamp for the NEXT wait
                                break # Exit wait loop and run trading cycle
                            elif current_kline_ts_wait is None and last_kline_timestamp is not None:
                                # Handle case where data becomes unavailable during wait
                                print("\n   ⚠️ WARN: Could not fetch latest data during wait. Retrying...")
                            elif last_kline_timestamp is None:
                                # Should have been initialized earlier, but handle just in case
                                print("\n   ⚠️ WARN: last_kline_timestamp is None during wait loop. Attempting to initialize...")
                                if current_kline_ts_wait is not None:
                                    last_kline_timestamp = current_kline_ts_wait
                                    print(f"   Initialized last_kline_timestamp to {last_kline_timestamp}")
                                else:
                                    print(f"   Could not initialize last_kline_timestamp. Still waiting...")


                            time.sleep(5) # Check every 5 seconds
                        # --- End Original Wait Logic ---
                    # --- END MODIFIED Kline Wait Logic ---

                    # Flag that the first cycle attempt has passed
                    first_cycle = False

                    # Now run the cycle using the data updated by WebSocket
                    print(f"\n{'='*70}")
                    print(f"📈 LIVE TRADING CYCLE #{cycle_count} (Kline: {last_kline_timestamp})") # Use the determined timestamp
                    print(f"🎯 Aggressiveness: {bot.aggressiveness.upper()}")
                    print(f"🛡️ Error Status: {bot.error_handler.get_health_status()}")
                    print(f"{'='*70}")

                    bot.run_trading_cycle()

                    # --- ADDED: Periodic ML Outcome Update ---
                    if cycle_count % 10 == 0: # Check every 10 cycles
                        print("💾 Updating ML prediction outcomes from closed trades...")
                        ml_update_thread = threading.Thread(
                            target=bot.strategy_orchestrator.ml_predictor.update_ml_prediction_outcomes,
                            daemon=True
                        )
                        ml_update_thread.start()
                    # --- END ADDED ---
                    # No long sleep needed here, loop will wait for the next kline

                except KeyboardInterrupt:
                    print("\n\n🛑 Bot stopping...")
                    portfolio_value = bot.get_portfolio_value()
                    performance = bot.get_performance_summary()

                    # --- ADD WebSocket Stop ---
                    print("🔌 Stopping WebSocket streams...")
                    bot.data_engine.stop()
                    # -------------------------

                    # Store stop event
                    bot.database.store_system_event( "BOT_STOPPED",
                        { 'final_portfolio_value': portfolio_value, 'total_cycles': cycle_count, 'performance': performance }, "INFO", "System Shutdown")

                    # Send Telegram stop message
                    if bot.telegram_bot:
                        bot.telegram_bot.send_channel_message(
                            f"🛑 <b>TRADING BOT STOPPED</b>\n\n"
                            f"💰 <b>Final Portfolio:</b> ${portfolio_value:,.2f}\n"
                            f"📈 <b>Total Trades:</b> {performance.get('total_trades', 0)}\n"
                            f"🎯 <b>Win Rate:</b> {performance.get('win_rate', 0):.1f}%\n"
                            f"🛡️ <b>Error Count:</b> {bot.error_handler.error_count}\n"
                            f"🕒 <b>Time:</b> {pd.Timestamp.now().strftime('%H:%M:%S')}"
                        )

                    # Save models
                    print("💾 Saving models before exit...")
                    bot.strategy_orchestrator.ml_predictor.save_models()

                    # Close database
                    print("💾 Closing database connection...")
                    bot.database.close()

                    print("✅ Bot stopped gracefully.")
                    break # Exit the main while loop

                except Exception as e:
                    # Main loop error handling
                    error_msg = f"Critical error in main loop: {e}"
                    print(f"\n❌ {error_msg}")
                    if hasattr(bot, 'logger'):
                        bot.logger.critical(error_msg, exc_info=True)
                    else:
                        import traceback
                        traceback.print_exc()

                    if bot: # Ensure bot object exists
                        bot.error_handler.handle_trading_error(e, "ALL", "main_loop_critical")
                        if bot.telegram_bot:
                            bot.telegram_bot.send_channel_message(f"🚨 <b>CRITICAL LOOP ERROR</b>\n\n{error_msg}\n\nCheck logs immediately!")
                    print("🔄 Attempting to recover in 1 minute...")
                    time.sleep(60)

        elif mode_choice == "2": # Backtest mode
            bot.run_backtest_mode(cycles=10) # Example: 10 cycles
            print("✅ Backtest finished. Exiting.")
            # Ensure cleanup even after backtest
            if bot:
                try:
                    print("🔌 Stopping WebSocket streams (if running)...")
                    bot.data_engine.stop()
                    print("💾 Closing database connection...")
                    bot.database.close()
                except Exception as cleanup_e:
                    print(f"⚠️ Error during backtest cleanup: {cleanup_e}")

    except Exception as e:
        # Fatal error handling
        error_msg = f"Fatal error during bot initialization or critical failure: {e}"
        print(f"❌ {error_msg}")
        if bot and hasattr(bot, 'logger'):
            bot.logger.critical(error_msg, exc_info=True)
        else:
            import traceback
            traceback.print_exc()

        if bot and bot.telegram_bot:
            bot.telegram_bot.send_channel_message(f"🚨 <b>FATAL BOT ERROR</b>\n\n{error_msg}\n\nBot will exit.")
        print("Bot cannot continue. Exiting.")
        if bot:
            try:
                bot.data_engine.stop() # Attempt WS stop on fatal error too
                bot.strategy_orchestrator.ml_predictor.save_models()
                bot.database.close()
            except: pass # Ignore errors during emergency shutdown

# This standard Python construct ensures main() runs when the script is executed directly
if __name__ == "__main__":
    main()