import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# --- FIX: Import the specific type for NaT ---
from pandas._libs.tslibs.nattype import NaTType
# --- END FIX ---

# Helper class for JSON serialization of NumPy/Pandas types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Check for NaN specifically, as json doesn't handle it well
            if np.isnan(obj):
                return None # Represent NaN as null in JSON
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
             return bool(obj)
        # --- FIX: Use NaTType instead of the value pd.NaT ---
        elif isinstance(obj, (pd.Timestamp, NaTType)):
             # Convert Timestamp to ISO string, NaTType to None
             return obj.isoformat() if isinstance(obj, pd.Timestamp) else None
        # --- END FIX ---
        elif isinstance(obj, datetime): # Add handling for standard datetime objects
             return obj.isoformat()
        elif pd.isna(obj): # General check for other pandas NA types
             return None
        # Let the base class default method raise the TypeError for unsupported types
        return super(NpEncoder, self).default(obj)

class TradingDatabase:
    """
    Enhanced Data persistence layer for trading bot with ML tracking
    Stores trades, performance metrics, system events, and ML model data
    """

    def __init__(self, db_path: str = "trading_data.db"):
        self.logger = logging.getLogger('TradingDatabase')
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self._init_database()

        print(f"💾 Enhanced Trading Database initialized: {db_path}")

    def _init_database(self):
        """Initialize database with required tables including ML tracking"""
        try:
            self.logger.info("Initializing enhanced trading database...")
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False,
                                             detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) # Added detect_types
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access

            self.cursor = self.connection.cursor()  # Initialize cursor

            # Trades table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,          -- FIX #3: Store timestamp as TEXT (ISO format)
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    position_size_usdt REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    exit_reason TEXT,
                    pnl_usdt REAL,
                    pnl_percent REAL,
                    confidence REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    risk_reward_ratio REAL NOT NULL,
                    aggressiveness TEXT NOT NULL,
                    order_id TEXT,
                    success TEXT NOT NULL,            -- FIX #3: Store boolean as TEXT ('True'/'False')
                    error_message TEXT,
                    trend_score REAL DEFAULT 0,
                    mr_score REAL DEFAULT 0,
                    breakout_score REAL DEFAULT 0,
                    ml_score REAL DEFAULT 0,
                    mtf_score REAL DEFAULT 0,
                    ml_prediction_details TEXT,       -- << NEW: Store ML prediction JSON
                    outcome_updated INTEGER DEFAULT 0,  -- << NEW: Flag (0=not updated, 1=updated)
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP -- Store as TEXT (ISO format)
                )
            ''')

            # Performance metrics table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,               -- Store date as TEXT (YYYY-MM-DD)
                    portfolio_value REAL NOT NULL,
                    daily_pnl_percent REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    avg_risk_reward REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, -- Store as TEXT (ISO format)
                    UNIQUE(date)
                )
            ''')

            # System events table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,          -- Store timestamp as TEXT (ISO format)
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,         -- Store JSON as TEXT
                    severity TEXT NOT NULL,           -- Original column name
                    context TEXT,                     -- Original column name
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP -- Store as TEXT (ISO format)
                )
            ''')

            # ML model performance table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,          -- Store timestamp as TEXT (ISO format)
                    symbol TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    rf_accuracy REAL,
                    gb_accuracy REAL,
                    rf_precision REAL,
                    gb_precision REAL,
                    rf_recall REAL,
                    gb_recall REAL,
                    rf_f1 REAL,
                    gb_f1 REAL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    model_version TEXT,
                    training_date TEXT NOT NULL,      -- Store date as TEXT (YYYY-MM-DD)
                    training_bars_used INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP -- Store as TEXT (ISO format)
                )
            ''')

            # Market data table (User's Original Schema - unchanged)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,          -- Store timestamp as TEXT (ISO format)
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, -- Store as TEXT (ISO format)
                    UNIQUE(symbol, timestamp, timeframe)
                )
            ''')

            # Prediction quality table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    actual INTEGER,
                    confidence REAL NOT NULL,
                    correct TEXT,                     -- Store boolean as TEXT ('True'/'False'/None)
                    timestamp TEXT NOT NULL,          -- Store timestamp as TEXT (ISO format)
                    model_used TEXT,
                    features_used TEXT,               -- Store JSON as TEXT
                    ensemble_used TEXT DEFAULT 'False', -- Store boolean as TEXT
                    feature_count INTEGER DEFAULT 0,
                    raw_prediction REAL DEFAULT 0,   -- Changed type to REAL
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP -- Store as TEXT (ISO format)
                )
            ''')

            # Feature importance table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance REAL NOT NULL,
                    ranking INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,          -- Store timestamp as TEXT (ISO format)
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, -- Store as TEXT (ISO format)
                    UNIQUE(symbol, model_type, feature_name, timestamp)
                )
            ''')

            # Model training history table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_date TEXT NOT NULL,      -- Store timestamp as TEXT (ISO format)
                    training_duration_seconds REAL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    final_accuracy REAL,
                    final_precision REAL,
                    final_recall REAL,
                    final_f1 REAL,
                    feature_count INTEGER,
                    parameters_used TEXT,             -- Store JSON as TEXT
                    training_bars_used INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP -- Store as TEXT (ISO format)
                )
            ''')

            # Model drift detection table (User's Original Schema)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_drift_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,          -- Store timestamp as TEXT (ISO format)
                    accuracy_drift REAL,
                    feature_drift REAL,
                    prediction_drift REAL,
                    drift_detected TEXT,              -- Store boolean as TEXT ('True'/'False')
                    drift_reason TEXT,
                    retraining_recommended TEXT,      -- Store boolean as TEXT ('True'/'False')
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP -- Store as TEXT (ISO format)
                )
            ''')

            self.connection.commit()
            self.logger.info("Enhanced database tables initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced database: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            raise

    def store_trade(self, trade_data: Dict[str, Any]) -> bool:
            """
            Store a trade record in the database including individual strategy scores
            and associated ML prediction details.
            """
            try:
                cursor = self.connection.cursor()

                # --- Convert Timestamp to ISO String ---
                ts_input = trade_data.get('timestamp', datetime.now())
                db_timestamp = ts_input.isoformat() if isinstance(ts_input, (datetime, pd.Timestamp)) else datetime.now().isoformat()

                # --- Convert Boolean to String ---
                db_success = str(trade_data.get('success', False))

                # --- Convert ML prediction details to JSON ---
                ml_details = trade_data.get('ml_prediction') # Get the raw dict
                db_ml_prediction_details = json.dumps(ml_details, cls=NpEncoder) if ml_details else None

                # --- Prepare data tuple with explicit type casting ---
                data_tuple = (
                    db_timestamp,
                    str(trade_data.get('symbol', '')),
                    str(trade_data.get('action', '')),
                    float(trade_data.get('quantity', 0.0) or 0.0),
                    float(trade_data.get('entry_price', 0.0) or 0.0),
                    float(trade_data.get('exit_price', 0.0) or 0.0) if trade_data.get('exit_price') is not None else None,
                    float(trade_data.get('position_size_usdt', 0.0) or 0.0),
                    float(trade_data.get('stop_loss', 0.0) or 0.0),
                    float(trade_data.get('take_profit', 0.0) or 0.0),
                    trade_data.get('exit_reason'), # Keep as TEXT
                    float(trade_data.get('pnl_usdt', 0.0) or 0.0) if trade_data.get('pnl_usdt') is not None else None,
                    float(trade_data.get('pnl_percent', 0.0) or 0.0) if trade_data.get('pnl_percent') is not None else None,
                    float(trade_data.get('confidence', 0.0) or 0.0),
                    float(trade_data.get('composite_score', 0.0) or 0.0),
                    float(trade_data.get('risk_reward_ratio', 0.0) or 0.0),
                    str(trade_data.get('aggressiveness', 'moderate')),
                    trade_data.get('order_id'), # Keep as TEXT
                    db_success, # Use converted string boolean
                    trade_data.get('error_message'), # Keep as TEXT
                    float(trade_data.get('trend_score', 0.0) or 0.0),
                    float(trade_data.get('mr_score', 0.0) or 0.0),
                    float(trade_data.get('breakout_score', 0.0) or 0.0),
                    float(trade_data.get('ml_score', 0.0) or 0.0),
                    float(trade_data.get('mtf_score', 0.0) or 0.0),
                    db_ml_prediction_details # Add the ML details JSON string
                )

                # Ensure SQL matches schema and tuple order (25 columns/placeholders now)
                sql = '''
                    INSERT INTO trades (
                        timestamp, symbol, action, quantity, entry_price, exit_price,
                        position_size_usdt, stop_loss, take_profit, exit_reason,
                        pnl_usdt, pnl_percent, confidence, composite_score,
                        risk_reward_ratio, aggressiveness, order_id, success, error_message,
                        trend_score, mr_score, breakout_score, ml_score, mtf_score,
                        ml_prediction_details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''' # 25 columns, 25 placeholders

                cursor.execute(sql, data_tuple)

                self.connection.commit()
                trade_id = cursor.lastrowid
                self.logger.info(f"Stored trade #{trade_id} for {trade_data.get('symbol')} with ML details")
                return True

            except sqlite3.Error as e:
                self.logger.error(f"SQLite error storing trade: {e} | Data: {trade_data}", exc_info=True)
                if self.connection: self.connection.rollback()
                return False
            except Exception as e:
                self.logger.error(f"Failed to store trade: {e} | Data: {trade_data}", exc_info=True)
                if self.connection: self.connection.rollback() # Rollback on general errors too
                return False

    def update_trade_outcome_updated_flag(self, trade_id: int) -> bool:
            """ Mark a trade as having its outcome processed for ML feedback. """
            try:
                cursor = self.connection.cursor()
                cursor.execute('UPDATE trades SET outcome_updated = 1 WHERE id = ?', (trade_id,))
                self.connection.commit()
                return cursor.rowcount > 0
            except Exception as e:
                self.logger.error(f"Failed to update outcome_updated flag for trade id {trade_id}: {e}", exc_info=True)
                if self.connection: self.connection.rollback()
                return False

    def update_trade_exit(self, trade_id: int, exit_price: float,
                          pnl_usdt: float, pnl_percent: float, exit_reason: str) -> bool:
        """ Update trade record with exit information (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()

            cursor.execute('''
                UPDATE trades
                SET exit_price = ?, pnl_usdt = ?, pnl_percent = ?, exit_reason = ?
                WHERE id = ?
            ''', (exit_price, pnl_usdt, pnl_percent, exit_reason, trade_id))

            self.connection.commit()
            if cursor.rowcount > 0:
                self.logger.info(f"Updated trade #{trade_id} exit: {exit_reason}")
                return True
            else:
                 self.logger.warning(f"Could not find trade with id {trade_id} to update exit.")
                 return False

        except Exception as e:
            self.logger.error(f"Failed to update trade exit for id {trade_id}: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False

    # --- FIX #3: Date/Type Conversion ---
    def store_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """ Store daily performance metrics (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()

            # Use YYYY-MM-DD date string
            today_str = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('SELECT id FROM performance_metrics WHERE date = ?', (today_str,))
            existing = cursor.fetchone()

            # Prepare values with casting
            values_tuple = (
                float(metrics.get('portfolio_value', 0.0) or 0.0),
                float(metrics.get('daily_pnl_percent', 0.0) or 0.0),
                int(metrics.get('total_trades', 0) or 0),
                int(metrics.get('winning_trades', 0) or 0),
                float(metrics.get('win_rate', 0.0) or 0.0),
                float(metrics.get('avg_confidence', 0.0) or 0.0),
                float(metrics.get('avg_risk_reward', 0.0) or 0.0),
                float(metrics.get('max_drawdown', 0.0) or 0.0),
                float(metrics.get('sharpe_ratio', 0.0) or 0.0) if metrics.get('sharpe_ratio') is not None else None,
                today_str # Date string for WHERE or VALUES clause
            )

            if existing:
                # Update existing entry
                sql = '''
                    UPDATE performance_metrics
                    SET portfolio_value = ?, daily_pnl_percent = ?, total_trades = ?,
                        winning_trades = ?, win_rate = ?, avg_confidence = ?,
                        avg_risk_reward = ?, max_drawdown = ?, sharpe_ratio = ?
                    WHERE date = ?
                '''
                cursor.execute(sql, values_tuple)
            else:
                # Insert new entry
                sql = '''
                    INSERT INTO performance_metrics (
                        portfolio_value, daily_pnl_percent, total_trades,
                        winning_trades, win_rate, avg_confidence, avg_risk_reward,
                        max_drawdown, sharpe_ratio, date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.execute(sql, values_tuple)

            self.connection.commit()
            self.logger.info(f"Stored performance metrics for {today_str}")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing performance metrics: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store performance metrics: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False

    # --- FIX #2: Use NpEncoder ---
    # --- FIX #3: Timestamp/Column Names ---
    def store_system_event(self, event_type: str, event_data: Dict,
                           severity: str = "INFO", context: str = None) -> bool:
        """ Store system event using original column names """
        try:
            cursor = self.connection.cursor()

            # Use NpEncoder for JSON data
            event_data_json = json.dumps(event_data, cls=NpEncoder)
            # Use ISO string for timestamp
            db_timestamp = datetime.now().isoformat()

            cursor.execute('''
                INSERT INTO system_events (timestamp, event_type, event_data, severity, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                db_timestamp,
                event_type,
                event_data_json,
                severity.upper(), # Match original column name
                context # Match original column name
            ))

            self.connection.commit()
            self.logger.debug(f"Stored system event: {event_type}")
            return True

        except TypeError as e:
             self.logger.error(f"JSON serialization error storing system event: {e}. Data: {event_data}", exc_info=True)
             if self.connection: self.connection.rollback()
             return False
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing system event: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store system event: {e}", exc_info=True)
            if self.connection: self.connection.rollback() # Rollback on general errors
            return False

    # --- FIX #3: Timestamp/Type Conversion ---
    def store_ml_model_performance(self, symbol: str, metrics: Dict[str, Any]) -> bool:
        """ Store ML model performance metrics (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()

            db_timestamp = datetime.now().isoformat()
            training_dt = metrics.get('training_date', datetime.now())
            # Convert training_date to YYYY-MM-DD string
            db_training_date = training_dt.strftime('%Y-%m-%d') if isinstance(training_dt, (datetime, pd.Timestamp)) else datetime.now().strftime('%Y-%m-%d')

            values_tuple = (
                db_timestamp,
                symbol,
                float(metrics.get('accuracy', 0.0) or 0.0),
                float(metrics.get('precision', 0.0) or 0.0),
                float(metrics.get('recall', 0.0) or 0.0),
                float(metrics.get('f1_score', 0.0) or 0.0),
                float(metrics.get('rf_accuracy', 0.0) or 0.0),
                float(metrics.get('gb_accuracy', 0.0) or 0.0),
                float(metrics.get('rf_precision', 0.0) or 0.0),
                float(metrics.get('gb_precision', 0.0) or 0.0),
                float(metrics.get('rf_recall', 0.0) or 0.0),
                float(metrics.get('gb_recall', 0.0) or 0.0),
                float(metrics.get('rf_f1', 0.0) or 0.0),
                float(metrics.get('gb_f1', 0.0) or 0.0),
                int(metrics.get('training_samples', 0) or 0),
                int(metrics.get('test_samples', 0) or 0),
                metrics.get('model_version'),
                db_training_date,
                int(metrics.get('training_bars_used', 0) or 0)
            )

            # Ensure SQL matches original schema and tuple order
            sql = '''
                INSERT INTO ml_model_performance (
                    timestamp, symbol, accuracy, precision, recall, f1_score,
                    rf_accuracy, gb_accuracy, rf_precision, gb_precision,
                    rf_recall, gb_recall, rf_f1, gb_f1, training_samples,
                    test_samples, model_version, training_date, training_bars_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''' # 19 columns, 19 placeholders

            cursor.execute(sql, values_tuple)

            self.connection.commit()
            self.logger.info(f"Stored ML performance for {symbol}")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing ML performance: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store ML performance: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False

    # --- FIX #2: Use NpEncoder ---
    # --- FIX #3: Timestamp/Type/Boolean Conversion ---
    def store_prediction_quality(self, symbol: str, prediction: int, actual: int = None,
                                 confidence: float = 0, model_used: str = None,
                                 features_used: List[str] = None, ensemble_used: bool = False,
                                 feature_count: int = 0, raw_prediction: Any = 0) -> bool: # Allow Any for raw
        """ Store individual prediction quality record (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()

            db_timestamp = datetime.now().isoformat()
            db_correct = None
            if actual is not None:
                # Use explicit int comparison
                db_correct = str(int(prediction) == int(actual)) # Store as 'True' or 'False'

            # Use NpEncoder for features_used list
            features_json = json.dumps(features_used, cls=NpEncoder) if features_used else None
            db_ensemble_used = str(ensemble_used) # Store boolean as 'True' or 'False'

            # Handle raw_prediction type - store as REAL
            db_raw_prediction = float(raw_prediction) if isinstance(raw_prediction, (int, float, np.number)) else 0.0


            values_tuple = (
                symbol,
                int(prediction),
                int(actual) if actual is not None else None,
                float(confidence or 0.0),
                db_correct, # Use string boolean or None
                db_timestamp,
                model_used,
                features_json, # Use JSON string
                db_ensemble_used, # Use string boolean
                int(feature_count or 0),
                db_raw_prediction # Store as REAL
            )

            # Ensure SQL matches original schema and tuple order
            sql = '''
                INSERT INTO prediction_quality
                (symbol, prediction, actual, confidence, correct, timestamp, model_used, features_used, ensemble_used, feature_count, raw_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''' # 11 columns, 11 placeholders

            cursor.execute(sql, values_tuple)

            self.connection.commit()
            self.logger.debug(f"Stored prediction quality for {symbol}")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing prediction quality: {e} | Data: {locals()}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store prediction quality: {e} | Data: {locals()}", exc_info=True)
            if self.connection: self.connection.rollback() # Rollback on general errors
            return False

    # --- FIX #3: Timestamp Conversion ---
    def store_feature_importance(self, symbol: str, model_type: str,
                                 feature_importance: Dict[str, float]) -> bool:
        """ Store feature importance for a model (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()
            timestamp_str = datetime.now().isoformat()

            # Store top 20 features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

            for rank, (feature_name, importance) in enumerate(sorted_features):
                cursor.execute('''
                    INSERT OR IGNORE INTO feature_importance
                    (symbol, model_type, feature_name, importance, ranking, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    model_type,
                    feature_name,
                    float(importance), # Ensure float
                    rank + 1,
                    timestamp_str # Use ISO string
                ))

            self.connection.commit()
            self.logger.info(f"Stored feature importance for {symbol} ({model_type})")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing feature importance: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store feature importance: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False

    # --- FIX #2: Use NpEncoder ---
    # --- FIX #3: Timestamp/Type Conversion ---
    def store_model_training_history(self, symbol: str, training_data: Dict[str, Any]) -> bool:
        """ Store model training history (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()

            training_dt = training_data.get('training_date', datetime.now())
            db_training_date = training_dt.isoformat() if isinstance(training_dt, (datetime, pd.Timestamp)) else datetime.now().isoformat()

            # Use NpEncoder for parameters_used dictionary
            params_json = json.dumps(training_data.get('parameters_used', {}), cls=NpEncoder)

            values_tuple = (
                symbol,
                training_data.get('model_version'),
                db_training_date, # Use ISO string
                float(training_data.get('training_duration_seconds', 0.0) or 0.0),
                int(training_data.get('training_samples', 0) or 0),
                int(training_data.get('test_samples', 0) or 0),
                float(training_data.get('final_accuracy', 0.0) or 0.0),
                float(training_data.get('final_precision', 0.0) or 0.0),
                float(training_data.get('final_recall', 0.0) or 0.0),
                float(training_data.get('final_f1', 0.0) or 0.0),
                int(training_data.get('feature_count', 0) or 0),
                params_json, # Use JSON string
                int(training_data.get('training_bars_used', 0) or 0)
            )

            # Ensure SQL matches original schema and tuple order
            sql = '''
                INSERT INTO model_training_history (
                    symbol, model_version, training_date, training_duration_seconds,
                    training_samples, test_samples, final_accuracy, final_precision,
                    final_recall, final_f1, feature_count, parameters_used, training_bars_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''' # 13 columns, 13 placeholders

            cursor.execute(sql, values_tuple)

            self.connection.commit()
            self.logger.info(f"Stored training history for {symbol}")
            return True

        except TypeError as e:
             self.logger.error(f"JSON/Type error storing training history: {e}. Data: {training_data}", exc_info=True)
             if self.connection: self.connection.rollback()
             return False
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing training history: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store training history: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False

    # --- FIX #3: Timestamp/Boolean Conversion ---
    def store_model_drift_detection(self, symbol: str, drift_data: Dict[str, Any]) -> bool:
        """ Store model drift detection results (Matches Original Schema) """
        try:
            cursor = self.connection.cursor()
            db_timestamp = datetime.now().isoformat()
            db_drift_detected = str(drift_data.get('drift_detected', False))
            db_retraining_rec = str(drift_data.get('retraining_recommended', False))

            values_tuple = (
                symbol,
                db_timestamp, # Use ISO string
                float(drift_data.get('accuracy_drift', 0.0) or 0.0),
                float(drift_data.get('feature_drift', 0.0) or 0.0),
                float(drift_data.get('prediction_drift', 0.0) or 0.0),
                db_drift_detected, # Use string boolean
                drift_data.get('drift_reason', ''),
                db_retraining_rec # Use string boolean
            )

            # Ensure SQL matches original schema and tuple order
            sql = '''
                INSERT INTO model_drift_detection (
                    symbol, timestamp, accuracy_drift, feature_drift, prediction_drift,
                    drift_detected, drift_reason, retraining_recommended
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''' # 8 columns, 8 placeholders

            cursor.execute(sql, values_tuple)

            self.connection.commit()
            self.logger.info(f"Stored drift detection for {symbol}")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"SQLite error storing drift detection: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Failed to store drift detection: {e}", exc_info=True)
            if self.connection: self.connection.rollback()
            return False

    # --- Retrieval Methods (Adjusted for TEXT Timestamps/Dates/Booleans) ---

    def get_ml_model_performance(self, symbol: str = None, limit: int = 50) -> list:
        """Get ML model performance history"""
        try:
            # No change needed if using sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            # and `detect_types` was added during connection. Otherwise, manual parsing might be needed.
            # Assuming detect_types works or data is processed later.
            if symbol:
                query = """
                    SELECT * FROM ml_model_performance
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                self.cursor.execute(query, (symbol, limit))
            else:
                query = """
                    SELECT * FROM ml_model_performance
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                self.cursor.execute(query, (limit,))

            results = self.cursor.fetchall()
            performance = [dict(row) for row in results] # Convert rows to dicts
            return performance
        except Exception as e:
            self.logger.error(f"Error getting ML performance: {e}", exc_info=True)
            return []

    def get_prediction_quality(self, symbol: str = None, limit: int = 100) -> list:
        """Get prediction quality data"""
        try:
            # Assuming detect_types handles timestamp conversion.
            # Need to manually handle JSON and boolean strings.
            if symbol:
                query = """
                SELECT * FROM prediction_quality
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """
                self.cursor.execute(query, (symbol, limit))
            else:
                query = """
                SELECT * FROM prediction_quality
                ORDER BY timestamp DESC
                LIMIT ?
                """
                self.cursor.execute(query, (limit,))

            results = self.cursor.fetchall()

            predictions = []
            for row in results:
                row_dict = dict(row)
                # Parse JSON string for features
                features_str = row_dict.get('features_used')
                row_dict['features_used'] = json.loads(features_str) if features_str else []
                # Convert boolean strings back to boolean
                row_dict['ensemble_used'] = row_dict.get('ensemble_used') == 'True'
                correct_str = row_dict.get('correct')
                row_dict['correct'] = correct_str == 'True' if correct_str is not None else None
                predictions.append(row_dict)

            return predictions
        except Exception as e:
            self.logger.error(f"Error getting prediction quality: {e}", exc_info=True)
            return []

    def get_feature_importance(self, symbol: str) -> dict:
        """Get latest feature importance data (Matches Original Logic)"""
        try:
            # Fetch latest timestamp first
            self.cursor.execute("""
                SELECT MAX(timestamp) as latest_ts
                FROM feature_importance
                WHERE symbol = ?
            """, (symbol,))
            latest_ts_row = self.cursor.fetchone()
            if not latest_ts_row or not latest_ts_row['latest_ts']:
                return {'features': [], 'rf_importance': [], 'gb_importance': []}
            latest_ts = latest_ts_row['latest_ts']

            # Fetch data only for the latest timestamp
            query = """
            SELECT model_type, feature_name, importance
            FROM feature_importance
            WHERE symbol = ? AND timestamp = ?
            ORDER BY ranking ASC
            """
            self.cursor.execute(query, (symbol, latest_ts))
            results = self.cursor.fetchall()

            importance_data = {'features': [], 'rf_importance': [], 'gb_importance': []}
            rf_data = {}
            gb_data = {}
            feature_set = set()

            for row in results:
                feature = row['feature_name']
                importance = row['importance']
                feature_set.add(feature)
                if row['model_type'] == 'rf':
                    rf_data[feature] = importance
                elif row['model_type'] == 'gb':
                    gb_data[feature] = importance

            # Order features alphabetically for consistency
            importance_data['features'] = sorted(list(feature_set))

            # Align importance scores
            for feature in importance_data['features']:
                importance_data['rf_importance'].append(rf_data.get(feature, 0.0))
                importance_data['gb_importance'].append(gb_data.get(feature, 0.0))

            return importance_data
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}", exc_info=True)
            return {'features': [], 'rf_importance': [], 'gb_importance': []}

    def get_model_training_history(self, symbol: str = None, limit: int = 50) -> list:
        """Get model training history for a symbol"""
        try:
            # Assuming detect_types handles timestamp. Need to parse JSON.
            if symbol:
                query = """
                SELECT * FROM model_training_history
                WHERE symbol = ?
                ORDER BY training_date DESC
                LIMIT ?
                """
                self.cursor.execute(query, (symbol, limit))
            else:
                query = """
                SELECT * FROM model_training_history
                ORDER BY training_date DESC
                LIMIT ?
                """
                self.cursor.execute(query, (limit,))

            results = self.cursor.fetchall()

            history = []
            for row in results:
                 row_dict = dict(row)
                 # Parse parameters JSON
                 params_str = row_dict.get('parameters_used')
                 row_dict['parameters_used'] = json.loads(params_str) if params_str else {}
                 history.append(row_dict)

            return history
        except Exception as e:
            self.logger.error(f"Error getting model training history: {e}", exc_info=True)
            return []

    def get_model_drift_history(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        """ Get model drift detection history (Adjusted for TEXT Timestamps/Booleans) """
        try:
            start_date_str = (datetime.now() - timedelta(days=days)).isoformat()

            query = '''
                SELECT * FROM model_drift_detection
                WHERE timestamp >= ?
            '''
            params = [start_date_str]

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)

            query += ' ORDER BY timestamp DESC'

            # Use pandas read_sql_query for easy DataFrame creation and parsing
            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['timestamp', 'created_at'])

            # Convert boolean strings back to boolean if needed for analysis
            if not df.empty:
                 if 'drift_detected' in df.columns:
                      df['drift_detected'] = df['drift_detected'].apply(lambda x: x == 'True' if x is not None else None)
                 if 'retraining_recommended' in df.columns:
                      df['retraining_recommended'] = df['retraining_recommended'].apply(lambda x: x == 'True' if x is not None else None)

            return df

        except Exception as e:
            self.logger.error(f"Failed to get drift history: {e}", exc_info=True)
            return pd.DataFrame()

    def get_historical_trades(self, days: int = 30, symbol: str = None) -> pd.DataFrame:
        """Get historical trades from database (Adjusted for TEXT Timestamps/Booleans)"""
        try:
            start_date_str = (datetime.now() - timedelta(days=days)).isoformat()

            query = '''
                SELECT * FROM trades
                WHERE timestamp >= ?
            '''
            params = [start_date_str]

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)

            query += ' ORDER BY timestamp DESC'

            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['timestamp', 'created_at'])

            # Convert success boolean string back
            if not df.empty and 'success' in df.columns:
                 df['success'] = df['success'].apply(lambda x: x == 'True' if x is not None else None)

            self.logger.debug(f"Retrieved {len(df)} trades from database")
            return df

        except Exception as e:
            self.logger.error(f"Failed to get historical trades: {e}", exc_info=True)
            return pd.DataFrame()

    def get_performance_history(self, days: int = 90) -> pd.DataFrame:
        """Get performance metrics history (Adjusted for TEXT Dates)"""
        try:
            start_date_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            query = '''
                SELECT * FROM performance_metrics
                WHERE date >= ?
                ORDER BY date DESC
            '''

            # Parse 'date' column as datetime objects when reading
            df = pd.read_sql_query(query, self.connection, params=[start_date_str], parse_dates={'date': '%Y-%m-%d', 'created_at': 'iso'})

            self.logger.debug(f"Retrieved {len(df)} performance records")
            return df

        except Exception as e:
            self.logger.error(f"Failed to get performance history: {e}", exc_info=True)
            return pd.DataFrame()

    def get_system_events(self, event_type: str = None, severity: str = None,
                          hours: int = 24) -> pd.DataFrame:
        """Get system events for monitoring (Adjusted for TEXT Timestamps)"""
        try:
            start_time_str = (datetime.now() - timedelta(hours=hours)).isoformat()

            query = 'SELECT * FROM system_events WHERE timestamp >= ?'
            params = [start_time_str]

            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)

            if severity:
                query += ' AND severity = ?'
                params.append(severity.upper()) # Match stored severity

            query += ' ORDER BY timestamp DESC'

            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['timestamp', 'created_at'])

            # Parse event_data JSON
            if not df.empty and 'event_data' in df.columns:
                 # Use a safe parsing function
                 def safe_json_loads(x):
                      try:
                           return json.loads(x) if isinstance(x, str) else x
                      except (json.JSONDecodeError, TypeError):
                           return None # Or return the original string x
                 df['event_data'] = df['event_data'].apply(safe_json_loads)

            self.logger.debug(f"Retrieved {len(df)} system events")
            return df

        except Exception as e:
            self.logger.error(f"Failed to get system events: {e}", exc_info=True)
            return pd.DataFrame()

    # --- Original get_trading_statistics, get_confusion_matrix, etc. ---
    # These methods primarily query and calculate based on retrieved data.
    # They should work fine as long as the data retrieval methods
    # (`get_historical_trades`, `get_prediction_quality`) correctly handle
    # the TEXT timestamp/boolean conversions when creating DataFrames.
    # Minor adjustments might be needed if calculations relied on native types.

    def get_trading_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive trading statistics (Adjusted for TEXT success)"""
        try:
            trades_df = self.get_historical_trades(days=days) # This now returns df with parsed dates/bools

            if trades_df.empty or 'pnl_percent' not in trades_df.columns:
                 return {'message': f'No trades with PnL found for {days} days'}

            # Filter for trades with PnL (implicitly closed)
            valid_trades = trades_df.dropna(subset=['pnl_percent'])
            if valid_trades.empty:
                return {'message': f'No closed trades with PnL found for {days} days'}

            stats = {}
            stats['total_trades'] = len(valid_trades)
            stats['successful_executions'] = int(valid_trades['success'].sum()) # Summing True values
            stats['avg_pnl_percent'] = valid_trades['pnl_percent'].mean()
            stats['avg_confidence'] = valid_trades['confidence'].mean()
            stats['avg_risk_reward'] = valid_trades['risk_reward_ratio'].mean()
            stats['winning_trades'] = len(valid_trades[valid_trades['pnl_percent'] > 0])
            stats['losing_trades'] = len(valid_trades[valid_trades['pnl_percent'] < 0])
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0

            # Best/Worst trade logic remains the same as it uses the DataFrame
            best_trade_row = valid_trades.loc[valid_trades['pnl_percent'].idxmax()] if stats['total_trades'] > 0 else None
            stats['best_trade'] = best_trade_row.to_dict() if best_trade_row is not None else None

            worst_trade_row = valid_trades.loc[valid_trades['pnl_percent'].idxmin()] if stats['total_trades'] > 0 else None
            stats['worst_trade'] = worst_trade_row.to_dict() if worst_trade_row is not None else None

            # Performance by symbol logic remains the same
            symbol_perf = valid_trades.groupby('symbol').agg(
                 trade_count=('id', 'count'),
                 avg_pnl_percent=('pnl_percent', 'mean'),
                 winning_trades=('pnl_percent', lambda x: (x > 0).sum())
            ).reset_index()
            symbol_perf['win_rate'] = (symbol_perf['winning_trades'] / symbol_perf['trade_count']) * 100
            stats['symbol_performance'] = symbol_perf.to_dict('records')


            # Add ML statistics if available
            try:
                ml_stats = self.get_prediction_quality_stats(days=days) # This needs check
                stats['prediction_quality'] = ml_stats
            except Exception as ml_e:
                self.logger.warning(f"Could not calculate prediction quality stats: {ml_e}")
                stats['prediction_quality'] = {}

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get trading statistics: {e}", exc_info=True)
            return {}

    def get_prediction_quality_stats(self, symbol: str = None, days: int = 7) -> Dict[str, Any]:
         """ Get prediction quality statistics (Adjusted for TEXT correct) """
         try:
             start_date_str = (datetime.now() - timedelta(days=days)).isoformat()

             # Query uses the TEXT representation of boolean
             query = '''
                 SELECT
                     symbol,
                     COUNT(*) as total_predictions,
                     SUM(CASE WHEN correct = 'True' THEN 1 ELSE 0 END) as correct_predictions,
                     AVG(confidence) as avg_confidence,
                     AVG(CASE WHEN correct = 'True' THEN confidence ELSE NULL END) as avg_correct_confidence,
                     AVG(CASE WHEN correct = 'False' THEN confidence ELSE NULL END) as avg_incorrect_confidence
                 FROM prediction_quality
                 WHERE timestamp >= ? AND correct IS NOT NULL
             '''
             params = [start_date_str]

             if symbol:
                 query += ' AND symbol = ?'
                 params.append(symbol)

             query += ' GROUP BY symbol'

             cursor = self.connection.cursor()
             cursor.execute(query, params)

             stats = {}
             for row in cursor.fetchall():
                 row_dict = dict(row)
                 symbol_key = row_dict['symbol']
                 total_preds = row_dict['total_predictions']
                 correct_preds = row_dict['correct_predictions']
                 stats[symbol_key] = {
                     'accuracy': (correct_preds / total_preds * 100) if total_preds > 0 else 0,
                     'total_predictions': total_preds,
                     'correct_predictions': correct_preds,
                     'avg_confidence': row_dict['avg_confidence'] or 0,
                     'avg_correct_confidence': row_dict['avg_correct_confidence'] or 0,
                     'avg_incorrect_confidence': row_dict['avg_incorrect_confidence'] or 0
                 }

             return stats

         except Exception as e:
             self.logger.error(f"Failed to get prediction quality stats: {e}", exc_info=True)
             return {}

    def get_confusion_matrix(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """ Get data for a 3x3 confusion matrix (Adjusted for TEXT correct) """
        try:
            start_date_str = (datetime.now() - timedelta(days=days)).isoformat()

            query = '''
                SELECT
                    prediction,
                    actual,
                    COUNT(*) as count
                FROM prediction_quality
                WHERE timestamp >= ? AND actual IS NOT NULL AND correct IS NOT NULL
            '''
            params = [start_date_str]

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)

            query += ' GROUP BY prediction, actual'

            cursor = self.connection.cursor()
            cursor.execute(query, params)

            matrix_data = [dict(row) for row in cursor.fetchall()]
            return matrix_data

        except Exception as e:
            self.logger.error(f"Failed to get confusion matrix data: {e}", exc_info=True)
            return []

    def get_confidence_distribution(self, symbol: str = None, days: int = 7) -> Dict[str, List[float]]:
         """ Get confidence scores grouped by correct/incorrect predictions (Adjusted for TEXT correct) """
         try:
             start_date_str = (datetime.now() - timedelta(days=days)).isoformat()

             query = '''
                 SELECT
                     confidence,
                     correct
                 FROM prediction_quality
                 WHERE timestamp >= ? AND actual IS NOT NULL AND correct IS NOT NULL
             '''
             params = [start_date_str]

             if symbol:
                 query += ' AND symbol = ?'
                 params.append(symbol)

             df = pd.read_sql_query(query, self.connection, params=params)

             if df.empty:
                 return {'correct': [], 'incorrect': []}

             # Filter based on the TEXT value of 'correct'
             distribution = {
                 'correct': df[df['correct'] == 'True']['confidence'].tolist(),
                 'incorrect': df[df['correct'] == 'False']['confidence'].tolist()
             }

             return distribution

         except Exception as e:
             self.logger.error(f"Failed to get confidence distribution: {e}", exc_info=True)
             return {'correct': [], 'incorrect': []}


    def store_market_data(self, symbol: str, data: pd.DataFrame, timeframe: str):
         """Store market data (Adjusted for TEXT Timestamps)"""
         try:
             cursor = self.connection.cursor()

             data_to_insert = []
             for idx, row in data.iterrows():
                  # Ensure index is a timestamp, convert to ISO string
                  timestamp_str = idx.isoformat() if isinstance(idx, (datetime, pd.Timestamp)) else datetime.now().isoformat()
                  data_to_insert.append((
                       symbol,
                       timestamp_str,
                       float(row['open']), float(row['high']), float(row['low']),
                       float(row['close']), float(row['volume']),
                       timeframe
                  ))

             if data_to_insert:
                  cursor.executemany('''
                       INSERT OR IGNORE INTO market_data
                       (symbol, timestamp, open, high, low, close, volume, timeframe)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                  ''', data_to_insert)

                  self.connection.commit()
                  self.logger.debug(f"Stored {len(data_to_insert)} market data records for {symbol} ({timeframe})")

         except Exception as e:
             self.logger.error(f"Failed to store market data: {e}", exc_info=True)
             if self.connection: self.connection.rollback()


    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data (Adjusted for TEXT Timestamps/Dates)"""
        try:
            # Use ISO format strings for comparison with TEXT columns
            cutoff_datetime_str = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            ml_cutoff_str = (datetime.now() - timedelta(days=days_to_keep * 2)).isoformat()
            cutoff_date_str = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d') # For date column

            cursor = self.connection.cursor()
            deleted_counts = {}

            tables_ts = ['trades', 'system_events', 'market_data', 'feature_importance',
                         'model_training_history', 'model_drift_detection', 'ml_model_performance',
                         'prediction_quality']
            for table in tables_ts:
                 cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff_datetime_str if table != 'model_training_history' else ml_cutoff_str,)) # Use longer cutoff for training history? Or consistent? Let's use ml_cutoff_str for all ML tables
                 deleted_counts[table] = cursor.rowcount

            # Delete old performance metrics by date string
            cursor.execute('DELETE FROM performance_metrics WHERE date < ?', (cutoff_date_str,))
            deleted_counts['performance_metrics'] = cursor.rowcount

            self.connection.commit()

            self.logger.info(f"Database cleanup completed: {deleted_counts}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}", exc_info=True)
            if self.connection: self.connection.rollback()

    def close(self):
        """Close database connection"""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                 self.logger.error(f"Error closing cursor: {e}")
            finally:
                 self.cursor = None
        if self.connection:
            try:
                self.connection.close()
                self.logger.info("Database connection closed")
            except Exception as e:
                 self.logger.error(f"Error closing database connection: {e}")
            finally:
                 self.connection = None

    def __del__(self):
         """Ensure connection is closed when object is destroyed."""
         self.close()


# Example usage and testing (Should now work with fixes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for detailed example logs
    # Use a different file for testing fixes
    db = TradingDatabase("test_fixed_trading.db")

    # Example: Store a system event with NumPy data
    event_data = {'version': "1.1", 'mode': "test", 'value': np.int64(10), 'float_val': np.float64(1.23)}
    db.store_system_event("STARTUP", event_data, severity="INFO", context="System")

    # Example: Store a trade using Pandas Timestamp and NumPy types
    trade_info = {
        'symbol': 'BTCUSDT',
        'timestamp': pd.Timestamp.now(), # Use Pandas Timestamp
        'action': 'BUY',
        'quantity': 0.001,
        'entry_price': 50000.0,
        'exit_price': None,
        'position_size_usdt': 50.0, # From original schema
        'stop_loss': 49500.0,
        'take_profit': 51000.0,
        'exit_reason': None,
        'pnl_usdt': None,
        'pnl_percent': None,
        'confidence': np.float64(75.0), # Use NumPy float
        'composite_score': np.float64(25.5), # Use NumPy float
        'risk_reward_ratio': 2.0,
        'aggressiveness': 'moderate',
        'order_id': 'test-order-fixed-123',
        'success': True, # Use Python boolean
        'error_message': None,
        'trend_score': 30.0,
        'mr_score': -10.0,
        'breakout_score': 5.0,
        'ml_score': np.int64(15), # Use NumPy int
        'mtf_score': 10.0
    }
    db.store_trade(trade_info)

    # Example: Retrieve trades
    trades_df = db.get_historical_trades(days=1)
    print("\nRecent Trades:")
    print(trades_df)
    if not trades_df.empty:
        print("Timestamp type:", trades_df['timestamp'].dtype)
        print("Success type:", trades_df['success'].dtype)


    # Example: Store performance
    perf_metrics = {
        'portfolio_value': 1010.5, 'daily_pnl_percent': 1.05, 'total_trades': 5,
        'winning_trades': 3, 'win_rate': 60.0, 'max_drawdown': -2.5,
        'avg_confidence': 70.1, 'avg_risk_reward': 1.8
    }
    db.store_performance_metrics(perf_metrics)

    # Example: Retrieve performance
    perf_df = db.get_performance_history(days=1)
    print("\nRecent Performance:")
    print(perf_df)
    if not perf_df.empty:
         print("Date type:", perf_df['date'].dtype)


    # Example: Store ML performance
    ml_perf = {
        'symbol': 'BTCUSDT', 'model_version': 'v20251027_fixed', 'accuracy': 0.62,
        'precision': 0.65, 'recall': 0.60, 'f1_score': 0.61,
        'rf_accuracy': 0.63, 'gb_accuracy': 0.61, 'training_samples': 800, 'test_samples': 200,
        'training_date': datetime.now(), # Use datetime object
         'training_bars_used': 1000
    }
    db.store_ml_model_performance('BTCUSDT', ml_perf)

    # Example: Retrieve ML performance
    ml_perf_list = db.get_ml_model_performance(symbol='BTCUSDT', limit=5)
    print("\nML Performance History:")
    print(ml_perf_list)


    # Example: Store Prediction Quality
    pred_data = {
         'symbol': 'BTCUSDT', 'prediction': 1, 'raw_prediction': np.float64(0.7),
         'actual': None, 'confidence': 0.78,
         'model_used': 'ml_ensemble_fixed', 'ensemble_used': True, 'feature_count': 25,
         'features_used': ['rsi_14', 'ema_diff', 'volume_ratio']
    }
    db.store_prediction_quality(**pred_data) # Pass dict as keyword args

    pred_qual_list = db.get_prediction_quality('BTCUSDT', limit=5)
    print("\nPrediction Quality History:")
    print(pred_qual_list)

    # Example: Store training history with dict parameters
    training_history = {
        'model_version': 'v20251027_fixed',
        'training_date': pd.Timestamp.now(), # Use pandas timestamp
        'training_duration_seconds': 45.2,
        'training_samples': 500,
        'test_samples': 100,
        'final_accuracy': 0.75,
        'final_precision': 0.72,
        'final_recall': 0.71,
        'final_f1': 0.715,
        'feature_count': 25,
        'parameters_used': {'n_estimators': np.int64(100), 'max_depth': 10}, # Include numpy type
        'training_bars_used': 1000
    }
    db.store_model_training_history('BTCUSDT', training_history)

    train_hist_list = db.get_model_training_history('BTCUSDT', limit=5)
    print("\nModel Training History:")
    print(train_hist_list)


    db.close()
    print("\nDatabase connection closed.")