import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

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
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            self.cursor = self.connection.cursor()  # Initialize cursor
            
            # Trades table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
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
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_pnl_percent REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    avg_risk_reward REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # System events table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ML model performance table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
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
                    training_date DATE NOT NULL,
                    training_bars_used INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Market data table (for backtesting)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            ''')
            
            # Prediction quality table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    actual INTEGER,
                    confidence REAL NOT NULL,
                    correct BOOLEAN,
                    timestamp DATETIME NOT NULL,
                    model_used TEXT,
                    features_used TEXT,
                    ensemble_used BOOLEAN DEFAULT FALSE,
                    feature_count INTEGER DEFAULT 0,
                    raw_prediction INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Feature importance table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance REAL NOT NULL,
                    ranking INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, model_type, feature_name, timestamp)
                )
            ''')
            
            # Model training history table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_date DATETIME NOT NULL,
                    training_duration_seconds REAL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    final_accuracy REAL,
                    final_precision REAL,
                    final_recall REAL,
                    final_f1 REAL,
                    feature_count INTEGER,
                    parameters_used TEXT,
                    training_bars_used INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model drift detection table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_drift_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    accuracy_drift REAL,
                    feature_drift REAL,
                    prediction_drift REAL,
                    drift_detected BOOLEAN,
                    drift_reason TEXT,
                    retraining_recommended BOOLEAN,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.connection.commit()
            self.logger.info("Enhanced database tables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced database: {e}")
            raise
    
    def store_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Store a trade record in the database
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, action, quantity, entry_price, exit_price,
                    position_size_usdt, stop_loss, take_profit, exit_reason,
                    pnl_usdt, pnl_percent, confidence, composite_score,
                    risk_reward_ratio, aggressiveness, order_id, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('symbol'),
                trade_data.get('action'),
                trade_data.get('quantity', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price'),
                trade_data.get('position_size_usdt', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('exit_reason'),
                trade_data.get('pnl_usdt'),
                trade_data.get('pnl_percent'),
                trade_data.get('confidence', 0),
                trade_data.get('composite_score', 0),
                trade_data.get('risk_reward_ratio', 0),
                trade_data.get('aggressiveness', 'moderate'),
                trade_data.get('order_id'),
                trade_data.get('success', False),
                trade_data.get('error_message')
            ))
            
            self.connection.commit()
            trade_id = cursor.lastrowid
            self.logger.info(f"Stored trade #{trade_id} for {trade_data.get('symbol')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store trade: {e}")
            return False
    
    def update_trade_exit(self, trade_id: int, exit_price: float, 
                         pnl_usdt: float, pnl_percent: float, exit_reason: str) -> bool:
        """
        Update trade record with exit information
        
        Args:
            trade_id: ID of the trade to update
            exit_price: Exit price
            pnl_usdt: PnL in USDT
            pnl_percent: PnL percentage
            exit_reason: Reason for exit (SL, TP, manual, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                UPDATE trades 
                SET exit_price = ?, pnl_usdt = ?, pnl_percent = ?, exit_reason = ?
                WHERE id = ?
            ''', (exit_price, pnl_usdt, pnl_percent, exit_reason, trade_id))
            
            self.connection.commit()
            self.logger.info(f"Updated trade #{trade_id} exit: {exit_reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update trade exit: {e}")
            return False
    
    def store_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Store daily performance metrics
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Check if entry exists for today
            today = datetime.now().date()
            cursor.execute('SELECT id FROM performance_metrics WHERE date = ?', (today,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entry
                cursor.execute('''
                    UPDATE performance_metrics 
                    SET portfolio_value = ?, daily_pnl_percent = ?, total_trades = ?,
                        winning_trades = ?, win_rate = ?, avg_confidence = ?,
                        avg_risk_reward = ?, max_drawdown = ?, sharpe_ratio = ?
                    WHERE date = ?
                ''', (
                    metrics.get('portfolio_value', 0),
                    metrics.get('daily_pnl_percent', 0),
                    metrics.get('total_trades', 0),
                    metrics.get('winning_trades', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('avg_confidence', 0),
                    metrics.get('avg_risk_reward', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('sharpe_ratio', 0),
                    today
                ))
            else:
                # Insert new entry
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        date, portfolio_value, daily_pnl_percent, total_trades,
                        winning_trades, win_rate, avg_confidence, avg_risk_reward,
                        max_drawdown, sharpe_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    today,
                    metrics.get('portfolio_value', 0),
                    metrics.get('daily_pnl_percent', 0),
                    metrics.get('total_trades', 0),
                    metrics.get('winning_trades', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('avg_confidence', 0),
                    metrics.get('avg_risk_reward', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('sharpe_ratio', 0)
                ))
            
            self.connection.commit()
            self.logger.info(f"Stored performance metrics for {today}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store performance metrics: {e}")
            return False
    
    def store_system_event(self, event_type: str, event_data: Dict, 
                          severity: str = "INFO", context: str = None) -> bool:
        """
        Store system event for monitoring and debugging
        
        Args:
            event_type: Type of event (ERROR, WARNING, INFO, etc.)
            event_data: Event data as dictionary
            severity: Event severity
            context: Context where event occurred
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO system_events (timestamp, event_type, event_data, severity, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                event_type,
                json.dumps(event_data),
                severity,
                context
            ))
            
            self.connection.commit()
            self.logger.debug(f"Stored system event: {event_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store system event: {e}")
            return False
    
    def store_ml_model_performance(self, symbol: str, metrics: Dict[str, Any]) -> bool:
        """
        Store ML model performance metrics
        
        Args:
            symbol: Trading symbol
            metrics: ML performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO ml_model_performance (
                    timestamp, symbol, accuracy, precision, recall, f1_score,
                    rf_accuracy, gb_accuracy, rf_precision, gb_precision,
                    rf_recall, gb_recall, rf_f1, gb_f1, training_samples,
                    test_samples, model_version, training_date, training_bars_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1_score'),
                metrics.get('rf_accuracy'),
                metrics.get('gb_accuracy'),
                metrics.get('rf_precision'),
                metrics.get('gb_precision'),
                metrics.get('rf_recall'),
                metrics.get('gb_recall'),
                metrics.get('rf_f1'),
                metrics.get('gb_f1'),
                metrics.get('training_samples'),
                metrics.get('test_samples'),
                metrics.get('model_version'),
                datetime.now().date(),
                metrics.get('training_bars_used', 0)
            ))
            
            self.connection.commit()
            self.logger.info(f"Stored ML performance for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store ML performance: {e}")
            return False
    
    def store_prediction_quality(self, symbol: str, prediction: int, actual: int = None, 
                               confidence: float = 0, model_used: str = None, 
                               features_used: List[str] = None, ensemble_used: bool = False,
                               feature_count: int = 0, raw_prediction: int = 0) -> bool:
        """
        Store individual prediction quality record
        
        Args:
            symbol: Trading symbol
            prediction: Model prediction (-1, 0, 1)
            actual: Actual outcome
            confidence: Prediction confidence
            model_used: Which model was used
            features_used: List of features used
            ensemble_used: Whether ensemble was used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            correct = None
            if actual is not None:
                correct = 1 if prediction == actual else 0
            
            cursor.execute('''
                INSERT INTO prediction_quality 
                (symbol, prediction, actual, confidence, correct, timestamp, model_used, features_used, ensemble_used, feature_count, raw_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                prediction,
                actual,
                confidence,
                correct,
                datetime.now(),
                model_used,
                json.dumps(features_used) if features_used else None,
                ensemble_used,
                feature_count,
                raw_prediction
            ))
            
            self.connection.commit()
            self.logger.debug(f"Stored prediction quality for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store prediction quality: {e}")
            return False
    
    def store_feature_importance(self, symbol: str, model_type: str, 
                               feature_importance: Dict[str, float]) -> bool:
        """
        Store feature importance for a model
        
        Args:
            symbol: Trading symbol
            model_type: 'rf' or 'gb'
            feature_importance: Dictionary of feature names to importance scores
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            timestamp = datetime.now()
            
            # Store top 20 features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            for rank, (feature_name, importance) in enumerate(sorted_features):
                cursor.execute('''
                    INSERT OR REPLACE INTO feature_importance 
                    (symbol, model_type, feature_name, importance, ranking, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    model_type,
                    feature_name,
                    importance,
                    rank + 1,
                    timestamp
                ))
            
            self.connection.commit()
            self.logger.info(f"Stored feature importance for {symbol} ({model_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store feature importance: {e}")
            return False
    
    def store_model_training_history(self, symbol: str, training_data: Dict[str, Any]) -> bool:
        """
        Store model training history
        
        Args:
            symbol: Trading symbol
            training_data: Training metadata and results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO model_training_history (
                    symbol, model_version, training_date, training_duration_seconds,
                    training_samples, test_samples, final_accuracy, final_precision,
                    final_recall, final_f1, feature_count, parameters_used, training_bars_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                training_data.get('model_version'),
                training_data.get('training_date', datetime.now()),
                training_data.get('training_duration_seconds'),
                training_data.get('training_samples'),
                training_data.get('test_samples'),
                training_data.get('final_accuracy'),
                training_data.get('final_precision'),
                training_data.get('final_recall'),
                training_data.get('final_f1'),
                training_data.get('feature_count'),
                json.dumps(training_data.get('parameters_used', {})),
                training_data.get('training_bars_used', 0)
            ))
            
            self.connection.commit()
            self.logger.info(f"Stored training history for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store training history: {e}")
            return False
    
    def store_model_drift_detection(self, symbol: str, drift_data: Dict[str, Any]) -> bool:
        """
        Store model drift detection results
        
        Args:
            symbol: Trading symbol
            drift_data: Drift detection results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO model_drift_detection (
                    symbol, timestamp, accuracy_drift, feature_drift, prediction_drift,
                    drift_detected, drift_reason, retraining_recommended
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now(),
                drift_data.get('accuracy_drift', 0),
                drift_data.get('feature_drift', 0),
                drift_data.get('prediction_drift', 0),
                drift_data.get('drift_detected', False),
                drift_data.get('drift_reason', ''),
                drift_data.get('retraining_recommended', False)
            ))
            
            self.connection.commit()
            self.logger.info(f"Stored drift detection for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store drift detection: {e}")
            return False
    
    def get_ml_model_performance(self, symbol: str = None, limit: int = 50) -> list:
        """Get ML model performance history"""
        try:
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
            performance = []
            for row in results:
                performance.append({
                    'timestamp': row['timestamp'],
                    'symbol': row['symbol'],
                    'accuracy': row['accuracy'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1_score': row['f1_score'],
                    'rf_accuracy': row['rf_accuracy'],
                    'gb_accuracy': row['gb_accuracy'],
                    'training_samples': row['training_samples'],
                    'test_samples': row['test_samples'],
                    'model_version': row['model_version'],
                    'training_bars_used': row.get('training_bars_used', 0)
                })
            
            return performance
        except Exception as e:
            print(f"Error getting ML performance: {e}")
            return []
    
    def get_prediction_quality(self, symbol: str = None, limit: int = 100) -> list:
        """Get prediction quality data"""
        try:
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
                predictions.append({
                    'timestamp': row['timestamp'],
                    'symbol': row['symbol'],
                    'prediction': row['prediction'],
                    'actual': row.get('actual'),
                    'confidence': row['confidence'],
                    'correct': row['correct'],
                    'model_used': row['model_used'],
                    'features_used': json.loads(row['features_used']) if row['features_used'] else [],
                    'ensemble_used': row.get('ensemble_used', False)
                })
            
            return predictions
        except Exception as e:
            print(f"Error getting prediction quality: {e}")
            return []
    
    def get_feature_importance(self, symbol: str) -> dict:
        """Get latest feature importance data"""
        try:
            query = """
            SELECT model_type, feature_name, importance, timestamp 
            FROM feature_importance 
            WHERE symbol = ? 
            ORDER BY timestamp DESC, ranking ASC
            LIMIT 40
            """
            self.cursor.execute(query, (symbol,))
            results = self.cursor.fetchall()
            
            importance_data = {'features': [], 'rf_importance': [], 'gb_importance': []}
            rf_data = []
            gb_data = []
            
            for row in results:
                model_type = row['model_type']
                feature_name = row['feature_name']
                importance = row['importance']
                
                if model_type == 'rf':
                    rf_data.append((feature_name, importance))
                elif model_type == 'gb':
                    gb_data.append((feature_name, importance))
            
            # Get top 10 features for each model type
            rf_top = sorted(rf_data, key=lambda x: x[1], reverse=True)[:10]
            gb_top = sorted(gb_data, key=lambda x: x[1], reverse=True)[:10]
            
            # Use RF features as base, add any missing from GB
            all_features = set()
            for feature, _ in rf_top + gb_top:
                all_features.add(feature)
            
            importance_data['features'] = list(all_features)
            
            # Create importance arrays aligned with features list
            for feature in importance_data['features']:
                rf_imp = next((imp for f, imp in rf_top if f == feature), 0)
                gb_imp = next((imp for f, imp in gb_top if f == feature), 0)
                importance_data['rf_importance'].append(rf_imp)
                importance_data['gb_importance'].append(gb_imp)
            
            return importance_data
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {'features': [], 'rf_importance': [], 'gb_importance': []}
    
    def get_model_training_history(self, symbol: str = None, limit: int = 50) -> list:
        """Get model training history for a symbol"""
        try:
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
                history.append({
                    'symbol': row['symbol'],
                    'training_date': row['training_date'],
                    'model_version': row['model_version'],
                    'training_duration_seconds': row['training_duration_seconds'],
                    'training_samples': row['training_samples'],
                    'test_samples': row['test_samples'],
                    'final_accuracy': row['final_accuracy'],
                    'final_precision': row['final_precision'],
                    'final_recall': row['final_recall'],
                    'final_f1': row['final_f1'],
                    'feature_count': row['feature_count'],
                    'training_bars_used': row.get('training_bars_used', 0)
                })
            
            return history
        except Exception as e:
            print(f"Error getting model training history: {e}")
            return []
    
    def get_prediction_quality_stats(self, symbol: str = None, days: int = 7) -> Dict[str, Any]:
        """
        Get prediction quality statistics
        
        Args:
            symbol: Filter by symbol (optional)
            days: Lookback period in days
            
        Returns:
            Dictionary with prediction quality statistics
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT 
                    symbol,
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN correct = 1 THEN confidence ELSE NULL END) as avg_correct_confidence,
                    AVG(CASE WHEN correct = 0 THEN confidence ELSE NULL END) as avg_incorrect_confidence
                FROM prediction_quality 
                WHERE timestamp >= ? AND correct IS NOT NULL
            '''
            params = [start_date]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
                
            query += ' GROUP BY symbol'
            
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            stats = {}
            for row in cursor.fetchall():
                row_dict = dict(row)
                symbol = row_dict['symbol']
                stats[symbol] = {
                    'accuracy': row_dict['correct_predictions'] / row_dict['total_predictions'] if row_dict['total_predictions'] > 0 else 0,
                    'total_predictions': row_dict['total_predictions'],
                    'correct_predictions': row_dict['correct_predictions'],
                    'avg_confidence': row_dict['avg_confidence'] or 0,
                    'avg_correct_confidence': row_dict['avg_correct_confidence'] or 0,
                    'avg_incorrect_confidence': row_dict['avg_incorrect_confidence'] or 0
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get prediction quality stats: {e}")
            return {}
    
    def get_feature_importance_history(self, symbol: str, model_type: str = None, 
                                     days: int = 30) -> pd.DataFrame:
        """
        Get feature importance history
        
        Args:
            symbol: Trading symbol
            model_type: 'rf' or 'gb' (optional)
            days: Lookback period in days
            
        Returns:
            DataFrame with feature importance history
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT * FROM feature_importance 
                WHERE symbol = ? AND timestamp >= ?
            '''
            params = [symbol, start_date]
            
            if model_type:
                query += ' AND model_type = ?'
                params.append(model_type)
                
            query += ' ORDER BY timestamp DESC, ranking ASC'
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get feature importance history: {e}")
            return pd.DataFrame()
    
    def get_model_drift_history(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        """
        Get model drift detection history
        
        Args:
            symbol: Filter by symbol (optional)
            days: Lookback period in days
            
        Returns:
            DataFrame with drift detection history
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT * FROM model_drift_detection 
                WHERE timestamp >= ?
            '''
            params = [start_date]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
                
            query += ' ORDER BY timestamp DESC'
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get drift history: {e}")
            return pd.DataFrame()
    
    def get_historical_trades(self, days: int = 30, symbol: str = None) -> pd.DataFrame:
        """Get historical trades from database"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT * FROM trades 
                WHERE timestamp >= ?
            '''
            params = [start_date]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            query += ' ORDER BY timestamp DESC'
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            # Convert datetime columns
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            self.logger.debug(f"Retrieved {len(df)} trades from database")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical trades: {e}")
            return pd.DataFrame()
    
    def get_performance_history(self, days: int = 90) -> pd.DataFrame:
        """Get performance metrics history"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            query = '''
                SELECT * FROM performance_metrics 
                WHERE date >= ?
                ORDER BY date DESC
            '''
            
            df = pd.read_sql_query(query, self.connection, params=[start_date])
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            self.logger.debug(f"Retrieved {len(df)} performance records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get performance history: {e}")
            return pd.DataFrame()
    
    def get_system_events(self, event_type: str = None, severity: str = None, 
                         hours: int = 24) -> pd.DataFrame:
        """Get system events for monitoring"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            query = 'SELECT * FROM system_events WHERE timestamp >= ?'
            params = [start_time]
            
            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity)
            
            query += ' ORDER BY timestamp DESC'
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Parse event_data JSON
                df['event_data'] = df['event_data'].apply(json.loads)
            
            self.logger.debug(f"Retrieved {len(df)} system events")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get system events: {e}")
            return pd.DataFrame()
    
    def get_trading_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Basic trade statistics
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_trades,
                    AVG(pnl_percent) as avg_pnl_percent,
                    AVG(confidence) as avg_confidence,
                    AVG(risk_reward_ratio) as avg_risk_reward,
                    SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl_percent < 0 THEN 1 ELSE 0 END) as losing_trades
                FROM trades 
                WHERE timestamp >= ? AND pnl_percent IS NOT NULL
            ''', (start_date,))
            
            stats = dict(cursor.fetchone())
            
            # Calculate win rate
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
            else:
                stats['win_rate'] = 0
            
            # Best and worst trades
            cursor.execute('''
                SELECT symbol, pnl_percent, action, timestamp
                FROM trades 
                WHERE timestamp >= ? AND pnl_percent IS NOT NULL
                ORDER BY pnl_percent DESC LIMIT 1
            ''', (start_date,))
            best_trade = cursor.fetchone()
            if best_trade:
                stats['best_trade'] = dict(best_trade)
            
            cursor.execute('''
                SELECT symbol, pnl_percent, action, timestamp
                FROM trades 
                WHERE timestamp >= ? AND pnl_percent IS NOT NULL
                ORDER BY pnl_percent ASC LIMIT 1
            ''', (start_date,))
            worst_trade = cursor.fetchone()
            if worst_trade:
                stats['worst_trade'] = dict(worst_trade)
            
            # Performance by symbol
            cursor.execute('''
                SELECT 
                    symbol,
                    COUNT(*) as trade_count,
                    AVG(pnl_percent) as avg_pnl_percent,
                    SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM trades 
                WHERE timestamp >= ? AND pnl_percent IS NOT NULL
                GROUP BY symbol
                ORDER BY avg_pnl_percent DESC
            ''', (start_date,))
            
            symbol_stats = []
            for row in cursor.fetchall():
                symbol_stats.append(dict(row))
            
            stats['symbol_performance'] = symbol_stats
            
            # Add ML statistics if available
            try:
                ml_stats = self.get_prediction_quality_stats(days=days)
                stats['prediction_quality'] = ml_stats
            except:
                pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get trading statistics: {e}")
            return {}
    
    def get_confusion_matrix(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get data for a 3x3 confusion matrix (prediction vs actual)
        
        Args:
            symbol: Filter by symbol (optional)
            days: Lookback period in days
            
        Returns:
            List of dictionaries with {'prediction': int, 'actual': int, 'count': int}
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT 
                    prediction,
                    actual,
                    COUNT(*) as count
                FROM prediction_quality 
                WHERE timestamp >= ? AND actual IS NOT NULL
            '''
            params = [start_date]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
                
            query += ' GROUP BY prediction, actual'
            
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            matrix_data = [dict(row) for row in cursor.fetchall()]
            return matrix_data
            
        except Exception as e:
            self.logger.error(f"Failed to get confusion matrix data: {e}")
            return []

    def get_confidence_distribution(self, symbol: str = None, days: int = 7) -> Dict[str, List[float]]:
        """
        Get confidence scores grouped by correct/incorrect predictions
        
        Args:
            symbol: Filter by symbol (optional)
            days: Lookback period in days
            
        Returns:
            Dictionary with {'correct': [conf1, conf2...], 'incorrect': [conf3, conf4...]}
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT 
                    confidence,
                    correct
                FROM prediction_quality 
                WHERE timestamp >= ? AND actual IS NOT NULL AND correct IS NOT NULL
            '''
            params = [start_date]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if df.empty:
                return {'correct': [], 'incorrect': []}
                
            distribution = {
                'correct': df[df['correct'] == 1]['confidence'].tolist(),
                'incorrect': df[df['correct'] == 0]['confidence'].tolist()
            }
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Failed to get confidence distribution: {e}")
            return {'correct': [], 'incorrect': []}

    def store_market_data(self, symbol: str, data: pd.DataFrame, timeframe: str):
        """Store market data for backtesting and analysis"""
        try:
            cursor = self.connection.cursor()
            
            for _, row in data.iterrows():
                cursor.execute('''
                    INSERT OR IGNORE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    row.name if hasattr(row, 'name') else datetime.now(),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    timeframe
                ))
            
            self.connection.commit()
            self.logger.debug(f"Stored market data for {symbol} ({timeframe})")
            
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to prevent database bloat"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cursor = self.connection.cursor()
            
            # Delete old trades
            cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_date,))
            trades_deleted = cursor.rowcount
            
            # Delete old system events
            cursor.execute('DELETE FROM system_events WHERE timestamp < ?', (cutoff_date,))
            events_deleted = cursor.rowcount
            
            # Delete old market data
            cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
            market_data_deleted = cursor.rowcount
            
            # Delete old ML data (keep longer for analysis)
            ml_cutoff = datetime.now() - timedelta(days=days_to_keep * 2)
            cursor.execute('DELETE FROM ml_model_performance WHERE timestamp < ?', (ml_cutoff,))
            ml_perf_deleted = cursor.rowcount
            
            cursor.execute('DELETE FROM prediction_quality WHERE timestamp < ?', (ml_cutoff,))
            pred_quality_deleted = cursor.rowcount
            
            self.connection.commit()
            
            self.logger.info(
                f"Database cleanup completed: "
                f"{trades_deleted} trades, "
                f"{events_deleted} events, "
                f"{market_data_deleted} market records, "
                f"{ml_perf_deleted} ML performance records, "
                f"{pred_quality_deleted} prediction quality records deleted"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced database
    db = TradingDatabase("test_enhanced_trading.db")
    
    # Test ML data storage
    test_ml_performance = {
        'accuracy': 0.75,
        'precision': 0.72,
        'recall': 0.71,
        'f1_score': 0.715,
        'rf_accuracy': 0.74,
        'gb_accuracy': 0.76,
        'rf_precision': 0.71,
        'gb_precision': 0.73,
        'rf_recall': 0.70,
        'gb_recall': 0.72,
        'rf_f1': 0.705,
        'gb_f1': 0.725,
        'training_samples': 500,
        'test_samples': 100,
        'model_version': 'v20240101_1200',
        'training_bars_used': 1000
    }
    
    db.store_ml_model_performance('BTCUSDT', test_ml_performance)
    
    # Test prediction quality storage
    db.store_prediction_quality('BTCUSDT', 1, 1, 0.85, 'ml_ensemble', ['rsi_14', 'volatility_20'], True)
    
    # Test feature importance storage
    feature_importance = {
        'rsi_14': 0.25,
        'volatility_20': 0.18,
        'sma_ratio': 0.15,
        'volume_ma': 0.12,
        'momentum_10': 0.10
    }
    db.store_feature_importance('BTCUSDT', 'rf', feature_importance)
    
    # Test training history
    training_history = {
        'model_version': 'v20240101_1200',
        'training_duration_seconds': 45.2,
        'training_samples': 500,
        'test_samples': 100,
        'final_accuracy': 0.75,
        'final_precision': 0.72,
        'final_recall': 0.71,
        'final_f1': 0.715,
        'feature_count': 25,
        'parameters_used': {'n_estimators': 100, 'max_depth': 10},
        'training_bars_used': 1000
    }
    db.store_model_training_history('BTCUSDT', training_history)
    
    # Test drift detection
    drift_data = {
        'accuracy_drift': 0.05,
        'feature_drift': 0.12,
        'prediction_drift': 0.08,
        'drift_detected': False,
        'drift_reason': 'Within acceptable limits',
        'retraining_recommended': False
    }
    db.store_model_drift_detection('BTCUSDT', drift_data)
    
    # Retrieve ML data
    ml_perf = db.get_ml_model_performance(days=7)
    print(f"Retrieved {len(ml_perf)} ML performance records")
    
    pred_stats = db.get_prediction_quality_stats(days=7)
    print(f"Prediction quality stats: {pred_stats}")
    
    feature_history = db.get_feature_importance_history('BTCUSDT', days=7)
    print(f"Retrieved {len(feature_history)} feature importance records")
    
    training_history = db.get_model_training_history(days=30)
    print(f"Retrieved {len(training_history)} training history records")
    
    drift_history = db.get_model_drift_history(days=30)
    print(f"Retrieved {len(drift_history)} drift detection records")
    
    db.close()