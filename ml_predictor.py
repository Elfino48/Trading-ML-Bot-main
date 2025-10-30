import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.multioutput import ClassifierChain
import joblib
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sklearn
from scipy import stats
import talib
import logging
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self, error_handler=None, database=None):
        self.models = {}
        self.previous_models = {}
        self.scalers = {}
        self.previous_scalers = {}
        self.feature_selectors = {}
        self.feature_columns = []
        self.error_handler = error_handler
        self.database = database
        self.model_performance = {}
        self.performance_threshold = 0.55
        self.feature_importance = {}
        self.walk_forward_performance = {}
        self.model_versions = {}
        self.performance_history = {}
        self.prediction_quality_metrics = {}
        self.feature_drift_detector = {}
        self.real_time_monitor = {}
        self.logger = logging.getLogger('MLPredictor')

        self.auto_save_interval = 10
        self.training_count = 0
        
        # Enhanced configuration
        self.quality_metrics_window = 100
        self.drift_threshold = 0.15
        self.performance_alert_threshold = 0.1
        self.ensemble_weight_current = 0.8
        self.ensemble_weight_previous = 0.2
        
        # NEW: Enhanced feature selection and regularization
        self.max_features = 25
        self.feature_selection_method = 'importance'
        self.use_pca = False
        self.pca_components = 15
        
        # NEW: Enhanced validation and multi-timeframe targets
        self.walk_forward_splits = 5
        self.validation_window = 200
        self.min_training_samples = 150
        
        # NEW: Multi-timeframe target configuration
        self.target_configs = [
            {'periods': 5, 'weight': 0.4, 'threshold_multiplier': 1.5},   # Short-term
            {'periods': 10, 'weight': 0.3, 'threshold_multiplier': 2.0},  # Medium-term
            {'periods': 20, 'weight': 0.3, 'threshold_multiplier': 2.5}   # Long-term
        ]
        
        # NEW: Enhanced model parameters with hyperparameter optimization
        self.rf_param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        self.gb_param_dist = {
            'n_estimators': randint(50, 150),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.6, 0.4)
        }
        
        # Default parameters (used when hyperparameter optimization is skipped)
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.gb_params = {
            'n_estimators': 80,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # NEW: Hyperparameter optimization settings
        self.enable_hyperparameter_optimization = True
        self.hpo_n_iter = 20
        self.hpo_cv = 3
        
        print(f"🤖 Enhanced ML Predictor initialized with scikit-learn {sklearn.__version__}")
        print(f"   • Multi-timeframe targets: {len(self.target_configs)} horizons")
        print(f"   • Hyperparameter optimization: {self.enable_hyperparameter_optimization}")
        print(f"   • Max features: {self.max_features}")

    def set_error_handler(self, error_handler):
        self.error_handler = error_handler

    def set_database(self, database):
        self.database = database

    # Add to MLPredictor class
    def should_retrain_model(self, symbol: str, current_accuracy_threshold: float = 0.55) -> bool:
        """Check if model needs retraining based on performance degradation"""
        if symbol not in self.model_versions:
            return True
            
        model_info = self.model_versions[symbol]
        
        # Check accuracy
        current_accuracy = model_info.get('accuracy', 0)
        if current_accuracy < current_accuracy_threshold:
            self.logger.info(f"Model for {symbol} needs retraining: accuracy {current_accuracy:.3f} < {current_accuracy_threshold}")
            return True
            
        # Check time since last training
        training_date = model_info.get('training_date')
        if training_date:
            if isinstance(training_date, str):
                try:
                    training_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                except ValueError:
                    training_date = datetime.now()
            
            days_since_training = (datetime.now() - training_date).days
            if days_since_training > 3:  # Retrain every 3 days max
                self.logger.info(f"Model for {symbol} needs retraining: {days_since_training} days since last training")
                return True
                
        # Check feature drift
        if symbol in self.real_time_monitor:
            recent_drift_alerts = self.real_time_monitor[symbol].get('drift_alerts', 0)
            if recent_drift_alerts > 10:  # Too many drift alerts
                self.logger.info(f"Model for {symbol} needs retraining: {recent_drift_alerts} drift alerts")
                return True
                
        # Check prediction quality degradation
        quality_metrics = self._get_current_quality_metrics(symbol)
        recent_accuracy = quality_metrics.get('recent_accuracy', 1.0)
        if recent_accuracy < 0.5:  # Recent accuracy below 50%
            self.logger.info(f"Model for {symbol} needs retraining: recent accuracy {recent_accuracy:.3f} < 0.5")
            return True
                
        return False

    def get_models_needing_retraining(self, accuracy_threshold: float = 0.55) -> List[str]:
        """Get list of symbols whose models need retraining"""
        symbols_needing_retraining = []
        for symbol in list(self.models.keys()):
            if self.should_retrain_model(symbol, accuracy_threshold):
                symbols_needing_retraining.append(symbol)
        return symbols_needing_retraining

    def prepare_features_point_in_time(self, df: pd.DataFrame, current_idx: int, lookback: int = 50) -> pd.DataFrame:
        """Enhanced feature preparation with volume and regime features"""
        if current_idx < lookback:
            return pd.DataFrame()

        features = {}
        current_data = df.iloc[:current_idx+1]
        
        close_prices = current_data['close'].astype(float)
        high_prices = current_data['high'].astype(float)
        low_prices = current_data['low'].astype(float)
        volume = current_data['volume'].astype(float)
        opens = current_data['open'].astype(float)

        try:
            # 1. Enhanced volume-based features
            features['volume_momentum'] = volume.iloc[-1] / volume.rolling(10).mean().iloc[-1] if len(volume) > 10 else 1
            features['volume_acceleration'] = (volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5] if len(volume) > 5 else 0
            
            # Volume volatility
            if len(volume) > 20:
                features['volume_volatility'] = volume.pct_change().rolling(20).std().iloc[-1]
            else:
                features['volume_volatility'] = 0
                
            # Price-volume correlation
            if len(close_prices) > 20:
                price_changes = close_prices.pct_change().tail(20)
                volume_changes = volume.pct_change().tail(20)
                features['price_volume_corr'] = price_changes.corr(volume_changes) if not price_changes.isna().all() else 0
            else:
                features['price_volume_corr'] = 0

            # 2. Regime-aware features
            if len(close_prices) > 20:
                price_change_20 = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
                features['in_trending_regime'] = 1 if abs(price_change_20) > 0.03 else 0
                
                volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1]
                features['in_high_vol_regime'] = 1 if volatility_20 > 0.025 else 0
            else:
                features['in_trending_regime'] = 0
                features['in_high_vol_regime'] = 0

            # 3. Existing price-based features (keep your current implementations)
            for period in [1, 3, 5, 10, 20]:
                if len(close_prices) > period:
                    features[f'returns_{period}'] = close_prices.iloc[-1] / close_prices.iloc[-period-1] - 1
            
            # Volatility features
            for period in [5, 10, 20]:
                if len(close_prices) > period:
                    features[f'volatility_{period}'] = close_prices.pct_change().rolling(period).std().iloc[-1]
            
            # Volume features
            for period in [5, 10, 20]:
                if len(volume) > period:
                    features[f'volume_ma_ratio_{period}'] = volume.iloc[-1] / volume.rolling(period).mean().iloc[-1]
            
            features['volume_volatility'] = volume.pct_change().rolling(20).std().iloc[-1]
            
            # 4. Advanced price action features
            features['high_low_ratio'] = (high_prices.iloc[-1] - low_prices.iloc[-1]) / close_prices.iloc[-1]
            features['close_open_ratio'] = (close_prices.iloc[-1] - opens.iloc[-1]) / opens.iloc[-1]
            features['body_size'] = abs(opens.iloc[-1] - close_prices.iloc[-1]) / (high_prices.iloc[-1] - low_prices.iloc[-1]) if (high_prices.iloc[-1] - low_prices.iloc[-1]) > 0 else 0
            
            # 5. Multi-timeframe trend features
            for short_period, long_period in [(5, 20), (10, 50), (20, 100)]:
                if len(close_prices) > long_period:
                    sma_short = close_prices.rolling(short_period).mean().iloc[-1]
                    sma_long = close_prices.rolling(long_period).mean().iloc[-1]
                    features[f'sma_ratio_{short_period}_{long_period}'] = sma_short / sma_long - 1
            
            # 6. Momentum features across multiple periods
            for period in [6, 14, 21]:
                if len(close_prices) > period:
                    features[f'rsi_{period}'] = self._calculate_rsi_point_in_time(close_prices, period)
            
            for period in [5, 10, 20]:
                if len(close_prices) > period:
                    features[f'momentum_{period}'] = close_prices.iloc[-1] / close_prices.iloc[-period-1] - 1
            
            # 7. Support/resistance levels
            for period in [20, 50]:
                if len(close_prices) > period:
                    features[f'price_vs_high_{period}'] = close_prices.iloc[-1] / high_prices.rolling(period).max().iloc[-1]
                    features[f'price_vs_low_{period}'] = close_prices.iloc[-1] / low_prices.rolling(period).min().iloc[-1]
            
            # 8. Volatility and risk features
            features['atr_14'] = self._calculate_atr_point_in_time(high_prices, low_prices, close_prices, 14)
            features['atr_ratio'] = features['atr_14'] / close_prices.iloc[-1]
            
            # 9. Bollinger Bands with multiple configurations
            for period, std in [(20, 2), (50, 2)]:
                if len(close_prices) > period:
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands_point_in_time(close_prices, period, std)
                    features[f'bb_position_{period}'] = (close_prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            # 10. MACD with different configurations
            features['macd'], features['macd_signal'] = self._calculate_macd_point_in_time(close_prices)
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # 11. Market regime and advanced features
            features['market_regime_feature'] = self._calculate_market_regime_feature(close_prices, high_prices, low_prices)
            features['volatility_regime_feature'] = self._calculate_volatility_regime_feature(close_prices)
            
            # 12. Time-based and cyclical features
            if hasattr(current_data.index[-1], 'weekday'):
                features['weekday_effect'] = current_data.index[-1].weekday()
                features['hour_effect'] = current_data.index[-1].hour
            
            # 13. Advanced statistical features
            features['price_efficiency'] = self._calculate_price_efficiency(close_prices)
            if len(close_prices) >= 100:
                features['hurst_exponent'] = self._calculate_hurst_exponent(close_prices.tail(100))
            else:
                features['hurst_exponent'] = 0.5
                
            # 14. Market microstructure features
            features['market_micro_1'] = self._calculate_market_microstructure_1(high_prices, low_prices, close_prices, volume)
            features['market_micro_2'] = self._calculate_market_microstructure_2(high_prices, low_prices, close_prices, volume)
            
            return pd.DataFrame([features]).fillna(0)
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_ml_error(e, "ALL", "feature_preparation")
            return pd.DataFrame()

    def create_multi_timeframe_targets(self, df: pd.DataFrame, current_idx: int) -> List[int]:
        """Create multi-timeframe targets for richer learning"""
        targets = []
        
        for config in self.target_configs:
            periods = config['periods']
            threshold_multiplier = config['threshold_multiplier']
            
            if current_idx + periods >= len(df):
                targets.append(0)
                continue
                
            current_price = df['close'].iloc[current_idx]
            future_price = df['close'].iloc[current_idx + periods]
            future_return = (future_price - current_price) / current_price
            
            # Dynamic threshold based on recent volatility
            volatility = df['close'].pct_change().rolling(20).std().iloc[current_idx - 1]
            threshold = max(0.015, volatility * threshold_multiplier)
            
            if future_return > threshold:
                targets.append(1)
            elif future_return < -threshold:
                targets.append(-1)
            else:
                targets.append(0)
        
        return targets

    def create_enhanced_target(self, df: pd.DataFrame, current_idx: int) -> int:
        """Create enhanced target using dynamic thresholds based on market regime"""
        multi_targets = self.create_multi_timeframe_targets(df, current_idx)
        
        if not multi_targets:
            return 0
        
        # Get market regime context
        close_prices = df['close'].astype(float)
        if current_idx >= 20:
            volatility = close_prices.pct_change().rolling(20).std().iloc[current_idx]
            trend_strength = abs(close_prices.iloc[current_idx] / close_prices.iloc[current_idx-20] - 1)
        else:
            volatility = 0.02
            trend_strength = 0
        
        # Weighted voting with regime consideration
        weighted_vote = 0
        total_weight = 0
        
        for i, target in enumerate(multi_targets):
            base_weight = self.target_configs[i]['weight']
            
            # Adjust weights based on market regime
            adjusted_weight = base_weight
            
            # Increase weight for shorter timeframes in trending markets
            if trend_strength > 0.04 and self.target_configs[i]['periods'] <= 10:
                adjusted_weight *= 1.3
            # Increase weight for medium timeframes in high volatility
            elif volatility > 0.03 and 10 <= self.target_configs[i]['periods'] <= 20:
                adjusted_weight *= 1.2
            # Decrease weight for all timeframes in low volatility ranging markets
            elif volatility < 0.01 and trend_strength < 0.02:
                adjusted_weight *= 0.7
                
            weighted_vote += target * adjusted_weight
            total_weight += adjusted_weight
        
        if total_weight > 0:
            normalized_vote = weighted_vote / total_weight
        else:
            normalized_vote = 0
        
        # Convert to trading signal with regime-aware thresholds
        if trend_strength > 0.05:  # Strong trend
            strong_threshold = 0.25
            weak_threshold = 0.15
        elif volatility > 0.03:  # High volatility
            strong_threshold = 0.30
            weak_threshold = 0.20
        else:  # Normal conditions
            strong_threshold = 0.20
            weak_threshold = 0.10
        
        if normalized_vote > strong_threshold:
            return 2  # Strong buy
        elif normalized_vote > weak_threshold:
            return 1  # Buy
        elif normalized_vote < -strong_threshold:
            return -2  # Strong sell
        elif normalized_vote < -weak_threshold:
            return -1  # Sell
        else:
            return 0  # Hold

    def prepare_training_data_enhanced(self, df: pd.DataFrame, min_samples: int = None) -> tuple:
        """Enhanced training data preparation with multi-timeframe targets"""
        if min_samples is None:
            min_samples = self.min_training_samples
            
        max_bars = min(2000, len(df) - 50)
        if max_bars < min_samples + 50:
            return pd.DataFrame(), pd.Series()
        
        df = df.tail(max_bars)
        
        features_list = []
        targets = []
        
        # Use stride to reduce autocorrelation
        stride = max(1, len(df) // 500)
        
        for i in range(50, len(df) - max([c['periods'] for c in self.target_configs]), stride):
            features = self.prepare_features_point_in_time(df, i)
            target = self.create_enhanced_target(df, i)
            
            if not features.empty:
                features_list.append(features.iloc[0])
                targets.append(target)
        
        if len(features_list) < min_samples:
            return pd.DataFrame(), pd.Series()
        
        features_df = pd.DataFrame(features_list).fillna(0)
        target_series = pd.Series(targets, index=features_df.index)
        
        # Remove constant and highly correlated features
        features_df = self._clean_features(features_df)
        
        return features_df, target_series

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by removing constant and highly correlated ones"""
        # Remove constant features
        constant_features = features_df.columns[features_df.std() == 0]
        if len(constant_features) > 0:
            features_df = features_df.drop(columns=constant_features)
        
        # Remove highly correlated features (threshold = 0.95)
        corr_matrix = features_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        if len(to_drop) > 0:
            features_df = features_df.drop(columns=to_drop)
        
        return features_df

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: pd.Series) -> Tuple[dict, dict]:
        """Enhanced hyperparameter optimization with time series cross-validation"""
        if not self.enable_hyperparameter_optimization:
            return self.rf_params, self.gb_params
        
        try:
            print("🔄 Optimizing hyperparameters...")
            
            # Time-series aware cross-validation
            tscv = TimeSeriesSplit(n_splits=self.hpo_cv)
            
            # Optimize Random Forest
            rf_search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                self.rf_param_dist,
                n_iter=self.hpo_n_iter,
                cv=tscv,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1
            )
            
            rf_search.fit(X_train, y_train)
            best_rf_params = rf_search.best_params_
            
            # Optimize Gradient Boosting
            gb_search = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=42),
                self.gb_param_dist,
                n_iter=self.hpo_n_iter,
                cv=tscv,
                scoring='accuracy',
                random_state=42
            )
            
            gb_search.fit(X_train, y_train)
            best_gb_params = gb_search.best_params_
            
            print(f"✅ RF best score: {rf_search.best_score_:.3f}")
            print(f"✅ GB best score: {gb_search.best_score_:.3f}")
            
            return best_rf_params, best_gb_params
            
        except Exception as e:
            print(f"⚠️ Hyperparameter optimization failed: {e}")
            return self.rf_params, self.gb_params

    def train_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """Enhanced model training with hyperparameter optimization and multi-timeframe targets"""
        try:
            print(f"🔄 Training enhanced model for {symbol}...")
            
            training_start = datetime.now()
            
            if not hasattr(self, 'model_versions'):
                self.model_versions = {}
            
            # Save current model as previous before training new one
            if symbol in self.models:
                self.previous_models[symbol] = self.models[symbol].copy()
                self.previous_scalers[symbol] = self.scalers[symbol]
                print(f"💾 Saved previous model for {symbol} for ensembling")
            
            # Use enhanced training data with multi-timeframe targets
            features, target = self.prepare_training_data_enhanced(df)
            
            if features.empty or target.empty:
                print(f"⚠️ Insufficient data for training {symbol}")
                self._record_training_failure(symbol, "insufficient_data")
                return False
                
            # Enhanced feature selection
            features_selected, selected_features = self._select_features(features, target, symbol)
            self.feature_importance[symbol]['selected_features'] = selected_features
            
            X_train, X_test, y_train, y_test = self.time_series_train_test_split(features_selected, target)
            
            if X_train is None:
                print(f"⚠️ Insufficient data after split for {symbol}")
                self._record_training_failure(symbol, "split_failed")
                return False

            # Enhanced scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply PCA if enabled
            if self.use_pca:
                X_train_scaled, X_test_scaled = self._apply_pca(X_train_scaled, X_test_scaled)

            # Hyperparameter optimization
            rf_params, gb_params = self._optimize_hyperparameters(X_train_scaled, y_train)
            
            # Train models with optimized parameters
            rf = RandomForestClassifier(**rf_params)
            gb = GradientBoostingClassifier(**gb_params)

            rf.fit(X_train_scaled, y_train)
            gb.fit(X_train_scaled, y_train)

            # Enhanced validation with multi-class metrics
            rf_pred = rf.predict(X_test_scaled)
            gb_pred = gb.predict(X_test_scaled)

            rf_accuracy = accuracy_score(y_test, rf_pred)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            
            rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
            gb_precision = precision_score(y_test, gb_pred, average='weighted', zero_division=0)
            
            rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
            gb_recall = recall_score(y_test, gb_pred, average='weighted', zero_division=0)
            
            rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
            gb_f1 = f1_score(y_test, gb_pred, average='weighted', zero_division=0)

            # Overfitting detection
            rf_train_pred = rf.predict(X_train_scaled)
            rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
            rf_overfit = rf_train_accuracy - rf_accuracy
            
            gb_train_pred = gb.predict(X_train_scaled)
            gb_train_accuracy = accuracy_score(y_train, gb_train_pred)
            gb_overfit = gb_train_accuracy - gb_accuracy
            
            if rf_overfit > 0.15 or gb_overfit > 0.15:
                print(f"⚠️ Potential overfitting detected for {symbol}: RF={rf_overfit:.3f}, GB={gb_overfit:.3f}")

            # Store models and metadata
            if not hasattr(self, 'models'):
                self.models = {}
            if not hasattr(self, 'scalers'):
                self.scalers = {}
                
            self.models[symbol] = {'rf': rf, 'gb': gb}
            self.scalers[symbol] = scaler
            
            if not hasattr(self, 'feature_importance'):
                self.feature_importance = {}
                
            self.feature_importance[symbol].update({
                'features': selected_features,
                'rf_importance': rf.feature_importances_.tolist(),
                'gb_importance': gb.feature_importances_.tolist(),
                'rf_overfit': rf_overfit,
                'gb_overfit': gb_overfit,
                'rf_params': rf_params,
                'gb_params': gb_params
            })
            
            self._initialize_feature_drift_detector(symbol, X_train_scaled)
            self._initialize_prediction_quality_tracking(symbol)

            # Enhanced model version tracking
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
            training_duration = (datetime.now() - training_start).total_seconds()
            
            self.model_versions[symbol] = {
                'version': model_version,
                'training_date': datetime.now(),
                'accuracy': (rf_accuracy + gb_accuracy) / 2,
                'rf_accuracy': rf_accuracy,
                'gb_accuracy': gb_accuracy,
                'rf_precision': rf_precision,
                'gb_precision': gb_precision,
                'rf_recall': rf_recall,
                'gb_recall': gb_recall,
                'rf_f1': rf_f1,
                'gb_f1': gb_f1,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'training_duration_seconds': training_duration,
                'status': 'trained',
                'training_bars_used': len(df),
                'feature_count': len(selected_features),
                'overfit_scores': {'rf': rf_overfit, 'gb': gb_overfit},
                'feature_selection_method': self.feature_selection_method,
                'target_type': 'multi_timeframe_enhanced',
                'hyperparameters_optimized': self.enable_hyperparameter_optimization
            }

            # Store in database
            if self.database:
                self._store_training_results(symbol, model_version, training_duration, 
                                        rf_accuracy, gb_accuracy, rf_precision, gb_precision,
                                        rf_recall, gb_recall, rf_f1, gb_f1,
                                        len(X_train), len(X_test), len(df), selected_features,
                                        rf_params, gb_params)

            print(f"✅ Trained enhanced models for {symbol}: RF={rf_accuracy:.3f}, GB={gb_accuracy:.3f}, Features={len(selected_features)}")
            
            # === ADDED: Training history tracking for evolution chart ===
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []
            
            training_record = {
                'timestamp': datetime.now(),
                'accuracy': (rf_accuracy + gb_accuracy) / 2,
                'rf_accuracy': rf_accuracy,
                'gb_accuracy': gb_accuracy,
                'training_samples': len(X_train),
                'feature_count': len(selected_features),
                'model_version': model_version,
                'training_duration_seconds': training_duration
            }
            
            self.performance_history[symbol].append(training_record)
            
            # Keep only last 50 records to prevent memory bloat
            if len(self.performance_history[symbol]) > 50:
                self.performance_history[symbol] = self.performance_history[symbol][-50:]
            
            print(f"📈 Training history updated for {symbol} (total cycles: {len(self.performance_history[symbol])})")
            # === END ADDED ===
            
            # Enhanced walk-forward validation
            try:
                wf_result = self.walk_forward_validation_enhanced(symbol, df)
                if wf_result.get('success'):
                    self.training_count += 1

                    if self.training_count % self.auto_save_interval == 0:
                        self.save_models()
                    print(f"📊 Enhanced walk-forward validation for {symbol}: {wf_result['avg_accuracy']:.3f} ± {wf_result['std_accuracy']:.3f}")
            except Exception as wf_error:
                print(f"⚠️ Walk-forward validation failed for {symbol}: {wf_error}")
            
            return True
        
        except Exception as e:
            print(f"❌ Error training model for {symbol}: {e}")
            if self.error_handler:
                self.error_handler.handle_ml_error(e, symbol, "training")
            
            self._record_training_failure(symbol, f"error: {str(e)[:100]}")
            return False
        
        except Exception as e:
            print(f"❌ Error training model for {symbol}: {e}")
            if self.error_handler:
                self.error_handler.handle_ml_error(e, symbol, "training")
            
            self._record_training_failure(symbol, f"error: {str(e)[:100]}")
            return False

    def walk_forward_validation_enhanced(self, symbol: str, df: pd.DataFrame, n_splits: int = None) -> Dict:
        """Enhanced walk-forward validation with multi-timeframe targets"""
        if n_splits is None:
            n_splits = self.walk_forward_splits
            
        try:
            features, target = self.prepare_training_data_enhanced(df)
            
            if features.empty or target.empty:
                return {'success': False, 'reason': 'Insufficient data'}

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=50, gap=10)
            performances = []
            feature_importances = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
                if len(train_idx) < 50 or len(test_idx) < 10:
                    continue
                    
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

                # Apply feature selection for each fold
                X_train_selected, selected_features = self._select_features(X_train, y_train, f"{symbol}_fold_{fold}")
                X_test_selected = X_test[selected_features]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                X_test_scaled = scaler.transform(X_test_selected)

                # Use optimized parameters for validation
                model = RandomForestClassifier(**self.rf_params)
                model.fit(X_train_scaled, y_train)

                pred = model.predict(X_test_scaled)
                pred_proba = model.predict_proba(X_test_scaled)
                
                accuracy = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                
                # Enhanced metrics
                confidence = np.max(pred_proba, axis=1).mean()
                balanced_accuracy = self._calculate_balanced_accuracy(y_test, pred)
                
                # Class distribution analysis
                unique, counts = np.unique(y_test, return_counts=True)
                class_distribution = dict(zip(unique, counts))
                
                performances.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confidence': confidence,
                    'balanced_accuracy': balanced_accuracy,
                    'fold_size': len(test_idx),
                    'class_distribution': class_distribution
                })
                
                feature_importances.append(model.feature_importances_)

            if performances:
                accuracies = [p['accuracy'] for p in performances]
                f1_scores = [p['f1'] for p in performances]
                confidences = [p['confidence'] for p in performances]
                
                result = {
                    'success': True,
                    'avg_accuracy': np.mean(accuracies),
                    'avg_precision': np.mean([p['precision'] for p in performances]),
                    'avg_recall': np.mean([p['recall'] for p in performances]),
                    'avg_f1': np.mean(f1_scores),
                    'avg_confidence': np.mean(confidences),
                    'std_accuracy': np.std(accuracies),
                    'std_f1': np.std(f1_scores),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'consistency_score': 1 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0,
                    'n_successful_folds': len(performances),
                    'fold_details': performances
                }
                
                self.walk_forward_performance[symbol] = result
                
                if self.database:
                    self.database.store_system_event(
                        "WALK_FORWARD_VALIDATION_ENHANCED",
                        {
                            'symbol': symbol,
                            'avg_accuracy': result['avg_accuracy'],
                            'avg_precision': result['avg_precision'],
                            'avg_recall': result['avg_recall'],
                            'avg_f1': result['avg_f1'],
                            'n_folds': result['n_successful_folds'],
                            'stability': result['consistency_score'],
                            'confidence': result['avg_confidence'],
                            'target_type': 'multi_timeframe_enhanced'
                        },
                        "INFO",
                        "ML Validation"
                    )
                
                return result
            else:
                return {'success': False, 'reason': 'No successful folds'}

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_ml_error(e, symbol, "walk_forward_validation")
            return {'success': False, 'reason': str(e)}

    def predict_enhanced(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Enhanced prediction with robust data validation"""
        if symbol not in self.models:
            return self.fallback_prediction_strategy(symbol, df)
            
        try:
            # --- DATA VALIDATION ---
            if df is None or len(df) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
                return self.fallback_prediction_strategy(symbol, df)
                
            # Check for valid close prices
            close_prices = df['close'].astype(float)
            if close_prices.isnull().all() or (close_prices == 0).all():
                self.logger.warning(f"Invalid close prices for {symbol} ML prediction")
                return self.fallback_prediction_strategy(symbol, df)
            
            if len(df) < 50:
                return self.fallback_prediction_strategy(symbol, df)
                
            features = self.prepare_features_point_in_time(df, len(df)-1)
            
            if features.empty:
                return self.fallback_prediction_strategy(symbol, df)
            
            # Apply feature selection
            selected_features = self.feature_importance.get(symbol, {}).get('selected_features', [])
            if selected_features:
                missing_features = set(selected_features) - set(features.columns)
                if missing_features:
                    available_features = [f for f in selected_features if f in features.columns]
                    if not available_features:
                        return self.fallback_prediction_strategy(symbol, df)
                    features_selected = features[available_features]
                else:
                    features_selected = features[selected_features]
            else:
                features_selected = features
            
            drift_result = self.detect_feature_drift(symbol, features_selected)
                
            scaled_features = self.scalers[symbol].transform(features_selected)

            # Get predictions with enhanced confidence measures
            rf_pred = self.models[symbol]['rf'].predict(scaled_features)[0]
            gb_pred = self.models[symbol]['gb'].predict(scaled_features)[0]

            rf_proba = self.models[symbol]['rf'].predict_proba(scaled_features)[0]
            gb_proba = self.models[symbol]['gb'].predict_proba(scaled_features)[0]

            rf_confidence = max(rf_proba)
            gb_confidence = max(gb_proba)
            
            # Enhanced confidence calibration
            rf_confidence_calibrated = self._calibrate_confidence(rf_proba, rf_pred)
            gb_confidence_calibrated = self._calibrate_confidence(gb_proba, gb_pred)

            # Ensemble with previous model if available
            ensemble_used = False
            if symbol in self.previous_models:
                try:
                    previous_scaled_features = self.previous_scalers[symbol].transform(features_selected)
                    previous_rf_pred = self.previous_models[symbol]['rf'].predict(previous_scaled_features)[0]
                    previous_gb_pred = self.previous_models[symbol]['gb'].predict(previous_scaled_features)[0]
                    
                    previous_rf_proba = self.previous_models[symbol]['rf'].predict_proba(previous_scaled_features)[0]
                    previous_gb_proba = self.previous_models[symbol]['gb'].predict_proba(previous_scaled_features)[0]
                    
                    previous_rf_confidence = max(previous_rf_proba)
                    previous_gb_confidence = max(previous_gb_proba)
                    
                    # Enhanced blending with calibrated confidence
                    rf_pred = self._blend_predictions_enhanced(rf_pred, previous_rf_pred, rf_confidence_calibrated, previous_rf_confidence)
                    gb_pred = self._blend_predictions_enhanced(gb_pred, previous_gb_pred, gb_confidence_calibrated, previous_gb_confidence)
                    
                    # Blend confidences
                    rf_confidence = (rf_confidence_calibrated * self.ensemble_weight_current + 
                                   previous_rf_confidence * self.ensemble_weight_previous)
                    gb_confidence = (gb_confidence_calibrated * self.ensemble_weight_current + 
                                   previous_gb_confidence * self.ensemble_weight_previous)
                    
                    ensemble_used = True
                except Exception as e:
                    print(f"⚠️ Error using previous model ensemble for {symbol}: {e}")

            # Enhanced ensemble decision making
            if rf_pred == gb_pred:
                ensemble_vote = rf_pred
                confidence = (rf_confidence + gb_confidence) / 2
            else:
                # Consider the strength of prediction (distance from decision boundary)
                rf_strength = abs(rf_proba[1] - rf_proba[0]) if len(rf_proba) > 1 else rf_confidence
                gb_strength = abs(gb_proba[1] - gb_proba[0]) if len(gb_proba) > 1 else gb_confidence
                
                if rf_strength > gb_strength:
                    ensemble_vote = rf_pred
                    confidence = rf_confidence
                else:
                    ensemble_vote = gb_pred
                    confidence = gb_confidence

            # Convert enhanced targets to trading signals
            trading_signal = self._convert_to_trading_signal(ensemble_vote)
            
            result = {
                'prediction': trading_signal,
                'raw_prediction': ensemble_vote,
                'confidence': confidence,
                'rf_pred': rf_pred,
                'gb_pred': gb_pred,
                'rf_confidence': rf_confidence,
                'gb_confidence': gb_confidence,
                'timestamp': datetime.now(),
                'model_used': 'ml_ensemble_enhanced',
                'feature_drift': drift_result,
                'quality_metrics': self._get_current_quality_metrics(symbol),
                'ensemble_used': ensemble_used,
                'training_bars_used': self.model_versions.get(symbol, {}).get('training_bars_used', 0),
                'feature_count_used': len(selected_features) if selected_features else len(features.columns),
                'target_type': 'multi_timeframe_enhanced'
            }
            
            if self.database:
                self.database.store_prediction_quality(
                    symbol=symbol,
                    prediction=trading_signal,
                    actual=None,
                    confidence=confidence,
                    model_used='ml_ensemble_enhanced',
                    features_used=selected_features if selected_features else features.columns.tolist(),
                    ensemble_used=ensemble_used,
                    feature_count=len(selected_features) if selected_features else len(features.columns),
                    raw_prediction=ensemble_vote
                )
            
            self._update_real_time_monitoring(symbol, result)
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            return self.fallback_prediction_strategy(symbol, df)

    def _calibrate_confidence(self, probabilities: np.ndarray, prediction: int) -> float:
        """Calibrate confidence based on probability distribution"""
        try:
            if len(probabilities) <= 1:
                return probabilities[0] if len(probabilities) == 1 else 0.5
            
            # For multi-class, use the difference between top two probabilities
            sorted_probs = np.sort(probabilities)[::-1]
            if len(sorted_probs) > 1:
                confidence = sorted_probs[0] - sorted_probs[1]
            else:
                confidence = sorted_probs[0]
            
            # Apply non-linear scaling to emphasize high confidence
            calibrated = 1.0 / (1.0 + np.exp(-10 * (confidence - 0.5)))
            
            return min(1.0, max(0.0, calibrated))
            
        except:
            return probabilities[prediction] if prediction < len(probabilities) else 0.5

    def _blend_predictions_enhanced(self, current_pred: int, previous_pred: int, 
                                  current_confidence: float, previous_confidence: float) -> int:
        """Enhanced prediction blending with confidence weighting"""
        if current_pred == previous_pred:
            return current_pred
        
        # Weight by confidence squared to favor high-confidence predictions
        current_weight = current_confidence ** 2 * self.ensemble_weight_current
        previous_weight = previous_confidence ** 2 * self.ensemble_weight_previous
        
        if current_weight > previous_weight:
            return current_pred
        else:
            return previous_pred

    def _convert_to_trading_signal(self, raw_prediction: int) -> int:
        """Convert enhanced multi-class predictions to trading signals"""
        # Map: 2 -> 1 (Strong buy), 1 -> 1 (Buy), 0 -> 0 (Hold), -1 -> -1 (Sell), -2 -> -1 (Strong sell)
        if raw_prediction >= 1:
            return 1
        elif raw_prediction <= -1:
            return -1
        else:
            return 0

    def get_model_analytics(self, symbol: str) -> Dict:
        """Get comprehensive model analytics with enhanced metrics"""
        if symbol not in self.model_versions:
            return {'error': 'Model not found'}
        
        model_info = self.model_versions[symbol]
        feature_info = self.feature_importance.get(symbol, {})
        wf_perf = self.walk_forward_performance.get(symbol, {})
        
        analytics = {
            'performance_metrics': {
                'accuracy': model_info.get('accuracy', 0),
                'precision': model_info.get('rf_precision', 0),
                'recall': model_info.get('rf_recall', 0),
                'f1_score': model_info.get('rf_f1', 0),
                'walk_forward_accuracy': wf_perf.get('avg_accuracy', 0),
                'consistency_score': wf_perf.get('consistency_score', 0)
            },
            'training_characteristics': {
                'training_samples': model_info.get('training_samples', 0),
                'feature_count': model_info.get('feature_count', 0),
                'target_type': model_info.get('target_type', 'unknown'),
                'hyperparameters_optimized': model_info.get('hyperparameters_optimized', False),
                'training_duration_seconds': model_info.get('training_duration_seconds', 0)
            },
            'model_health': self._assess_model_health_enhanced(symbol),
            'feature_analysis': {
                'selected_features': feature_info.get('selected_features', []),
                'rf_importance': feature_info.get('rf_importance', []),
                'gb_importance': feature_info.get('gb_importance', []),
                'original_feature_count': feature_info.get('original_feature_count', 0)
            },
            'validation_metrics': {
                'overfit_scores': model_info.get('overfit_scores', {}),
                'walk_forward_folds': wf_perf.get('n_successful_folds', 0),
                'fold_performance_range': f"{wf_perf.get('min_accuracy', 0):.3f}-{wf_perf.get('max_accuracy', 0):.3f}"
            }
        }
        
        return analytics

    def _assess_model_health_enhanced(self, symbol: str) -> Dict:
        """Enhanced model health assessment"""
        model_info = self.model_versions.get(symbol, {})
        wf_perf = self.walk_forward_performance.get(symbol, {})
        
        health_score = 0
        issues = []
        recommendations = []
        
        # Accuracy assessment (30 points)
        accuracy = model_info.get('accuracy', 0)
        if accuracy > 0.65:
            health_score += 30
        elif accuracy > 0.55:
            health_score += 20
            issues.append("Moderate accuracy")
            recommendations.append("Consider feature engineering or more training data")
        else:
            health_score += 10
            issues.append("Low accuracy")
            recommendations.append("Retrain with different parameters or more data")
        
        # Overfitting assessment (25 points)
        overfit_scores = model_info.get('overfit_scores', {})
        max_overfit = max(overfit_scores.get('rf', 0), overfit_scores.get('gb', 0))
        if max_overfit < 0.08:
            health_score += 25
        elif max_overfit < 0.15:
            health_score += 15
            issues.append("Mild overfitting")
            recommendations.append("Increase regularization or reduce model complexity")
        else:
            health_score += 5
            issues.append("Significant overfitting")
            recommendations.append("Reduce feature count or increase regularization")
        
        # Walk-forward consistency (25 points)
        consistency = wf_perf.get('consistency_score', 0)
        if consistency > 0.8:
            health_score += 25
        elif consistency > 0.6:
            health_score += 15
            issues.append("Moderate performance variance")
            recommendations.append("Model may be sensitive to market regime changes")
        else:
            health_score += 5
            issues.append("High performance variance")
            recommendations.append("Consider ensemble methods or regime-specific models")
        
        # Feature quality (20 points)
        feature_count = model_info.get('feature_count', 0)
        if 15 <= feature_count <= 30:
            health_score += 20
        elif 10 <= feature_count <= 40:
            health_score += 15
            issues.append("Suboptimal feature count")
            recommendations.append("Review feature selection strategy")
        else:
            health_score += 5
            issues.append("Poor feature selection")
            recommendations.append("Re-evaluate feature engineering and selection")
        
        # Determine health rating
        if health_score >= 85:
            rating = 'EXCELLENT'
        elif health_score >= 70:
            rating = 'GOOD'
        elif health_score >= 50:
            rating = 'FAIR'
        else:
            rating = 'POOR'
        
        return {
            'health_score': health_score,
            'health_rating': rating,
            'issues': issues,
            'recommendations': recommendations,
            'components': {
                'accuracy_score': min(30, int(accuracy * 30)),
                'overfitting_score': 25 - min(20, int(max_overfit * 100)),
                'consistency_score': min(25, int(consistency * 25)),
                'feature_score': min(20, feature_count)
            }
        }

    # Keep all existing helper methods with their original implementations
    # (_calculate_rsi_point_in_time, _calculate_atr_point_in_time, etc.)
    
    def _calculate_rsi_point_in_time(self, prices: pd.Series, period: int) -> float:
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_atr_point_in_time(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else (high.iloc[-1] - low.iloc[-1])
        except:
            return high.iloc[-1] - low.iloc[-1]

    def _calculate_bollinger_bands_point_in_time(self, series: pd.Series, period: int, std_dev: float) -> Tuple[float, float, float]:
        try:
            middle = series.rolling(period).mean()
            std = series.rolling(period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return (upper.iloc[-1], middle.iloc[-1], lower.iloc[-1])
        except:
            current_price = series.iloc[-1]
            return (current_price * 1.1, current_price, current_price * 0.9)

    def _calculate_macd_point_in_time(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        try:
            exp1 = series.ewm(span=fast).mean()
            exp2 = series.ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            return macd.iloc[-1], macd_signal.iloc[-1]
        except:
            return 0.0, 0.0

    def _calculate_market_regime_feature(self, close: pd.Series, high: pd.Series, low: pd.Series) -> float:
        try:
            if len(close) < 50:
                return 0.0
            
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_100 = close.rolling(100).mean()
            
            trend_strength = 0.0
            if sma_20.iloc[-1] > sma_50.iloc[-1] > sma_100.iloc[-1]:
                trend_strength = 1.0
            elif sma_20.iloc[-1] < sma_50.iloc[-1] < sma_100.iloc[-1]:
                trend_strength = -1.0
                
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            if volatility > 0.03:
                trend_strength *= 0.7
            elif volatility < 0.01:
                trend_strength *= 1.3
                
            return trend_strength
        except:
            return 0.0

    def _calculate_volatility_regime_feature(self, close: pd.Series) -> float:
        try:
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            if volatility > 0.03:
                return 1.0
            elif volatility < 0.01:
                return -1.0
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_price_efficiency(self, close: pd.Series) -> float:
        try:
            if len(close) < 20:
                return 0.5
                
            returns = close.pct_change().dropna()
            if len(returns) < 10:
                return 0.5
                
            abs_returns = returns.abs()
            net_movement = returns.sum()
            total_movement = abs_returns.sum()
            
            if total_movement == 0:
                return 0.5
                
            return abs(net_movement / total_movement)
        except:
            return 0.5

    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        try:
            lags = range(2, min(max_lag, len(series)//2))
            tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5

    def _calculate_market_microstructure_1(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        try:
            if len(close) < 10:
                return 0.0
                
            price_range = (high - low) / close
            volume_weighted_range = (price_range * volume).rolling(10).mean()
            return volume_weighted_range.iloc[-1] if not pd.isna(volume_weighted_range.iloc[-1]) else 0.0
        except:
            return 0.0

    def _calculate_market_microstructure_2(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        try:
            if len(close) < 10:
                return 0.0
                
            close_returns = close.pct_change()
            volume_returns = volume.pct_change()
            correlation = close_returns.rolling(10).corr(volume_returns)
            return correlation.iloc[-1] if not pd.isna(correlation.iloc[-1]) else 0.0
        except:
            return 0.0

    # Keep all other existing methods (feature selection, drift detection, etc.)
    # They remain largely the same but now work with enhanced targets

    def _select_features(self, X_train: pd.DataFrame, y_train: pd.Series, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection implementation"""
        try:
            if len(X_train.columns) <= self.max_features:
                return X_train, X_train.columns.tolist()
            
            if self.feature_selection_method == 'importance':
                rf_temp = RandomForestClassifier(**self.rf_params)
                rf_temp.fit(X_train, y_train)
                
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_temp.feature_importances_
                }).sort_values('importance', ascending=False)
                
                selected_features = importance_df.head(self.max_features)['feature'].tolist()
                X_selected = X_train[selected_features]
                
            elif self.feature_selection_method == 'rfe':
                estimator = RandomForestClassifier(**self.rf_params)
                selector = RFE(estimator, n_features_to_select=self.max_features, step=5)
                selector.fit(X_train, y_train)
                selected_features = X_train.columns[selector.support_].tolist()
                X_selected = X_train[selected_features]
                
            elif self.feature_selection_method == 'lasso':
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(X_train, y_train)
                
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'coefficient': lasso.coef_
                }).sort_values('coefficient', key=abs, ascending=False)
                
                selected_features = importance_df.head(self.max_features)['feature'].tolist()
                X_selected = X_train[selected_features]
                
            else:
                corr_matrix = X_train.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                selected_features = [col for col in X_train.columns if col not in to_drop][:self.max_features]
                X_selected = X_train[selected_features]
            
            print(f"🔍 Feature selection for {symbol}: {len(selected_features)} features selected from {len(X_train.columns)}")
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
            
            self.feature_importance[symbol]['selected_features'] = selected_features
            self.feature_importance[symbol]['original_feature_count'] = len(X_train.columns)
            
            return X_selected, selected_features
            
        except Exception as e:
            print(f"⚠️ Feature selection failed for {symbol}: {e}")
            variances = X_train.var().sort_values(ascending=False)
            selected_features = variances.head(self.max_features).index.tolist()
            return X_train[selected_features], selected_features

    def _apply_pca(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            pca = PCA(n_components=self.pca_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            
            if X_test is not None:
                X_test_pca = pca.transform(X_test)
                return X_train_pca, X_test_pca
            else:
                return X_train_pca, None
                
        except Exception as e:
            print(f"⚠️ PCA failed: {e}")
            if X_test is not None:
                return X_train.values, X_test.values
            else:
                return X_train.values, None

    def time_series_train_test_split(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2):
        if len(features) < 100:
            return None, None, None, None
            
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        if len(X_train) < 50 or len(X_test) < 10:
            return None, None, None, None
            
        return X_train, X_test, y_train, y_test

    def _record_training_failure(self, symbol: str, reason: str):
        if not hasattr(self, 'model_versions'):
            self.model_versions = {}
            
        self.model_versions[symbol] = {
            'version': f"v{datetime.now().strftime('%Y%m%d_%H%M')}_{reason}",
            'training_date': datetime.now(),
            'accuracy': 0.5,
            'rf_accuracy': 0.5,
            'gb_accuracy': 0.5,
            'rf_precision': 0.5,
            'gb_precision': 0.5,
            'rf_recall': 0.5,
            'gb_recall': 0.5,
            'rf_f1': 0.5,
            'gb_f1': 0.5,
            'training_samples': 0,
            'test_samples': 0,
            'status': reason
        }

    def _store_training_results(self, symbol: str, model_version: str, training_duration: float,
                              rf_accuracy: float, gb_accuracy: float, rf_precision: float, gb_precision: float,
                              rf_recall: float, gb_recall: float, rf_f1: float, gb_f1: float,
                              train_samples: int, test_samples: int, bars_used: int, features: List[str],
                              rf_params: dict, gb_params: dict):
        try:
            self.database.store_ml_model_performance(symbol, {
                'accuracy': (rf_accuracy + gb_accuracy) / 2,
                'precision': (rf_precision + gb_precision) / 2,
                'recall': (rf_recall + gb_recall) / 2,
                'f1_score': (rf_f1 + gb_f1) / 2,
                'rf_accuracy': rf_accuracy,
                'gb_accuracy': gb_accuracy,
                'rf_precision': rf_precision,
                'gb_precision': gb_precision,
                'rf_recall': rf_recall,
                'gb_recall': gb_recall,
                'rf_f1': rf_f1,
                'gb_f1': gb_f1,
                'training_samples': train_samples,
                'test_samples': test_samples,
                'model_version': model_version,
                'training_bars_used': bars_used,
                'feature_count': len(features),
                'target_type': 'multi_timeframe_enhanced',
                'hyperparameters_optimized': self.enable_hyperparameter_optimization
            })
            
            # Store hyperparameters
            self.database.store_system_event(
                "HYPERPARAMETER_OPTIMIZATION",
                {
                    'symbol': symbol,
                    'rf_params': rf_params,
                    'gb_params': gb_params,
                    'feature_selection_method': self.feature_selection_method
                },
                "INFO",
                "ML Training"
            )
            
        except Exception as db_error:
            print(f"⚠️ Failed to save ML data to database for {symbol}: {db_error}")

    def _calculate_balanced_accuracy(self, y_true, y_pred):
        try:
            classes = np.unique(y_true)
            class_accuracies = []
            
            for cls in classes:
                if np.sum(y_true == cls) > 0:
                    cls_accuracy = np.mean(y_pred[y_true == cls] == cls)
                    class_accuracies.append(cls_accuracy)
            
            return np.mean(class_accuracies) if class_accuracies else 0.0
        except:
            return accuracy_score(y_true, y_pred)

    def _initialize_feature_drift_detector(self, symbol: str, X_train: np.ndarray):
        try:
            detector = EllipticEnvelope(contamination=0.1, random_state=42)
            detector.fit(X_train)
            self.feature_drift_detector[symbol] = detector
        except Exception as e:
            self.logger.warning(f"Could not initialize feature drift detector for {symbol}: {e}")

    def _initialize_prediction_quality_tracking(self, symbol: str):
        self.prediction_quality_metrics[symbol] = {
            'predictions': [],
            'actuals': [],
            'timestamps': [],
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': []
        }

    def detect_feature_drift(self, symbol: str, current_features: pd.DataFrame) -> Dict:
        try:
            if symbol not in self.feature_drift_detector:
                return {'drift_detected': False, 'reason': 'No drift detector initialized'}
            
            if current_features.empty:
                return {'drift_detected': False, 'reason': 'No current features'}
            
            if symbol not in self.scalers:
                return {'drift_detected': False, 'reason': 'No scaler available'}
                
            scaler = self.scalers[symbol]
            
            if len(current_features) == 1:
                current_scaled = scaler.transform(current_features)
            else:
                expected_features = getattr(scaler, 'n_features_in_', current_features.shape[1])
                if current_features.shape[1] != expected_features:
                    return {'drift_detected': False, 'reason': f'Feature dimension mismatch: {current_features.shape[1]} vs {expected_features}'}
                current_scaled = scaler.transform(current_features)
            
            detector = self.feature_drift_detector[symbol]
            drift_scores = detector.decision_function(current_scaled)
            
            threshold = np.percentile(drift_scores, 10)
            drift_detected = np.any(drift_scores < threshold)
            
            drift_ratio = np.sum(drift_scores < threshold) / len(drift_scores) if len(drift_scores) > 0 else 0
            
            result = {
                'drift_detected': drift_detected,
                'drift_ratio': drift_ratio,
                'drift_scores': drift_scores.tolist(),
                'threshold_violations': np.sum(drift_scores < 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting feature drift for {symbol}: {e}")
            return {'drift_detected': False, 'reason': str(e)}

    def monitor_prediction_quality(self, symbol: str, prediction: int, actual: int, timestamp: datetime):
        try:
            if symbol not in self.prediction_quality_metrics:
                self._initialize_prediction_quality_tracking(symbol)
            
            metrics = self.prediction_quality_metrics[symbol]
            metrics['predictions'].append(prediction)
            metrics['actuals'].append(actual)
            metrics['timestamps'].append(timestamp)
            
            if len(metrics['predictions']) > self.quality_metrics_window:
                metrics['predictions'].pop(0)
                metrics['actuals'].pop(0)
                metrics['timestamps'].pop(0)
            
            if len(metrics['predictions']) >= 10:
                accuracy = accuracy_score(metrics['actuals'], metrics['predictions'])
                precision = precision_score(metrics['actuals'], metrics['predictions'], average='weighted', zero_division=0)
                recall = recall_score(metrics['actuals'], metrics['predictions'], average='weighted', zero_division=0)
                f1 = f1_score(metrics['actuals'], metrics['predictions'], average='weighted', zero_division=0)
                
                metrics['accuracies'].append(accuracy)
                metrics['precisions'].append(precision)
                metrics['recalls'].append(recall)
                metrics['f1_scores'].append(f1)
                
                if len(metrics['accuracies']) > 20:
                    metrics['accuracies'].pop(0)
                    metrics['precisions'].pop(0)
                    metrics['recalls'].pop(0)
                    metrics['f1_scores'].pop(0)
                
        except Exception as e:
            self.logger.error(f"Error monitoring prediction quality for {symbol}: {e}")

    def _get_current_quality_metrics(self, symbol: str) -> Dict:
        try:
            if symbol not in self.prediction_quality_metrics:
                return {}
                
            metrics = self.prediction_quality_metrics[symbol]
            if not metrics['accuracies']:
                return {}
                
            return {
                'recent_accuracy': metrics['accuracies'][-1] if metrics['accuracies'] else 0,
                'recent_precision': metrics['precisions'][-1] if metrics['precisions'] else 0,
                'recent_recall': metrics['recalls'][-1] if metrics['recalls'] else 0,
                'recent_f1': metrics['f1_scores'][-1] if metrics['f1_scores'] else 0,
                'accuracy_trend': np.mean(metrics['accuracies'][-5:]) if len(metrics['accuracies']) >= 5 else 0,
                'monitoring_window': len(metrics['predictions'])
            }
        except:
            return {}

    def _update_real_time_monitoring(self, symbol: str, prediction_result: Dict):
        try:
            if symbol not in self.real_time_monitor:
                self.real_time_monitor[symbol] = {
                    'prediction_count': 0,
                    'confidence_sum': 0,
                    'drift_alerts': 0,
                    'quality_alerts': 0,
                    'last_update': datetime.now()
                }
            
            monitor = self.real_time_monitor[symbol]
            monitor['prediction_count'] += 1
            monitor['confidence_sum'] += prediction_result['confidence']
            monitor['last_update'] = datetime.now()
            
            if prediction_result.get('feature_drift', {}).get('drift_detected', False):
                monitor['drift_alerts'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating real-time monitoring for {symbol}: {e}")

    def fallback_prediction_strategy(self, symbol: str, df: pd.DataFrame) -> Dict:
        try:
            close_prices = df['close'].astype(float)
            
            momentum_5 = close_prices.pct_change(5).iloc[-1]
            sma_5 = close_prices.rolling(5).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1]
            rsi = self._calculate_rsi_point_in_time(close_prices, 14)
            
            if (sma_5 > sma_20 > sma_50 and momentum_5 > 0 and rsi < 70):
                prediction = 1
                confidence = 0.65
            elif (sma_5 < sma_20 < sma_50 and momentum_5 < 0 and rsi > 30):
                prediction = -1
                confidence = 0.65
            else:
                prediction = 0
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'fallback_method': 'enhanced_technical_analysis',
                'momentum_5': momentum_5,
                'sma_5': sma_5,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi
            }
            
        except Exception as e:
            return {
                'prediction': 0,
                'confidence': 0.3,
                'fallback_method': 'error_fallback'
            }

    def update_prediction_actual(self, symbol: str, prediction_timestamp: datetime, actual: int):
        try:
            if self.database:
                self.logger.info(f"Updated prediction actual for {symbol} at {prediction_timestamp}: {actual}")
                
            if symbol in self.prediction_quality_metrics:
                metrics = self.prediction_quality_metrics[symbol]
                if len(metrics['actuals']) > 0:
                    metrics['actuals'][-1] = actual
                        
        except Exception as e:
            self.logger.error(f"Error updating prediction actual for {symbol}: {e}")

    def get_model_info(self) -> Dict:
        info = {
            'total_models': len(self.models),
            'model_versions': self.model_versions,
            'walk_forward_performance': self.walk_forward_performance,
            'feature_importance_available': len(self.feature_importance) > 0,
            'prediction_quality_metrics': {
                symbol: self._get_current_quality_metrics(symbol)
                for symbol in self.prediction_quality_metrics
            },
            'real_time_monitoring': self.real_time_monitor,
            'previous_models_available': len(self.previous_models),
            'ensemble_config': {
                'current_weight': self.ensemble_weight_current,
                'previous_weight': self.ensemble_weight_previous
            },
            'model_health': self._get_model_health_summary(),
            'feature_selection_config': {
                'max_features': self.max_features,
                'method': self.feature_selection_method,
                'use_pca': self.use_pca
            },
            'training_config': {
                'target_type': 'multi_timeframe_enhanced',
                'hyperparameter_optimization': self.enable_hyperparameter_optimization,
                'multi_timeframe_configs': self.target_configs
            }
        }
        return info
    
    def _get_model_health_summary(self) -> Dict:
        health_summary = {}
        for symbol, version_info in self.model_versions.items():
            accuracy = version_info.get('accuracy', 0.5)
            training_date = version_info.get('training_date')
            
            if training_date:
                if isinstance(training_date, str):
                    try:
                        training_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                    except ValueError:
                        training_date = datetime.now()
                
                days_since_training = (datetime.now() - training_date).days
                needs_retraining = days_since_training > 30 or accuracy < 0.55
            else:
                needs_retraining = accuracy < 0.55
            
            health_summary[symbol] = {
                'accuracy': accuracy,
                'rf_accuracy': version_info.get('rf_accuracy', 0),
                'gb_accuracy': version_info.get('gb_accuracy', 0),
                'training_date': training_date.strftime('%Y-%m-%d') if training_date else 'Unknown',
                'days_since_training': days_since_training if training_date else 999,
                'needs_retraining': needs_retraining,
                'status': 'excellent' if accuracy > 0.7 else 'good' if accuracy > 0.6 else 'fair' if accuracy > 0.55 else 'poor',
                'feature_count': version_info.get('feature_count', 0),
                'overfit_score': version_info.get('overfit_scores', {}).get('rf', 0),
                'target_type': version_info.get('target_type', 'unknown')
            }
        
        return health_summary

    def force_reset_models(self):
        print("🔄 FORCE RESETTING ALL MODELS AND DATA...")
        
        self.models = {}
        self.previous_models = {}
        self.scalers = {}
        self.previous_scalers = {}
        self.model_versions = {}
        self.feature_importance = {}
        self.walk_forward_performance = {}
        self.performance_history = {}  # Add this line
        self.prediction_quality_metrics = {}
        self.feature_drift_detector = {}
        self.real_time_monitor = {}
        
        if hasattr(self, 'training_in_progress'):
            self.training_in_progress = False
        if hasattr(self, 'current_training_symbol'):
            self.current_training_symbol = None
        
        print("✅ All models and data have been reset!")
        
        import shutil
        try:
            shutil.rmtree("ml_models", ignore_errors=True)
            print("✅ Deleted ml_models directory")
        except:
            pass
        
        return True

    def save_models(self, base_path: str = "ml_models") -> bool:
        try:
            if not os.path.exists(base_path):
                os.makedirs(base_path)
                
            saved_count = 0
            
            try:
                metadata_path = os.path.join(base_path, "_metadata.joblib")
                metadata_to_save = {
                    'model_versions': self.model_versions,
                    'feature_importance': self.feature_importance,
                    'performance_history': self.performance_history,
                    'ensemble_config': {
                        'current_weight': self.ensemble_weight_current,
                        'previous_weight': self.ensemble_weight_previous
                    },
                    'feature_selection_config': {
                        'max_features': self.max_features,
                        'method': self.feature_selection_method,
                        'use_pca': self.use_pca
                    },
                    'training_config': {
                        'target_configs': self.target_configs,
                        'enable_hpo': self.enable_hyperparameter_optimization
                    }
                }
                joblib.dump(metadata_to_save, metadata_path, compress=3)
                print("✅ Saved enhanced ML metadata (_metadata.joblib)")
            except Exception as meta_e:
                print(f"⚠️ Failed to save ML metadata: {meta_e}")

            for symbol, model_data in self.models.items():
                model_path = os.path.join(base_path, f"{symbol}_model.joblib")
                scaler_path = os.path.join(base_path, f"{symbol}_scaler.joblib")
                drift_detector_path = os.path.join(base_path, f"{symbol}_drift_detector.joblib")
                
                previous_model_path = os.path.join(base_path, f"{symbol}_previous_model.joblib")
                previous_scaler_path = os.path.join(base_path, f"{symbol}_previous_scaler.joblib")
                
                model_data_to_save = {
                    'rf': model_data['rf'],
                    'gb': model_data['gb'],
                    'model_version': self.model_versions.get(symbol, {}).get('version', 'v1'),
                    'trained_date': datetime.now().isoformat()
                }
                
                joblib.dump(model_data_to_save, model_path, compress=3)
                joblib.dump(self.scalers[symbol], scaler_path, compress=3)
                
                if symbol in self.feature_drift_detector:
                    joblib.dump(self.feature_drift_detector[symbol], drift_detector_path, compress=3)
                
                if symbol in self.previous_models:
                    previous_model_data_to_save = {
                        'rf': self.previous_models[symbol]['rf'],
                        'gb': self.previous_models[symbol]['gb'],
                        'model_version': 'previous',
                        'trained_date': 'previous'
                    }
                    joblib.dump(previous_model_data_to_save, previous_model_path, compress=3)
                    joblib.dump(self.previous_scalers[symbol], previous_scaler_path, compress=3)
                    print(f"💾 Saved previous model for {symbol}")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    saved_count += 1
            
            print(f"✅ Saved {saved_count} enhanced models with multi-timeframe targets and HPO")
            return saved_count > 0
            
        except Exception as e:
            print(f"❌ Error saving enhanced models: {e}")
            return False

    def load_models(self, base_path: str = "ml_models") -> bool:
        try:
            if not os.path.exists(base_path):
                return False
                
            loaded_count = 0

            metadata_path = os.path.join(base_path, "_metadata.joblib")
            if os.path.exists(metadata_path):
                try:
                    metadata = joblib.load(metadata_path)
                    self.model_versions = metadata.get('model_versions', {})
                    self.feature_importance = metadata.get('feature_importance', {})
                    self.performance_history = metadata.get('performance_history', {})
                    
                    ensemble_config = metadata.get('ensemble_config', {})
                    self.ensemble_weight_current = ensemble_config.get('current_weight', 0.8)
                    self.ensemble_weight_previous = ensemble_config.get('previous_weight', 0.2)
                    
                    feature_config = metadata.get('feature_selection_config', {})
                    self.max_features = feature_config.get('max_features', 25)
                    self.feature_selection_method = feature_config.get('method', 'importance')
                    self.use_pca = feature_config.get('use_pca', False)
                    
                    training_config = metadata.get('training_config', {})
                    self.target_configs = training_config.get('target_configs', self.target_configs)
                    self.enable_hyperparameter_optimization = training_config.get('enable_hpo', True)
                    
                    print("✅ Loaded enhanced ML metadata (_metadata.joblib)")
                    
                    if not hasattr(self, 'model_versions'):
                         self.model_versions = {}
                    if not hasattr(self, 'feature_importance'):
                         self.feature_importance = {}
                    if not hasattr(self, 'previous_models'):
                         self.previous_models = {}
                    if not hasattr(self, 'previous_scalers'):
                         self.previous_scalers = {}
                         
                except Exception as meta_e:
                    print(f"⚠️ Failed to load ML metadata: {meta_e}")
                    self.model_versions = {}
                    self.feature_importance = {}
                    self.previous_models = {}
                    self.previous_scalers = {}
            else:
                print("⚠️ No _metadata.joblib file found. Metadata will be empty.")
                self.model_versions = {}
                self.feature_importance = {}
                self.previous_models = {}
                self.previous_scalers = {}

            for filename in os.listdir(base_path):
                if filename.endswith("_model.joblib") and not filename.endswith("_previous_model.joblib"):
                    symbol = filename.replace("_model.joblib", "")
                    model_path = os.path.join(base_path, filename)
                    scaler_path = os.path.join(base_path, f"{symbol}_scaler.joblib")
                    drift_detector_path = os.path.join(base_path, f"{symbol}_drift_detector.joblib")
                    
                    previous_model_path = os.path.join(base_path, f"{symbol}_previous_model.joblib")
                    previous_scaler_path = os.path.join(base_path, f"{symbol}_previous_scaler.joblib")
                    
                    if os.path.exists(scaler_path):
                        try:
                            model_data = joblib.load(model_path)
                            scaler = joblib.load(scaler_path)
                            
                            self.models[symbol] = {'rf': model_data['rf'], 'gb': model_data['gb']}
                            self.scalers[symbol] = scaler
                            
                            if os.path.exists(drift_detector_path):
                                self.feature_drift_detector[symbol] = joblib.load(drift_detector_path)
                            
                            if os.path.exists(previous_model_path) and os.path.exists(previous_scaler_path):
                                previous_model_data = joblib.load(previous_model_path)
                                previous_scaler = joblib.load(previous_scaler_path)
                                
                                self.previous_models[symbol] = {'rf': previous_model_data['rf'], 'gb': previous_model_data['gb']}
                                self.previous_scalers[symbol] = previous_scaler
                                print(f"💾 Loaded previous model for {symbol}")
                            
                            self._initialize_prediction_quality_tracking(symbol)

                            if symbol not in self.model_versions:
                                print(f"ℹ️ No metadata found for {symbol}, creating fallback entry.")
                                self.model_versions[symbol] = {
                                    'version': model_data.get('model_version', 'v_loaded'),
                                    'training_date': datetime.fromisoformat(model_data.get('trained_date')) if 'trained_date' in model_data else datetime.now(),
                                    'accuracy': 0.5,
                                    'rf_accuracy': 0.5, 'gb_accuracy': 0.5,
                                    'rf_precision': 0.5, 'gb_precision': 0.5,
                                    'rf_recall': 0.5, 'gb_recall': 0.5,
                                    'rf_f1': 0.5, 'gb_f1': 0.5,
                                    'status': 'loaded_no_metadata'
                                }

                            loaded_count += 1
                        except Exception as e:
                            print(f"⚠️ Error loading model file {filename}: {e}")
                            continue
            
            print(f"📊 Loaded {loaded_count} enhanced models with multi-timeframe targets and HPO")
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def force_save_models(self):
        success = self.save_models()
        if success:
            print("💾 Models saved successfully")
        else:
            print("❌ Failed to save models")
        return success

    # Backward compatibility
    def predict(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Main prediction method - uses enhanced prediction by default"""
        return self.predict_enhanced(symbol, df)

    def prepare_training_data(self, df: pd.DataFrame, min_samples: int = None) -> tuple:
        """Main training data method - uses enhanced data by default"""
        return self.prepare_training_data_enhanced(df, min_samples)

    def walk_forward_validation(self, symbol: str, df: pd.DataFrame, n_splits: int = None) -> Dict:
        """Main validation method - uses enhanced validation by default"""
        return self.walk_forward_validation_enhanced(symbol, df, n_splits)
    
    def update_ml_prediction_outcomes(self, lookback_hours: int = 24):
        """
        Periodically checks recent closed trades, determines the actual outcome
        relative to the ML prediction, and stores it in the prediction_quality table.
        """
        if not self.database:
            self.logger.warning("Database not available, skipping ML outcome update.")
            return

        self.logger.info(f"Starting ML prediction outcome update (checking last {lookback_hours} hours)...")
        updated_count = 0
        failed_count = 0
        try:
            # 1. Fetch recently closed trades that haven't been updated yet
            start_time = datetime.now() - timedelta(hours=lookback_hours)
            query = """
                SELECT id, symbol, ml_prediction_details, pnl_percent, timestamp
                FROM trades
                WHERE exit_price IS NOT NULL
                  AND outcome_updated = 0
                  AND ml_prediction_details IS NOT NULL
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 100 -- Process in batches
            """ # Ensure timestamp comparison works with TEXT ISO format
            params = (start_time.isoformat(),)

            # Use pandas read_sql for easier processing
            closed_trades_df = pd.read_sql_query(query, self.database.connection, params=params, parse_dates=['timestamp'])

            if closed_trades_df.empty:
                self.logger.info("No recently closed trades found needing ML outcome update.")
                return

            self.logger.info(f"Found {len(closed_trades_df)} closed trades to process for ML outcome.")

            for index, trade in closed_trades_df.iterrows():
                try:
                    trade_id = trade['id']
                    pnl_percent = trade['pnl_percent']
                    ml_details_json = trade['ml_prediction_details']
                    symbol = trade['symbol']

                    if pd.isna(pnl_percent) or not ml_details_json:
                        self.logger.warning(f"Skipping trade {trade_id}: Missing PnL ({pnl_percent}) or ML details ({ml_details_json})")
                        # Mark as updated anyway to avoid reprocessing invalid data
                        self.database.update_trade_outcome_updated_flag(trade_id)
                        continue

                    # Parse ML prediction details from JSON
                    ml_pred_info = json.loads(ml_details_json)

                    # --- Determine Actual Outcome based on PnL ---
                    # Define a small threshold for 'hold' outcomes
                    hold_threshold = 0.1 # e.g., +/- 0.1% PnL is considered neutral
                    actual_outcome = 0
                    if pnl_percent > hold_threshold:
                        actual_outcome = 1 # Profit -> Actual 'BUY' direction was correct (or 'SELL' direction was wrong)
                    elif pnl_percent < -hold_threshold:
                        actual_outcome = -1 # Loss -> Actual 'SELL' direction was correct (or 'BUY' direction was wrong)
                    else:
                        actual_outcome = 0 # Neutral/Hold

                    # --- Extract prediction details ---
                    # Use the raw prediction score (e.g., -2, -1, 0, 1, 2) if available
                    raw_prediction = ml_pred_info.get('raw_prediction', ml_pred_info.get('prediction', 0)) # Fallback to final signal
                    prediction_signal = self._convert_to_trading_signal(raw_prediction) # Ensure it's -1, 0, or 1

                    # Important: Determine 'correctness' based on the *signal*, not the raw score
                    # Example: Predicted BUY (1), Actual was profit (1) -> Correct
                    # Example: Predicted SELL (-1), Actual was profit (1) -> Incorrect
                    # Example: Predicted BUY (1), Actual was loss (-1) -> Incorrect
                    # Example: Predicted SELL (-1), Actual was loss (-1) -> Correct
                    # Example: Predicted HOLD (0), Actual was profit/loss (1/-1) -> Incorrect (treat 0 PNL as correct hold?)
                    # Example: Predicted BUY/SELL (1/-1), Actual was neutral (0) -> Incorrect? Or partially correct? Let's treat as incorrect for simplicity.

                    is_correct = (prediction_signal == actual_outcome)

                    # Get other details needed for store_prediction_quality
                    pred_confidence = ml_pred_info.get('confidence', 0.5)
                    pred_model_used = ml_pred_info.get('model_used', 'unknown')
                    pred_features_used = ml_pred_info.get('features_used', []) # Assuming features were stored
                    pred_ensemble_used = ml_pred_info.get('ensemble_used', False)
                    pred_feature_count = ml_pred_info.get('feature_count_used', 0)

                    # Use the prediction timestamp if stored, otherwise fallback to trade timestamp (less accurate)
                    pred_timestamp_iso = ml_pred_info.get('timestamp')
                    if pred_timestamp_iso:
                        # Attempt to parse ISO string back to datetime
                        try:
                             pred_timestamp = datetime.fromisoformat(pred_timestamp_iso.replace('Z', '+00:00'))
                        except:
                             pred_timestamp = trade['timestamp'] # Fallback
                    else:
                        pred_timestamp = trade['timestamp'] # Fallback

                    # 3. Store the prediction quality record
                    store_success = self.database.store_prediction_quality(
                        symbol=symbol,
                        prediction=prediction_signal, # Use the simplified signal
                        actual=actual_outcome,
                        confidence=pred_confidence,
                        # 'correct' will be calculated by store_prediction_quality based on prediction/actual
                        model_used=pred_model_used,
                        features_used=pred_features_used,
                        ensemble_used=pred_ensemble_used,
                        feature_count=pred_feature_count,
                        raw_prediction=raw_prediction,
                        timestamp=pred_timestamp # Pass the timestamp object
                    )

                    # 4. Mark the trade as updated in the trades table
                    if store_success:
                        update_flag_success = self.database.update_trade_outcome_updated_flag(trade_id)
                        if update_flag_success:
                            updated_count += 1
                            self.logger.debug(f"Successfully updated ML outcome for trade {trade_id}")
                        else:
                            failed_count += 1
                            self.logger.warning(f"Stored prediction quality for trade {trade_id}, but failed to update outcome flag.")
                    else:
                        failed_count += 1
                        self.logger.warning(f"Failed to store prediction quality for trade {trade_id}.")

                except json.JSONDecodeError as json_e:
                    failed_count += 1
                    self.logger.error(f"Error decoding ML details for trade {trade.get('id', 'N/A')}: {json_e}")
                    # Mark as updated to avoid retrying bad JSON
                    self.database.update_trade_outcome_updated_flag(trade.get('id'))
                except Exception as inner_e:
                    failed_count += 1
                    self.logger.error(f"Error processing outcome for trade {trade.get('id', 'N/A')}: {inner_e}", exc_info=True)
                    # Optionally mark as updated to avoid infinite loops on problematic trades
                    # self.database.update_trade_outcome_updated_flag(trade.get('id'))


            self.logger.info(f"ML prediction outcome update complete. Updated: {updated_count}, Failed/Skipped: {failed_count}")

        except Exception as e:
            self.logger.error(f"Critical error during ML outcome update process: {e}", exc_info=True)
            if self.error_handler:
                self.error_handler.handle_ml_error(e, "ALL", "outcome_update_process")

    # Add this method to the MLPredictor class
    def analyze_model_performance_issues(self) -> Dict:
        """Analyze why models are producing too many 'Hold' predictions"""
        analysis = {}
        self.logger.info("Analyzing ML model performance issues (Hold prediction ratio)...") # Added logging

        for symbol, model_info in self.model_versions.items():
            if symbol not in self.models:
                continue

            # Check model performance metrics
            accuracy = model_info.get('accuracy', 0)
            training_samples = model_info.get('training_samples', 0)
            feature_count = model_info.get('feature_count', 0)

            # Analyze prediction distribution from recent predictions
            hold_ratio = self._calculate_hold_prediction_ratio(symbol)

            analysis[symbol] = {
                'accuracy': accuracy,
                'training_samples': training_samples,
                'feature_count': feature_count,
                'hold_prediction_ratio': hold_ratio,
                'issues': [],
                'recommendations': []
            }

            # Identify issues
            if hold_ratio > 0.7: # If more than 70% of recent predictions are HOLD
                analysis[symbol]['issues'].append("High proportion of 'Hold' predictions (>70%)")
                # Removed the retraining recommendation from here, as it's just analysis
            elif hold_ratio < 0.1: # Added check for unusually LOW hold ratio
                 analysis[symbol]['issues'].append("Very low proportion of 'Hold' predictions (<10%)")
                 analysis[symbol]['recommendations'].append("Check target definition or model calibration")


            if accuracy < 0.55:
                analysis[symbol]['issues'].append(f"Low accuracy ({accuracy:.3f})")
                analysis[symbol]['recommendations'].append("Consider retraining or feature review")

            if training_samples < self.min_training_samples: # Use class attribute
                analysis[symbol]['issues'].append(f"Insufficient training samples ({training_samples})")
                analysis[symbol]['recommendations'].append("Collect more historical data before training")

        self.logger.info(f"ML performance issue analysis complete for {len(analysis)} symbols.")
        return analysis

    # Add this helper method to the MLPredictor class
    def _calculate_hold_prediction_ratio(self, symbol: str) -> float:
        """Calculate ratio of hold predictions from recent predictions"""
        if symbol not in self.prediction_quality_metrics:
            self.logger.warning(f"_calculate_hold_prediction_ratio: No quality metrics found for {symbol}")
            return 0.5 # Return neutral if no data

        metrics = self.prediction_quality_metrics[symbol]
        predictions = metrics.get('predictions', []) # Get the list of raw predictions (-2 to 2)

        if not predictions:
            self.logger.debug(f"_calculate_hold_prediction_ratio: No recent predictions tracked for {symbol}")
            return 0.5 # Return neutral if no predictions tracked

        # Count HOLD predictions (raw prediction == 0)
        hold_count = sum(1 for pred in predictions if pred == 0)
        total_predictions = len(predictions)
        hold_ratio = hold_count / total_predictions if total_predictions > 0 else 0.0

        self.logger.debug(f"_calculate_hold_prediction_ratio for {symbol}: {hold_count}/{total_predictions} holds ({hold_ratio:.2f})")
        return hold_ratio