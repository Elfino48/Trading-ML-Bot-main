import json
import time
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
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
        self.performance_threshold = 0.60
        self.feature_importance = {}
        self.walk_forward_performance = {}
        self.model_versions = {}
        self.performance_history = {}
        self.prediction_quality_metrics = {}
        self.feature_drift_detector = {}
        self.real_time_monitor = {}
        self.logger = logging.getLogger('MLPredictor')
        self.detailed_logger = logging.getLogger('MLDetailed') # For enhanced logging

        # Configuration for Deepseek's enhanced pipeline
        self.enhanced_config = {
            'min_training_samples': 400,
            'test_size': 0.25,
            'validation_gap': 15,
            'max_overfit_threshold': 0.15,
            'feature_selection_method': 'advanced',
            'target_lookahead': [5, 10, 15],  # Multiple timeframes
            'volatility_adjustment': True,
            'enable_smote': True,
            'symbol_specific_params': {
                'BTCUSDT': {'n_estimators': 200, 'max_depth': 10},
                'ETHUSDT': {'n_estimators': 180, 'max_depth': 9},
                'BNBUSDT': {'n_estimators': 150, 'max_depth': 8},
                'XRPUSDT': {'n_estimators': 120, 'max_depth': 7},
                'SOLUSDT': {'n_estimators': 120, 'max_depth': 7},
                'DOGEUSDT': {'n_estimators': 100, 'max_depth': 6}
            }
        }
        
        self.auto_save_interval = 10
        self.training_count = 0
        
        # Enhanced configuration - DEFINE ALL CONFIGS BEFORE USING THEM
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
        self.walk_forward_splits = 3
        self.validation_window = 200
        self.min_training_samples = 200

        # BTC correlation configuration
        self.btc_correlation_enabled = True
        self.btc_correlation_cache = {}
        self.btc_correlation_cache_ttl = 300  # 5 minutes
        
        print(f"   • BTC correlation features: {self.btc_correlation_enabled}")
        
        # NEW: Multi-timeframe target configuration - DEFINE BEFORE PRINT
        self.target_configs = [
            {'periods': 8, 'weight': 0.6, 'threshold_multiplier': 1.8},   # Primary timeframe
            {'periods': 15, 'weight': 0.4, 'threshold_multiplier': 2.2},  # Secondary timeframe
        ]
        
        # NEW: Symbol-specific configurations for accuracy improvement
        self.symbol_volatility_profiles = {
            'BTCUSDT': {'volatility_tier': 'low', 'multiplier': 1.3, 'timeframes': [5, 10, 20]},
            'ETHUSDT': {'volatility_tier': 'medium', 'multiplier': 1.6, 'timeframes': [5, 10, 15]},
            'BNBUSDT': {'volatility_tier': 'high', 'multiplier': 2.0, 'timeframes': [5, 8, 12]},
            'XRPUSDT': {'volatility_tier': 'high', 'multiplier': 2.0, 'timeframes': [5, 8, 12]},
            'SOLUSDT': {'volatility_tier': 'extreme', 'multiplier': 2.5, 'timeframes': [5, 8]},
            'DOGEUSDT': {'volatility_tier': 'extreme', 'multiplier': 2.5, 'timeframes': [5, 8]}
        }

        self.symbol_model_complexity = {
            'BTCUSDT': {'max_features': 12, 'enable_hpo': True, 'model_depth': 'medium'},
            'ETHUSDT': {'max_features': 12, 'enable_hpo': True, 'model_depth': 'medium'},
            'BNBUSDT': {'max_features': 10, 'enable_hpo': False, 'model_depth': 'simple'},
            'XRPUSDT': {'max_features': 8, 'enable_hpo': False, 'model_depth': 'simple'},
            'SOLUSDT': {'max_features': 8, 'enable_hpo': False, 'model_depth': 'simple'},
            'DOGEUSDT': {'max_features': 8, 'enable_hpo': False, 'model_depth': 'simple'}
        }
        
        # Enhanced model parameters with hyperparameter optimization
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
        # --- UPDATED with Deepseek's regularization parameters to reduce overfitting ---
        rf_params = {
            'n_estimators': 50,  # Reduced from 100
            'max_depth': 4,      # Reduced from 8
            'min_samples_split': 20,  # Increased from 10
            'min_samples_leaf': 10,   # Increased from 5
            'max_features': 0.3,      # More restrictive
            'bootstrap': True,
            'max_samples': 0.7,       # Use subset of data
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        self.gb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.05,  # Lower learning rate
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': 42
        }

        # NEW: Hyperparameter optimization settings
        self.enable_hyperparameter_optimization = True
        self.hpo_n_iter = 20
        self.hpo_cv = 3

        # ✅ NOW PRINT AFTER ALL CONFIGS ARE DEFINED
        print(f"🤖 Enhanced ML Predictor initialized with scikit-learn {sklearn.__version__}")
        print(f"   • Multi-timeframe targets: {len(self.target_configs)} horizons")
        print(f"   • Hyperparameter optimization: {self.enable_hyperparameter_optimization}")
        print(f"   • Max features: {self.max_features}")
        print(f"   • Symbol-specific configurations: {len(self.symbol_volatility_profiles)} symbols")

        self.enable_conservative_mode()

    def set_error_handler(self, error_handler):
        self.error_handler = error_handler

    def set_database(self, database):
        self.database = database

    # =========================================================================
    # 2. DEEPSEEK ENHANCED PIPELINE (NEW METHODS)
    # =========================================================================

    # --- 2.1: New Main Training Function ---

    def train_model_enhanced(self, symbol: str, df: pd.DataFrame) -> bool:
        try:
            print(f"🚀 ENHANCED TRAINING FOR {symbol}")
            
            features, target = self.prepare_training_data_enhanced_v2(df, symbol=symbol)
            
            min_samples = self.enhanced_config.get('min_training_samples', 300)
            if features.empty or len(features) < min_samples:
                print(f"⚠️ Insufficient quality data for {symbol}: {len(features)} samples")
                return False
            
            self.log_feature_analysis(symbol, features, target)
            
            target_counts = target.value_counts()
            print(f"Target distribution: {target_counts.to_dict()}")
            
            # Remove SMOTE entirely and use proper class weights
            if min(target_counts.values) < len(target) * 0.3:
                print(f"⚠️ Class imbalance detected. Using class weights instead of SMOTE.")
                # Class weights will be handled in model parameters
            
            X_train, X_test, y_train, y_test = self.time_series_split_enhanced(
                features, 
                target, 
                test_size=self.enhanced_config.get('test_size', 0.25), 
                gap=self.enhanced_config.get('validation_gap', 15)
            )
            
            if X_train is None:
                print(f"⚠️ Data split failed for {symbol}")
                return False
            
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            
            self.scalers[symbol] = scaler
            
            selected_features = self._advanced_feature_selection(
                X_train_scaled, y_train, X_test_scaled, y_test, symbol, n_features=15
            )
            X_train_selected = X_train_scaled[selected_features]
            X_test_selected = X_test_scaled[selected_features]
            
            print(f"🔍 Selected {len(selected_features)} robust features")
            
            return self._train_regularized_models(symbol, X_train_selected, X_test_selected, y_train, y_test, selected_features)
            
        except Exception as e:
            print(f"❌ Enhanced training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # --- 2.2: New Data Preparation ---

    def train_model_safe(self, symbol: str, df: pd.DataFrame) -> bool:
        """Safe training wrapper that ensures proper method selection"""
        print(f"🔧 SAFE TRAINING FOR {symbol}")
        
        # First try robust training with simpler models
        if self.train_robust_model_fixed(symbol, df):
            return True
        
        # Fallback to ultra-conservative training
        print(f"⚠️ Robust training failed, trying ultra-conservative for {symbol}")
        return self.train_ultra_conservative(symbol, df)

    def train_robust_model_fixed(self, symbol: str, df: pd.DataFrame) -> bool:
        """Fixed version of robust training"""
        try:
            print(f"🎯 FIXED ROBUST TRAINING FOR {symbol}")
            
            # 1. Prepare data with strict validation
            features, target = self.prepare_training_data_enhanced_v2(df, symbol=symbol)
            
            if features.empty or len(features) < 500:  # Increased minimum
                print(f"⚠️ Insufficient data for {symbol}: {len(features)} samples")
                return False
            
            # 2. More conservative split with larger gap
            split_idx = int(len(features) * 0.6)  # 60% train, 40% test
            X_train = features.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            if len(X_train) < 200 or len(X_test) < 100:
                print(f"⚠️ Insufficient data after split for {symbol}")
                return False
            
            # 3. Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
            # 4. Very conservative feature selection
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k=min(8, X_train_scaled.shape[1]))  # Max 8 features
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            selected_features = X_train_scaled.columns[selector.get_support()].tolist()
            
            print(f"🔍 Selected {len(selected_features)} features for {symbol}")
            
            # 5. Train with VERY conservative parameters
            rf_params = {
                'n_estimators': 50,
                'max_depth': 4,
                'min_samples_split': 25,
                'min_samples_leaf': 15,
                'max_features': 0.3,
                'bootstrap': True,
                'max_samples': 0.6,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            
            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train_selected, y_train)
            
            # 6. Validate with realistic thresholds
            train_pred_rf = rf.predict(X_train_selected)
            test_pred_rf = rf.predict(X_test_selected)
            
            train_acc_rf = accuracy_score(y_train, train_pred_rf)
            test_acc_rf = accuracy_score(y_test, test_pred_rf)
            overfit_rf = train_acc_rf - test_acc_rf
            
            print(f"📊 {symbol} RF - Train: {train_acc_rf:.3f}, Test: {test_acc_rf:.3f}, Overfit: {overfit_rf:.3f}")
            
            # MUCH stricter acceptance criteria
            if test_acc_rf > 0.52 and overfit_rf < 0.15:  # Reduced overfit threshold
                self.models[symbol] = {'rf': rf}
                self.scalers[symbol] = scaler
                self.feature_importance[symbol] = {
                    'selected_features': selected_features,
                    'test_accuracy': test_acc_rf
                }
                print(f"✅ ACCEPTED model for {symbol} with test accuracy: {test_acc_rf:.3f}")
                return True
            else:
                print(f"❌ REJECTED model for {symbol}: test_acc={test_acc_rf:.3f}, overfit={overfit_rf:.3f}")
                return False
                
        except Exception as e:
            print(f"❌ Fixed robust training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_training_data_enhanced_v2(self, df: pd.DataFrame, symbol: str = None) -> tuple:
        
        min_samples = 600
            
        if len(df) < 1500:  # Need at least 1500 bars
            return pd.DataFrame(), pd.Series()
        max_bars = min(4000, len(df) - 100)  # Use more data if available
        
        df = df.tail(max_bars).copy()
        
        features_list = []
        targets = []
        
        stride = max(1, len(df) // 500)
        max_target_period = 20

        for i in range(50, len(df) - max_target_period, stride):
            
            window_data = df.iloc[:i+1]
            if len(window_data) < 50:
                continue

            close_prices = window_data['close'].astype(float)
            high_prices = window_data['high'].astype(float)
            low_prices = window_data['low'].astype(float)
            volume = window_data['volume'].astype(float)
            
            features_dict = self._calculate_high_signal_features(close_prices, high_prices, low_prices, volume, symbol)
            features = pd.DataFrame([features_dict]).fillna(0)

            target = self.create_regime_aware_target(df, i, symbol)
            
            if not features.empty:
                features_list.append(features.iloc[0])
                targets.append(target)
        
        if len(features_list) < min_samples:
            return pd.DataFrame(), pd.Series()
        
        features_df = pd.DataFrame(features_list).fillna(0)
        target_series = pd.Series(targets, index=features_df.index)
        
        features_df = self._clean_features(features_df)
        
        return features_df, target_series

    # --- 2.3: New Feature Engineering ---

    def _calculate_enhanced_features(self, close, high, low, volume, symbol):
        """Enhanced feature set with better market microstructure (Deepseek)"""
        features = {}
        
        try:
            if len(close) < 50:
                return {}

            # 1. Price-based features
            returns = close.pct_change()
            
            # Volatility features
            features['volatility_5'] = returns.rolling(5).std().iloc[-1]
            features['volatility_20'] = returns.rolling(20).std().iloc[-1]
            features['volatility_ratio'] = features['volatility_5'] / max(features['volatility_20'], 0.001)
            
            # Momentum features
            features['momentum_5'] = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 6 else 0
            features['momentum_10'] = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 11 else 0
            features['momentum_ratio'] = features['momentum_5'] / max(abs(features['momentum_10']), 0.001)
            
            # 2. Advanced TA features
            # RSI variations
            rsi_14 = talib.RSI(close, timeperiod=14)[-1]
            rsi_7 = talib.RSI(close, timeperiod=7)[-1]
            features['rsi_14'] = rsi_14 if not pd.isna(rsi_14) else 50
            features['rsi_7'] = rsi_7 if not pd.isna(rsi_7) else 50
            features['rsi_momentum'] = features['rsi_7'] - features['rsi_14']
            
            # MACD features
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd[-1] if not pd.isna(macd[-1]) else 0
            features['macd_signal'] = macd_signal[-1] if not pd.isna(macd_signal[-1]) else 0
            features['macd_hist'] = macd_hist[-1] if not pd.isna(macd_hist[-1]) else 0
            
            # 3. Volume-price relationship
            features['volume_price_corr'] = self._calculate_volume_price_correlation(close, volume, 20)
            features['volume_velocity'] = volume.pct_change(5).iloc[-1] if len(volume) >= 6 else 0
            
            # 4. Market regime features
            features['trend_strength'] = self._calculate_trend_strength(close, 20) # Re-uses your existing helper
            features['market_regime'] = self._classify_market_regime(close, volume)
            
            # 5. Support/resistance levels (NEW HELPERS)
            features['support_distance'] = self._calculate_support_distance(close, low, 20)
            features['resistance_distance'] = self._calculate_resistance_distance(close, high, 20)
            
            # 6. Symbol-specific features (NEW HELPERS)
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                features.update(self._calculate_institutional_features(close, volume))
            elif symbol in ['SOLUSDT', 'DOGEUSDT']:
                features.update(self._calculate_retail_momentum_features(close, volume, high, low))
                
        except Exception as e:
            self.logger.error(f"Error calculating enhanced features for {symbol}: {e}")
        
        return features

    def _calculate_volume_price_correlation(self, close, volume, window):
        """Calculate correlation between price changes and volume (Deepseek)"""
        try:
            price_changes = close.pct_change().dropna()
            volume_changes = volume.pct_change().dropna()
            
            if len(price_changes) < window or len(volume_changes) < window:
                return 0
                
            # Align the series
            common_index = price_changes.index.intersection(volume_changes.index)
            if len(common_index) < window:
                return 0.0

            price_aligned = price_changes.loc[common_index].tail(window)
            volume_aligned = volume_changes.loc[common_index].tail(window)
            
            if len(price_aligned) < 10:
                return 0

            correlation = price_aligned.corr(volume_aligned)
            return correlation if not pd.isna(correlation) else 0
        except:
            return 0

    def _classify_market_regime(self, close, volume):
        """Classify current market regime (Deepseek)"""
        try:
            returns = close.pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0
            trend = self._calculate_trend_strength(close, 20) # Re-uses your existing helper
            volume_trend = volume.rolling(20).mean().iloc[-1] / volume.rolling(50).mean().iloc[-1] if len(volume) >= 50 else 1
            
            if pd.isna(volatility) or pd.isna(trend) or pd.isna(volume_trend):
                return 0.5 # Neutral on bad data

            if volatility > 0.03 and abs(trend) < 0.2:
                return 2  # High volatility, low trend (choppy)
            elif volatility < 0.01 and abs(trend) < 0.1:
                return 0  # Low volatility, low trend (consolidation)
            elif trend > 0.3:
                return 1  # Strong uptrend
            elif trend < -0.3:
                return -1  # Strong downtrend
            else:
                return 0.5  # Neutral
        except:
            return 0

    # --- 2.4: *MISSING* Feature Helpers (I've implemented these) ---

    def _calculate_support_distance(self, close: pd.Series, low: pd.Series, window: int) -> float:
        """Calculate distance from current close to recent support"""
        try:
            if len(low) < window:
                return 0
            support = low.rolling(window).min().iloc[-1]
            if support <= 0:
                return 0
            distance = (close.iloc[-1] / support) - 1
            return max(0, distance) # Distance can only be positive
        except:
            return 0

    def _calculate_resistance_distance(self, close: pd.Series, high: pd.Series, window: int) -> float:
        """Calculate distance from current close to recent resistance"""
        try:
            if len(high) < window:
                return 0
            resistance = high.rolling(window).max().iloc[-1]
            if resistance <= 0:
                return 0
            distance = (close.iloc[-1] / resistance) - 1
            return min(0, distance) # Distance can only be negative
        except:
            return 0

    def _calculate_institutional_features(self, close: pd.Series, volume: pd.Series) -> Dict:
        """Features for large caps (BTC/ETH) - steady volume"""
        features = {}
        try:
            if len(volume) < 20:
                return {}
            vol_mean = volume.rolling(20).mean().iloc[-1]
            vol_std = volume.rolling(20).std().iloc[-1]
            
            # Coefficient of variation (lower is more consistent)
            features['volume_consistency'] = vol_std / vol_mean if vol_mean > 0 else 1.0
            
            # Percentage of recent days with high volume
            high_vol_threshold = volume.quantile(0.75)
            features['high_volume_concentration'] = (volume.tail(20) > high_vol_threshold).mean()
            return features
        except:
            return {}

    def _calculate_retail_momentum_features(self, close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """Features for volatile caps (SOL/DOGE) - spiky volume, high range"""
        features = {}
        try:
            if len(volume) < 20 or len(close) < 20:
                return {}
            vol_mean = volume.rolling(20).mean().iloc[-1]
            
            # Current volume vs. recent average
            features['volume_spike'] = volume.iloc[-1] / vol_mean if vol_mean > 0 else 1.0
            
            # Recent price range as % of close
            day_range = (high.tail(20) - low.tail(20)) / close.tail(20)
            features['avg_day_range_perc'] = day_range.mean()
            return features
        except:
            return {}

    # --- 2.5: New Target Definition ---

    def create_enhanced_target_v2(self, df: pd.DataFrame, current_idx: int, symbol: str = None) -> int:
        """More sophisticated target definition with regime awareness (Deepseek)"""
        
        if current_idx + 20 >= len(df):
            return 0
            
        current_data = df.iloc[:current_idx+1]
        future_data = df.iloc[current_idx:current_idx+21]  # Look 20 periods ahead
        
        if len(future_data) < 2 or len(current_data) < 20:
            return 0
            
        current_price = current_data['close'].iloc[-1]
        future_prices = future_data['close']
        
        # Calculate multiple metrics
        max_price = future_prices.max()
        min_price = future_prices.min()
        final_price = future_prices.iloc[-1]
        
        # Dynamic thresholds based on recent volatility
        recent_volatility = current_data['close'].pct_change().rolling(20).std().iloc[-1]
        if pd.isna(recent_volatility) or recent_volatility == 0:
            recent_volatility = 0.02
            
        # Symbol-specific thresholds
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            threshold = recent_volatility * 1.5
            strong_threshold = recent_volatility * 2.5
        elif symbol in ['SOLUSDT', 'DOGEUSDT']:
            threshold = recent_volatility * 1.2
            strong_threshold = recent_volatility * 2.0
        else:
            threshold = recent_volatility * 1.3
            strong_threshold = recent_volatility * 2.2
        
        # Calculate various return metrics
        max_return = (max_price - current_price) / current_price
        min_return = (min_price - current_price) / current_price
        final_return = (final_price - current_price) / current_price
        
        # Weighted decision based on multiple factors
        score = 0
        
        # Primary: Final return
        if final_return > strong_threshold:
            score += 2
        elif final_return > threshold:
            score += 1
        elif final_return < -strong_threshold:
            score -= 2
        elif final_return < -threshold:
            score -= 1
            
        # Secondary: Maximum potential
        if max_return > threshold * 1.5:
            score += 1
        if min_return < -threshold * 1.5:
            score -= 1
            
        # Convert to signal
        if score >= 2:
            return 1
        elif score <= -2:
            return -1
        else:
            return 0

    # --- 2.6: New Training Helpers ---

    def _apply_smote_balancing(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes using SMOTE, with a fallback to undersampling."""
        try:
            smote = SMOTE(random_state=42, k_neighbors=max(1, min(target.value_counts()) - 1))
            X_res, y_res = smote.fit_resample(features, target)
            return X_res, y_res
        except ImportError:
            self.logger.warning("imblearn not found. Falling back to simple random undersampling.")
            return self._balance_classes(features, target) # Re-uses your existing helper
        except Exception as e:
            self.logger.warning(f"SMOTE failed: {e}. Falling back to simple random undersampling.")
            return self._balance_classes(features, target)

    def time_series_split_enhanced(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.25, gap: int = 15):
        """Rigorous time-based split (from Deepseek)"""
        if len(features) < 100:
            return None, None, None, None
            
        split_idx = int(len(features) * (1 - test_size))
        train_end_idx = split_idx - gap
        
        if train_end_idx < 50:
            train_end_idx = split_idx # Not enough data for a gap
            
        X_train = features.iloc[:train_end_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:train_end_idx]
        y_test = target.iloc[split_idx:]
        
        if len(X_train) < 50 or len(X_test) < 10:
            return None, None, None, None
            
        return X_train, X_test, y_train, y_test

    def _advanced_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_test: pd.DataFrame, y_test: pd.Series, 
                                symbol: str, n_features: int = 15) -> List[str]:
        """Feature selection using only training data"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Method 1: ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X_train.shape[1]))
        selector.fit(X_train, y_train)
        anova_features = X_train.columns[selector.get_support()].tolist()
        
        # Method 2: Random Forest importance (training only)
        rf_selector = RandomForestClassifier(
            n_estimators=50, 
            max_depth=5, 
            random_state=42
        )
        rf_selector.fit(X_train, y_train)
        
        # Get top features from RF
        importances = rf_selector.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        rf_features = feature_importance_df.head(n_features)['feature'].tolist()
        
        # Combine methods
        combined_features = list(set(anova_features + rf_features))
        
        print(f"🔍 Feature selection for {symbol}: {len(combined_features)} features from {len(X_train.columns)}")
        
        return combined_features[:n_features]

    def _train_regularized_models(self, symbol, X_train, X_test, y_train, y_test, features):
        """Training with balanced regularization and realistic thresholds"""
        
        # Get symbol-specific configuration
        enhanced_params = self.enhanced_config.get('symbol_specific_params', {}).get(symbol, {})
        
        # Symbol-specific overfit thresholds
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            max_overfit_threshold = 0.20  # More lenient for stable coins
            min_test_accuracy = 0.48
        elif symbol in ['SOLUSDT', 'DOGEUSDT']:
            max_overfit_threshold = 0.25  # Even more lenient for volatile coins  
            min_test_accuracy = 0.45
        else:
            max_overfit_threshold = 0.22  # Default
            min_test_accuracy = 0.47
        
        # Use enhanced parameters if available, otherwise use balanced defaults
        if enhanced_params:
            rf_params_enhanced = {
                'n_estimators': enhanced_params.get('n_estimators', 100),
                'max_depth': enhanced_params.get('max_depth', 6),
                'min_samples_split': 10,  # Reduced from 30
                'min_samples_leaf': 5,    # Reduced from 40
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            rf_params_enhanced = {
                'n_estimators': 100,
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }

        # Gradient Boosting parameters
        gb_params_enhanced = {
            'n_estimators': 80,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': 42
        }
        
        # Initialize and train models
        rf = RandomForestClassifier(**rf_params_enhanced)
        gb = GradientBoostingClassifier(**gb_params_enhanced)
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # Calculate performance metrics
        train_pred_rf = rf.predict(X_train)
        test_pred_rf = rf.predict(X_test)
        
        train_accuracy_rf = accuracy_score(y_train, train_pred_rf)
        test_accuracy_rf = accuracy_score(y_test, test_pred_rf)
        overfit_rf = train_accuracy_rf - test_accuracy_rf
        
        # Additional metrics for RF
        rf_precision = precision_score(y_test, test_pred_rf, average='weighted', zero_division=0)
        rf_recall = recall_score(y_test, test_pred_rf, average='weighted', zero_division=0)
        rf_f1 = f1_score(y_test, test_pred_rf, average='weighted', zero_division=0)
        
        print(f"📊 {symbol} RF - Train: {train_accuracy_rf:.3f}, Test: {test_accuracy_rf:.3f}, Overfit: {overfit_rf:.3f}")
        print(f"    Precision: {rf_precision:.3f}, Recall: {rf_recall:.3f}, F1: {rf_f1:.3f}")

        # GB metrics
        train_pred_gb = gb.predict(X_train)
        test_pred_gb = gb.predict(X_test)
        train_accuracy_gb = accuracy_score(y_train, train_pred_gb)
        test_accuracy_gb = accuracy_score(y_test, test_pred_gb)
        overfit_gb = train_accuracy_gb - test_accuracy_gb
        
        gb_precision = precision_score(y_test, test_pred_gb, average='weighted', zero_division=0)
        gb_recall = recall_score(y_test, test_pred_gb, average='weighted', zero_division=0)
        gb_f1 = f1_score(y_test, test_pred_gb, average='weighted', zero_division=0)
        
        print(f"📊 {symbol} GB - Train: {train_accuracy_gb:.3f}, Test: {test_accuracy_gb:.3f}, Overfit: {overfit_gb:.3f}")
        print(f"    Precision: {gb_precision:.3f}, Recall: {gb_recall:.3f}, F1: {gb_f1:.3f}")
        
        # Check class distribution in predictions
        unique_rf, counts_rf = np.unique(test_pred_rf, return_counts=True)
        unique_gb, counts_gb = np.unique(test_pred_gb, return_counts=True)
        
        print(f"📈 {symbol} Test Prediction Distribution:")
        print(f"    RF: {dict(zip(unique_rf, counts_rf))}")
        print(f"    GB: {dict(zip(unique_gb, counts_gb))}")
        
        # Enhanced model acceptance criteria
        rf_acceptable = (overfit_rf < max_overfit_threshold and 
                        test_accuracy_rf >= min_test_accuracy)
        gb_acceptable = (overfit_gb < max_overfit_threshold and 
                        test_accuracy_gb >= min_test_accuracy)
        
        # At least one model should be acceptable
        model_acceptable = rf_acceptable or gb_acceptable
        
        if model_acceptable:
            # Save previous models if they exist
            if symbol in self.models:
                self.previous_models[symbol] = self.models[symbol].copy()
                self.previous_scalers[symbol] = self.scalers[symbol]
            
            # Store current models (even if only one is good)
            self.models[symbol] = {'rf': rf, 'gb': gb}
            
            # Initialize feature importance tracking if needed
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
                
            # Update feature importance with detailed metrics
            self.feature_importance[symbol].update({
                'selected_features': features,
                'rf_importance': rf.feature_importances_.tolist() if hasattr(rf, 'feature_importances_') else [],
                'gb_importance': gb.feature_importances_.tolist() if hasattr(gb, 'feature_importances_') else [],
                'test_accuracy_rf': test_accuracy_rf,
                'test_accuracy_gb': test_accuracy_gb,
                'overfit_rf': overfit_rf,
                'overfit_gb': overfit_gb,
                'rf_precision': rf_precision,
                'rf_recall': rf_recall,
                'rf_f1': rf_f1,
                'gb_precision': gb_precision,
                'gb_recall': gb_recall,
                'gb_f1': gb_f1
            })

            # Create model version info
            model_version = f"v_enhanced_{datetime.now().strftime('%Y%m%d_%H%M')}"
            avg_test_accuracy = (test_accuracy_rf + test_accuracy_gb) / 2
            
            self.model_versions[symbol] = {
                'version': model_version,
                'training_date': datetime.now(),
                'accuracy': avg_test_accuracy,
                'rf_accuracy': test_accuracy_rf,
                'gb_accuracy': test_accuracy_gb,
                'rf_precision': rf_precision,
                'gb_precision': gb_precision,
                'rf_recall': rf_recall,
                'gb_recall': gb_recall,
                'rf_f1': rf_f1,
                'gb_f1': gb_f1,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(features),
                'overfit_scores': {'rf': overfit_rf, 'gb': overfit_gb},
                'target_type': 'enhanced_v2',
                'acceptance_criteria': {
                    'rf_acceptable': rf_acceptable,
                    'gb_acceptable': gb_acceptable,
                    'max_overfit_threshold': max_overfit_threshold,
                    'min_test_accuracy': min_test_accuracy
                }
            }
            
            # Log detailed acceptance info
            if rf_acceptable and gb_acceptable:
                print(f"✅ ACCEPTED both models for {symbol}")
            elif rf_acceptable:
                print(f"✅ ACCEPTED RF model for {symbol} (GB rejected)")
            else:
                print(f"✅ ACCEPTED GB model for {symbol} (RF rejected)")
                
            print(f"💾 Saved new ENHANCED model for {symbol} with avg test accuracy: {avg_test_accuracy:.3f}")
            return True
        else:
            # Detailed rejection reasons
            rejection_reasons = []
            if overfit_rf >= max_overfit_threshold:
                rejection_reasons.append(f"RF overfit ({overfit_rf:.3f} >= {max_overfit_threshold})")
            if test_accuracy_rf < min_test_accuracy:
                rejection_reasons.append(f"RF low accuracy ({test_accuracy_rf:.3f} < {min_test_accuracy})")
            if overfit_gb >= max_overfit_threshold:
                rejection_reasons.append(f"GB overfit ({overfit_gb:.3f} >= {max_overfit_threshold})")
            if test_accuracy_gb < min_test_accuracy:
                rejection_reasons.append(f"GB low accuracy ({test_accuracy_gb:.3f} < {min_test_accuracy})")
                
            print(f"🚫 REJECTED model for {symbol}: {', '.join(rejection_reasons)}")
            return False


    # Add this method to replace the problematic ones:
    def train_robust_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """Unified robust training pipeline"""
        try:
            print(f"🎯 ROBUST TRAINING FOR {symbol}")
            
            # 1. Prepare data with proper validation
            features, target = self.prepare_training_data_enhanced_v2(df, symbol=symbol)
            
            if features.empty or len(features) < 300:
                print(f"⚠️ Insufficient data for {symbol}: {len(features)} samples")
                return False
            
            # 2. Simple time series split
            split_idx = int(len(features) * 0.7)
            X_train = features.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            if len(X_train) < 100 or len(X_test) < 50:
                print(f"⚠️ Insufficient data after split for {symbol}")
                return False
            
            # 3. Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
            # 4. Simple feature selection
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k=min(10, X_train_scaled.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            selected_features = X_train_scaled.columns[selector.get_support()].tolist()
            
            print(f"🔍 Selected {len(selected_features)} features for {symbol}")
            
            # 5. Train with balanced parameters
            rf_params = {
                'n_estimators': 50,  # Reduced from 100
                'max_depth': 4,      # Reduced from 8
                'min_samples_split': 20,  # Increased from 10
                'min_samples_leaf': 10,   # Increased from 5
                'max_features': 0.3,      # More restrictive
                'bootstrap': True,
                'max_samples': 0.7,       # Use subset of data
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            
            gb_params = {
                'n_estimators': 80,
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'subsample': 0.8,
                'random_state': 42
            }
            
            rf = RandomForestClassifier(**rf_params)
            gb = GradientBoostingClassifier(**gb_params)
            
            rf.fit(X_train_selected, y_train)
            gb.fit(X_train_selected, y_train)
            
            # 6. Validate with realistic thresholds
            train_pred_rf = rf.predict(X_train_selected)
            test_pred_rf = rf.predict(X_test_selected)
            
            train_acc_rf = accuracy_score(y_train, train_pred_rf)
            test_acc_rf = accuracy_score(y_test, test_pred_rf)
            overfit_rf = train_acc_rf - test_acc_rf
            
            print(f"📊 {symbol} RF - Train: {train_acc_rf:.3f}, Test: {test_acc_rf:.3f}, Overfit: {overfit_rf:.3f}")
            
            # Accept model if reasonable performance
            if test_acc_rf > 0.48 and overfit_rf < 0.25:
                self.models[symbol] = {'rf': rf, 'gb': gb}
                self.scalers[symbol] = scaler
                self.feature_importance[symbol] = {
                    'selected_features': selected_features,
                    'test_accuracy': test_acc_rf
                }
                print(f"✅ ACCEPTED model for {symbol} with test accuracy: {test_acc_rf:.3f}")
                return True
            else:
                print(f"❌ REJECTED model for {symbol}: test_acc={test_acc_rf:.3f}, overfit={overfit_rf:.3f}")
                return False
                
        except Exception as e:
            print(f"❌ Robust training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False
    # --- 2.7: New Logging Functions ---

    def enable_detailed_logging(self):
        """Enable comprehensive logging for debugging (Deepseek)"""
        if logging.getLogger('MLDetailed').hasHandlers():
            return # Already configured

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_detailed.log'),
                logging.StreamHandler()
            ]
        )
        
        self.detailed_logger = logging.getLogger('MLDetailed')
        print("✅ Detailed ML logging enabled (writing to ml_detailed.log)")

    def log_feature_analysis(self, symbol, features, target):
        """Detailed feature analysis logging (Deepseek)"""
        self.detailed_logger.info(f"🔍 FEATURE ANALYSIS FOR {symbol}")
        self.detailed_logger.info(f"Feature matrix shape: {features.shape}")
        self.detailed_logger.info(f"Target distribution: {target.value_counts().to_dict()}")
        
        # Feature statistics
        self.detailed_logger.info("Feature statistics:")
        for col in features.columns:
            self.detailed_logger.info(f"  {col}: mean={features[col].mean():.4f}, std={features[col].std():.4f}, "
                                    f"nulls={features[col].isnull().sum()}")
        
        # Correlation analysis
        corr_with_target = []
        for col in features.columns:
            if features[col].std() > 0:
                try:
                    corr = np.corrcoef(features[col], target)[0, 1] if len(target) == len(features) else 0
                    corr_with_target.append((col, corr))
                except Exception:
                    corr_with_target.append((col, 0.0))
        
        corr_with_target.sort(key=lambda x: abs(x[1]), reverse=True)
        self.detailed_logger.info("Top feature correlations with target:")
        for col, corr in corr_with_target[:10]:
            self.detailed_logger.info(f"  {col}: {corr:.3f}")

    def log_training_progress(self, symbol, iteration, metrics):
        """Log training progress with detailed metrics (Deepseek)"""
        self.detailed_logger.info(
            f"🔄 {symbol} Training Iteration {iteration}: "
            f"Accuracy={metrics.get('accuracy', 0):.3f}, "
            f"Precision={metrics.get('precision', 0):.3f}, "
            f"Recall={metrics.get('recall', 0):.3f}, "
            f"F1={metrics.get('f1', 0):.3f}, "
            f"Overfit={metrics.get('overfit', 0):.3f}"
        )

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

    def set_data_engine(self, data_engine):
        """Set data engine for BTC correlation features"""
        self.data_engine = data_engine

    def prepare_features_point_in_time(self, df: pd.DataFrame, current_idx: int, lookback: int = 50, symbol: str = None) -> pd.DataFrame:
        features = {}
        
        if current_idx < lookback:
            return pd.DataFrame()

        historical_data = df.iloc[:current_idx].copy()
        
        if len(historical_data) < lookback:
            return pd.DataFrame()
            
        window_data = historical_data.iloc[-lookback:]
        
        close_prices = window_data['close'].astype(float)
        high_prices = window_data['high'].astype(float)
        low_prices = window_data['low'].astype(float)
        volume = window_data['volume'].astype(float)

        try:
            # NEW: Core TA-Lib Features
            core_ta_features = self._calculate_core_ta_features(close_prices, high_prices, low_prices, volume)
            features.update(core_ta_features)

            # 1. Symbol-specific volatility scaling
            volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1]
            features['symbol_specific_volatility'] = volatility_20
            
            # 2. BTC Dominance correlation (enhanced implementation)
            if symbol and symbol != 'BTCUSDT' and hasattr(self, 'data_engine') and self.data_engine and self.btc_correlation_enabled:
                btc_features = self._calculate_btc_correlation_features(symbol, historical_data)
                features.update(btc_features)
            
            # 3. Market cap tier features (now numerical)
            if symbol:
                market_cap_tier = self._get_market_cap_tier(symbol)  # Returns 0, 1, or 2
                features['market_cap_tier'] = market_cap_tier
                
                # Adjust feature complexity based on symbol type
                if market_cap_tier == 2:  # large_cap
                    features.update(self._calculate_advanced_features(close_prices, high_prices, low_prices, volume))
                elif market_cap_tier == 1:  # mid_cap
                    features.update(self._calculate_medium_complexity_features(close_prices, high_prices, low_prices, volume))
                else:  # small_cap
                    features.update(self._calculate_simple_features(close_prices, high_prices, low_prices, volume))

            # 4. Volume profile features
            volume_profile = self._calculate_volume_profile(volume, symbol)
            features.update(volume_profile)
            
            # 5. Existing features (keep but with volatility adjustment)
            for period in [1, 3, 5, 10, 20]:
                if len(close_prices) > period:
                    vol_adjusted_return = (close_prices.iloc[-1] / close_prices.iloc[-period-1] - 1) / max(volatility_20, 0.001)
                    features[f'vol_adjusted_returns_{period}'] = vol_adjusted_return
            
            # Ensure all features are numerical
            features = self._ensure_numerical_features(features)
            
            return pd.DataFrame([features]).fillna(0)
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_ml_error(e, symbol, "feature_preparation")
            return pd.DataFrame()

    def _ensure_numerical_features(self, features: Dict) -> Dict:
        """Convert all feature values to numerical types"""
        numerical_features = {}
        
        for key, value in features.items():
            try:
                # Convert to float if possible, otherwise use 0
                if isinstance(value, (int, float, np.number)):
                    numerical_features[key] = float(value)
                elif isinstance(value, (str, bool)):
                    # Convert strings and booleans to numerical
                    if isinstance(value, bool):
                        numerical_features[key] = 1.0 if value else 0.0
                    elif value.lower() in ['true', 'false']:
                        numerical_features[key] = 1.0 if value.lower() == 'true' else 0.0
                    else:
                        # Try to convert string to float, if fails use 0
                        try:
                            numerical_features[key] = float(value)
                        except (ValueError, TypeError):
                            numerical_features[key] = 0.0
                else:
                    numerical_features[key] = 0.0
            except Exception as e:
                self.logger.warning(f"Could not convert feature {key} with value {value} to float: {e}")
                numerical_features[key] = 0.0
        
        return numerical_features

    def _calculate_btc_correlation_features(self, symbol: str, current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate BTC correlation features with caching"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{len(current_data)}"
            if cache_key in self.btc_correlation_cache:
                cache_entry = self.btc_correlation_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.btc_correlation_cache_ttl:
                    return cache_entry['features']
            
            # Calculate fresh features
            if hasattr(self, 'data_engine') and self.data_engine:
                features = self.data_engine.get_btc_correlation_features(symbol, current_data)
                
                # Apply symbol-specific interpretation
                features = self._apply_symbol_specific_btc_interpretation(symbol, features)
                
                # Cache the results
                self.btc_correlation_cache[cache_key] = {
                    'features': features,
                    'timestamp': time.time()
                }
                
                # Clean old cache entries
                self._clean_btc_correlation_cache()
                
                return features
            else:
                return self._get_default_btc_features()
                
        except Exception as e:
            self.logger.error(f"Error calculating BTC correlation features for {symbol}: {e}")
            return self._get_default_btc_features()

    def _apply_symbol_specific_btc_interpretation(self, symbol: str, btc_features: Dict[str, float]) -> Dict[str, float]:
        """Apply symbol-specific interpretation to BTC correlation features"""
        enhanced_features = btc_features.copy()
        
        # Large caps (BTC/ETH) should have high BTC correlation
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            # High correlation is normal for large caps
            correlation_penalty = 0.0
        else:
            # Altcoins: very high correlation might indicate lack of independent movement
            high_corr_penalty = max(0, abs(btc_features.get('btc_correlation_20', 0)) - 0.7)
            correlation_penalty = high_corr_penalty * 0.5
        
        # Beta interpretation
        beta_20 = btc_features.get('btc_beta_20', 1.0)
        if symbol in ['SOLUSDT', 'DOGEUSDT']:
            # High beta expected for volatile altcoins
            beta_deviation = abs(beta_20 - 2.0) / 2.0
        elif symbol in ['BNBUSDT', 'XRPUSDT']:
            # Medium beta expected
            beta_deviation = abs(beta_20 - 1.5) / 1.5
        else:
            # Low beta expected for large caps
            beta_deviation = abs(beta_20 - 1.0)
        
        enhanced_features['btc_correlation_quality'] = 1.0 - (correlation_penalty + beta_deviation * 0.5)
        enhanced_features['btc_beta_deviation'] = beta_deviation
        
        return enhanced_features

    def _get_default_btc_features(self) -> Dict[str, float]:
        """Return default BTC correlation features"""
        return {
            'btc_correlation_5': 0.0,
            'btc_correlation_10': 0.0,
            'btc_correlation_20': 0.0,
            'btc_correlation_50': 0.0,
            'btc_correlation_overall': 0.0,
            'btc_correlation_high_vol': 0.0,
            'btc_beta_5': 1.0,
            'btc_beta_10': 1.0,
            'btc_beta_20': 1.0,
            'btc_beta_50': 1.0,
            'relative_perf_vs_btc_5': 0.0,
            'relative_perf_vs_btc_10': 0.0,
            'relative_perf_vs_btc_20': 0.0,
            'relative_strength_momentum': 0.0,
            'outperformance_ratio_20': 0.5,
            'btc_dominance_trend': 0.0,
            'altcoin_season_score': 1.0,
            'btc_market_regime': 0.0,
            'btc_correlation_quality': 0.5,
            'btc_beta_deviation': 0.5
        }

    def _clean_btc_correlation_cache(self):
        """Clean old entries from BTC correlation cache"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.btc_correlation_cache.items():
            if current_time - entry['timestamp'] > self.btc_correlation_cache_ttl * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.btc_correlation_cache[key]

    def _adjust_target_weight(self, base_weight: float, symbol: str, trend_strength: float, volatility: float) -> float:
        """Adjust target weight based on symbol and market conditions"""
        adjusted_weight = base_weight
        
        # Reduce weight during high volatility for all symbols
        if volatility > 0.03:
            adjusted_weight *= 0.8
        
        # Symbol-specific adjustments
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] in ['high', 'extreme']:
                # Reduce weight for volatile symbols in strong trends
                if trend_strength > 0.6:
                    adjusted_weight *= 0.7
        
        return max(0.1, adjusted_weight)  # Ensure minimum weight

    def _calculate_volume_profile(self, volume: pd.Series, symbol: str) -> Dict[str, float]:
        """Calculate volume-based features with symbol-specific thresholds"""
        features = {}
        
        try:
            if len(volume) < 20:
                return {
                    'volume_ma_20': 1.0,
                    'volume_ratio': 1.0,
                    'volume_trend': 0.0,
                    'volume_volatility': 0.0,
                    'volume_zscore': 0.0
                }
            
            # Symbol-specific volume thresholds
            if symbol in self.symbol_volatility_profiles:
                profile = self.symbol_volatility_profiles[symbol]
                if profile['volatility_tier'] in ['high', 'extreme']:
                    # Higher thresholds for volatile symbols
                    volume_spike_threshold = 3.0
                    low_volume_threshold = 0.3
                else:
                    # Standard thresholds for stable symbols
                    volume_spike_threshold = 2.5
                    low_volume_threshold = 0.5
            else:
                volume_spike_threshold = 2.5
                low_volume_threshold = 0.5
            
            # Volume moving averages
            volume_ma_20 = volume.rolling(20).mean().iloc[-1]
            volume_ma_50 = volume.rolling(50).mean().iloc[-1]
            
            # Current volume ratios
            current_volume = volume.iloc[-1]
            volume_ratio_20 = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
            volume_ratio_50 = current_volume / volume_ma_50 if volume_ma_50 > 0 else 1.0
            
            # Volume trend (slope of linear regression)
            if len(volume) >= 20:
                x = np.arange(len(volume.tail(20)))
                y = volume.tail(20).values
                volume_trend = np.polyfit(x, y, 1)[0] / np.mean(y) if np.mean(y) > 0 else 0.0
            else:
                volume_trend = 0.0
            
            # Volume volatility (coefficient of variation)
            volume_volatility = volume.rolling(20).std().iloc[-1] / volume_ma_20 if volume_ma_20 > 0 else 0.0
            
            # Volume z-score (how extreme is current volume)
            volume_zscore = (current_volume - volume_ma_20) / volume.rolling(20).std().iloc[-1] if volume.rolling(20).std().iloc[-1] > 0 else 0.0
            
            # Volume regime detection
            volume_regime = 0
            if volume_ratio_20 > volume_spike_threshold:
                volume_regime = 1  # High volume regime
            elif volume_ratio_20 < low_volume_threshold:
                volume_regime = -1  # Low volume regime
            
            features = {
                'volume_ma_20': volume_ma_20,
                'volume_ratio_20': volume_ratio_20,
                'volume_ratio_50': volume_ratio_50,
                'volume_trend': volume_trend,
                'volume_volatility': volume_volatility,
                'volume_zscore': volume_zscore,
                'volume_regime': volume_regime,
                'volume_above_average': 1 if volume_ratio_20 > 1.0 else 0
            }
            
            # Add symbol-specific volume features
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                # Institutional volume patterns for large caps
                features['volume_consistency'] = volume.rolling(10).std().iloc[-1] / volume_ma_20 if volume_ma_20 > 0 else 0.0
            elif symbol in ['XRPUSDT', 'DOGEUSDT']:
                # Retail volume patterns for meme coins
                features['volume_spikiness'] = (volume_ratio_20 > 2.0) * 1.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile for {symbol}: {e}")
            return {
                'volume_ma_20': 1.0,
                'volume_ratio_20': 1.0,
                'volume_ratio_50': 1.0,
                'volume_trend': 0.0,
                'volume_volatility': 0.0,
                'volume_zscore': 0.0,
                'volume_regime': 0,
                'volume_above_average': 0
            }

    def _get_market_cap_tier(self, symbol: str) -> int:
        """Categorize symbols by market cap tier - returns numerical value"""
        large_cap = ['BTCUSDT', 'ETHUSDT']
        mid_cap = ['BNBUSDT', 'SOLUSDT']
        small_cap = ['XRPUSDT', 'DOGEUSDT']
        
        if symbol in large_cap:
            return 2  # large_cap
        elif symbol in mid_cap:
            return 1  # mid_cap
        else:
            return 0  # small_cap

    def _calculate_advanced_features(self, close, high, low, volume):
        """Advanced features for large-cap symbols"""
        features = {}
        
        # Hurst exponent, microstructure, etc.
        if len(close) >= 100:
            features['hurst_exponent'] = self._calculate_hurst_exponent(close.tail(100))
            features['market_micro_1'] = self._calculate_market_microstructure_1(high, low, close, volume)
        
        return features

    def _calculate_price_volume_correlation(self, close_prices: pd.Series, volume: pd.Series, window: int = 20) -> float:
        """Calculate correlation between price changes and volume"""
        try:
            if len(close_prices) < window + 1:
                return 0.0
            
            # Calculate price returns and volume changes
            price_returns = close_prices.pct_change().dropna()
            volume_changes = volume.pct_change().dropna()
            
            # Align the series
            common_index = price_returns.index.intersection(volume_changes.index)
            if len(common_index) < window:
                return 0.0
            
            price_aligned = price_returns.loc[common_index]
            volume_aligned = volume_changes.loc[common_index]
            
            # Calculate rolling correlation
            correlation = price_aligned.rolling(window).corr(volume_aligned)
            
            return correlation.iloc[-1] if not pd.isna(correlation.iloc[-1]) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating price-volume correlation: {e}")
            return 0.0

    def _calculate_medium_complexity_features(self, close, high, low, volume):
        """Medium complexity features for mid-cap symbols"""
        features = {}
        
        # Basic indicators + volume features
        features['volume_momentum'] = volume.iloc[-1] / volume.rolling(10).mean().iloc[-1] if len(volume) > 10 else 1
        features['price_volume_corr'] = self._calculate_price_volume_correlation(close, volume)
        
        return features

    def _calculate_core_ta_features(self, close, high, low, volume) -> Dict:
        """Calculates core TA-Lib features safely."""
        features = {}
        try:
            if len(close) < 25: 
                return {}
            
            # Volatility
            atr_14 = talib.ATR(high, low, close, timeperiod=14)[-1]
            features['atr_14_perc'] = (atr_14 / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0
            
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            if middle.iloc[-1] > 0:
                features['bb_width'] = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            else:
                features['bb_width'] = 0
            
            # Trend Strength
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)[-1]
            
            # Momentum
            features['rsi_14'] = talib.RSI(close, timeperiod=14)[-1]
            
        except Exception as e:
            self.logger.warning(f"Error calculating core TA features: {e}")
        return features

    def _calculate_simple_features(self, close, high, low, volume):
        """Simple features for small-cap/volatile symbols"""
        features = {}
        
        # Only essential features to avoid overfitting
        for period in [5, 10, 20]:
            if len(close) > period:
                features[f'returns_{period}'] = close.iloc[-1] / close.iloc[-period-1] - 1
        
        # features['rsi_14'] = self._calculate_rsi_point_in_time(close, 14) # Now handled by _calculate_core_ta_features
        # features['atr_14'] = self._calculate_atr_point_in_time(high, low, close, 14) # Now handled by _calculate_core_ta_features
        
        return features

    def create_multi_timeframe_targets(self, df: pd.DataFrame, current_idx: int, target_configs: List[Dict] = None) -> List[int]:
        """Create multi-timeframe targets for richer learning"""
        if target_configs is None:
            target_configs = self.target_configs
            
        targets = []
        
        for config in target_configs:
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
            
            # NEW: Use ATR for a dynamic, non-noisy threshold
            try:
                atr_data = df.iloc[:current_idx]
                atr = talib.ATR(atr_data['high'], atr_data['low'], atr_data['close'], timeperiod=14)[-1]
                atr_percent = atr / current_price if current_price > 0 else 0
                
                # Set threshold to a fraction of ATR (e.g., 0.75 * ATR)
                # This ensures we only target moves that are significant relative to noise
                dynamic_atr_threshold = atr_percent * 0.75 
            except Exception as e:
                self.logger.warning(f"Could not calculate ATR for target, using default: {e}")
                dynamic_atr_threshold = 0.001
            
            # Use the higher of the ATR-based threshold or a 0.1% minimum
            threshold = max(0.001, dynamic_atr_threshold)
            
            if future_return > threshold:
                targets.append(1)
            elif future_return < -threshold:
                targets.append(-1)
            else:
                targets.append(0)
        
        return targets

    def train_model_improved(self, symbol: str, df: pd.DataFrame) -> bool:
        """Improved training with better regularization and validation"""
        try:
            print(f"🔄 IMPROVED TRAINING FOR {symbol}")
            
            # First, analyze feature quality
            feature_analysis = self.analyze_feature_quality(symbol, df)
            print(f"Feature analysis: {feature_analysis}")
            
            # Debug feature generation
            self.debug_feature_generation(symbol, df)
            
            # Prepare training data with more aggressive filtering
            features, target = self.prepare_training_data_enhanced(df, symbol=symbol)
            
            if features.empty or len(features) < 200:  # Increased minimum samples
                print(f"⚠️ Insufficient quality data for {symbol}: {len(features)} samples")
                return False
            
            # Analyze target distribution
            target_counts = target.value_counts()
            print(f"Target distribution: {target_counts.to_dict()}")
            
            # Balance classes if needed
            if 0 in target_counts and target_counts[0] / len(target) > 0.7:
                print("⚠️ High proportion of 'Hold' targets, applying balancing...")
                features, target = self._balance_classes(features, target)
            
            # More rigorous train-test split
            X_train, X_test, y_train, y_test = self.time_series_train_test_split_improved(features, target)
            
            if X_train is None:
                return False
            
            # Aggressive feature selection
            X_train_selected, selected_features = self._select_features_improved(X_train, y_train, symbol)
            X_test_selected = X_test[selected_features]
            
            print(f"🔍 Selected {len(selected_features)} features from {len(features.columns)}")
            
            # Train with improved parameters
            return self._train_with_cross_validation(symbol, X_train_selected, X_test_selected, y_train, y_test, selected_features)
            
        except Exception as e:
            print(f"❌ Improved training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _balance_classes(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes by undersampling the majority class"""
        from sklearn.utils import resample
        
        # Separate classes
        hold_mask = target == 0
        buy_mask = target == 1
        sell_mask = target == -1
        
        hold_features = features[hold_mask]
        hold_target = target[hold_mask]
        buy_features = features[buy_mask]
        buy_target = target[buy_mask]
        sell_features = features[sell_mask]
        sell_target = target[sell_mask]
        
        # Find the smallest class size
        min_size = min(len(hold_features), len(buy_features), len(sell_features))
        
        if min_size < 10:  # If any class is too small, don't balance
            return features, target
        
        # Undersample majority classes
        hold_features_balanced = resample(hold_features, n_samples=min_size, random_state=42)
        hold_target_balanced = resample(hold_target, n_samples=min_size, random_state=42)
        buy_features_balanced = resample(buy_features, n_samples=min_size, random_state=42)
        buy_target_balanced = resample(buy_target, n_samples=min_size, random_state=42)
        sell_features_balanced = resample(sell_features, n_samples=min_size, random_state=42)
        sell_target_balanced = resample(sell_target, n_samples=min_size, random_state=42)
        
        # Combine balanced data
        features_balanced = pd.concat([hold_features_balanced, buy_features_balanced, sell_features_balanced])
        target_balanced = pd.concat([hold_target_balanced, buy_target_balanced, sell_target_balanced])
        
        # Shuffle
        indices = np.random.permutation(len(features_balanced))
        return features_balanced.iloc[indices], target_balanced.iloc[indices]

    def _select_features_improved(self, X_train: pd.DataFrame, y_train: pd.Series, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """More aggressive feature selection"""
        # Remove low variance features
        variances = X_train.var()
        low_variance = variances[variances < 0.001].index
        X_filtered = X_train.drop(columns=low_variance)
        
        # Remove highly correlated features
        corr_matrix = X_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
        X_filtered = X_filtered.drop(columns=high_corr)
        
        # Use mutual information for feature selection
        from sklearn.feature_selection import mutual_info_classif
        
        mi_scores = mutual_info_classif(X_filtered, y_train, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_filtered.columns)
        top_features = mi_series.nlargest(min(15, len(mi_series))).index.tolist()  # Reduced to 15 max
        
        return X_filtered[top_features], top_features

    def analyze_feature_quality(self, symbol: str, df: pd.DataFrame):
        """Analyze feature quality and identify potential issues"""
        features, target = self.prepare_training_data_enhanced(df, symbol=symbol)
        
        if features.empty:
            return {"error": "No features generated"}
        
        analysis = {
            'symbol': symbol,
            'feature_count': len(features.columns),
            'target_distribution': target.value_counts().to_dict(),
            'constant_features': [],
            'correlated_features': {},
            'feature_target_correlation': {}
        }
        
        # Check for constant features
        for col in features.columns:
            if features[col].std() == 0:
                analysis['constant_features'].append(col)
        
        # Check feature correlations
        corr_matrix = features.corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    analysis['correlated_features'][corr_matrix.columns[i]] = corr_matrix.columns[j]
        
        # Feature-target correlation
        for col in features.columns:
            if features[col].std() > 0:  # Skip constant features
                correlation = np.corrcoef(features[col], target)[0, 1] if len(target) == len(features) else 0
                analysis['feature_target_correlation'][col] = correlation
        
        return analysis

    def debug_feature_generation(self, symbol: str, df: pd.DataFrame):
        """Debug feature generation process"""
        print(f"\n🔍 DEBUGGING FEATURE GENERATION FOR {symbol}")
        print(f"Data shape: {df.shape}")
        print(f"Data columns: {df.columns.tolist()}")
        
        # Test feature generation at multiple points
        test_indices = [len(df)-1, len(df)-50, len(df)-100]
        
        for idx in test_indices:
            if idx < 50:
                continue
                
            features = self.prepare_features_point_in_time(df, idx, symbol=symbol)
            print(f"Index {idx}: {len(features.columns) if not features.empty else 0} features")
            
            if not features.empty:
                print(f"Sample features: {features.iloc[0].to_dict()}")

    def create_enhanced_target(self, df: pd.DataFrame, current_idx: int, symbol: str = None) -> int:
        """Improved target creation with dynamic, symbol-specific thresholds"""
        
        # Get symbol-specific configuration
        if symbol and symbol in self.symbol_volatility_profiles:
            target_configs = self._get_symbol_specific_target_configs(symbol)
        else:
            target_configs = self.target_configs
        
        # Calculate recent volatility for dynamic thresholds
        recent_data = df.iloc[max(0, current_idx-50):current_idx+1]
        if len(recent_data) < 20:
            return 0
            
        volatility = recent_data['close'].pct_change().std()
        if pd.isna(volatility) or volatility == 0:
            volatility = 0.02
        
        # Symbol-specific base thresholds
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            base_threshold = volatility * 1.2  # Tighter thresholds for stable coins
        elif symbol in ['SOLUSDT', 'DOGEUSDT']:
            base_threshold = volatility * 0.8  # Looser thresholds for volatile coins
        else:
            base_threshold = volatility
        
        # Multi-timeframe prediction with confidence weighting
        predictions = []
        confidences = []
        
        for config in target_configs:
            periods = config['periods']
            
            if current_idx + periods >= len(df):
                continue
                
            current_price = df['close'].iloc[current_idx]
            future_price = df['close'].iloc[current_idx + periods]
            price_change = (future_price - current_price) / current_price
            
            # Dynamic threshold based on timeframe and volatility
            timeframe_threshold = base_threshold * np.sqrt(periods/5)  # Scale with sqrt(time)
            
            # Strong signal detection
            if abs(price_change) > timeframe_threshold * 2.0:
                confidence = 0.9
            elif abs(price_change) > timeframe_threshold * 1.5:
                confidence = 0.7
            elif abs(price_change) > timeframe_threshold:
                confidence = 0.6
            else:
                continue  # Skip weak signals
                
            direction = 1 if price_change > 0 else -1
            predictions.append(direction)
            confidences.append(confidence * config['weight'])
        
        if not predictions:
            return 0
        
        # Weighted ensemble prediction
        weighted_sum = sum(p * c for p, c in zip(predictions, confidences))
        total_confidence = sum(confidences)
        
        if total_confidence == 0:
            return 0
        
        final_vote = weighted_sum / total_confidence
        
        # Convert to trading signal with confidence threshold
        if abs(final_vote) > 0.3:  # Reduced threshold for more signals
            return 1 if final_vote > 0 else -1
        else:
            return 0

    def create_simple_target(self, df: pd.DataFrame, current_idx: int, symbol: str = None) -> int:
        """Simplified target creation to reduce noise"""
        if current_idx + 10 >= len(df):
            return 0
            
        current_price = df['close'].iloc[current_idx]
        future_price = df['close'].iloc[current_idx + 10]  # Fixed 10-period lookahead
        
        price_change = (future_price - current_price) / current_price
        
        # Use fixed thresholds instead of dynamic ones
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            threshold = 0.008  # 0.8% for stable coins
        else:
            threshold = 0.012  # 1.2% for volatile coins
        
        if price_change > threshold:
            return 1
        elif price_change < -threshold:
            return -1
        else:
            return 0

    def enable_conservative_mode(self):
        print("🛡️ ENABLING ULTRA-CONSERVATIVE MODE")
        
        self.max_features = 6  # Further reduced
        self.min_training_samples = 600  # Increased
        self.enable_hyperparameter_optimization = False
        
        # Set the safe training method as default
        self.train_model = self.train_model_safe
        
        # Update all symbol configurations
        for symbol in self.symbol_model_complexity:
            self.symbol_model_complexity[symbol].update({
                'max_features': 4,  # Much fewer features
                'enable_hpo': False,
                'model_depth': 'ultra_simple'  # New level
            })
        
        self.train_model = self.train_conservative_model

    def train_conservative_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """Ultra-conservative training to prevent overfitting"""
        try:
            print(f"🎯 CONSERVATIVE TRAINING FOR {symbol}")
            
            features, target = self.prepare_training_data_enhanced_v2(df, symbol=symbol)
            
            if features.empty or len(features) < 400:
                print(f"⚠️ Insufficient data for {symbol}: {len(features)} samples")
                return False
            
            features = features.iloc[:, :15]
            
            X_train, X_test, y_train, y_test = self.time_series_split_conservative(features, target)
            
            if X_train is None:
                return False

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            self.scalers[symbol] = scaler

            selected_features = self._stable_feature_selection(X_train_scaled, y_train, X_test_scaled, y_test, symbol, n_features=8)
            X_train_sel = X_train_scaled[selected_features]
            X_test_sel = X_test_scaled[selected_features]
            
            rf_params, gb_params = self._get_ultra_regularized_params(symbol)
            
            rf = RandomForestClassifier(**rf_params)
            gb = GradientBoostingClassifier(**gb_params)
            
            rf.fit(X_train_sel, y_train)
            gb.fit(X_train_sel, y_train)
            
            train_pred_rf = rf.predict(X_train_sel)
            test_pred_rf = rf.predict(X_test_sel)
            
            train_acc_rf = accuracy_score(y_train, train_pred_rf)
            test_acc_rf = accuracy_score(y_test, test_pred_rf)
            overfit_rf = train_acc_rf - test_acc_rf

            train_pred_gb = gb.predict(X_train_sel)
            test_pred_gb = gb.predict(X_test_sel)
            train_acc_gb = accuracy_score(y_train, train_pred_gb)
            test_acc_gb = accuracy_score(y_test, test_pred_gb)
            overfit_gb = train_acc_gb - test_acc_gb
            
            max_allowed_overfit = 0.10
            
            if overfit_rf > max_allowed_overfit or overfit_gb > max_allowed_overfit:
                print(f"🚫 REJECTED {symbol}: Overfit RF={overfit_rf:.3f}, GB={overfit_gb:.3f} > {max_allowed_overfit}")
                return False
            
            if test_acc_rf < 0.45 or test_acc_gb < 0.45:
                print(f"🚫 REJECTED {symbol}: Test accuracy RF={test_acc_rf:.3f}, GB={test_acc_gb:.3f} too low")
                return False

            if symbol in self.models:
                self.previous_models[symbol] = self.models[symbol].copy()
                self.previous_scalers[symbol] = self.scalers[symbol]
            
            self.models[symbol] = {'rf': rf, 'gb': gb}
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
                
            self.feature_importance[symbol].update({
                'selected_features': selected_features,
                'rf_importance': rf.feature_importances_.tolist(),
                'gb_importance': gb.feature_importances_.tolist(),
                'test_accuracy': (test_acc_rf + test_acc_gb) / 2,
                'overfit_score': (overfit_rf + overfit_gb) / 2
            })
            
            print(f"✅ CONSERVATIVE MODEL FOR {symbol}: "
                  f"RF Train={train_acc_rf:.3f}, Test={test_acc_rf:.3f}, Overfit={overfit_rf:.3f} | "
                  f"GB Train={train_acc_gb:.3f}, Test={test_acc_gb:.3f}, Overfit={overfit_gb:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Conservative training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_ultra_regularized_params(self, symbol):
        """Extremely regularized parameters to combat overfitting"""
        
        ultra_rf_params = {
            'n_estimators': 80,
            'max_depth': 3,
            'min_samples_split': 50,
            'min_samples_leaf': 40,
            'max_features': 0.3,
            'bootstrap': True,
            'max_samples': 0.7,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        ultra_gb_params = {
            'n_estimators': 60,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_samples_split': 40,
            'min_samples_leaf': 20,
            'subsample': 0.6,
            'max_features': 0.4,
            'random_state': 42
        }
        
        return ultra_rf_params, ultra_gb_params

    def time_series_split_conservative(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.3, gap: int = 20):
        if len(features) < 100:
            return None, None, None, None
            
        split_idx = int(len(features) * (1 - test_size))
        train_end_idx = split_idx - gap
        
        if train_end_idx < 50:
            train_end_idx = split_idx
            
        X_train = features.iloc[:train_end_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:train_end_idx]
        y_test = target.iloc[split_idx:]
        
        if len(X_train) < 50 or len(X_test) < 20:
            return None, None, None, None
            
        return X_train, X_test, y_train, y_test

    def _stable_feature_selection(self, X_train, y_train, X_test, y_test, symbol, n_features=12):
        """Stable feature selection that preserves feature names"""
        try:
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train_common = X_train[common_features]
            
            variances = X_train_common.var()
            stable_features = variances[variances > 0.001].index.tolist()
            
            if len(stable_features) < 5:
                print(f"⚠️ Too few stable features for {symbol}, using fallback")
                return common_features[:min(8, len(common_features))]
            
            selector = SelectKBest(score_func=f_classif, k=min(n_features, len(stable_features)))
            selector.fit(X_train_common[stable_features], y_train)
            
            selected_mask = selector.get_support()
            selected_features = [stable_features[i] for i in range(len(stable_features)) if selected_mask[i]]
            
            print(f"✅ Stable feature selection for {symbol}: {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            print(f"⚠️ Stable feature selection failed for {symbol}: {e}")
            variances = X_train.var()
            return variances.nlargest(min(8, len(variances))).index.tolist()

    def create_regime_aware_target(self, df: pd.DataFrame, current_idx: int, symbol: str = None) -> int:
        """Simplified robust target creation"""
        if current_idx + 15 >= len(df):
            return 0
            
        current_price = df['close'].iloc[current_idx]
        future_prices = df['close'].iloc[current_idx:current_idx+16]
        
        if len(future_prices) < 15:
            return 0
            
        # Calculate future returns at different horizons
        returns_5 = (future_prices.iloc[4] - current_price) / current_price
        returns_10 = (future_prices.iloc[9] - current_price) / current_price
        returns_15 = (future_prices.iloc[14] - current_price) / current_price
        
        # Use ATR for dynamic threshold
        recent_data = df.iloc[max(0, current_idx-20):current_idx+1]
        if len(recent_data) < 14:
            return 0
            
        try:
            atr = talib.ATR(recent_data['high'], recent_data['low'], recent_data['close'], timeperiod=14)
            atr_percent = atr.iloc[-1] / current_price if current_price > 0 else 0.01
            threshold = atr_percent * 1.5  # 1.5x ATR as threshold
        except:
            threshold = 0.01  # Fallback threshold
        
        # Weighted decision
        weighted_return = returns_5 * 0.4 + returns_10 * 0.3 + returns_15 * 0.3
        
        if weighted_return > threshold:
            return 1
        elif weighted_return < -threshold:
            return -1
        else:
            return 0

    def _detect_current_regime(self, data):
        """Detect current market regime"""
        close = data['close']
        returns = close.pct_change().dropna()
        
        if len(returns) < 20:
            return "unknown"
        
        volatility = returns.rolling(10).std().iloc[-1]
        trend_strength = abs(self._calculate_trend_strength(close, 15))
        
        if pd.isna(volatility) or pd.isna(trend_strength):
            return "unknown"

        if volatility > 0.025:
            return "high_volatility"
        elif trend_strength > 0.4:
            return "trending"
        else:
            return "consolidation"

    def _calculate_high_signal_features(self, close, high, low, volume, symbol):
        """Calculate robust features with proper error handling"""
        features = {}
        try:
            if len(close) < 50:  # Increased minimum
                return {}
                
            # Use iloc safely for pandas operations
            close_vals = close.values if hasattr(close, 'values') else close
            high_vals = high.values if hasattr(high, 'values') else high  
            low_vals = low.values if hasattr(low, 'values') else low
            volume_vals = volume.values if hasattr(volume, 'values') else volume
            
            # 1. Momentum acceleration (safe calculation)
            if len(close_vals) >= 11:
                mom_5 = (close_vals[-1] / close_vals[-6] - 1) if len(close_vals) >= 6 else 0
                mom_10 = (close_vals[-1] / close_vals[-11] - 1) if len(close_vals) >= 11 else 0
                features['momentum_accel'] = mom_5 - mom_10
            else:
                features['momentum_accel'] = 0
                
            # 2. Breakout strength (safe rolling calculations)
            if len(high_vals) >= 20:
                resistance = np.max(high_vals[-20:])
                features['breakout_strength'] = (close_vals[-1] - resistance) / close_vals[-1] if close_vals[-1] > resistance else 0
            else:
                features['breakout_strength'] = 0
                
            # 3. Volume breakout (safe calculation)
            if len(volume_vals) >= 20:
                volume_sma = np.mean(volume_vals[-20:])
                features['volume_breakout'] = (volume_vals[-1] / volume_sma - 1) if volume_sma > 0 else 0
            else:
                features['volume_breakout'] = 0
                
            # 4. Add core TA features that we know work
            core_features = self._calculate_core_ta_features(close, high, low, volume)
            features.update(core_features)
            
        except Exception as e:
            print(f"Error in high signal features for {symbol}: {e}")
            # Return minimal feature set instead of empty
            features = {'momentum_accel': 0, 'volume_breakout': 0, 'atr_14_perc': 0.01}
            
        return features

    def _calculate_support_quality(self, close, low):
        """Calculate quality of support levels"""
        try:
            if len(low) < 10:
                return 0
            recent_lows = low.tail(10)
            support_level = recent_lows.min()
            touches = (recent_lows <= support_level * 1.001).sum()
            return touches / len(recent_lows)
        except:
            return 0

    def _calculate_resistance_quality(self, close, high):
        """Calculate quality of resistance levels"""
        try:
            if len(high) < 10:
                return 0
            recent_highs = high.tail(10)
            resistance_level = recent_highs.max()
            touches = (recent_highs >= resistance_level * 0.999).sum()
            return touches / len(recent_highs)
        except:
            return 0

    def _classify_market_structure(self, close, high, low, window=10):
        """Classify market structure based on recent highs and lows"""
        try:
            if len(close) < window * 2:
                return 0
            
            recent_highs = high.tail(window)
            prev_highs = high.iloc[-window*2:-window]
            
            recent_lows = low.tail(window)
            prev_lows = low.iloc[-window*2:-window]

            if recent_highs.max() > prev_highs.max() and recent_lows.max() > prev_lows.max():
                return 1
            elif recent_highs.min() < prev_highs.min() and recent_lows.min() < prev_lows.min():
                return -1
            else:
                return 0
        except:
            return 0

    def _get_symbol_specific_target_configs(self, symbol: str) -> List[Dict]:
        """Get symbol-specific target configurations"""
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            periods = profile['timeframes']
            multiplier = profile['multiplier']
        else:
            periods = [5, 10, 20]
            multiplier = 2.0
        
        # Create configs based on available periods
        weights = [0.4, 0.3, 0.3] if len(periods) == 3 else [0.6, 0.4] if len(periods) == 2 else [1.0]
        
        configs = []
        for i, period in enumerate(periods):
            configs.append({
                'periods': period,
                'weight': weights[i] if i < len(weights) else 1.0/len(periods),
                'threshold_multiplier': multiplier * (1 + i * 0.2)  # Increase multiplier for longer timeframes
            })
        
        return configs

    def _get_symbol_thresholds(self, symbol: str, volatility: float) -> Dict[str, float]:   
        """Get symbol-specific decision thresholds"""
        base_strong = 0.15
        base_weak = 0.05
        
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] == 'low':
                # Tighter thresholds for BTC
                strong_threshold = base_strong * 0.8
                weak_threshold = base_weak * 0.8
            elif profile['volatility_tier'] == 'extreme':
                # Wider thresholds for DOGE/SOL
                strong_threshold = base_strong * 1.5
                weak_threshold = base_weak * 1.5
            else:
                # Moderate adjustments
                strong_threshold = base_strong
                weak_threshold = base_weak
        else:
            strong_threshold = base_strong
            weak_threshold = base_weak
        
        # Adjust for current volatility
        vol_adjustment = 1.0 + (volatility - 0.02) * 10  # Scale adjustment
        strong_threshold *= vol_adjustment
        weak_threshold *= vol_adjustment
        
        return {
            'strong': max(0.10, min(0.25, strong_threshold)),
            'weak': max(0.03, min(0.15, weak_threshold))
        }
    def prepare_training_data_enhanced(self, df: pd.DataFrame, min_samples: int = None, symbol: str = None) -> tuple:
        """Enhanced training data preparation with multi-timeframe targets"""
        if min_samples is None:
            min_samples = self.min_training_samples
            
        max_bars = min(2000, len(df) - 50)
        if max_bars < min_samples + 50:
            return pd.DataFrame(), pd.Series()
        
        df = df.tail(max_bars).copy()
        
        features_list = []
        targets = []
        
        stride = max(1, len(df) // 400) # Reduced stride to get more samples
        
        # --- FIX: Determine max target period from the correct config ---
        if symbol and symbol in self.symbol_volatility_profiles:
            target_configs = self._get_symbol_specific_target_configs(symbol)
        else:
            target_configs = self.target_configs
        max_target_period = max([c['periods'] for c in target_configs])
        # --- END FIX ---

        # --- FIX: Align regime filtering inside the feature generation loop ---
        tradable_regimes = ['bull_trend', 'bear_trend', 'neutral', 'ranging', 'high_volatility']
        
        for i in range(50, len(df) - max_target_period, stride):
            
            # --- FIX 2: Calculate regime filter AT point-in-time 'i' ---
            regime_data_for_features = df.iloc[:i]
            if len(regime_data_for_features) < 100:
                continue
            regime = self._detect_training_regime(regime_data_for_features, symbol)
            
            # --- FIX 2: Apply filter *before* generating features/targets ---
            if regime in tradable_regimes:
                features = self.prepare_features_point_in_time(df, i, symbol=symbol)
                target = self.create_enhanced_target(df, i, symbol)
                
                if not features.empty:
                    features_list.append(features.iloc[0])
                    targets.append(target)
        # --- END FIX 2 ---
        
        if len(features_list) < min_samples:
            return pd.DataFrame(), pd.Series()
        
        features_df = pd.DataFrame(features_list).fillna(0)
        target_series = pd.Series(targets, index=features_df.index)
        
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

    def _get_symbol_specific_parameters(self, symbol: str, X_train_scaled: np.ndarray, y_train: pd.Series) -> Tuple[dict, dict]:
        """Get symbol-specific model parameters"""
        if symbol in self.symbol_model_complexity and self.symbol_model_complexity[symbol].get('enable_hpo', False):
            # Use hyperparameter optimization for complex models
            return self._optimize_hyperparameters(X_train_scaled, y_train)
        else:
            # Use default parameters for simple models
            return self.rf_params.copy(), self.gb_params.copy()

    def train_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """Enhanced training with symbol-specific configuration"""
        try:
            leakage_report = self.detect_data_leakage(symbol, df)
            if leakage_report['issues']:
                print(f"⚠️ Potential data leakage detected for {symbol}:")
                for issue in leakage_report['issues']:
                    print(f"   - {issue}")
            
            print(f"🔄 Training symbol-specific model for {symbol}...")
            
            self._apply_symbol_specific_config(symbol)
            
            training_start = datetime.now()
            
            if not hasattr(self, 'model_versions'):
                self.model_versions = {}
            
            # ✅ FIX: Initialize feature_importance for this symbol
            if not hasattr(self, 'feature_importance'):
                self.feature_importance = {}
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
            
            if symbol in self.models:
                self.previous_models[symbol] = self.models[symbol].copy()
                self.previous_scalers[symbol] = self.scalers[symbol]
                print(f"💾 Saved previous model for {symbol} for ensembling")
            
            # --- FIX 2: Regime filtering is now done inside prepare_training_data_enhanced ---
            features, target = self.prepare_training_data_enhanced(df, symbol=symbol)
            
            if features.empty or target.empty:
                print(f"⚠️ Insufficient data for training {symbol}")
                self._record_training_failure(symbol, "insufficient_data")
                return False
                
            # --- FIX 2: No longer need to call _filter_training_data_by_regime ---
            # We now use 'features' and 'target' directly as they are already filtered
            
            X_train, X_test, y_train, y_test = self.time_series_train_test_split(features, target)
            
            if X_train is None:
                print(f"⚠️ Insufficient data after split for {symbol}")
                self._record_training_failure(symbol, "split_failed")
                return False

            # --- NEW: Log training details ---
            self.log_training_details(symbol, features, target, X_train, X_test)
            # --- FIX 1: Perform Feature Selection *AFTER* split, *ONLY* on (X_train, y_train) ---
            X_train_selected, selected_features = self._select_features(X_train, y_train, symbol)
            # Apply the *same* selected features to the test set
            X_test_selected = X_test[selected_features]
            # --- END FIX 1 ---

            self.feature_importance[symbol]['selected_features'] = selected_features
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            rf_params, gb_params = self._get_symbol_specific_parameters(symbol, X_train_scaled, y_train)
            
            rf = RandomForestClassifier(**rf_params)
            gb = GradientBoostingClassifier(**gb_params)

            rf.fit(X_train_scaled, y_train)
            gb.fit(X_train_scaled, y_train)

            # Enhanced validation with symbol-specific metrics
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

            print(f"\n--- {symbol} RF Classification Report ---")
            print(classification_report(y_test, rf_pred, zero_division=0))
            print(f"--- {symbol} GB Classification Report ---")
            print(classification_report(y_test, gb_pred, zero_division=0))

            # Overfitting detection
            rf_train_pred = rf.predict(X_train_scaled)
            rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
            rf_overfit = rf_train_accuracy - rf_accuracy
            
            gb_train_pred = gb.predict(X_train_scaled)
            gb_train_accuracy = accuracy_score(y_train, gb_train_pred)
            gb_overfit = gb_train_accuracy - gb_accuracy
            
            if rf_overfit > 0.15 or gb_overfit > 0.15:
                print(f"⚠️ Potential overfitting detected for {symbol}: RF={rf_overfit:.3f}, GB={gb_overfit:.3f}")
                # --- ADDED: Reject model if overfitting is too high ---
                print(f"❌ Rejecting model for {symbol} due to high overfitting.")
                self._record_training_failure(symbol, "overfitting_reject")
                return False
                # --- END ADDED ---

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

    def _apply_symbol_specific_config(self, symbol: str):
        """Apply symbol-specific configuration"""
        if symbol in self.symbol_model_complexity:
            config = self.symbol_model_complexity[symbol]
            self.max_features = config['max_features']
            self.enable_hyperparameter_optimization = config['enable_hpo']
            
            # Adjust model parameters based on complexity
            if config['model_depth'] == 'deep':
                self.rf_params['n_estimators'] = 150
                self.rf_params['max_depth'] = 12
            elif config['model_depth'] == 'simple':
                self.rf_params['n_estimators'] = 80
                self.rf_params['max_depth'] = 6
            # medium keeps default values
        else:
            # Default configuration for unknown symbols
            self.max_features = 20
            self.enable_hyperparameter_optimization = False

    def _filter_training_data_by_regime(self, features: pd.DataFrame, target: pd.Series, 
                                    df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter training data to only include tradable market regimes"""
        try:
            # Use technical analyzer to detect regimes for each training point
            tradable_indices = []
            
            for i in range(len(features)):
                if i < 50:  # Need enough data for regime detection
                    continue
                    
                # Get historical data up to this point
                historical_window = df.iloc[:i+1]
                if len(historical_window) < 100:
                    continue
                    
                # Detect market regime
                regime = self._detect_training_regime(historical_window, symbol)
                
                # Only include data from tradable regimes
                if regime in ['bull_trend', 'bear_trend', 'neutral']:
                    tradable_indices.append(i)
            
            if len(tradable_indices) > self.min_training_samples:
                return features.iloc[tradable_indices], target.iloc[tradable_indices]
            else:
                # Fallback: use all data if not enough tradable samples
                return features, target
                
        except Exception as e:
            self.logger.error(f"Error filtering training data by regime for {symbol}: {e}")
            return features, target  # Fallback to all data

    def _calculate_trend_strength(self, close_prices: pd.Series, period: int = 20) -> float:
        """Calculate trend strength using linear regression slope"""
        try:
            if len(close_prices) < period:
                return 0.0
            
            # Use linear regression to determine trend strength
            x = np.arange(len(close_prices.tail(period)))
            y = close_prices.tail(period).values
            
            # Handle NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 0.0
                
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            
            # Normalize slope by average price to get relative strength
            avg_price = np.mean(y_clean)
            if avg_price == 0:
                return 0.0
                
            trend_strength = slope / avg_price
            
            # Scale to reasonable range and cap
            return max(-1.0, min(1.0, trend_strength * 100))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0

    def _detect_training_regime(self, df: pd.DataFrame, symbol: str) -> str:
        """Detect market regime for training data filtering"""
        try:
            close = df['close'].astype(float)
            returns = close.pct_change().dropna()
            
            if len(returns) < 20:
                return 'neutral'
            
            volatility = returns.rolling(20).std().iloc[-1]
            trend_strength = self._calculate_trend_strength(close)
            
            # Symbol-specific regime detection
            if symbol in self.symbol_volatility_profiles:
                profile = self.symbol_volatility_profiles[symbol]
                vol_threshold_high = 0.04 if profile['volatility_tier'] == 'low' else 0.06
                vol_threshold_low = 0.015 if profile['volatility_tier'] == 'low' else 0.025
            else:
                vol_threshold_high = 0.05
                vol_threshold_low = 0.02
            
            if volatility > vol_threshold_high:
                return 'high_volatility'
            elif volatility < vol_threshold_low and trend_strength < 0.3:
                return 'ranging'
            elif trend_strength > 0.5:
                return 'bull_trend' if close.iloc[-1] > close.iloc[-20] else 'bear_trend'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error detecting training regime for {symbol}: {e}")
            return 'neutral'

    def walk_forward_validation_enhanced(self, symbol: str, df: pd.DataFrame, n_splits: int = None) -> Dict:
        """Enhanced walk-forward validation with multi-timeframe targets"""
        if n_splits is None:
            n_splits = self.walk_forward_splits
            
        try:
            features, target = self.prepare_training_data_enhanced(df)
            
            if features.empty or target.empty:
                return {'success': False, 'reason': 'Insufficient data'}

            # --- DYNAMIC SPLIT CALCULATION ---
            min_train_size = 50
            test_size = 50
            gap = 10
            n_samples = len(features)
            
            # Calculate max possible splits
            # Formula: (n_samples - min_train_size - gap) // (test_size + gap)
            max_possible_splits = (n_samples - min_train_size - gap) // (test_size + gap)
            
            if n_splits > max_possible_splits:
                n_splits = max(1, max_possible_splits)
                print(f"⚠️ Walk-forward splits reduced to {n_splits} due to limited sample size ({n_samples})")
            
            if n_splits < 1:
                return {'success': False, 'reason': f'Not enough samples ({n_samples}) for even 1 walk-forward split'}
            # --- END DYNAMIC SPLIT ---

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
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
        """Enhanced prediction with symbol-specific confidence calibration"""
        if symbol not in self.models:
            return self.fallback_prediction_strategy(symbol, df)
            
        try:
            # Apply symbol-specific configuration
            self._apply_symbol_specific_config(symbol)
            
            # Data validation with symbol-specific checks
            if df is None or len(df) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
                return self.fallback_prediction_strategy(symbol, df)
                
            # Check for valid close prices with symbol-specific validation
            close_prices = df['close'].astype(float)
            if close_prices.isnull().all() or (close_prices == 0).all():
                self.logger.warning(f"Invalid close prices for {symbol} ML prediction")
                return self.fallback_prediction_strategy(symbol, df)
            
            # Symbol-specific data quality checks
            if not self._validate_symbol_data_quality(df, symbol):
                self.logger.warning(f"Poor data quality for {symbol}, using fallback")
                return self.fallback_prediction_strategy(symbol, df)
            
            # Prepare features with symbol-specific configuration
            features = self.prepare_features_point_in_time(df, len(df), symbol=symbol)
            
            if features.empty:
                return self.fallback_prediction_strategy(symbol, df)
            
            # Apply symbol-specific feature selection
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
            
            # Validate feature consistency
            if not self.validate_feature_consistency(symbol, features_selected):
                self.logger.warning(f"Feature consistency check failed for {symbol}, using fallback")
                return self.fallback_prediction_strategy(symbol, df)

            # Scale features
            scaled_features = self.scalers[symbol].transform(features_selected)

            # Get predictions
            rf_pred = self.models[symbol]['rf'].predict(scaled_features)[0]
            gb_pred = self.models[symbol]['gb'].predict(scaled_features)[0]

            rf_proba = self.models[symbol]['rf'].predict_proba(scaled_features)[0]
            gb_proba = self.models[symbol]['gb'].predict_proba(scaled_features)[0]

            # Symbol-specific confidence calibration
            rf_confidence = self._calibrate_confidence_symbol(rf_proba, rf_pred, symbol)
            gb_confidence = self._calibrate_confidence_symbol(gb_proba, gb_pred, symbol)
            
            # Apply symbol-specific confidence thresholds
            min_confidence = self._get_symbol_min_confidence(symbol)
            if rf_confidence < min_confidence and gb_confidence < min_confidence:
                self.logger.info(f"Low confidence for {symbol} (RF: {rf_confidence:.2f}, GB: {gb_confidence:.2f}), using fallback")
                return self.fallback_prediction_strategy(symbol, df)

            # Ensemble decision making with symbol-specific weights
            ensemble_vote, ensemble_confidence = self._symbol_specific_ensemble(
                rf_pred, gb_pred, rf_confidence, gb_confidence, symbol
            )

            # Convert to trading signal
            trading_signal = self._convert_to_trading_signal(ensemble_vote)
            
            result = {
                'prediction': trading_signal,
                'raw_prediction': ensemble_vote,
                'confidence': ensemble_confidence,
                'rf_pred': rf_pred,
                'gb_pred': gb_pred,
                'rf_confidence': rf_confidence,
                'gb_confidence': gb_confidence,
                'timestamp': datetime.now(),
                'model_used': 'ml_ensemble_enhanced_symbol_specific',
                'symbol_specific_config': True,
                'min_confidence_threshold': min_confidence
            }
            
            # Store prediction quality with symbol context
            if self.database:
                self.database.store_prediction_quality(
                    symbol=symbol,
                    prediction=trading_signal,
                    actual=None,
                    confidence=ensemble_confidence,
                    model_used='ml_ensemble_enhanced_symbol_specific',
                    features_used=selected_features if selected_features else features.columns.tolist(),
                    symbol_volatility_tier=self.symbol_volatility_profiles.get(symbol, {}).get('volatility_tier', 'unknown'),
                    market_cap_tier=self._get_market_cap_tier(symbol)
                )
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            return self.fallback_prediction_strategy(symbol, df)

    def _calibrate_confidence_symbol(self, probabilities: np.ndarray, prediction: int, symbol: str) -> float:
        """Symbol-specific confidence calibration"""
        base_confidence = self._calibrate_confidence(probabilities, prediction)
        
        # Apply symbol-specific adjustments
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] in ['high', 'extreme']:
                # Reduce confidence for volatile symbols
                base_confidence *= 0.8
            elif profile['volatility_tier'] == 'low':
                # Boost confidence for stable symbols
                base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence

    def _get_symbol_min_confidence(self, symbol: str) -> float:
        """Get symbol-specific minimum confidence threshold"""
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] == 'low':
                return 0.55  # Lower threshold for BTC
            elif profile['volatility_tier'] == 'extreme':
                return 0.75  # Higher threshold for DOGE/SOL
            else:
                return 0.65  # Medium threshold
        return 0.60  # Default

    def _symbol_specific_ensemble(self, rf_pred: int, gb_pred: int, 
                                rf_confidence: float, gb_confidence: float, 
                                symbol: str) -> Tuple[int, float]:
        """Symbol-specific ensemble decision making"""
        
        # Base ensemble logic
        if rf_pred == gb_pred:
            ensemble_vote = rf_pred
            confidence = (rf_confidence + gb_confidence) / 2
        else:
            # Consider prediction strength
            if rf_confidence > gb_confidence:
                ensemble_vote = rf_pred
                confidence = rf_confidence
            else:
                ensemble_vote = gb_pred
                confidence = gb_confidence
        
        # Symbol-specific adjustments
        if symbol in self.symbol_model_complexity:
            config = self.symbol_model_complexity[symbol]
            if config['model_depth'] == 'simple':
                # For simple models, be more conservative
                if confidence < 0.7:
                    ensemble_vote = 0  # Force hold on low confidence
                    confidence = max(confidence, 0.5)
        
        return ensemble_vote, confidence

    def _validate_symbol_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """Symbol-specific data quality validation"""
        try:
            close_prices = df['close'].astype(float)
            volume = df['volume'].astype(float)
            
            # Check for sufficient price movement
            price_range = (close_prices.max() - close_prices.min()) / close_prices.mean()
            if price_range < 0.005:  # Less than 0.5% movement
                self.logger.warning(f"Low price movement for {symbol}: {price_range:.4f}")
                return False
            
            # Check volume requirements
            avg_volume = volume.mean()
            if symbol in ['XRPUSDT', 'DOGEUSDT'] and avg_volume < 1000000:  # Lower threshold for small caps
                self.logger.warning(f"Low volume for {symbol}: {avg_volume:.0f}")
                return False
            
            # Check for extreme volatility spikes
            returns = close_prices.pct_change().dropna()
            max_daily_move = returns.abs().max()
            if max_daily_move > 0.15:  # Filter 15%+ daily moves
                self.logger.warning(f"Extreme volatility spike for {symbol}: {max_daily_move:.4f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data quality for {symbol}: {e}")
            return True  # Allow proceeding on error

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
            
            # Use a fixed rolling window (e.g., 20 periods) to prevent lookback leak
            window = 20
            returns = close.pct_change()
            
            # Calculate rolling sums
            net_movement = returns.rolling(window).sum()
            total_movement = returns.abs().rolling(window).sum()

            if net_movement.empty or total_movement.empty:
                return 0.5

            # Get the last value
            last_net_movement = net_movement.iloc[-1]
            last_total_movement = total_movement.iloc[-1]

            if last_total_movement == 0:
                return 0.5
            
            efficiency = abs(last_net_movement / last_total_movement)
            return efficiency if not pd.isna(efficiency) else 0.5
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


    def analyze_feature_quality(self, symbol: str, df: pd.DataFrame):
        """Analyze feature quality and identify potential issues"""
        features, target = self.prepare_training_data_enhanced(df, symbol=symbol)
        
        if features.empty:
            return {"error": "No features generated"}
        
        analysis = {
            'symbol': symbol,
            'feature_count': len(features.columns),
            'target_distribution': target.value_counts().to_dict(),
            'constant_features': [],
            'correlated_features': {},
            'feature_target_correlation': {}
        }
        
        # Check for constant features
        for col in features.columns:
            if features[col].std() == 0:
                analysis['constant_features'].append(col)
        
        # Check feature correlations
        corr_matrix = features.corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    if col1 not in analysis['correlated_features']:
                         analysis['correlated_features'][col1] = []
                    analysis['correlated_features'][col1].append(col2)
        
        # Feature-target correlation
        for col in features.columns:
            if features[col].std() > 0:  # Skip constant features
                try:
                    correlation = np.corrcoef(features[col], target)[0, 1] if len(target) == len(features) else 0
                    analysis['feature_target_correlation'][col] = correlation
                except Exception:
                    analysis['feature_target_correlation'][col] = 0.0
        
        return analysis

    def debug_feature_generation(self, symbol: str, df: pd.DataFrame):
        """Debug feature generation process"""
        print(f"\n🔍 DEBUGGING FEATURE GENERATION FOR {symbol}")
        print(f"Data shape: {df.shape}")
        
        # Test feature generation at multiple points
        test_indices = [len(df)-1, len(df)-50, len(df)-100]
        
        for idx in test_indices:
            if idx < 50:
                continue
                
            features = self.prepare_features_point_in_time(df, idx, symbol=symbol)
            print(f"Index {idx}: {len(features.columns) if not features.empty else 0} features")
            
            if not features.empty:
                print(f"   Sample features (first 5): {dict(list(features.iloc[0].to_dict().items())[:5])}")

    def log_training_details(self, symbol: str, features: pd.DataFrame, target: pd.Series, 
                            X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Log detailed training information"""
        try:
            log_message = f"""
            🎯 TRAINING DETAILS FOR {symbol}
            =================================
            Overall Statistics:
            - Total samples: {len(features)}
            - Training samples: {len(X_train)}
            - Test samples: {len(X_test)}
            - Feature count (original): {len(features.columns)}
            
            Target Distribution:
            {target.value_counts(normalize=True).to_dict()}
            
            Feature Statistics:
            - Mean features per sample: {features.mean().mean():.4f}
            - Std features per sample: {features.std().mean():.4f}
            - Features with zero variance: {len(features.columns[features.std() == 0])}
            """
            print(log_message)
        except Exception as e:
            print(f"Error logging training details: {e}")

    # Keep all other existing methods (feature selection, drift detection, etc.)
    # They remain largely the same but now work with enhanced targets

    def _select_features(self, X_train: pd.DataFrame, y_train: pd.Series, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """More aggressive feature selection"""
        try:
            # Get max features from symbol config
            max_features = self.max_features
            if symbol in self.symbol_model_complexity:
                max_features = self.symbol_model_complexity[symbol]['max_features']
            
            # Remove low variance features
            variances = X_train.var()
            low_variance = variances[variances < 0.001].index
            X_filtered = X_train.drop(columns=low_variance)
            
            if len(low_variance) > 0:
                print(f"   Removed {len(low_variance)} low-variance features")

            # Remove highly correlated features
            corr_matrix = X_filtered.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
            X_filtered = X_filtered.drop(columns=high_corr)
            
            if len(high_corr) > 0:
                print(f"   Removed {len(high_corr)} highly-correlated features")

            if X_filtered.empty:
                print("⚠️ No features left after variance and correlation filtering.")
                return X_train.iloc[:, :max_features], X_train.columns.tolist()[:max_features]

            # Use mutual information for feature selection
            mi_scores = mutual_info_classif(X_filtered, y_train, random_state=42)
            mi_series = pd.Series(mi_scores, index=X_filtered.columns)
            top_features = mi_series.nlargest(min(max_features, len(mi_series))).index.tolist()
            
            if not top_features:
                 print("⚠️ Mutual information selection returned no features. Using fallback.")
                 return X_train.iloc[:, :max_features], X_train.columns.tolist()[:max_features]
            
            print(f"🔍 Feature selection for {symbol}: {len(top_features)} features selected from {len(X_train.columns)}")
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
            
            self.feature_importance[symbol]['selected_features'] = top_features
            self.feature_importance[symbol]['original_feature_count'] = len(X_train.columns)
            self.feature_importance[symbol]['all_available_features'] = X_train.columns.tolist()
            
            return X_filtered[top_features], top_features

        except Exception as e:
            print(f"⚠️ Feature selection failed for {symbol}: {e}")
            max_features = self.max_features
            if symbol in self.symbol_model_complexity:
                max_features = self.symbol_model_complexity[symbol]['max_features']
                
            selected_features = X_train.columns.tolist()[:max_features]
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
            self.feature_importance[symbol]['selected_features'] = selected_features
            self.feature_importance[symbol]['selection_failed'] = True
            
            return X_train[selected_features], selected_features

    def validate_feature_consistency(self, symbol: str, current_features: pd.DataFrame) -> bool:
        """Validate that current features match training features"""
        if symbol not in self.feature_importance:
            return False
            
        selected_features = self.feature_importance[symbol].get('selected_features', [])
        if not selected_features:
            return False
            
        # Check if all selected features are present in current data
        missing_features = set(selected_features) - set(current_features.columns)
        if missing_features:
            self.logger.warning(f"Feature mismatch for {symbol}: Missing {len(missing_features)} features")
            return False
            
        return True

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

    def time_series_train_test_split(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2, gap: int = 10):
        if len(features) < 100:
            return None, None, None, None
            
        split_idx = int(len(features) * (1 - test_size))
        train_end_idx = split_idx - gap
        
        if train_end_idx < 50:
            train_end_idx = split_idx
            
        X_train = features.iloc[:train_end_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:train_end_idx]
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
            
            # === FIX: Handle feature dimension mismatch ===
            expected_features = getattr(scaler, 'n_features_in_', None)
            if expected_features is not None and current_features.shape[1] != expected_features:
                return {
                    'drift_detected': True, 
                    'reason': f'Feature dimension mismatch: {current_features.shape[1]} vs {expected_features}',
                    'feature_mismatch': True
                }
            
            if len(current_features) == 1:
                current_scaled = scaler.transform(current_features)
            else:
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
                'threshold_violations': np.sum(drift_scores < 0),
                'feature_mismatch': False
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting feature drift for {symbol}: {e}")
            return {'drift_detected': False, 'reason': str(e), 'feature_mismatch': True}

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

    def detect_data_leakage(self, symbol: str, df: pd.DataFrame) -> Dict:
        leakage_report = {
            'symbol': symbol,
            'issues': [],
            'suggestions': []
        }
        
        features, target = self.prepare_training_data_enhanced(df, symbol=symbol)
        
        if not features.empty:
            constant_features = features.columns[features.std() == 0]
            if len(constant_features) > 0:
                leakage_report['issues'].append(f"Constant features detected: {list(constant_features)}")
                leakage_report['suggestions'].append("Review feature calculation; constant features provide no value")
        
            if symbol and symbol in self.symbol_volatility_profiles:
                target_configs = self._get_symbol_specific_target_configs(symbol)
            else:
                target_configs = self.target_configs
            max_target_period = max([c['periods'] for c in target_configs])
            
            future_returns = df['close'].pct_change(max_target_period).shift(-max_target_period).dropna()
            
            if not future_returns.empty:
                aligned_data = features.iloc[:len(future_returns)].copy()
                aligned_returns = future_returns.iloc[:len(aligned_data)]
                
                if not aligned_data.empty and not aligned_returns.empty:
                    for col in aligned_data.columns:
                        try:
                            corr = np.corrcoef(aligned_data[col], aligned_returns)[0,1]
                            if abs(corr) > 0.5:
                                leakage_report['issues'].append(f"High correlation with future returns in {col}: {corr:.3f}")
                                leakage_report['suggestions'].append(f"Feature {col} may be leaking future information")
                        except Exception as e:
                            self.logger.warning(f"Could not calculate leakage correlation for {col}: {e}")

        if not target.empty:
            target_counts = target.value_counts()
            if 0 in target_counts and target_counts[0] / len(target) > 0.8:
                leakage_report['issues'].append("High proportion of 'Hold' targets (>80%)")
                leakage_report['suggestions'].append("Review target creation thresholds and volatility calculations")
        
        return leakage_report

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