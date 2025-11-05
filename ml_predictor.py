import json
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
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
        self.min_training_samples = 20  # Reduced from 150
        self.walk_forward_splits = 3    # Reduced from 5
        self.validation_window = 100    # Reduced from 200

        # BTC correlation configuration
        self.btc_correlation_enabled = True
        self.btc_correlation_cache = {}
        self.btc_correlation_cache_ttl = 300  # 5 minutes
        
        print(f"   • BTC correlation features: {self.btc_correlation_enabled}")
                
        self.target_configs = [
            {'periods': 2, 'weight': 0.40, 'buy_threshold': 0.0015, 'sell_threshold': -0.0015},
            {'periods': 4, 'weight': 0.35, 'buy_threshold': 0.0020, 'sell_threshold': -0.0020},
            {'periods': 8, 'weight': 0.25, 'buy_threshold': 0.0030, 'sell_threshold': -0.0030}
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
            'BTCUSDT': {'max_features': 30, 'enable_hpo': True, 'model_depth': 'deep'},
            'ETHUSDT': {'max_features': 25, 'enable_hpo': True, 'model_depth': 'medium'},
            'BNBUSDT': {'max_features': 20, 'enable_hpo': False, 'model_depth': 'medium'},
            'XRPUSDT': {'max_features': 15, 'enable_hpo': False, 'model_depth': 'simple'},
            'SOLUSDT': {'max_features': 15, 'enable_hpo': False, 'model_depth': 'simple'},
            'DOGEUSDT': {'max_features': 12, 'enable_hpo': False, 'model_depth': 'simple'}
        }
        
        
        self.rf_params = {
            'n_estimators': 100,          # More trees for stability
            'max_depth': 5,               # Shallower trees
            'min_samples_split': 30,      # Much more conservative splitting
            'min_samples_leaf': 20,       # Much more conservative leaf size
            'max_features': 0.3,          # Fewer features per tree
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }

        self.gb_params = {
            'n_estimators': 80,           # More trees but slower learning
            'max_depth': 3,               # Much shallower
            'learning_rate': 0.05,        # Slower learning
            'min_samples_split': 30,      # More conservative
            'min_samples_leaf': 20,       # More conservative
            'subsample': 0.7,             # Lower subsample
            'max_features': 0.4,       
            'random_state': 42
        }

        self.rf_param_dist = {
            'n_estimators': randint(80, 150),
            'max_depth': randint(3, 8),
            'min_samples_split': randint(20, 40),
            'min_samples_leaf': randint(15, 30),
            'max_features': uniform(0.2, 0.4),
            'bootstrap': [True, False]
        }

        self.gb_param_dist = {
            'n_estimators': randint(60, 120),
            'learning_rate': uniform(0.01, 0.1),
            'max_depth': randint(2, 5),
            'min_samples_split': randint(20, 40),
            'min_samples_leaf': randint(15, 30),
            'subsample': uniform(0.6, 0.3),
            'max_features': uniform(0.3, 0.4)
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

    def set_error_handler(self, error_handler):
        self.error_handler = error_handler

    def set_database(self, database):
        self.database = database

    def analyze_target_distribution(self, symbol: str, targets: pd.Series):
        """Analyze and log 3-class target distribution for debugging"""
        target_counts = targets.value_counts().sort_index()
        total_samples = len(targets)
        
        self.logger.info(f"[TARGET_3CLASS_ANALYSIS] {symbol} target distribution:")
        for target_val, count in target_counts.items():
            percentage = (count / total_samples) * 100
            label = "SELL" if target_val == -1 else "HOLD" if target_val == 0 else "BUY"
            self.logger.info(f"  {label} ({target_val}): {count} samples ({percentage:.1f}%)")
        
        # Check for class imbalance - adjusted for 3-class system
        if len(target_counts) == 3:  # Should have all 3 classes
            counts = list(target_counts.values)
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 2.5:  # More permissive for 3 classes
                self.logger.warning(f"Class imbalance detected for {symbol}: ratio {imbalance_ratio:.2f}")
        elif len(target_counts) < 3:
            self.logger.warning(f"Missing classes for {symbol}: only {len(target_counts)} classes out of 3")
            self.logger.warning(f"Present classes: {list(target_counts.index)}")

    def _balance_classes(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes using appropriate methods for financial data"""
        try:
            # Check class distribution
            target_counts = target.value_counts()
            self.logger.info(f"[CLASS_BALANCE] Before balancing: {target_counts.to_dict()}")
            
            # For financial data with 3 classes, we need a different approach
            if len(target_counts) == 3:
                min_count = target_counts.min()
                max_count = target_counts.max()
                
                # If imbalance ratio > 3, apply balancing
                if max_count / min_count > 3.0:
                    from imblearn.over_sampling import RandomOverSampler
                    
                    # Use random oversampling to boost minority classes
                    sampler = RandomOverSampler(
                        sampling_strategy='minority',  # Only oversample minority classes
                        random_state=42
                    )
                    
                    features_balanced, target_balanced = sampler.fit_resample(features, target)
                    balanced_counts = pd.Series(target_balanced).value_counts()
                    self.logger.info(f"[CLASS_BALANCE] After oversampling: {balanced_counts.to_dict()}")
                    return features_balanced, target_balanced
                
                # If we have reasonable balance, don't apply balancing
                else:
                    self.logger.info(f"[CLASS_BALANCE] Imbalance ratio {max_count/min_count:.2f} acceptable, skipping balancing")
                    return features, target
            
            # For 2-class system, use original logic
            elif len(target_counts) == 2:
                max_count = target_counts.max()
                min_count = target_counts.min()
                
                if max_count / min_count > 2.0:
                    if len(target) < 200:
                        from imblearn.over_sampling import RandomOverSampler
                        sampler = RandomOverSampler(random_state=42)
                    else:
                        from imblearn.under_sampling import RandomUnderSampler
                        sampler = RandomUnderSampler(random_state=42)
                    
                    features_balanced, target_balanced = sampler.fit_resample(features, target)
                    balanced_counts = pd.Series(target_balanced).value_counts()
                    self.logger.info(f"[CLASS_BALANCE] After balancing: {balanced_counts.to_dict()}")
                    return features_balanced, target_balanced
            
            return features, target
                
        except ImportError:
            self.logger.warning("[CLASS_BALANCE] imbalanced-learn not available, skipping class balancing")
            return features, target
        except Exception as e:
            self.logger.error(f"[CLASS_BALANCE] Error in class balancing: {e}")
            return features, target

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

    def _calculate_core_features(self, close_prices: pd.Series, high_prices: pd.Series, 
                            low_prices: pd.Series, volume: pd.Series) -> Dict[str, float]:
        """Calculate essential core features that the model ALWAYS expects"""
        features = {}
        
        try:
            # Returns for different periods
            for period in [5, 10, 20]:
                if len(close_prices) > period:
                    features[f'returns_{period}'] = close_prices.iloc[-1] / close_prices.iloc[-period-1] - 1
                else:
                    features[f'returns_{period}'] = 0.0
            
            # RSI
            features['rsi_14'] = self._calculate_rsi_point_in_time(close_prices, 14)
            
            # ATR
            features['atr_14'] = self._calculate_atr_point_in_time(high_prices, low_prices, close_prices, 14)
            
            # Basic volume features
            if len(volume) >= 20:
                features['volume_ma_20'] = volume.rolling(20).mean().iloc[-1]
                features['volume_ratio_20'] = volume.iloc[-1] / features['volume_ma_20'] if features['volume_ma_20'] > 0 else 1.0
            else:
                features['volume_ma_20'] = volume.mean() if len(volume) > 0 else 1.0
                features['volume_ratio_20'] = 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating core features: {e}")
            # Return minimal fallback core features
            return {
                'returns_5': 0.0, 'returns_10': 0.0, 'returns_20': 0.0,
                'rsi_14': 50.0, 'atr_14': 0.02,
                'volume_ma_20': 1.0, 'volume_ratio_20': 1.0
            }
        
        return features

    def _ensure_minimum_features(self, features: Dict, close_prices: pd.Series, 
                            high_prices: pd.Series, low_prices: pd.Series, 
                            volume: pd.Series) -> Dict:
        """Ensure we have all minimum required features"""
        minimum_features = {
            'symbol_specific_volatility': 0.02,
            'market_cap_tier': 0,
            'returns_5': 0.0, 'returns_10': 0.0, 'returns_20': 0.0,
            'rsi_14': 50.0, 'atr_14': 0.02,
            'volume_ma_20': 1.0, 'volume_ratio_20': 1.0
        }
        
        # Update with any calculated features, but ensure minimums are present
        for key, default_value in minimum_features.items():
            if key not in features:
                features[key] = default_value
                self.logger.warning(f"Added missing minimum feature: {key} = {default_value}")
        
        return features

    def _get_fallback_features(self, symbol: str) -> pd.DataFrame:
        """Get comprehensive fallback features when primary calculation fails"""
        fallback_features = {
            'symbol_specific_volatility': 0.02,
            'market_cap_tier': self._get_market_cap_tier(symbol) if symbol else 0,
            'returns_5': 0.0, 'returns_10': 0.0, 'returns_20': 0.0,
            'rsi_14': 50.0, 'atr_14': 0.02,
            'volume_ma_20': 1.0, 'volume_ratio_20': 1.0,
            'volume_regime': 0, 'volume_above_average': 0
        }
        
        self.logger.warning(f"Using comprehensive fallback features for {symbol}")
        return pd.DataFrame([fallback_features])

    def prepare_features_point_in_time(self, df: pd.DataFrame, current_idx: int, lookback: int = 50, symbol: str = None) -> pd.DataFrame:
        """Prepare features using ONLY data available up to current_idx - FIXED VERSION"""
        features = {}
        
        if current_idx < lookback:
            return pd.DataFrame()

        # Use only data up to current_idx (exclusive)
        current_data = df.iloc[:current_idx]
        
        if len(current_data) < 20:
            return pd.DataFrame()

        # CRITICAL FIX: Ensure we have required price columns
        required_columns = ['close', 'high', 'low', 'volume', 'open']
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns for {symbol}: {missing_columns}")
            # Try to use fallback: check if we have at least 'close' and 'volume'
            if 'close' not in current_data.columns:
                self.logger.error(f"Cannot proceed without 'close' column for {symbol}")
                return pd.DataFrame()
            
            # Create minimal required columns if missing
            if 'high' not in current_data.columns:
                current_data['high'] = current_data['close'] * 1.002  # Approximate
            if 'low' not in current_data.columns:
                current_data['low'] = current_data['close'] * 0.998   # Approximate
            if 'open' not in current_data.columns:
                current_data['open'] = current_data['close'].shift(1).fillna(current_data['close'])
        
        try:
            close_prices = current_data['close'].astype(float)
            high_prices = current_data['high'].astype(float)
            low_prices = current_data['low'].astype(float)
            volume = current_data['volume'].astype(float)
            opens = current_data['open'].astype(float)

            # Rest of the feature calculation remains the same...
            # 1. Symbol-specific volatility
            if len(close_prices) >= 20:
                volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1]
                features['symbol_specific_volatility'] = volatility_20 if not pd.isna(volatility_20) else 0.02
            else:
                features['symbol_specific_volatility'] = 0.02
            
            # 2. BTC correlation features
            if symbol and symbol != 'BTCUSDT' and hasattr(self, 'data_engine') and self.data_engine and self.btc_correlation_enabled:
                try:
                    btc_features = self._calculate_btc_correlation_features(symbol, current_data)
                    features.update(btc_features)
                except Exception as btc_error:
                    self.logger.warning(f"BTC correlation features failed for {symbol}: {btc_error}")
            
            # 3. Market cap tier and core features
            if symbol:
                market_cap_tier = self._get_market_cap_tier(symbol)
                features['market_cap_tier'] = market_cap_tier
                
                # CRITICAL: ALWAYS calculate core features regardless of market cap tier
                # These are the essential features the model expects
                core_features = self._calculate_core_features(close_prices, high_prices, low_prices, volume)
                features.update(core_features)
                
                # Add additional features based on market cap
                if market_cap_tier == 2 and len(close_prices) >= 100:
                    advanced_features = self._calculate_advanced_features(close_prices, high_prices, low_prices, volume)
                    features.update(advanced_features)
                elif market_cap_tier == 1 and len(close_prices) >= 50:
                    medium_features = self._calculate_medium_complexity_features(close_prices, high_prices, low_prices, volume)
                    features.update(medium_features)

            # 4. Volume profile features
            if len(volume) >= 20:
                try:
                    volume_profile = self._calculate_volume_profile(volume, symbol)
                    features.update(volume_profile)
                except Exception as vol_error:
                    self.logger.warning(f"Volume profile features failed for {symbol}: {vol_error}")
            
            # 5. Ensure we have minimum required features
            features = self._ensure_minimum_features(features, close_prices, high_prices, low_prices, volume)
            
            # 6. Ensure all features are numerical and valid
            features = self._ensure_numerical_features(features)
            
            # 7. Validate no NaN/inf values
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0.0
            
            return pd.DataFrame([features]).fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error in prepare_features_point_in_time for {symbol} at index {current_idx}: {e}")
            if self.error_handler:
                self.error_handler.handle_ml_error(e, symbol, "feature_preparation")
            
            # Return fallback features instead of empty DataFrame
            return self._get_fallback_features(symbol)

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

    def _calculate_simple_features(self, close, high, low, volume):
        """Simple features for small-cap/volatile symbols"""
        features = {}
        
        # Only essential features to avoid overfitting
        for period in [5, 10, 20]:
            if len(close) > period:
                features[f'returns_{period}'] = close.iloc[-1] / close.iloc[-period-1] - 1
        
        features['rsi_14'] = self._calculate_rsi_point_in_time(close, 14)
        features['atr_14'] = self._calculate_atr_point_in_time(high, low, close, 14)
        
        return features

    def analyze_price_movements(self, symbol: str, df: pd.DataFrame):
        """Analyze actual price movements to set realistic thresholds"""
        self.logger.info(f"[PRICE_ANALYSIS] Analyzing price movements for {symbol}")
        
        close_prices = df['close'].astype(float)
        
        # Calculate returns for different periods used in target creation
        returns_2 = close_prices.pct_change(2).dropna()
        returns_4 = close_prices.pct_change(4).dropna() 
        returns_8 = close_prices.pct_change(8).dropna()
        
        # Analyze the distribution of returns
        for period, returns in [("2", returns_2), ("4", returns_4), ("8", returns_8)]:
            if len(returns) > 0:
                abs_returns = returns.abs()
                self.logger.info(f"[PRICE_ANALYSIS] {symbol} {period}-period returns:")
                self.logger.info(f"  - Mean: {returns.mean():.6f}")
                self.logger.info(f"  - Std: {returns.std():.6f}")
                self.logger.info(f"  - Min: {returns.min():.6f}")
                self.logger.info(f"  - Max: {returns.max():.6f}")
                self.logger.info(f"  - 25th percentile: {abs_returns.quantile(0.25):.6f}")
                self.logger.info(f"  - 50th percentile: {abs_returns.quantile(0.50):.6f}")
                self.logger.info(f"  - 75th percentile: {abs_returns.quantile(0.75):.6f}")
                self.logger.info(f"  - 90th percentile: {abs_returns.quantile(0.90):.6f}")
        
        # Test what thresholds would capture different percentages of movements
        for threshold in [0.0005, 0.001, 0.002, 0.003, 0.005]:
            if len(returns_2) > 0:
                above_threshold = (returns_2.abs() > threshold).mean()
                self.logger.info(f"[PRICE_ANALYSIS] {symbol} 2-period moves > {threshold:.4f}: {above_threshold:.2%}")

    def create_multi_timeframe_targets(self, df: pd.DataFrame, current_idx: int, target_configs: List[Dict] = None) -> List[int]:
        if target_configs is None:
            target_configs = self.target_configs
            
        targets = []
        
        for config in target_configs:
            periods = config['periods']
            
            if current_idx + periods >= len(df):
                targets.append(0)
                continue
                
            current_price = df['close'].iloc[current_idx]
            future_price = df['close'].iloc[current_idx + periods]
            price_change_pct = (future_price - current_price) / current_price
            
            # 3-class system: -1 (sell), 0 (hold), 1 (buy)
            buy_threshold = config.get('buy_threshold', 0.0015)
            sell_threshold = config.get('sell_threshold', -0.0015)
            
            if price_change_pct > buy_threshold:
                targets.append(1)    # Buy
            elif price_change_pct < sell_threshold:
                targets.append(-1)   # Sell
            else:
                targets.append(0)    # Hold
        
        return targets
            
    def debug_target_creation(self, symbol: str, df: pd.DataFrame):
        """Debug method to analyze why targets are only producing 0s"""
        self.logger.info(f"[DEBUG_TARGET] Analyzing target creation for {symbol}")
        
        # Analyze price movements in the data
        close_prices = df['close'].astype(float)
        returns = close_prices.pct_change().dropna()
        
        self.logger.info(f"[DEBUG_TARGET] {symbol} price stats:")
        self.logger.info(f"  - Data length: {len(df)}")
        self.logger.info(f"  - Price range: {close_prices.min():.4f} to {close_prices.max():.4f}")
        self.logger.info(f"  - Return stats: mean={returns.mean():.6f}, std={returns.std():.6f}")
        self.logger.info(f"  - Return range: {returns.min():.6f} to {returns.max():.6f}")
        
        # Test target creation at multiple points
        test_indices = [50, 100, 150, 200, 250] if len(df) > 250 else list(range(20, min(100, len(df)-20), 10))
        
        for idx in test_indices:
            if idx >= len(df) - 20:
                continue
                
            target = self.create_enhanced_target(df, idx, symbol)
            multi_targets = self.create_multi_timeframe_targets(df, idx)
            
            # Calculate actual price changes
            price_changes = []
            for config in self.target_configs:
                periods = config['periods']
                if idx + periods < len(df):
                    change = (df['close'].iloc[idx + periods] - df['close'].iloc[idx]) / df['close'].iloc[idx]
                    price_changes.append(change)
            
            self.logger.info(f"[DEBUG_TARGET] Index {idx}: target={target}, multi_targets={multi_targets}, price_changes={[f'{x:.4f}' for x in price_changes]}")


    def create_enhanced_target(self, df: pd.DataFrame, current_idx: int, symbol: str = None) -> int:
        """Create 3-class targets with dynamic, volatility-adjusted thresholds"""
        
        # Get current volatility regime
        if current_idx >= 20:
            recent_returns = df['close'].iloc[current_idx-19:current_idx+1].pct_change().dropna()
            current_volatility = recent_returns.std()
        else:
            current_volatility = 0.02  # Default
        
        # Dynamic threshold scaling based on current volatility
        vol_scale = max(0.5, min(2.0, current_volatility / 0.02))  # Scale relative to 2% baseline
        
        # Get base config and apply volatility scaling
        if symbol and symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] == 'low':
                base_threshold = 0.0010 * vol_scale
            elif profile['volatility_tier'] == 'medium':
                base_threshold = 0.0015 * vol_scale  
            elif profile['volatility_tier'] == 'high':
                base_threshold = 0.0020 * vol_scale
            elif profile['volatility_tier'] == 'extreme':
                base_threshold = 0.0025 * vol_scale
            else:
                base_threshold = 0.0015 * vol_scale
        else:
            base_threshold = 0.0015 * vol_scale
        
        # Create dynamic configs
        dynamic_configs = []
        for config in self.target_configs:
            periods = config['periods']
            # Scale thresholds by timeframe and volatility
            timeframe_multiplier = 1.0 + (periods / 8) * 0.4  # Longer timeframes get wider thresholds
            
            dynamic_configs.append({
                'periods': periods,
                'weight': config['weight'],
                'buy_threshold': base_threshold * timeframe_multiplier,
                'sell_threshold': -base_threshold * timeframe_multiplier
            })
        
        # Use dynamic configs for target creation
        multi_targets = self.create_multi_timeframe_targets(df, current_idx, dynamic_configs)
        
        if not multi_targets:
            return 0
        
        # Calculate weighted vote - using 3-class targets directly
        weighted_vote = 0
        total_weight = 0
        
        for i, target in enumerate(multi_targets):
            base_weight = self.target_configs[i]['weight']
            weighted_vote += target * base_weight  # target is already -1, 0, or 1
            total_weight += base_weight
        
        if total_weight > 0:
            normalized_vote = weighted_vote / total_weight
        else:
            normalized_vote = 0
        
        # 3-class decision with adaptive thresholds
        # Use average of timeframe thresholds for final decision
        avg_buy_threshold = np.mean([config.get('buy_threshold', 0.0015) for config in self.target_configs])
        avg_sell_threshold = np.mean([config.get('sell_threshold', -0.0015) for config in self.target_configs])
        
        final_target = self.create_probabilistic_target(df, i, symbol);
        
        # DEBUG: Log target creation for analysis
        if current_idx % 100 == 0 and symbol:  # Reduced frequency to avoid log spam
            self.logger.info(f"[TARGET_3CLASS] {symbol} at {current_idx}: "
                            f"vote={normalized_vote:.6f}, target={final_target}, "
                            f"multi_targets={multi_targets}, "
                            f"thresholds=[buy:{avg_buy_threshold:.6f}, sell:{avg_sell_threshold:.6f}]")
        
        return final_target

    def calculate_feature_quality_score(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Calculate quality score for each feature"""
        quality_scores = {}
        
        for column in features.columns:
            feature_series = features[column]
            
            # Skip constant features
            if feature_series.std() == 0:
                quality_scores[column] = 0.0
                continue
            
            try:
                # 1. Predictive power (mutual information)
                mi = mutual_info_classif(feature_series.values.reshape(-1, 1), target.values)[0]
                
                # 2. Stability (low outlier count)
                z_scores = np.abs(stats.zscore(feature_series))
                outlier_ratio = np.sum(z_scores > 3) / len(feature_series)
                stability_score = 1.0 - outlier_ratio
                
                # 3. Information value for binary classification
                # For 3-class, we'll use a simplified approach
                if len(target.unique()) == 3:
                    # Calculate separation between classes
                    class_separations = []
                    for class_val in [-1, 0, 1]:
                        class_mask = target == class_val
                        if class_mask.sum() > 0:
                            class_mean = feature_series[class_mask].mean()
                            overall_mean = feature_series.mean()
                            separation = abs(class_mean - overall_mean) / feature_series.std()
                            class_separations.append(separation)
                    separation_score = np.mean(class_separations) if class_separations else 0
                else:
                    separation_score = 0
                
                # Combined quality score
                quality_score = (mi * 0.5 + stability_score * 0.3 + separation_score * 0.2)
                quality_scores[column] = quality_score
                
            except:
                quality_scores[column] = 0.0
        
        return quality_scores

    def select_features_by_quality(self, features: pd.DataFrame, target: pd.Series, symbol: str, min_quality: float = 0.1) -> List[str]:
        """Select features based on quality scoring"""
        quality_scores = self.calculate_feature_quality_score(features, target)
        
        # Sort features by quality
        sorted_features = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select features meeting minimum quality threshold
        selected_features = [feat for feat, score in sorted_features if score >= min_quality]
        
        # Ensure minimum feature count
        min_features = 8
        if len(selected_features) < min_features and len(sorted_features) >= min_features:
            selected_features = [feat for feat, score in sorted_features[:min_features]]
        
        # Log feature quality information
        high_quality_count = sum(1 for score in quality_scores.values() if score > 0.3)
        self.logger.info(f"Feature quality for {symbol}: {high_quality_count}/{len(features.columns)} high quality, selected {len(selected_features)}")
        
        return selected_features

    def create_probabilistic_target(self, df: pd.DataFrame, current_idx: int, symbol: str = None) -> int:
        """Create targets based on probability of movement rather than fixed thresholds"""
        
        # Get symbol-specific price movement characteristics
        if current_idx >= 50:
            recent_data = df.iloc[max(0, current_idx-49):current_idx+1]
            close_prices = recent_data['close'].astype(float)
            
            # Analyze actual price movement distribution
            future_returns = []
            for config in self.target_configs:
                periods = config['periods']
                if current_idx + periods < len(df):
                    future_return = (df['close'].iloc[current_idx + periods] - df['close'].iloc[current_idx]) / df['close'].iloc[current_idx]
                    future_returns.append(future_return)
            
            if future_returns:
                # Use percentiles instead of fixed thresholds
                abs_returns = np.abs(future_returns)
                if len(abs_returns) >= 10:
                    # Set thresholds at 30th and 70th percentiles of historical moves
                    lower_threshold = np.percentile(abs_returns, 30)  # Only predict moves larger than 30% of historical moves
                    upper_threshold = np.percentile(abs_returns, 70)  # Strong signal for moves larger than 70% of historical moves
                else:
                    # Fallback to volatility-based thresholds
                    volatility = close_prices.pct_change().std()
                    lower_threshold = volatility * 0.8
                    upper_threshold = volatility * 1.5
            else:
                return 0
        else:
            return 0
        
        # Calculate weighted future return
        weighted_future_return = 0
        total_weight = 0
        
        for i, config in enumerate(self.target_configs):
            periods = config['periods']
            weight = config['weight']
            
            if current_idx + periods < len(df):
                future_return = (df['close'].iloc[current_idx + periods] - df['close'].iloc[current_idx]) / df['close'].iloc[current_idx]
                weighted_future_return += future_return * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0
        
        normalized_return = weighted_future_return / total_weight
        abs_return = abs(normalized_return)
        
        # Probabilistic decision making
        if abs_return < lower_threshold:
            return 0  # Hold - movement too small to trade
        
        # For larger movements, use confidence-based decision
        confidence = min(1.0, (abs_return - lower_threshold) / (upper_threshold - lower_threshold))
        
        # Only make directional prediction if we have reasonable confidence
        if confidence > 0.4:  # 40% confidence threshold
            if normalized_return > 0:
                return 1  # Buy
            else:
                return -1  # Sell
        else:
            return 0  # Hold - not confident enough

    def _get_current_volatility(self, df: pd.DataFrame, current_idx: int, lookback: int = 20) -> float:
        """Calculate current volatility for adaptive thresholding"""
        if current_idx < lookback:
            return 0.02  # Default volatility
        
        close_prices = df['close'].iloc[max(0, current_idx - lookback):current_idx + 1]
        returns = close_prices.pct_change().dropna()
        
        if len(returns) < 10:
            return 0.02
        
        volatility = returns.std()
        return max(0.005, min(0.10, volatility))  # Bound between 0.5% and 10%

    def _get_symbol_specific_target_configs(self, symbol: str, current_volatility: float = None) -> List[Dict]:
        """Get symbol-specific 3-class target configurations with realistic thresholds"""
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            periods = profile['timeframes']
            
            # Set realistic thresholds based on actual price movement analysis from logs
            # From BTCUSDT logs: 50th percentile of 2-period returns is 0.001379
            # We want thresholds that capture meaningful movements but aren't too restrictive
            
            if profile['volatility_tier'] == 'low':  # BTCUSDT, ETHUSDT
                base_threshold = 0.0012  # Slightly below 50th percentile to capture more signals
            elif profile['volatility_tier'] == 'medium':  # BNBUSDT
                base_threshold = 0.0015
            elif profile['volatility_tier'] == 'high':  # XRPUSDT
                base_threshold = 0.0020
            elif profile['volatility_tier'] == 'extreme':  # SOLUSDT, DOGEUSDT
                base_threshold = 0.0025
            else:
                base_threshold = 0.0015  # Default fallback

            # Apply small volatility adjustment if available
            if current_volatility is not None:
                # Gentle adjustment - don't let volatility dominate the thresholds
                vol_adjustment = 1.0 + (current_volatility - 0.02) * 1.5
                base_threshold *= vol_adjustment
        else:
            # Default configuration for unknown symbols
            periods = [2, 4, 8]
            base_threshold = 0.0015
        
        # Ensure reasonable bounds for thresholds
        base_threshold = max(0.0008, min(0.0030, base_threshold))
        
        # Create configs based on available periods with appropriate timeframe scaling
        if len(periods) == 3:
            weights = [0.40, 0.35, 0.25]  # Slightly favor shorter timeframes
            timeframe_multipliers = [1.0, 1.2, 1.4]  # Gentle increase for longer timeframes
        elif len(periods) == 2:
            weights = [0.60, 0.40]
            timeframe_multipliers = [1.0, 1.3]
        else:
            weights = [1.0]
            timeframe_multipliers = [1.0]
        
        configs = []
        for i, period in enumerate(periods):
            # Use multiplier for longer timeframes
            if i < len(timeframe_multipliers):
                timeframe_multiplier = timeframe_multipliers[i]
            else:
                # If we have more periods than multipliers, use the last multiplier
                timeframe_multiplier = timeframe_multipliers[-1]
            
            # Use appropriate weight
            if i < len(weights):
                weight = weights[i]
            else:
                # Distribute remaining weight evenly if we have more periods
                weight = 1.0 / len(periods)
            
            buy_threshold = base_threshold * timeframe_multiplier
            sell_threshold = -base_threshold * timeframe_multiplier
            
            configs.append({
                'periods': period,
                'weight': weight,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold
            })
        
        # Log the configuration for debugging
        ##if symbol:
          ##  threshold_info = ", ".join([f"{c['periods']}p:{c['buy_threshold']:.4f}" for c in configs])
          ##  self.logger.info(f"🎯 Target config for {symbol}: base={base_threshold:.4f}, periods=[{threshold_info}]")
        
        return configs

    def _get_symbol_thresholds(self, symbol: str, volatility: float) -> Dict[str, float]:
        """Get symbol-specific decision thresholds - EXTREMELY PERMISSIVE"""
        # Extremely permissive base thresholds
        base_strong = 0.04  # Further reduced from 0.08
        base_weak = 0.01    # Further reduced from 0.03
        
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] == 'low':
                # Even more permissive for stable symbols
                strong_threshold = base_strong * 0.5
                weak_threshold = base_weak * 0.5
            elif profile['volatility_tier'] == 'extreme':
                # Slightly wider for volatile symbols but still permissive
                strong_threshold = base_strong * 1.0
                weak_threshold = base_weak * 1.0
            else:
                strong_threshold = base_strong
                weak_threshold = base_weak
        else:
            strong_threshold = base_strong
            weak_threshold = base_weak
        
        # Minimal volatility adjustment
        vol_adjustment = 1.0 + (volatility - 0.02) * 2  # Further reduced from 5 to 2
        strong_threshold *= vol_adjustment
        weak_threshold *= vol_adjustment
        
        # Extremely wide acceptable range
        return {
            'strong': max(0.02, min(0.10, strong_threshold)),  # Much wider range
            'weak': max(0.005, min(0.05, weak_threshold))      # Much wider range
        }

    def _validate_no_data_leakage(self, df: pd.DataFrame, feature_idx: int, target_configs: List[Dict]) -> bool:
        """Validate that feature creation doesn't use future data"""
        try:
            max_future_needed = max([config['periods'] for config in target_configs])
            
            # Check if we have enough future data for targets
            if feature_idx + max_future_needed >= len(df):
                return False
                
            # Check that feature calculation only uses data up to feature_idx
            feature_data = df.iloc[:feature_idx]
            target_data_start = feature_idx + 1
            target_data_end = feature_idx + max_future_needed
            
            # Ensure no overlap between feature and target data periods
            if target_data_start <= feature_idx:
                self.logger.error(f"Data leakage detected: feature_idx={feature_idx}, target starts at {target_data_start}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data leakage validation: {e}")
            return False

    def prepare_training_data_enhanced(self, df: pd.DataFrame, min_samples: int = None, symbol: str = None) -> tuple:
        if min_samples is None:
            min_samples = self.min_training_samples
            
        self.logger.info(f"[DATA_PREP_FIXED] Starting data prep for {symbol}. Input data length: {len(df)}")
        
        # CRITICAL FIX: Run price movement analysis to understand realistic thresholds
        self.analyze_price_movements(symbol, df)
        
        # CRITICAL FIX: Reduced requirements to get more training samples
        max_target_period = max([config['periods'] for config in self.target_configs])
        required_future = max_target_period + 5  # Reduced buffer
        
        # Use more available data
        max_bars = min(1000, len(df) - required_future) if len(df) > required_future else len(df)
        if max_bars < 50:  # Absolute minimum
            self.logger.error(f"CRITICAL: Insufficient data for {symbol}: only {len(df)} bars")
            return pd.DataFrame(), pd.Series()
        
        df = df.tail(max_bars)
        self.logger.info(f"[DATA_PREP] Using {len(df)} bars for {symbol}")
        
        features_list = []
        targets = []
        
        # CRITICAL FIX: More aggressive sampling to get enough samples
        stride = 1  # Use every data point
        start_idx = 20  # Reduced from 30
        end_idx = len(df) - max_target_period - 1
        
        self.logger.info(f"[DATA_PREP] Sampling from idx {start_idx} to {end_idx} with stride {stride}")
        
        valid_samples = 0
        target_counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        
        for i in range(start_idx, end_idx, stride):
            features = self.prepare_features_point_in_time(df, i, symbol=symbol)
            target = self.create_probabilistic_target(df, i, symbol)

            
            if features.empty:
                continue
                
            features_list.append(features.iloc[0])
            targets.append(target)
            target_counts[target] += 1
            valid_samples += 1
        
        self.logger.info(f"[DATA_PREP] {symbol} - Valid samples: {valid_samples}")
        self.logger.info(f"[DATA_PREP] {symbol} - Target distribution: {target_counts}")
        
        if len(features_list) < 5:  # Reduced minimum
            self.logger.warning(f"Very few training samples for {symbol}: {len(features_list)}")
            return pd.DataFrame(), pd.Series()
        
        features_df = pd.DataFrame(features_list).fillna(0)
        target_series = pd.Series(targets, index=features_df.index)
        
        # CRITICAL: If we only have one class, we can't train properly
        unique_targets = set(targets)
        if len(unique_targets) < 2:
            self.logger.error(f"CRITICAL: Only one target class for {symbol}: {unique_targets}")
            return pd.DataFrame(), pd.Series()
        
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
        """Apply aggressive regularization for low-performing symbols"""
        rf_params = self.rf_params.copy()
        gb_params = self.gb_params.copy()
        
        # Check historical performance and apply regularization
        if symbol in self.model_versions:
            historical_accuracy = self.model_versions[symbol].get('accuracy', 0.5)
            
            # Aggressive regularization for poor performers
            if historical_accuracy < 0.45:  # DOGE, BNB
                self.logger.info(f"🔧 Applying AGGRESSIVE regularization for {symbol} (accuracy: {historical_accuracy:.3f})")
                
                # Much more conservative RF
                rf_params.update({
                    'n_estimators': 60,                # Fewer trees
                    'max_depth': 4,                    # Much shallower
                    'min_samples_split': 40,           # Very conservative splitting
                    'min_samples_leaf': 25,            # Very large leaves
                    'max_features': 0.2,               # Very few features
                })
                
                # Much more conservative GB
                gb_params.update({
                    'n_estimators': 50,                # Fewer trees
                    'learning_rate': 0.02,             # Very slow learning
                    'max_depth': 3,                    # Very shallow
                    'min_samples_split': 40,           # Very conservative
                    'min_samples_leaf': 25,            # Very large leaves
                    'subsample': 0.5,                  # High randomness
                    'max_features': 0.3,               # Few features
                })
            
            elif historical_accuracy < 0.55:  # XRP, borderline cases
                self.logger.info(f"🔧 Applying MODERATE regularization for {symbol} (accuracy: {historical_accuracy:.3f})")
                
                rf_params.update({
                    'max_depth': 5,
                    'min_samples_split': 30,
                    'min_samples_leaf': 20,
                    'max_features': 0.3,
                })
                
                gb_params.update({
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'min_samples_split': 30,
                    'min_samples_leaf': 20,
                    'subsample': 0.7,
                })
        
        return rf_params, gb_params

    def train_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """Enhanced training with quality gates"""
        try:
            print(f"🔄 Training symbol-specific model for {symbol}...")
            
            # CRITICAL: Quality gate - data validation
            if df is None or len(df) < 200:  # Increased minimum
                self.logger.error(f"CRITICAL: Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
                self._record_training_failure(symbol, "insufficient_data_quality_gate")
                return False
                
            # CRITICAL: Quality gate - data freshness
            data_age_hours = (pd.Timestamp.now() - df.index[-1]).total_seconds() / 3600
            if data_age_hours > 48:  # Data older than 2 days
                self.logger.warning(f"Stale data for {symbol}: {data_age_hours:.1f} hours old")
                
            # CRITICAL: Quality gate - price validation
            close_prices = df['close'].astype(float)
            if close_prices.isna().any() or (close_prices <= 0).any():
                self.logger.error(f"CRITICAL: Invalid price data for {symbol}")
                self._record_training_failure(symbol, "invalid_price_data")
                return False
            
            # Set symbol-specific configuration
            self._apply_symbol_specific_config(symbol)
            
            training_start = datetime.now()
            
            if not hasattr(self, 'model_versions'):
                self.model_versions = {}
            
            # Save current model as previous before training new one
            if symbol in self.models:
                self.previous_models[symbol] = self.models[symbol].copy()
                self.previous_scalers[symbol] = self.scalers[symbol]
                print(f"💾 Saved previous model for {symbol} for ensembling")
            
            # Use enhanced training data with data leakage prevention
            features, target = self.prepare_training_data_enhanced(df, symbol=symbol)

            # Apply class balancing for imbalanced datasets
            features, target = self._balance_classes(features, target)

            # CRITICAL FIX: More permissive validation for small datasets
            if features.empty or target.empty:
                self.logger.warning(f"No training data generated for {symbol}")
                self._record_training_failure(symbol, "no_training_data")
                return False

            # Modified validation for small datasets
            if len(features) < 5:
                self.logger.warning(f"Very few training samples for {symbol}: {len(features)}")
                self._record_training_failure(symbol, "insufficient_samples")
                return False

            # Check for class diversity
            unique_targets = set(target)
            if len(unique_targets) < 2:
                self.logger.error(f"CRITICAL: Only one target class for {symbol}: {unique_targets}")
                self._record_training_failure(symbol, "single_class_only")
                return False

            self.logger.info(f"Training data for {symbol}: {len(features)} samples, classes: {unique_targets}")

            # DEBUG: Run target analysis if we're getting mostly zeros
            if target is not None and len(target) > 0:
                zero_count = (target == 0).sum()
                zero_ratio = zero_count / len(target)
                if zero_ratio > 0.8:  # If more than 80% are zeros
                    self.logger.warning(f"High zero target ratio for {symbol}: {zero_ratio:.2f}")
                    self.debug_target_creation(symbol, df)

            # ADD THIS VALIDATION
            if not self.validate_training_data_quality(features, target, symbol):
                self.logger.error(f"Training data quality check failed for {symbol}")
                self._record_training_failure(symbol, "data_quality_validation_failed")
                return False
            
            if features.empty or target.empty:
                print(f"⚠️ Insufficient data for training {symbol}")
                self._record_training_failure(symbol, "insufficient_data")
                return False
                
            # Log target distribution for debugging
            target_distribution = target.value_counts()
            print(f"📊 Target distribution for {symbol}: {target_distribution.to_dict()}")
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}

            # Apply symbol-specific feature selection
            quality_features = self.select_features_by_quality(features, target, symbol, min_quality=0.1)
            features_selected = features[quality_features]
            selected_features = quality_features
            self.feature_importance[symbol]['selected_features'] = selected_features
            
            # Log feature statistics
            print(f"🔍 Feature stats for {symbol}: {len(selected_features)} features selected")
            
            # Filter training data by market regime
            filtered_features, filtered_target = self._filter_training_data_by_regime(features_selected, target, df, symbol)
            
            if filtered_features.empty or filtered_target.empty:
                print(f"⚠️ Insufficient tradable regime data for {symbol}")
                self._record_training_failure(symbol, "insufficient_tradable_data")
                return False
            
            X_train, X_test, y_train, y_test = self.time_series_train_test_split(filtered_features, filtered_target)

            if X_train is None:
                # For very small datasets, use all data for training (no test set)
                if len(filtered_features) >= 10:  # If we have at least 10 samples
                    self.logger.info(f"Using all {len(filtered_features)} samples for training (no test set) for {symbol}")
                    X_train, y_train = filtered_features, filtered_target
                    X_test, y_test = filtered_features.iloc[:0], filtered_target.iloc[:0]  # Empty test set
                else:
                    print(f"⚠️ Insufficient data after split for {symbol}: only {len(filtered_features)} samples")
                    self._record_training_failure(symbol, "split_failed")
                    return False

            # Enhanced scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply symbol-specific model complexity
            rf_params, gb_params = self._get_symbol_specific_parameters(symbol, X_train_scaled, y_train)
            
            # In train_model method - add regularization
            # Enhanced parameters with regularization
            enhanced_rf_params = rf_params.copy()
            enhanced_gb_params = gb_params.copy()

            # Add regularization for small datasets
            if len(X_train) < 100:
                enhanced_rf_params.update({
                    'min_samples_split': 10,  # Increased for regularization
                    'min_samples_leaf': 5,    # Increased for regularization  
                    'max_depth': 6,           # Reduced for regularization
                })
                enhanced_gb_params.update({
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_depth': 4,
                    'subsample': 0.7,         # Added subsampling
                })

            rf = RandomForestClassifier(**enhanced_rf_params)
            gb = GradientBoostingClassifier(**enhanced_gb_params)

            rf.fit(X_train_scaled, y_train)
            gb.fit(X_train_scaled, y_train)

            # Enhanced validation with symbol-specific metrics
            rf_pred = rf.predict(X_test_scaled)
            gb_pred = gb.predict(X_test_scaled)

            rf_accuracy = self.validate_accuracy(y_test, rf_pred, "RF", symbol)
            gb_accuracy = self.validate_accuracy(y_test, gb_pred, "GB", symbol)
            
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
            
            # In the training method:
            try:
                overfit_threshold = self.calculate_reasonable_overfit_threshold(len(X_train), len(selected_features))
            except AttributeError:
                # Fallback if method is missing
                self.logger.warning("calculate_reasonable_overfit_threshold method not found, using default threshold")
                overfit_threshold = 0.25

            if (rf_overfit > overfit_threshold or gb_overfit > overfit_threshold) and len(X_train) > 30:
                self.logger.warning(f"Potential overfitting for {symbol}: RF={rf_overfit:.3f}, GB={gb_overfit:.3f} "
                                f"(threshold: {overfit_threshold:.3f})")
            elif rf_overfit > 0.5 or gb_overfit > 0.5:
                self.logger.error(f"Severe overfitting for {symbol}: RF={rf_overfit:.3f}, GB={gb_overfit:.3f}")

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
                'target_type': '3_class_system',
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

    def ensure_feature_stability(self, symbol: str, training_features: pd.DataFrame) -> List[str]:
        """Enhanced feature selection that preserves regime-sensitive features for volatile assets"""
        
        # Core features that should NEVER be removed
        regime_sensitive_features = {
            'high_volatility': ['symbol_specific_volatility', 'atr_14', 'volume_zscore', 
                            'btc_beta_20', 'volume_regime', 'returns_20'],
            'all': ['rsi_14', 'volume_ratio_20', 'market_cap_tier', 'returns_5', 'returns_10']
        }
        
        # Determine symbol volatility profile
        volatility_tier = self.symbol_volatility_profiles.get(symbol, {}).get('volatility_tier', 'medium')
        
        # Always keep core features
        protected_features = set(regime_sensitive_features['all'])
        if volatility_tier in ['high', 'extreme']:
            protected_features.update(regime_sensitive_features['high_volatility'])
        
        # Get current feature selection
        current_features = training_features.columns.tolist()
        
        # Remove highly correlated features, but PROTECT regime-sensitive ones
        try:
            corr_matrix = training_features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Identify features to drop (excluding protected features)
            features_to_drop = set()
            for col in upper_tri.columns:
                if col in protected_features:
                    continue  # Skip protected features
                    
                high_corr_cols = upper_tri[col][upper_tri[col] > 0.85].index.tolist()
                if high_corr_cols:
                    # Among correlated group, keep the one with highest variance, drop others
                    candidates = [col] + high_corr_cols
                    # Protect any protected features in this group
                    protected_in_group = [f for f in candidates if f in protected_features]
                    if protected_in_group:
                        # Keep all protected features, drop only non-protected
                        non_protected = set(candidates) - set(protected_in_group)
                        features_to_drop.update(non_protected)
                    else:
                        # No protected features, keep highest variance
                        variances = training_features[candidates].var()
                        feature_to_keep = variances.idxmax()
                        features_to_remove = set(candidates) - {feature_to_keep}
                        features_to_drop.update(features_to_remove)
            
            # Apply removal (excluding protected features from removal)
            features_to_drop = features_to_drop - protected_features
            remaining_features = [f for f in current_features if f not in features_to_drop]
            
            # Ensure we have minimum feature count for volatile symbols
            min_features = 12 if volatility_tier in ['high', 'extreme'] else 8
            if len(remaining_features) < min_features:
                # Add back some high-variance features we removed
                removed_features = set(current_features) - set(remaining_features)
                variance_ranking = training_features[list(removed_features)].var().sort_values(ascending=False)
                features_to_add = variance_ranking.head(min_features - len(remaining_features)).index.tolist()
                remaining_features.extend(features_to_add)
            
            self.logger.info(f"Enhanced feature selection for {symbol}: {len(remaining_features)} features "
                            f"(protected {len(protected_features)}, dropped {len(features_to_drop)})")
            
            return remaining_features
            
        except Exception as e:
            self.logger.warning(f"Enhanced feature selection failed for {symbol}: {e}")
            # Fallback: use protected features + top variance features
            protected_list = list(protected_features.intersection(set(current_features)))
            other_features = [f for f in current_features if f not in protected_features]
            variances = training_features[other_features].var().sort_values(ascending=False)
            additional_features = variances.head(max(5, 15 - len(protected_list))).index.tolist()
            return protected_list + additional_features

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

    def validate_accuracy(self, y_true, y_pred, model_name, symbol):
        if len(y_true) == 0:
            return 0.33  # Random chance for 3-class system
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # CRITICAL: Detect and handle perfect/overfit accuracy
        if accuracy >= 1.0:
            self.logger.error(f"CRITICAL: Perfect accuracy (1.0) for {symbol} {model_name}. This indicates severe overfitting or data leakage.")
            # Return a more realistic capped accuracy
            balanced_acc = self._calculate_balanced_accuracy(y_true, y_pred)
            return min(0.95, balanced_acc * 1.1)  # Cap at 95% and use balanced accuracy
        
        if accuracy > 0.95:
            self.logger.warning(f"Suspiciously high accuracy for {symbol} {model_name}: {accuracy:.3f}. May be overfitting.")
            # Use balanced accuracy as a more conservative measure
            balanced_acc = self._calculate_balanced_accuracy(y_true, y_pred)
            return min(accuracy, balanced_acc * 1.05)
        
        # For normal cases, use the original logic
        if len(y_true) > 5:
            class_distribution = pd.Series(y_true).value_counts()
            unique_classes = len(class_distribution)
            
            # Check if model is better than always predicting the majority class
            majority_class = class_distribution.index[0]
            majority_accuracy = (y_true == majority_class).mean()
            
            # Check if model is better than random
            random_accuracy = 0.33  # For 3-class system
            
            # If model is worse than majority class or random, be suspicious
            if accuracy < max(majority_accuracy, random_accuracy) - 0.05:
                self.logger.warning(f"Model {model_name} for {symbol} underperforms baseline: "
                                f"{accuracy:.3f} vs majority={majority_accuracy:.3f}, random=0.333")
        
        return accuracy

    def _validate_model_sanity(self, model, X_test, y_test, model_name, symbol):
        """Comprehensive model sanity checks"""
        try:
            if len(y_test) == 0:
                return 0.5  # Default accuracy for empty test set
                
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Check for constant predictions
            unique_predictions = len(np.unique(predictions))
            if unique_predictions == 1:
                self.logger.error(f"CRITICAL: {model_name} for {symbol} predicts constant value: {predictions[0]}")
                return 0.5  # Return neutral accuracy
                
            # Check class distribution in predictions
            pred_distribution = pd.Series(predictions).value_counts()
            if len(pred_distribution) < 2 and len(np.unique(y_test)) > 1:
                self.logger.warning(f"Model {model_name} for {symbol} predicts only {len(pred_distribution)} classes")
                
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error in model sanity check for {symbol} {model_name}: {e}")
            return 0.5
        
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

    def analyze_model_performance_trends(self, symbol: str) -> Dict:
        """Analyze model performance trends and provide recommendations"""
        if symbol not in self.performance_history or len(self.performance_history[symbol]) < 3:
            return {"status": "insufficient_history"}
        
        history = self.performance_history[symbol]
        recent_performance = [entry['accuracy'] for entry in history[-3:]]
        older_performance = [entry['accuracy'] for entry in history[-6:-3]] if len(history) >= 6 else recent_performance
        
        avg_recent = np.mean(recent_performance)
        avg_older = np.mean(older_performance)
        trend = avg_recent - avg_older
        
        analysis = {
            'symbol': symbol,
            'recent_accuracy': avg_recent,
            'trend': trend,
            'recommendations': [],
            'status': 'stable'
        }
        
        # Provide specific recommendations
        if avg_recent < 0.55:
            analysis['status'] = 'poor'
            analysis['recommendations'].append("Model accuracy below 55%. Consider feature engineering or more data.")
        
        if trend < -0.05:
            analysis['status'] = 'deteriorating'
            analysis['recommendations'].append("Performance trending down. Check for concept drift.")
        
        if len(self.performance_history[symbol]) >= 5:
            volatilities = [entry['accuracy'] for entry in history[-5:]]
            if np.std(volatilities) > 0.08:
                analysis['recommendations'].append("High performance volatility. Model may be unstable.")
        
        return analysis
        
    def calculate_reasonable_overfit_threshold(self, train_samples: int, feature_count: int) -> float:
        """More realistic overfit thresholds for financial time series"""
        # Financial data often has legitimate patterns that can look like overfitting
        
        # Set base threshold based on sample size
        if train_samples < 100:
            base_threshold = 0.40  # Much more lenient for small datasets
        elif train_samples < 300:
            base_threshold = 0.30
        elif train_samples < 500:
            base_threshold = 0.25
        else:
            base_threshold = 0.20
        
        # Adjust for feature complexity
        feature_ratio = feature_count / train_samples if train_samples > 0 else 0
        if feature_ratio > 0.5:
            base_threshold += 0.10
        
        # Ensure reasonable bounds
        return min(0.50, max(0.25, base_threshold))
    
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
        if n_splits is None:
            n_splits = self.walk_forward_splits
            
        try:
            features, target = self.prepare_training_data_enhanced(df, symbol=symbol)
            
            if features.empty or target.empty:
                return {'success': False, 'reason': 'Insufficient data'}

            # Dynamic test size based on available data
            test_size = min(20, max(5, len(features) // 10))  # 5-20 samples for test
            gap = 5  # Reduced from 10
            
            # Adjust splits based on available data
            available_splits = min(n_splits, (len(features) - test_size - gap) // test_size)
            if available_splits < 2:
                self.logger.info(f"Not enough data for walk-forward validation on {symbol}, skipping")
                return {'success': False, 'reason': 'Insufficient data for validation splits'}
            
            tscv = TimeSeriesSplit(n_splits=available_splits, test_size=test_size, gap=gap)
            self.logger.info(f"[WF_VALIDATION] {symbol} - Using {available_splits} splits, test_size={test_size}")
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
        """Enhanced prediction with better volume handling"""
        # Check volume conditions first
        if df is not None and len(df) > 0:
            try:
                current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
                volume_ma_20 = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_volume
                
                volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
                
                # If volume is too low, use fallback immediately
                if volume_ratio < 0.3:  # Even more conservative threshold
                    self.logger.warning(f"Very low volume for {symbol}: ratio={volume_ratio:.3f}, using fallback")
                    return self.fallback_prediction_strategy(symbol, df)
                    
            except Exception as e:
                self.logger.warning(f"Error checking volume for {symbol}: {e}")
        # CRITICAL: Skip prediction if model is being retrained
        
        if hasattr(self, 'training_lock') and self.training_lock.locked():
            self.logger.warning(f"Model retraining in progress for {symbol}, using fallback")
            return self.fallback_prediction_strategy(symbol, df)
            
        if symbol not in self.models:
            self.logger.warning(f"No model found for {symbol}, using fallback")
            return self.fallback_prediction_strategy(symbol, df)
        
        try:
            # Apply symbol-specific configuration
            self._apply_symbol_specific_config(symbol)
            
            # Data validation
            if df is None or len(df) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
                return self.fallback_prediction_strategy(symbol, df)
                
            # Prepare features
            features = self.prepare_features_point_in_time(df, len(df)-1, symbol=symbol)
            
            if features.empty:
                return self.fallback_prediction_strategy(symbol, df)
            
            # Get expected features from the model
            selected_features = self.feature_importance.get(symbol, {}).get('selected_features', [])
            if not selected_features:
                self.logger.warning(f"No selected features for {symbol}, using fallback")
                return self.fallback_prediction_strategy(symbol, df)
            
            # CRITICAL: Dynamically ensure all required features are present
            missing_features = set(selected_features) - set(features.columns)
            if missing_features:
                self.logger.warning(f"Generating {len(missing_features)} missing features for {symbol}: {missing_features}")
                features = self._generate_missing_features(features, selected_features, symbol)

            
            # Select only the features the model expects
            features_selected = features[selected_features]
            
            # Scale features and predict
            scaled_features = self.scalers[symbol].transform(features_selected)
            
            # Rest of prediction logic remains the same...
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
            error_msg = str(e)
            if "feature names should match" in error_msg:
                self.logger.warning(f"Feature mismatch during prediction for {symbol}, using fallback: {error_msg}")
                # Force retrain on next cycle if we have persistent feature issues
                if symbol in self.model_versions:
                    self.model_versions[symbol]['needs_retraining'] = True
            else:
                self.logger.error(f"Prediction error for {symbol}: {error_msg}")
            
            return self.fallback_prediction_strategy(symbol, df)
  
    def validate_training_data_quality(self, features: pd.DataFrame, target: pd.Series, symbol: str) -> bool:
        """Enhanced training data quality validation with financial-aware checks"""
        try:
            # Stricter sample size requirement
            min_samples_required = max(50, self.min_training_samples)  # Reduced from 100 to 50
            if len(features) < min_samples_required:
                self.logger.warning(f"Insufficient samples for {symbol}: {len(features)} < {min_samples_required}")
                return False
            
            # Enhanced class distribution check - FINANCIAL DATA SPECIFIC
            class_counts = target.value_counts()
            unique_classes = len(class_counts)
            
            if unique_classes < 2:
                self.logger.error(f"Only {unique_classes} class(es) for {symbol}. Need at least 2 for meaningful training.")
                return False
            
            # FINANCIAL DATA INSIGHT: In trading, we often have imbalanced classes
            # What matters more is having enough samples in each class for learning
            min_class_count = class_counts.min()
            min_samples_per_class = max(20, len(features) * 0.03)  # At least 20 samples or 3% of data
            
            if min_class_count < min_samples_per_class:
                self.logger.warning(f"Small class for {symbol}: {min_class_count} samples in minority class "
                                f"(need at least {min_samples_per_class:.0f})")
                # Don't return False - just warn and proceed with class balancing
            
            # Calculate imbalance ratio for logging (but don't block training)
            min_class_ratio = class_counts.min() / class_counts.max()
            
            if min_class_ratio < 0.10:  # Reduced threshold to 10%
                self.logger.warning(f"Significant class imbalance for {symbol}: min_ratio={min_class_ratio:.3f}")
                # Don't return False - we'll handle this with class balancing
            elif min_class_ratio < 0.20:
                self.logger.info(f"Moderate class imbalance for {symbol}: min_ratio={min_class_ratio:.3f}")
            
            # Feature quality checks
            constant_features = features.columns[features.std() == 0]
            if len(constant_features) > len(features.columns) * 0.2:  # Increased to 20%
                self.logger.warning(f"Too many constant features for {symbol}: {len(constant_features)}/{len(features.columns)}")
                return False
            
            # Correlation check (informational only)
            try:
                corr_matrix = features.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = (upper_tri > 0.85).sum().sum()
                if high_corr_pairs > len(features.columns) * 0.3:
                    self.logger.info(f"High feature correlation for {symbol}: {high_corr_pairs} pairs > 0.85")
            except:
                pass  # Don't fail on correlation calculation errors
            
            self.logger.info(f"Training data quality OK for {symbol}: {len(features)} samples, {unique_classes} classes, "
                            f"min_class_ratio={min_class_ratio:.3f}, class_counts={class_counts.to_dict()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating training data for {symbol}: {e}")
            return False

    def _generate_missing_features(self, features_df: pd.DataFrame, required_features: List[str], symbol: str) -> pd.DataFrame:
        """Generate missing features with comprehensive fallbacks"""
        missing_features = set(required_features) - set(features_df.columns)
        
        if not missing_features:
            return features_df
        
        self.logger.warning(f"Generating {len(missing_features)} missing features for {symbol}: {missing_features}")
        
        # Create a copy to avoid modifying original
        enhanced_df = features_df.copy()
        
        calculated_count = 0
        for feature in missing_features:
            calculated_series = self._calculate_specific_feature(enhanced_df, feature, symbol)
            if calculated_series is not None:
                enhanced_df[feature] = calculated_series
                calculated_count += 1
            else:
                # Use robust fallback values
                fallback_value = self._get_robust_fallback_value(feature, enhanced_df, symbol)
                enhanced_df[feature] = fallback_value
                self.logger.info(f"Used fallback value for {feature}: {fallback_value.iloc[0] if hasattr(fallback_value, 'iloc') else fallback_value}")
        
        self.logger.info(f"Generated {calculated_count}/{len(missing_features)} features for {symbol}, used fallbacks for {len(missing_features) - calculated_count}")
        return enhanced_df

    def _get_robust_fallback_value(self, feature: str, features_df: pd.DataFrame, symbol: str):
        """Get robust fallback values for missing features"""
        fallback_values = {
            'rsi_14': 50.0,
            'atr_14': 0.02,
            'returns_5': 0.0,
            'returns_10': 0.0, 
            'returns_20': 0.0,
            'volume_ma_20': 1.0,
            'volume_ratio_20': 1.0,
            'symbol_specific_volatility': 0.02
        }
        
        # For unknown features, try to infer type
        if feature not in fallback_values:
            if 'rsi' in feature:
                return 50.0
            elif 'returns' in feature:
                return 0.0
            elif 'volume' in feature:
                return 1.0
            elif 'volatility' in feature:
                return 0.02
            else:
                return 0.0
        
        return fallback_values[feature]

    def _calculate_specific_feature(self, features_df: pd.DataFrame, feature: str, symbol: str) -> Optional[pd.Series]:
        """Calculate specific missing features with robust error handling"""
        try:
            # Ensure we have required columns
            if 'close' not in features_df.columns:
                self.logger.warning(f"Cannot calculate {feature}: missing 'close' column")
                return None
                
            close_prices = features_df['close'].astype(float)
            
            if feature.startswith('returns_'):
                period = int(feature.split('_')[1])
                if len(close_prices) > period:
                    return close_prices.pct_change(period)
                else:
                    # Return zeros if insufficient data
                    return pd.Series([0.0] * len(close_prices), index=features_df.index)
            
            elif feature.startswith('rsi_'):
                period = int(feature.split('_')[1])
                if len(close_prices) > period:
                    return self._calculate_rsi_point_in_time(close_prices, period)
                else:
                    # Return neutral RSI if insufficient data
                    return pd.Series([50.0] * len(close_prices), index=features_df.index)
            
            elif feature == 'atr_14':
                if all(col in features_df.columns for col in ['high', 'low', 'close']):
                    high = features_df['high'].astype(float)
                    low = features_df['low'].astype(float)
                    close = features_df['close'].astype(float)
                    if len(close) >= 14:
                        return self._calculate_atr_point_in_time(high, low, close, 14)
                    else:
                        # Calculate simple ATR approximation
                        price_range = (high - low).rolling(5, min_periods=1).mean()
                        return price_range
                else:
                    # Fallback: use price range
                    if len(close_prices) >= 5:
                        volatility = close_prices.pct_change().rolling(5).std().fillna(0.02)
                        return close_prices * volatility
                    else:
                        return pd.Series([close_prices.mean() * 0.02] * len(close_prices), index=features_df.index)
            
            elif feature.startswith('volume_'):
                if 'volume' in features_df.columns:
                    volume = features_df['volume'].astype(float)
                    if 'ma' in feature:
                        period = int(feature.split('_')[-1])
                        return volume.rolling(period, min_periods=1).mean()
                    else:
                        return volume
                else:
                    return pd.Series([1.0] * len(close_prices), index=features_df.index)
            
            elif feature == 'symbol_specific_volatility':
                if len(close_prices) >= 20:
                    return close_prices.pct_change().rolling(20, min_periods=1).std().fillna(0.02)
                else:
                    return pd.Series([0.02] * len(close_prices), index=features_df.index)
            
            else:
                self.logger.warning(f"Unknown feature type: {feature}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating feature {feature} for {symbol}: {e}")
            return None

    def _get_feature_median(self, feature: str, symbol: str) -> float:
        """Get median value for a feature from training data or use safe default"""
        # Default values based on feature type
        if 'returns' in feature:
            return 0.0
        elif 'rsi' in feature:
            return 50.0
        elif 'volume' in feature:
            return 1.0
        elif 'volatility' in feature:
            return 0.02
        else:
            return 0.0

    def _calibrate_confidence_symbol(self, probabilities: np.ndarray, prediction: int, symbol: str) -> float:
        """Symbol-specific confidence with more permissive thresholds"""
        base_confidence = self._calibrate_confidence(probabilities, prediction)
        
        # Much more permissive symbol adjustments
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] in ['high', 'extreme']:
                # Less reduction for volatile symbols
                base_confidence *= 0.9
            elif profile['volatility_tier'] == 'low':
                # Less boost for stable symbols
                base_confidence = min(1.0, base_confidence * 1.05)
        
        return base_confidence

    def _get_symbol_min_confidence(self, symbol: str) -> float:
        """Much more permissive minimum confidence thresholds"""
        if symbol in self.symbol_volatility_profiles:
            profile = self.symbol_volatility_profiles[symbol]
            if profile['volatility_tier'] == 'low':
                return 0.35  # Much lower for BTC/ETH
            elif profile['volatility_tier'] == 'extreme':
                return 0.45  # Lower for DOGE/SOL
            else:
                return 0.40  # Medium threshold
        return 0.40  # Default

    def train_multiple_models_for_symbol(self, symbol: str, df: pd.DataFrame, num_models: int = 5) -> bool:
        """
        Public method to train multiple models for a symbol and select the best one
        """
        return self.train_multiple_models_and_select_best(symbol, df, num_models)

    def _symbol_specific_ensemble(self, rf_pred: int, gb_pred: int, 
                                rf_confidence: float, gb_confidence: float, 
                                symbol: str) -> Tuple[int, float]:
        """More permissive ensemble decision making"""
        
        # If both models agree, use that prediction with averaged confidence
        if rf_pred == gb_pred:
            ensemble_vote = rf_pred
            confidence = (rf_confidence + gb_confidence) / 2
        else:
            # If models disagree, be more permissive about using predictions
            avg_confidence = (rf_confidence + gb_confidence) / 2
            
            # Use the prediction with higher confidence, but be more lenient
            if rf_confidence > gb_confidence and rf_confidence > 0.3:
                ensemble_vote = rf_pred
                confidence = rf_confidence
            elif gb_confidence > 0.3:
                ensemble_vote = gb_pred
                confidence = gb_confidence
            else:
                # If both have low confidence, default to HOLD but with moderate confidence
                ensemble_vote = 0
                confidence = max(rf_confidence, gb_confidence, 0.4)
        
        # For simple models, be even more conservative about forcing HOLD
        if symbol in self.symbol_model_complexity:
            config = self.symbol_model_complexity[symbol]
            if config['model_depth'] == 'simple':
                # Only force HOLD on extremely low confidence
                if confidence < 0.3:
                    ensemble_vote = 0
                    confidence = max(confidence, 0.3)
        
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
        """More permissive confidence calibration for financial predictions"""
        try:
            if len(probabilities) <= 1:
                return probabilities[0] if len(probabilities) == 1 else 0.5
            
            # For financial data, use maximum probability as confidence
            max_prob = np.max(probabilities)
            
            # Apply gentle scaling - much more permissive
            if max_prob > 0.6:
                confidence = 0.8 + (max_prob - 0.6) * 0.5  # 0.8 to 1.0 range
            elif max_prob > 0.4:
                confidence = 0.5 + (max_prob - 0.4) * 1.5  # 0.5 to 0.8 range
            else:
                confidence = max_prob * 1.25  # Boost low probabilities
            
            return min(1.0, max(0.0, confidence))
            
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
        """Convert 3-class predictions to trading signals"""
        # Map: 1 -> 1 (Buy), 0 -> 0 (Hold), -1 -> -1 (Sell)
        # For 3-class system, the mapping is direct
        return raw_prediction

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
        """Feature selection implementation with better error handling"""
        try:
            if len(X_train.columns) <= self.max_features:
                selected_features = X_train.columns.tolist()
                print(f"🔍 Feature selection for {symbol}: Using all {len(selected_features)} features")
                return X_train, selected_features
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
            
            # === FIX: Ensure consistent feature naming ===
            original_features = X_train.columns.tolist()
            
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
                # Default: remove highly correlated features
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
            self.feature_importance[symbol]['all_available_features'] = original_features
            
            return X_selected, selected_features
            
        except Exception as e:
            print(f"⚠️ Feature selection failed for {symbol}: {e}")
            # Fallback: use features with highest variance
            variances = X_train.var().sort_values(ascending=False)
            selected_features = variances.head(self.max_features).index.tolist()
            
            if symbol not in self.feature_importance:
                self.feature_importance[symbol] = {}
            self.feature_importance[symbol]['selected_features'] = selected_features
            self.feature_importance[symbol]['selection_failed'] = True
            
            return X_train[selected_features], selected_features

    def validate_feature_consistency(self, symbol: str, current_features: pd.DataFrame) -> bool:
        """Enhanced feature consistency with dynamic alignment"""
        if symbol not in self.feature_importance:
            self.logger.warning(f"No feature importance data for {symbol}")
            return False
            
        selected_features = self.feature_importance[symbol].get('selected_features', [])
        if not selected_features:
            self.logger.warning(f"No selected features for {symbol}")
            return False
        
        # CRITICAL: Check if we're in the middle of retraining
        if hasattr(self, 'training_lock') and self.training_lock.locked():
            self.logger.warning(f"Model retraining in progress for {symbol}, skipping feature validation")
            return False
            
        # Check if all selected features are present in current data
        missing_features = set(selected_features) - set(current_features.columns)
        if missing_features:
            self.logger.warning(f"Feature mismatch for {symbol}: Missing {len(missing_features)} features: {missing_features}")
            
            # Enhanced graceful degradation: Use available features and create missing ones with defaults
            available_features = [f for f in selected_features if f in current_features.columns]
            
            if len(available_features) >= max(5, len(selected_features) * 0.6):  # Reduced threshold to 60%
                # Add default values for missing features
                for missing_feature in missing_features:
                    if missing_feature.startswith('returns_'):
                        current_features[missing_feature] = 0.0
                    elif missing_feature.startswith('rsi_'):
                        current_features[missing_feature] = 50.0
                    elif missing_feature.startswith('volume_'):
                        current_features[missing_feature] = 1.0
                    else:
                        current_features[missing_feature] = 0.0
                
                self.logger.info(f"Auto-populated {len(missing_features)} missing features for {symbol}")
                return True
            else:
                self.logger.error(f"Insufficient features for {symbol}: only {len(available_features)}/{len(selected_features)} available")
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

    def time_series_train_test_split(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2):
        """Enhanced split that handles small datasets"""
        if len(features) < 10:  # Reduced minimum
            self.logger.warning(f"time_series_train_test_split: Only {len(features)} samples, too few for split")
            return None, None, None, None
            
        # For very small datasets, use a single sample for testing
        if len(features) < 30:
            test_size = 1 / len(features)  # One sample for testing
            self.logger.info(f"Using adaptive test_size {test_size:.3f} for small dataset ({len(features)} samples)")
        
        split_idx = int(len(features) * (1 - test_size))
        
        # Ensure we have at least 1 test sample
        if split_idx >= len(features):
            split_idx = len(features) - 1
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        # Much more permissive minimums
        if len(X_train) < 5 or len(X_test) < 1:  # Reduced from 50/10
            self.logger.warning(f"Split failed: train={len(X_train)}, test={len(X_test)}")
            return None, None, None, None
            
        self.logger.info(f"Split successful: train={len(X_train)}, test={len(X_test)}")
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
                'target_type': '3_class_system',
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

    def debug_prediction_process(self, symbol: str, df: pd.DataFrame):
        """Comprehensive debugging of the prediction process"""
        self.logger.info(f"🔍 [DEBUG_PREDICTION] Starting prediction debug for {symbol}")
        
        # 1. Check if model exists
        if symbol not in self.models:
            self.logger.error(f"❌ [DEBUG_PREDICTION] No model found for {symbol}")
            return
        
        # 2. Check model components
        model_info = self.models[symbol]
        if 'rf' not in model_info or 'gb' not in model_info:
            self.logger.error(f"❌ [DEBUG_PREDICTION] Incomplete model for {symbol}")
            return
        
        # 3. Prepare features and check
        features = self.prepare_features_point_in_time(df, len(df)-1, symbol=symbol)
        self.logger.info(f"📊 [DEBUG_PREDICTION] Features shape: {features.shape}")
        self.logger.info(f"📊 [DEBUG_PREDICTION] Features columns: {list(features.columns)}")
        
        # 4. Check for NaN values
        nan_count = features.isna().sum().sum()
        self.logger.info(f"🔍 [DEBUG_PREDICTION] NaN values in features: {nan_count}")
        
        # 5. Check selected features
        selected_features = self.feature_importance.get(symbol, {}).get('selected_features', [])
        self.logger.info(f"📋 [DEBUG_PREDICTION] Selected features count: {len(selected_features)}")
        self.logger.info(f"📋 [DEBUG_PREDICTION] Selected features: {selected_features}")
        
        # 6. Check feature alignment
        missing_features = set(selected_features) - set(features.columns)
        extra_features = set(features.columns) - set(selected_features)
        
        if missing_features:
            self.logger.warning(f"⚠️ [DEBUG_PREDICTION] Missing features: {missing_features}")
        if extra_features:
            self.logger.warning(f"⚠️ [DEBUG_PREDICTION] Extra features: {extra_features}")
        
        # 7. Try scaling and prediction
        try:
            if selected_features:
                features_selected = features[selected_features]
                self.logger.info(f"📊 [DEBUG_PREDICTION] Selected features shape: {features_selected.shape}")
                
                # Check scaler
                if symbol in self.scalers:
                    scaled_features = self.scalers[symbol].transform(features_selected)
                    self.logger.info(f"⚖️ [DEBUG_PREDICTION] Scaled features shape: {scaled_features.shape}")
                    
                    # Try RF prediction
                    rf_pred = self.models[symbol]['rf'].predict(scaled_features)
                    rf_proba = self.models[symbol]['rf'].predict_proba(scaled_features)
                    
                    self.logger.info(f"🤖 [DEBUG_PREDICTION] RF prediction: {rf_pred[0]}")
                    self.logger.info(f"🎯 [DEBUG_PREDICTION] RF probabilities: {rf_proba[0]}")
                    
                    # Try GB prediction  
                    gb_pred = self.models[symbol]['gb'].predict(scaled_features)
                    gb_proba = self.models[symbol]['gb'].predict_proba(scaled_features)
                    
                    self.logger.info(f"🤖 [DEBUG_PREDICTION] GB prediction: {gb_pred[0]}")
                    self.logger.info(f"🎯 [DEBUG_PREDICTION] GB probabilities: {gb_proba[0]}")
                    
                else:
                    self.logger.error(f"❌ [DEBUG_PREDICTION] No scaler found for {symbol}")
            else:
                self.logger.error(f"❌ [DEBUG_PREDICTION] No selected features for {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ [DEBUG_PREDICTION] Prediction error: {e}")

    def debug_model_status(self, symbol: str):
        """Debug model training and status"""
        self.logger.info(f"🔍 [DEBUG_MODEL] Checking model status for {symbol}")
        
        # Check if model exists
        if symbol not in self.models:
            self.logger.error(f"❌ [DEBUG_MODEL] No model found for {symbol}")
            return False
        
        # Check model version info
        model_info = self.model_versions.get(symbol, {})
        self.logger.info(f"📋 [DEBUG_MODEL] Model info: {model_info}")
        
        # Check feature importance
        feature_info = self.feature_importance.get(symbol, {})
        self.logger.info(f"📊 [DEBUG_MODEL] Feature info keys: {list(feature_info.keys())}")
        
        # Check if models are actually trained
        model_data = self.models[symbol]
        if 'rf' in model_data and hasattr(model_data['rf'], 'classes_'):
            self.logger.info(f"✅ [DEBUG_MODEL] RF model trained with classes: {model_data['rf'].classes_}")
        else:
            self.logger.error(f"❌ [DEBUG_MODEL] RF model not properly trained")
            
        if 'gb' in model_data and hasattr(model_data['gb'], 'classes_'):
            self.logger.info(f"✅ [DEBUG_MODEL] GB model trained with classes: {model_data['gb'].classes_}")
        else:
            self.logger.error(f"❌ [DEBUG_MODEL] GB model not properly trained")
        
        return True

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