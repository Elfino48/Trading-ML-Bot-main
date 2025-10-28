import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TA_AVAILABLE = True
except ImportError:
    try:
        import ta
        TA_AVAILABLE = True
    except ImportError:
        TA_AVAILABLE = False
        print("Warning: No TA library available. Install ta or talib.")

class EnhancedTechnicalAnalyzer:
    def __init__(self):
        self.regime_threshold = 0.5
        self.regime_history = []
        self.macro_indicators = {}
        self.logger = logging.getLogger(__name__)

    def calculate_momentum_alignment(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum alignment across multiple timeframes and indicators"""
        if len(df) < 100:
            return {}
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        momentum_indicators = {}
        
        try:
            # 1. Price Momentum (20, 50, 100 periods)
            for period in [20, 50, 100]:
                if len(close) > period:
                    momentum = (close.iloc[-1] / close.iloc[-period] - 1) * 100
                    momentum_indicators[f'price_momentum_{period}'] = momentum
            
            # 2. Moving Average Alignment
            ema_8 = self._calculate_ema(close, 8)
            ema_21 = self._calculate_ema(close, 21)
            ema_55 = self._calculate_ema(close, 55)
            
            ma_alignment = 0
            if ema_8 > ema_21 > ema_55:
                ma_alignment = 1  # Strong bullish
            elif ema_8 < ema_21 < ema_55:
                ma_alignment = -1  # Strong bearish
            
            momentum_indicators['ma_alignment'] = ma_alignment
            momentum_indicators['ma_alignment_strength'] = self._calculate_ma_alignment_strength(ema_8, ema_21, ema_55)
            
            # 3. RSI Momentum
            rsi_14 = self._calculate_rsi(close, 14)
            rsi_21 = self._calculate_rsi(close, 21)
            
            momentum_indicators['rsi_trend'] = 1 if rsi_14 > rsi_21 else -1
            momentum_indicators['rsi_momentum'] = rsi_14 - rsi_21
            
            # 4. MACD Momentum
            macd, macd_signal = self._calculate_macd_momentum(close)
            momentum_indicators['macd_momentum'] = macd - macd_signal
            momentum_indicators['macd_trend'] = 1 if macd > macd_signal else -1
            
            # 5. Volume-Weighted Momentum
            volume = df['volume'].astype(float)
            vwap = self._calculate_vwap(high, low, close, volume)
            price_vs_vwap = (close.iloc[-1] / vwap - 1) * 100
            momentum_indicators['vwap_momentum'] = price_vs_vwap
            
            # 6. Composite Momentum Score
            composite_momentum = self._calculate_composite_momentum(momentum_indicators)
            momentum_indicators['composite_momentum'] = composite_momentum
            momentum_indicators['momentum_regime'] = self._classify_momentum_regime(composite_momentum)
            
            return momentum_indicators
            
        except Exception as e:
            print(f"Error calculating momentum alignment: {e}")
            return {}

    def _calculate_ma_alignment_strength(self, ema_8: float, ema_21: float, ema_55: float) -> float:
        """Calculate the strength of moving average alignment"""
        if ema_8 > ema_21 > ema_55:
            # Bullish alignment strength
            strength_8_21 = (ema_8 - ema_21) / ema_21
            strength_21_55 = (ema_21 - ema_55) / ema_55
            return (strength_8_21 + strength_21_55) / 2 * 100
        elif ema_8 < ema_21 < ema_55:
            # Bearish alignment strength
            strength_8_21 = (ema_21 - ema_8) / ema_8
            strength_21_55 = (ema_55 - ema_21) / ema_21
            return (strength_8_21 + strength_21_55) / 2 * 100
        else:
            return 0

    def _calculate_macd_momentum(self, close: pd.Series) -> Tuple[float, float]:
        """Calculate MACD momentum"""
        try:
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            return macd.iloc[-1], macd_signal.iloc[-1]
        except:
            return 0, 0

    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap.iloc[-1]
        except:
            return close.iloc[-1]

    def _calculate_composite_momentum(self, momentum_indicators: Dict) -> float:
        """Calculate composite momentum score from all indicators"""
        score = 0
        weights = {
            'price_momentum_20': 0.15,
            'price_momentum_50': 0.20,
            'price_momentum_100': 0.15,
            'ma_alignment': 0.20,
            'ma_alignment_strength': 0.10,
            'rsi_momentum': 0.10,
            'macd_momentum': 0.10
        }
        
        for indicator, weight in weights.items():
            value = momentum_indicators.get(indicator, 0)
            # Normalize different indicators to similar scale
            if 'momentum' in indicator:
                normalized_value = max(-100, min(100, value)) / 100
            elif 'alignment' in indicator:
                normalized_value = value
            else:
                normalized_value = value / 100  # Assume percentage values
            
            score += normalized_value * weight
        
        return score * 100  # Return as percentage

    def _classify_momentum_regime(self, composite_momentum: float) -> str:
        """Classify momentum regime based on composite score"""
        if composite_momentum > 30:
            return "strong_bullish"
        elif composite_momentum > 10:
            return "bullish"
        elif composite_momentum < -30:
            return "strong_bearish"
        elif composite_momentum < -10:
            return "bearish"
        else:
            return "neutral"

    def set_macro_indicators(self, indicators: Dict):
        self.macro_indicators = indicators
        
    def calculate_regime_indicators(self, df: pd.DataFrame) -> Dict:
            if len(df) < 100:
                return {}

            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            volume = df['volume'].astype(float)

            indicators = {}

            try:
                for period in [8, 21, 34, 55, 89]:
                    indicators[f'ema_{period}'] = self._calculate_ema(close, period)

                # --- FIX 1: Calculate and add SMA 50 ---
                indicators['sma_50'] = self._calculate_sma(close, 50)
                # --- END FIX 1 ---

                hma_length = 20
                indicators['hma'] = self._calculate_hma(close, hma_length)

                indicators['super_trend'] = self._calculate_supertrend(high, low, close)

                for period in [6, 14, 21]:
                    indicators[f'rsi_{period}'] = self._calculate_rsi(close, period)

                stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d

                indicators['williams_r'] = self._calculate_williams_r(high, low, close)

                indicators['atr'] = self._calculate_atr(high, low, close)
                indicators['atr_percent'] = (indicators['atr'] / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0

                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
                indicators['bb_upper'] = bb_upper
                indicators['bb_lower'] = bb_lower
                indicators['bb_middle'] = bb_middle
                indicators['bb_position'] = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

                indicators['volume_sma_20'] = volume.rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else 1

                obv = self._calculate_obv(close, volume)
                indicators['obv_trend'] = self._calculate_slope(obv, 5)

                regime_result = self._detect_market_regime_enhanced(close, high, low, volume)
                indicators['market_regime'] = regime_result['regime']
                indicators['regime_confidence'] = regime_result['confidence']
                indicators['regime_transition_prob'] = regime_result['transition_prob']

                sr_levels = self._find_support_resistance(close, high, low)
                indicators['support'] = sr_levels['support']
                indicators['resistance'] = sr_levels['resistance']

                indicators['is_anomaly'] = self._detect_anomalies(close)

                indicators['price_vs_high_20'] = close.iloc[-1] / high.rolling(20).max().iloc[-1] if high.rolling(20).max().iloc[-1] > 0 else 1
                indicators['price_vs_low_20'] = close.iloc[-1] / low.rolling(20).min().iloc[-1] if low.rolling(20).min().iloc[-1] > 0 else 1

                # Add momentum alignment indicators
                momentum_indicators = self.calculate_momentum_alignment(df)
                indicators.update(momentum_indicators)

                self.regime_history.append(indicators['market_regime'])
                if len(self.regime_history) > 50:
                    self.regime_history.pop(0)

            except Exception as e:
                print(f"Error calculating enhanced indicators: {e}")
                return {}

            return indicators
        
    def _calculate_sma(self, series, period):
            # Add basic calculation if TA libraries are not guaranteed
            if len(series) < period:
                return series.iloc[-1] if not series.empty else 0
            sma = series.rolling(window=period).mean().iloc[-1]
            return sma if not pd.isna(sma) else series.iloc[-1]
    
    def _calculate_ema(self, series, period):
        if TA_AVAILABLE:
            try:
                if 'talib' in globals():
                    return talib.EMA(series, timeperiod=period).iloc[-1]
                else:
                    return ta.trend.EMAIndicator(series, window=period).ema_indicator().iloc[-1]
            except:
                return series.ewm(span=period).mean().iloc[-1]
        else:
            return series.ewm(span=period).mean().iloc[-1]
    
    def _calculate_rsi(self, series, period):
        if TA_AVAILABLE:
            try:
                if 'talib' in globals():
                    return talib.RSI(series, timeperiod=period).iloc[-1]
                else:
                    return ta.momentum.RSIIndicator(series, window=period).rsi().iloc[-1]
            except:
                return self._calculate_rsi_manual(series, period)
        else:
            return self._calculate_rsi_manual(series, period)
    
    def _calculate_rsi_manual(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            stoch_d = stoch_k.rolling(window=d_period).mean()
            return stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50, stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
        except:
            return 50, 50
    
    def _calculate_williams_r(self, high, low, close, period=14):
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50
        except:
            return -50
    
    def _calculate_atr(self, high, low, close, period=14):
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else (high.iloc[-1] - low.iloc[-1])
        except:
            return high.iloc[-1] - low.iloc[-1]
    
    def _calculate_bollinger_bands(self, series, period=20, std_dev=2):
        try:
            middle = series.rolling(period).mean()
            std = series.rolling(period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return (upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else series.iloc[-1] * 1.1,
                    middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else series.iloc[-1],
                    lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else series.iloc[-1] * 0.9)
        except:
            current_price = series.iloc[-1]
            return current_price * 1.1, current_price, current_price * 0.9
    
    def _calculate_obv(self, close, volume):
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
    
    def _calculate_hma(self, series, period):
        try:
            half_length = int(period / 2)
            sqrt_length = int(np.sqrt(period))
            
            wma_half = series.rolling(half_length).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=False
            )
            wma_full = series.rolling(period).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=False
            )
            
            hma_series = 2 * wma_half - wma_full
            hma = hma_series.rolling(sqrt_length).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=False
            )
            
            return hma.iloc[-1] if not pd.isna(hma.iloc[-1]) else series.iloc[-1]
        except:
            return series.iloc[-1]
    
    def _calculate_supertrend(self, high, low, close, period=10, multiplier=3):
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            st = pd.Series(index=close.index, dtype=float)
            st.iloc[0] = upper_band.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > st.iloc[i-1]:
                    st.iloc[i] = max(lower_band.iloc[i], st.iloc[i-1])
                else:
                    st.iloc[i] = min(upper_band.iloc[i], st.iloc[i-1])
            
            return st.iloc[-1] if not pd.isna(st.iloc[-1]) else close.iloc[-1]
        except:
            return close.iloc[-1]
    
    def _calculate_slope(self, series, period):
        if len(series) < period:
            return 0
        y = series.iloc[-period:].values
        x = np.arange(len(y))
        if len(y) < 2:
            return 0
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def _detect_market_regime_enhanced(self, close, high, low, volume, lookback=50):
        if len(close) < lookback:
            return {'regime': 'neutral', 'confidence': 0.5, 'transition_prob': 0.5}
        
        returns = close.pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        y = close.iloc[-lookback:].values
        x = np.arange(len(y))
        slope, _, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        adx = self._calculate_adx(high, low, close)
        ema_8 = self._calculate_ema(close, 8)
        ema_21 = self._calculate_ema(close, 21)
        ema_55 = self._calculate_ema(close, 55)
        
        trend_alignment = 0
        if ema_8 > ema_21 > ema_55:
            trend_alignment = 1
        elif ema_8 < ema_21 < ema_55:
            trend_alignment = -1
        
        volume_trend = self._calculate_slope(volume, 20)
        price_range = (high.rolling(lookback).max() - low.rolling(lookback).min()) / close.rolling(lookback).mean()
        
        macro_score = self._calculate_macro_impact()
        
        trend_score = (slope * 1000 + r_squared * trend_alignment + adx / 50) / 3
        vol_score = volatility * 100
        
        regime_score = 0
        if r_squared > 0.6 and abs(trend_score) > 0.1: # Strong trend condition
             regime_score = trend_score # Use trend direction
        elif r_squared < 0.3: # Low R-squared suggests ranging or weak trend
             regime_score = 0 # Map towards neutral/ranging
        else: # Moderate R-squared, mixed signals
             regime_score = trend_score * 0.5 # Weaken trend signal

        regime_score = np.clip(regime_score + macro_score, -1, 1)

        # --- Ensure output matches the standard set ---
        # Maybe adjust thresholds slightly based on observation
        # --- Standardized Classification Logic ---
        threshold_trend = 0.3 # Example threshold for trend score
        threshold_ranging_upper = 0.15 # Upper bound for ranging/neutral based on score
        threshold_ranging_lower = -0.15 # Lower bound for ranging/neutral based on score

        volatility = close.pct_change().rolling(20).std().iloc[-1] # Recalculate or get volatility
        vol_high_threshold = 0.03
        vol_low_threshold = 0.01

        # Determine regime based on score and volatility
        if trend_score > threshold_trend:
            regime = "bull_trend"
            confidence = min(abs(trend_score), 0.9) # Confidence based on score strength
        elif trend_score < -threshold_trend:
            regime = "bear_trend"
            confidence = min(abs(trend_score), 0.9)
        elif volatility >= vol_high_threshold:
             regime = "high_volatility" # Prioritize high vol over weak trend/ranging
             confidence = min(volatility / 0.05, 0.8) # Confidence based on vol level
        elif volatility <= vol_low_threshold and abs(trend_score) < threshold_ranging_upper:
             regime = "ranging" # Low vol and weak score -> ranging
             confidence = max(0.6, 1 - abs(trend_score) / threshold_ranging_upper)
        # --- Explicitly map low absolute scores between ranging bounds to 'neutral' ---
        elif threshold_ranging_lower <= trend_score <= threshold_ranging_upper:
             regime = "neutral" # Genuinely weak/mixed signal
             confidence = 0.5
        else: # Default case if something unexpected happens
             regime = "neutral"
             confidence = 0.4

        transition_prob = self._calculate_regime_transition_prob(regime)

        self.logger.debug(f"_detect_market_regime_enhanced: score={trend_score:.3f}, vol={volatility:.4f} -> Regime='{regime}', Conf={confidence:.2f}") # Add debug log

        return {
            'regime': regime, # Ensure output is one of the standard names
            'confidence': confidence,
            'transition_prob': transition_prob
        }
    
    def _calculate_adx(self, high, low, close, period=14):
        try:
            if TA_AVAILABLE and 'talib' in globals():
                return talib.ADX(high, low, close, timeperiod=period).iloc[-1]
            else:
                return 25
        except:
            return 25
    
    def _calculate_macro_impact(self):
        if not self.macro_indicators:
            return 0
        
        macro_score = 0
        weight_total = 0
        
        if 'vix' in self.macro_indicators:
            vix = self.macro_indicators['vix']
            if vix > 25:
                macro_score -= 0.3
            elif vix < 15:
                macro_score += 0.2
            weight_total += 1
        
        if 'interest_rate' in self.macro_indicators:
            rate = self.macro_indicators['interest_rate']
            if rate > 5:
                macro_score -= 0.2
            elif rate < 2:
                macro_score += 0.1
            weight_total += 1
        
        if 'inflation' in self.macro_indicators:
            inflation = self.macro_indicators['inflation']
            if inflation > 4:
                macro_score -= 0.2
            elif inflation < 2:
                macro_score += 0.1
            weight_total += 1
        
        if weight_total > 0:
            return macro_score / weight_total
        return 0
    
    def _calculate_regime_transition_prob(self, current_regime):
        if len(self.regime_history) < 10:
            return 0.5
        
        transitions = []
        for i in range(1, len(self.regime_history)):
            transitions.append((self.regime_history[i-1], self.regime_history[i]))
        
        if not transitions:
            return 0.5
        
        same_regime_count = sum(1 for prev, curr in transitions if prev == current_regime and curr == current_regime)
        total_from_current = sum(1 for prev, curr in transitions if prev == current_regime)
        
        if total_from_current == 0:
            return 0.5
        
        stability_prob = same_regime_count / total_from_current
        
        recent_changes = sum(1 for i in range(1, min(10, len(self.regime_history))) 
                         if self.regime_history[i] != self.regime_history[i-1])
        volatility_factor = recent_changes / min(10, len(self.regime_history)-1)
        
        transition_prob = (1 - stability_prob) * (1 + volatility_factor)
        
        return min(transition_prob, 0.9)
    
    def _find_support_resistance(self, close, high, low, window=20):
        if len(close) < window * 2:
            return {'support': close.iloc[-1] * 0.98, 'resistance': close.iloc[-1] * 1.02}
        
        recent_high = high.rolling(window).max().iloc[-1]
        recent_low = low.rolling(window).min().iloc[-1]
        
        return {
            'support': recent_low if not pd.isna(recent_low) else close.iloc[-1] * 0.98,
            'resistance': recent_high if not pd.isna(recent_high) else close.iloc[-1] * 1.02
        }
    
    def _detect_anomalies(self, close, contamination=0.1):
        if len(close) < 50:
            return False
            
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            return False
            
        X = returns.values.reshape(-1, 1)
        
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        
        return predictions[-1] == -1
    
    def _calculate_fibonacci(self, high, low, close, lookback=30):
        if len(close) < lookback:
            return {}
            
        swing_high = high.rolling(lookback).max().iloc[-1]
        swing_low = low.rolling(lookback).min().iloc[-1]
        
        diff = swing_high - swing_low
        
        return {
            'fib_236': swing_high - 0.236 * diff,
            'fib_382': swing_high - 0.382 * diff,
            'fib_500': swing_high - 0.5 * diff,
            'fib_618': swing_high - 0.618 * diff,
            'fib_786': swing_high - 0.786 * diff
        }

    def calculate_advanced_features(self, df: pd.DataFrame) -> Dict:
        features = {}
        
        try:
            features['hurst_exponent'] = self._calculate_hurst_exponent(df['close'])
            
            features['sample_entropy'] = self._calculate_sample_entropy(df['close'])
            
            features['regime_change_prob'] = self._detect_regime_change(df)
            
            features['volatility_clustering'] = self._detect_volatility_clustering(df)
            
            return features
            
        except Exception as e:
            return {}

    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        try:
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5

    def _calculate_sample_entropy(self, series: pd.Series, m: int = 2, r: float = 0.2) -> float:
        try:
            n = len(series)
            std = series.std()
            if std == 0:
                return 0
                
            r_val = r * std
            patterns = [series[i:i+m] for i in range(n - m + 1)]
            
            if len(patterns) < 2:
                return 0
                
            matches = 0
            total = 0
            
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r_val:
                        matches += 1
                    total += 1
                    
            return -np.log(matches / total) if matches > 0 else 0
            
        except:
            return 0

    def detect_market_regime_advanced(self, df: pd.DataFrame) -> Dict:
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            regime_result = self._detect_market_regime_enhanced(close, high, low, volume)
            
            returns = close.pct_change().dropna()
            volatility_regime = self._classify_volatility_regime(returns)
            market_state = self._classify_market_state(close, returns)
            
            regimes = {
                'trend_strength': self._calculate_trend_strength(close),
                'volatility_regime': volatility_regime,
                'market_state': market_state,
                'regime_confidence': regime_result['confidence'],
                'composite_regime': regime_result['regime'],
                'transition_probability': regime_result['transition_prob']
            }
            
            return regimes
            
        except Exception as e:
            return {'composite_regime': 'neutral', 'regime_confidence': 0.0, 'transition_probability': 0.5}

    def _detect_regime_change(self, df):
        if len(self.regime_history) < 5:
            return 0.5
        
        recent_changes = sum(1 for i in range(1, len(self.regime_history)) 
                          if self.regime_history[i] != self.regime_history[i-1])
        
        return recent_changes / (len(self.regime_history) - 1)

    def _detect_volatility_clustering(self, df):
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()
        
        if len(volatility) < 2:
            return "medium"
            
        vol_change = volatility.pct_change().dropna()
        clustering_score = vol_change.abs().mean()
        
        if clustering_score > 0.5:
            return "high"
        elif clustering_score < 0.2:
            return "low"
        else:
            return "medium"

    def _calculate_trend_strength(self, close):
        if len(close) < 50:
            return 0.0
            
        y = close.iloc[-50:].values
        x = np.arange(len(y))
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        strength = abs(slope) * (r_value ** 2)
        return float(strength * 1000)

    def _classify_volatility_regime(self, returns):
        if len(returns) < 20:
            return "medium"
            
        volatility = returns.rolling(20).std().iloc[-1]
        
        if volatility > 0.03:
            return "high"
        elif volatility < 0.01:
            return "low"
        else:
            return "medium"

    def _classify_market_state(self, close, returns):
        if len(close) < 50:
            return "neutral"
            
        trend_strength = self._calculate_trend_strength(close)
        volatility = returns.rolling(20).std().iloc[-1]
        
        if trend_strength > 0.5 and volatility < 0.02:
            return "strong_trend"
        elif trend_strength > 0.3:
            return "trending"
        elif volatility > 0.03:
            return "high_volatility"
        else:
            return "neutral"

    def generate_enhanced_signals(self, indicators: Dict) -> Dict:
        signals = {
            'trend_strength': 0,
            'momentum_score': 0,
            'volatility_adjusted': 0,
            'volume_confirmation': 0,
            'regime_score': 0,
            'composite_score': 0,
            'signal_strength': 'NEUTRAL',
            'recommended_action': 'HOLD',
            'regime_confidence': indicators.get('regime_confidence', 0.5),
            'transition_prob': indicators.get('regime_transition_prob', 0.5)
        }
        
        if not indicators:
            return signals
        
        try:
            trend_score = 0
            current_price = indicators.get('ema_8', indicators.get('bb_middle', 0))
            
            if (indicators.get('ema_8', 0) > indicators.get('ema_21', 0) > 
                indicators.get('ema_55', 0)):
                trend_score += 40
            elif (indicators.get('ema_8', 0) < indicators.get('ema_21', 0) < 
                    indicators.get('ema_55', 0)):
                trend_score -= 40
                
            if indicators.get('hma', 0) > current_price:
                trend_score -= 20
            else:
                trend_score += 20
                
            signals['trend_strength'] = trend_score
            
            momentum_score = 0
            rsi_14 = indicators.get('rsi_14', 50)
            
            if rsi_14 > 60:
                momentum_score += 30
            elif rsi_14 < 40:
                momentum_score -= 30
                
            stoch_k = indicators.get('stoch_k', 50)
            if stoch_k > 80:
                momentum_score += 20
            elif stoch_k < 20:
                momentum_score -= 20
                
            signals['momentum_score'] = momentum_score
            
            vol_score = 0
            bb_position = indicators.get('bb_position', 0.5)
            atr_percent = indicators.get('atr_percent', 0)
            
            if bb_position < 0.2:
                vol_score += 25
            elif bb_position > 0.8:
                vol_score -= 25
                
            if atr_percent < 1.5:
                vol_score += 15
                
            signals['volatility_adjusted'] = vol_score
            
            volume_score = 0
            volume_ratio = indicators.get('volume_ratio', 1)
            obv_trend = indicators.get('obv_trend', 0)
            
            if volume_ratio > 1.2:
                volume_score += 20
            if obv_trend > 0:
                volume_score += 10
                
            signals['volume_confirmation'] = volume_score
            
            regime = indicators.get('market_regime', 'neutral')
            regime_map = {
                'bull_trend': 20,
                'bear_trend': -20,
                'ranging': 0,
                'neutral': 0
            }
            signals['regime_score'] = regime_map.get(regime, 0)
            
            transition_factor = 1 - indicators.get('regime_transition_prob', 0.5)
            confidence_factor = indicators.get('regime_confidence', 0.5)
            
            composite = (
                signals['trend_strength'] * 0.25 * transition_factor +
                signals['momentum_score'] * 0.25 * confidence_factor +
                signals['volatility_adjusted'] * 0.2 +
                signals['volume_confirmation'] * 0.15 +
                signals['regime_score'] * 0.15
            )
            
            signals['composite_score'] = composite
            
            if composite > 30:
                signals['signal_strength'] = 'STRONG_BUY'
                signals['recommended_action'] = 'BUY'
            elif composite > 15:
                signals['signal_strength'] = 'MODERATE_BUY'
                signals['recommended_action'] = 'BUY'
            elif composite < -30:
                signals['signal_strength'] = 'STRONG_SELL'
                signals['recommended_action'] = 'SELL'
            elif composite < -15:
                signals['signal_strength'] = 'MODERATE_SELL'
                signals['recommended_action'] = 'SELL'
            else:
                signals['signal_strength'] = 'NEUTRAL'
                signals['recommended_action'] = 'HOLD'
                
        except Exception as e:
            print(f"Error generating enhanced signals: {e}")
            
        return signals