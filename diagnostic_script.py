# strategy_debug.py
from typing import Dict
import pandas as pd
import numpy as np
from bybit_client import BybitClient
from enhanced_strategy_orchestrator import EnhancedStrategyOrchestrator

def create_clear_bullish_data():
    """Create data with very clear bullish signals"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1h')
    
    # Strong uptrend with healthy pullbacks
    base_trend = [100 + i * 0.8 for i in range(200)]  # Strong uptrend
    noise = np.random.normal(0, 0.5, 200)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_trend + noise * 0.1,
        'high': base_trend + np.abs(noise) * 0.3 + 2,
        'low': base_trend - np.abs(noise) * 0.3 - 1,
        'close': base_trend + noise * 0.1,
        'volume': [5000 + i * 20 + np.random.randint(-500, 500) for i in range(200)]
    })
    
    # Ensure bullish characteristics
    df['close'] = df['close'].rolling(5).mean()  # Smooth trend
    df['volume'] = df['volume'] * 1.5  # High volume
    
    return df

def debug_strategy_components():
    """
    Debug each strategy component individually
    """
    print("🔧 DEBUGGING STRATEGY COMPONENTS")
    print("=" * 70)
    
    bybit_client = BybitClient()
    orchestrator = EnhancedStrategyOrchestrator(bybit_client, aggressiveness="moderate")
    test_data = create_clear_bullish_data()
    
    print("📈 TEST DATA CHARACTERISTICS:")
    print(f"• Trend: {test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1:.1%} gain")
    print(f"• Current RSI: {calculate_rsi(test_data['close'], 14):.1f}")
    print(f"• Volume trend: {(test_data['volume'].iloc[-1] / test_data['volume'].iloc[0] - 1):.1%}")
    
    # Calculate indicators
    indicators = orchestrator.technical_analyzer.calculate_regime_indicators(test_data)
    signals = orchestrator.technical_analyzer.generate_enhanced_signals(indicators)
    ml_result = orchestrator.ml_predictor.predict("DEBUG", test_data)
    mtf_signals = orchestrator._multi_timeframe_analysis(test_data)
    
    print(f"\n🎯 INDICATOR VALUES:")
    print(f"• Market Regime: {indicators.get('market_regime', 'unknown')}")
    print(f"• RSI 14: {indicators.get('rsi_14', 50):.1f}")
    print(f"• Volume Ratio: {indicators.get('volume_ratio', 1):.2f}")
    print(f"• BB Position: {indicators.get('bb_position', 0.5):.2f}")
    print(f"• ATR %: {indicators.get('atr_percent', 0):.2f}%")
    
    print(f"\n⚡ STRATEGY SIGNALS:")
    print(f"• Trend Strength: {signals.get('trend_strength', 0)}")
    print(f"• Momentum Score: {signals.get('momentum_score', 0)}")
    print(f"• Volatility Adjusted: {signals.get('volatility_adjusted', 0)}")
    print(f"• Volume Confirmation: {signals.get('volume_confirmation', 0)}")
    
    print(f"\n🤖 ML PREDICTION:")
    print(f"• Prediction: {ml_result.get('prediction', 0)}")
    print(f"• Confidence: {ml_result.get('confidence', 0):.1%}")
    
    print(f"\n⏰ MULTI-TIMEFRAME:")
    print(f"• Alignment: {mtf_signals.get('timeframe_alignment', 0)}")
    print(f"• Strength: {mtf_signals.get('alignment_strength', 0):.2f}")
    
    # Debug individual strategies
    print(f"\n🔍 STRATEGY BREAKDOWN:")
    
    # Trend Following
    trend_score = orchestrator._trend_following_strategy_enhanced(indicators)
    print(f"• Trend Following Score: {trend_score}")
    
    # Mean Reversion  
    mean_reversion_score = orchestrator._mean_reversion_strategy_enhanced(indicators)
    print(f"• Mean Reversion Score: {mean_reversion_score}")
    
    # Breakout
    breakout_score = orchestrator._breakout_strategy(indicators, mtf_signals)
    print(f"• Breakout Score: {breakout_score}")
    
    # ML Score
    ml_raw = ml_result.get('prediction', 0)
    ml_score = ml_raw * 50  # How ML score is calculated
    print(f"• ML Score: {ml_score}")
    
    # Calculate composite manually
    weights = orchestrator.strategy_weights
    composite_manual = (
        trend_score * weights['trend_following'] +
        mean_reversion_score * weights['mean_reversion'] + 
        breakout_score * weights['breakout'] +
        ml_score * weights['ml_prediction']
    )
    
    print(f"\n🧮 COMPOSITE CALCULATION:")
    print(f"• Trend ({trend_score}) × {weights['trend_following']} = {trend_score * weights['trend_following']:.1f}")
    print(f"• Mean Reversion ({mean_reversion_score}) × {weights['mean_reversion']} = {mean_reversion_score * weights['mean_reversion']:.1f}")
    print(f"• Breakout ({breakout_score}) × {weights['breakout']} = {breakout_score * weights['breakout']:.1f}") 
    print(f"• ML ({ml_score}) × {weights['ml_prediction']} = {ml_score * weights['ml_prediction']:.1f}")
    print(f"• TOTAL COMPOSITE: {composite_manual:.1f}")
    
    # Check what's causing negative score
    if composite_manual < 0:
        print(f"\n🚨 NEGATIVE COMPOSITE ANALYSIS:")
        negative_components = []
        if trend_score * weights['trend_following'] < 0:
            negative_components.append("Trend Following")
        if mean_reversion_score * weights['mean_reversion'] < 0:
            negative_components.append("Mean Reversion") 
        if breakout_score * weights['breakout'] < 0:
            negative_components.append("Breakout")
        if ml_score * weights['ml_prediction'] < 0:
            negative_components.append("ML")
            
        print(f"Negative components: {negative_components}")

def calculate_rsi(series, period):
    """Calculate RSI manually for debugging"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def test_strategy_sensitivity():
    """
    Test how sensitive each strategy is to clear signals
    """
    print(f"\n{'='*70}")
    print("🎛️  STRATEGY SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    
    bybit_client = BybitClient()
    
    # Test different market conditions
    test_cases = [
        ("STRONG BULL", create_strong_bull_market()),
        ("STRONG BEAR", create_strong_bear_market()), 
        ("HIGH VOLATILITY", create_high_volatility_market()),
        ("RANGING", create_ranging_market())
    ]
    
    for case_name, test_data in test_cases:
        print(f"\n📊 TEST CASE: {case_name}")
        orchestrator = EnhancedStrategyOrchestrator(bybit_client, aggressiveness="moderate")
        
        try:
            decision = orchestrator.analyze_symbol("TEST", test_data, 1000)
            print(f"   • Action: {decision.get('action', 'HOLD')}")
            print(f"   • Confidence: {decision.get('confidence', 0):.1f}%")
            print(f"   • Composite: {decision.get('composite_score', 0):.1f}")
            
            # Quick diagnosis
            composite = decision.get('composite_score', 0)
            if abs(composite) < 10:
                print(f"   • ISSUE: Weak signals even in {case_name}")
            elif composite > 0 and decision.get('action') == 'SELL':
                print(f"   • ISSUE: Positive composite but SELL action")
            elif composite < 0 and decision.get('action') == 'BUY':
                print(f"   • ISSUE: Negative composite but BUY action")
                
        except Exception as e:
            print(f"   • Error: {e}")

def create_strong_bull_market():
    """Create data that should trigger strong buy signals"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1h')
    
    # Very strong uptrend
    prices = [100 + i * 1.0 for i in range(200)]  # $1 per hour increase
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + 2 for p in prices],
        'low': [p - 1 for p in prices], 
        'close': prices,
        'volume': [10000] * 200  # High consistent volume
    })
    return df

def create_strong_bear_market():
    """Create data that should trigger strong sell signals"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1h')
    
    # Strong downtrend
    prices = [200 - i * 1.0 for i in range(200)]  # $1 per hour decrease
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 2 for p in prices],
        'close': prices, 
        'volume': [10000] * 200
    })
    return df

def create_high_volatility_market():
    """Create high volatility market data"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1h')
    
    base = 100
    prices = [base + np.random.normal(0, 5) for _ in range(200)]  # High volatility
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 2)) for p in prices],
        'low': [p - abs(np.random.normal(0, 2)) for p in prices],
        'close': prices,
        'volume': [5000 + np.random.randint(-1000, 1000) for _ in range(200)]
    })
    return df

def create_ranging_market():
    """Create sideways market data"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1h')
    
    base = 100
    prices = [base + np.sin(i * 0.1) * 5 for i in range(200)]  # Sine wave pattern
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 1 for p in prices],
        'close': prices,
        'volume': [3000 + np.random.randint(-500, 500) for _ in range(200)]
    })
    return df

def immediate_fix_test():
    """
    Test with some immediate fixes to the strategy logic
    """
    print(f"\n{'='*70}")
    print("🔧 IMMEDIATE FIX TEST")
    print(f"{'='*70}")
    
    # Create a patched orchestrator with fixed strategy weights
    class FixedOrchestrator(EnhancedStrategyOrchestrator):
        def _calculate_aggressive_composite_score(self, signals: Dict, ml_result: Dict, mtf_signals: Dict, aggressiveness: str) -> float:
            """Fixed composite score calculation"""
            try:
                # Use more balanced weights
                weights = {
                    'trend_following': 0.35,  # Increased from 0.25
                    'mean_reversion': 0.25,   # Decreased from 0.35  
                    'breakout': 0.25,         # Increased from 0.20
                    'ml_prediction': 0.15     # Decreased from 0.20
                }
                
                trend_score = self._trend_following_strategy_enhanced(signals)
                mean_reversion_score = self._mean_reversion_strategy_enhanced(signals)
                breakout_score = self._breakout_strategy(signals, mtf_signals)
                
                # Fix ML score calculation
                ml_raw = ml_result.get('prediction', 0)
                if ml_raw == 1:
                    ml_score = 40
                elif ml_raw == -1:
                    ml_score = -40
                else:
                    ml_score = 0
                
                mtf_score = mtf_signals.get('timeframe_alignment', 0) * 30 * mtf_signals.get('alignment_strength', 1)
                
                composite = (
                    trend_score * weights['trend_following'] +
                    mean_reversion_score * weights['mean_reversion'] +
                    breakout_score * weights['breakout'] +
                    ml_score * weights['ml_prediction'] +
                    mtf_score * 0.1
                )
                
                print(f"   [FIXED] Trend: {trend_score}, MeanRev: {mean_reversion_score}, Breakout: {breakout_score}, ML: {ml_score}")
                print(f"   [FIXED] Composite: {composite:.1f}")
                
                return composite
                
            except Exception as e:
                print(f"Error in fixed composite: {e}")
                return 0
    
    bybit_client = BybitClient()
    orchestrator = FixedOrchestrator(bybit_client, aggressiveness="moderate")
    test_data = create_clear_bullish_data()
    
    print("Testing with fixed composite calculation...")
    decision = orchestrator.analyze_symbol("FIXED_TEST", test_data, 1000)
    
    print(f"📊 FIXED RESULTS:")
    print(f"• Action: {decision.get('action', 'HOLD')}")
    print(f"• Confidence: {decision.get('confidence', 0):.1f}%")
    print(f"• Composite Score: {decision.get('composite_score', 0):.1f}")

if __name__ == "__main__":
    print("Starting Strategy Component Debug...")
    
    # Step 1: Debug individual components
    debug_strategy_components()
    
    # Step 2: Test different market conditions
    test_strategy_sensitivity()
    
    # Step 3: Test immediate fixes
    immediate_fix_test()
    
    print(f"\n{'='*70}")
    print("🎯 NEXT STEPS:")
    print("1. Look at which strategy components are negative")
    print("2. Check if mean reversion is fighting against trend following") 
    print("3. Verify ML predictions are working correctly")
    print("4. Adjust strategy weights and thresholds")