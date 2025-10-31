import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from trading_database import TradingDatabase, NpEncoder
from datetime import datetime
import time 
import json
import numpy as np

# --- NEW IMPORTS ---
from bybit_client import BybitClient
from config import BYBIT_CONFIG
# --- END NEW IMPORTS ---

# =============================================
# 🎨 CUSTOM STYLING & CONFIGURATION
# =============================================

def apply_custom_styling():
    """Apply custom CSS styling for a modern, professional look"""
    st.markdown("""
    <style>
    /* Main background and text colors */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #4CC9F0 0%, #4361EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem !important;
    }
    
    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #4CC9F0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #8898aa !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Copy button styling */
    .copy-button {
        background: linear-gradient(90deg, #4CC9F0 0%, #4361EE 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
    
    /* JSON container styling */
    .json-container {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(76, 201, 240, 0.3) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        max-height: 400px !important;
        overflow-y: auto !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        font-weight: 600 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
    }
    
    /* Divider styling */
    hr {
        border: 0.5px solid rgba(255, 255, 255, 0.1) !important;
        margin: 2rem 0 !important;
    }
    
    /* Custom cards */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-positive {
        color: #4CC9F0;
        font-weight: 600;
    }
    
    .status-negative {
        color: #F72585;
        font-weight: 600;
    }
    
    .status-neutral {
        color: #7209B7;
        font-weight: 600;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CC9F0 0%, #4361EE 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================
# 📊 DATA LOADING FUNCTIONS (Updated TTL to 10s)
# =============================================

@st.cache_data(ttl=10)
def load_pnl_data(_db):
    df = _db.get_performance_history(days=1)
    if df.empty:
        return pd.DataFrame(columns=['timestamp_dt', 'portfolio_value_num'])

    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['portfolio_value_num'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
    df = df.dropna(subset=['timestamp_dt', 'portfolio_value_num'])
    
    if df.empty:
        return pd.DataFrame(columns=['timestamp_dt', 'portfolio_value_num'])
        
    df_clean = df[['timestamp_dt', 'portfolio_value_num']]
    return df_clean.sort_values('timestamp_dt')

@st.cache_data(ttl=10)
def load_trade_stats(_db):
    bot_start_time = _db.get_first_trade_timestamp_iso()
    trades_df = _db.get_historical_trades(start_timestamp_iso=bot_start_time)
    closed_trades = trades_df[trades_df['pnl_percent'].notna()].copy()
    
    if closed_trades.empty:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    # Ensure pnl_usdt is numeric for profit factor calculation
    closed_trades['pnl_usdt'] = pd.to_numeric(closed_trades['pnl_usdt'], errors='coerce').fillna(0)
        
    wins = len(closed_trades[closed_trades['pnl_percent'] > 0])
    losses = len(closed_trades[closed_trades['pnl_percent'] < 0])
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    # Calculate average PnL
    avg_pnl = closed_trades['pnl_percent'].mean() if not closed_trades.empty else 0.0
    
    # --- START ADDED CALCULATIONS ---
    
    # 1. Profit Factor
    gross_profit = closed_trades[closed_trades['pnl_usdt'] > 0]['pnl_usdt'].sum()
    gross_loss = abs(closed_trades[closed_trades['pnl_usdt'] < 0]['pnl_usdt'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    # 2. Best / Worst Trade (using pnl_percent)
    best_trade_pnl = closed_trades['pnl_percent'].max() if not closed_trades.empty else 0.0
    worst_trade_pnl = closed_trades['pnl_percent'].min() if not closed_trades.empty else 0.0
    
    # --- END ADDED CALCULATIONS ---
    
    return wins, losses, win_rate, avg_pnl, profit_factor, best_trade_pnl, worst_trade_pnl

@st.cache_data(ttl=10)
def load_ml_history(_db, symbol: str = None):
    df_list = _db.get_ml_model_performance(symbol=symbol, limit=100)
    if not df_list:
        return pd.DataFrame(columns=['training_date', 'accuracy', 'rf_accuracy', 'gb_accuracy', 'symbol'])
        
    df = pd.DataFrame(df_list)
    df['training_date'] = pd.to_datetime(df['training_date'])
    df = df.sort_values('training_date')
    df = df.set_index('training_date')
    return df

@st.cache_data(ttl=10)
def get_available_symbols(_db):
    df_list = _db.get_ml_model_performance(limit=1000)
    if not df_list:
        return []
    df = pd.DataFrame(df_list)
    return sorted(df['symbol'].unique())

@st.cache_data(ttl=10)
def load_all_trades(_db):
    bot_start_time = _db.get_first_trade_timestamp_iso()

    # 1. Get all *successful* trades (open and closed) from the bot's perspective
    trades_df_all = _db.get_historical_trades(start_timestamp_iso=bot_start_time)
    trades_df = trades_df_all[trades_df_all['success'] == True].copy()
    
    # 2. Normalize and sort
    trades_df['type'] = 'BOT_TRADE'
    trades_df['sort_time'] = pd.to_datetime(trades_df['timestamp'])
    
    return trades_df.sort_values('sort_time', ascending=False)

@st.cache_data(ttl=10)
def load_live_positions_api() -> pd.DataFrame:
    """Fetches live position data directly from the Bybit API."""
    try:
        client = BybitClient()
        
        if not client.api_key or not client.api_secret:
            st.error("🔑 API keys are not configured. Cannot fetch live positions.")
            return pd.DataFrame()

        positions_response = client.get_position_info(category="linear", settleCoin="USDT")
        
        if positions_response and positions_response.get('retCode') == 0:
            positions_list = positions_response['result'].get('list', [])
            open_positions = [p for p in positions_list if float(p.get('size', 0)) > 0]
            
            if not open_positions:
                return pd.DataFrame()

            df = pd.DataFrame(open_positions)
            
            numeric_cols = ['size', 'avgPrice', 'positionValue', 'unrealisedPnl', 'liqPrice', 'leverage']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['unrealisedPnl', 'symbol'])
            return df.sort_values('unrealisedPnl', ascending=False)
        else:
            msg = positions_response.get('retMsg', 'Unknown API error')
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Failed to fetch live positions from API: {e}")
        return pd.DataFrame()

# =============================================
# 🎯 VISUALIZATION COMPONENTS
# =============================================

def display_pnl_chart(db):
    """Enhanced PnL chart with better visuals"""
    st.markdown("### 📈 Live Portfolio Performance")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        pnl_df = load_pnl_data(db)
        
        if pnl_df.empty:
            st.info("⏳ No portfolio data found yet. Waiting for the first bot cycle to complete.")
            st.metric("Current Portfolio Value", "$0.00")
            return
        
        current_value = pnl_df['portfolio_value_num'].iloc[-1]
        initial_value = pnl_df['portfolio_value_num'].iloc[0] if len(pnl_df) > 1 else current_value
        total_pnl = current_value - initial_value
        pnl_percent = ((current_value / initial_value) - 1) * 100 if initial_value > 0 else 0
        
        with col1:
            st.metric(
                "💰 Portfolio Value", 
                f"${current_value:,.2f}",
                f"{pnl_percent:+.2f}%"
            )
        
        with col2:
            st.metric(
                "📊 Total PnL", 
                f"${total_pnl:+,.2f}"
            )
        
        with col3:
            st.metric(
                "🔄 Data Points", 
                f"{len(pnl_df)}"
            )

        # Enhanced Chart
        if len(pnl_df) >= 2:
            fig = go.Figure()
            
            # Main area chart
            fig.add_trace(go.Scatter(
                x=pnl_df['timestamp_dt'], 
                y=pnl_df['portfolio_value_num'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#4CC9F0', width=4, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(76, 201, 240, 0.1)',
                hovertemplate='<b>%{x}</b><br>$%{y:,.2f}<extra></extra>'
            ))
            
            # Add markers for significant points
            if len(pnl_df) > 10:
                fig.add_trace(go.Scatter(
                    x=[pnl_df['timestamp_dt'].iloc[0], pnl_df['timestamp_dt'].iloc[-1]],
                    y=[pnl_df['portfolio_value_num'].iloc[0], pnl_df['portfolio_value_num'].iloc[-1]],
                    mode='markers',
                    marker=dict(size=12, color=['#F72585', '#4CC9F0']),
                    name='Start/End',
                    hovertemplate='<b>%{x}</b><br>$%{y:,.2f}<extra></extra>'
                ))

            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode='x unified',
                showlegend=False,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Collecting data points... Chart will appear with more data.")

def display_open_positions_api():
    """Enhanced live positions display"""
    st.markdown("### 💰 Live Open Positions")
    
    with st.container():
        positions_df = load_live_positions_api()
        
        if positions_df.empty:
            st.info("🤝 No open positions. Ready for new opportunities!")
            return
            
        total_pnl = positions_df['unrealisedPnl'].sum()
        total_value = positions_df['positionValue'].sum()
        total_leverage = positions_df['leverage'].mean() if 'leverage' in positions_df.columns else 0
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Positions", 
                f"{len(positions_df)}"
            )
        
        with col2:
            st.metric(
                "💰 Position Value", 
                f"${total_value:,.0f}"
            )
            
        with col3:
            st.metric(
                "🎯 Unrealized PnL", 
                f"${total_pnl:+,.2f}"
            )
            
        with col4:
            st.metric(
                "⚡ Avg Leverage", 
                f"{total_leverage:.1f}x"
            )

        st.markdown("---")
        
        # Individual positions
        st.markdown("#### 📋 Position Details")
        for _, pos in positions_df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                # Position header
                side_color = "🟢" if pos['side'] == 'Buy' else "🔴"
                pnl_emoji = "🚀" if pos['unrealisedPnl'] >= 0 else "🔻"
                pnl_color = "#4CC9F0" if pos['unrealisedPnl'] >= 0 else "#F72585"
                
                with col1:
                    st.markdown(f"**{side_color} {pos['symbol']}**")
                    st.caption(f"Side: {pos['side']} | Size: {pos['size']}")
                    
                with col2:
                    st.markdown(f"<span style='color: {pnl_color}; font-weight: 600;'>${pos['unrealisedPnl']:+.2f} {pnl_emoji}</span>", unsafe_allow_html=True)
                    st.caption(f"Value: ${pos['positionValue']:,.2f}")
                    
                with col3:
                    st.metric("Entry", f"${float(pos['avgPrice']):.4f}")
                    
                with col4:
                    leverage = pos.get('leverage', 0)
                    st.metric("Leverage", f"{leverage:.1f}x" if leverage else "N/A")
                
                # Progress-like bar for risk visualization
                if pos.get('liqPrice') and float(pos['liqPrice']) > 0:
                    current_price = float(pos['avgPrice'])  # Simplified
                    liq_price = float(pos['liqPrice'])
                    distance_to_liq = abs((current_price - liq_price) / current_price * 100)
                    
                    st.progress(min(100, distance_to_liq) / 100, 
                               text=f"Liquidation Distance: {distance_to_liq:.2f}%")
                
                st.markdown("---")

def display_performance_stats(db):
    """Enhanced performance statistics with better visuals"""
    st.markdown("### 📊 Trading Performance Analytics")
    
    wins, losses, win_rate, avg_pnl, profit_factor, best_trade, worst_trade = load_trade_stats(db)
    total_trades = wins + losses
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "✅ Total Wins", 
                f"{wins}",
                delta=f"{(wins/total_trades*100 if total_trades > 0 else 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "❌ Total Losses", 
                f"{losses}",
                delta=f"{(losses/total_trades*100 if total_trades > 0 else 0):.1f}%"
            )
            
        with col3:
            st.metric(
                "🎯 Win Rate", 
                f"{win_rate:.1f}%",
                delta=f"{win_rate - 50:+.1f}% vs 50%"
            )
            
        with col4:
            st.metric(
                "📈 Avg PnL %", 
                f"{avg_pnl:+.2f}%"
            )
        
        # Additional metrics in expanded view
        with st.expander("📈 Detailed Performance Metrics"):
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Total Trades", f"{total_trades}")
            with col6:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            with col7:
                st.metric("Best Trade", f"{best_trade:+.2f}%")
            with col8:
                st.metric("Worst Trade", f"{worst_trade:+.2f}%")

def display_ml_insights(db):
    """Enhanced ML model performance display"""
    st.markdown("### 🧠 AI Model Intelligence")
    
    symbols = get_available_symbols(db)
    if not symbols:
        st.info("🤖 No ML model training history found. Models are learning...")
        return
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol:",
            options=symbols,
            key="ml_symbol_select"
        )

    if selected_symbol:
        ml_df = load_ml_history(db, symbol=selected_symbol)
        
        if ml_df.empty:
            st.info(f"📊 No ML data for {selected_symbol} yet.")
            return
            
        with col2:
            # Current model performance
            latest = ml_df.iloc[-1]
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Latest Accuracy", f"{latest.get('accuracy', 0)*100:.1f}%")
            with col2_2:
                st.metric("RF Accuracy", f"{latest.get('rf_accuracy', 0)*100:.1f}%")
            with col2_3:
                st.metric("GB Accuracy", f"{latest.get('gb_accuracy', 0)*100:.1f}%")

        # Enhanced chart
        fig = go.Figure()
        
        colors = ['#4CC9F0', '#7209B7', '#F72585']
        models = ['accuracy', 'rf_accuracy', 'gb_accuracy']
        names = ['Ensemble', 'Random Forest', 'Gradient Boost']
        
        for i, (model, name) in enumerate(zip(models, names)):
            if model in ml_df.columns:
                fig.add_trace(go.Scatter(
                    x=ml_df.index,
                    y=ml_df[model] * 100,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i], width=3, shape='spline'),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{name}</b><br>%{{x}}<br>Accuracy: %{{y:.1f}}%<extra></extra>'
                ))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title=f"Model Accuracy Over Time - {selected_symbol}",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', title="Accuracy %"),
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_trade_intelligence(db):
    st.markdown("### 📋 Combined Trade & Closure History")
    
    combined_df = load_all_trades(db)

    if combined_df.empty:
        st.info("💼 No trades or closure events found. The bot is analyzing the markets...")
        return

    # Summary statistics (from closed bot trades only)
    closed_trades = combined_df[
        (combined_df['type'] == 'BOT_TRADE') & (combined_df['pnl_percent'].notna())
    ]
    
    if not closed_trades.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recent Win Rate", f"{(len(closed_trades[closed_trades['pnl_percent'] > 0]) / len(closed_trades) * 100):.1f}%")
        with col2:
            st.metric("Avg PnL %", f"{closed_trades['pnl_percent'].mean():.2f}%")
        with col3:
            st.metric("Total Volume", f"${closed_trades['position_size_usdt'].sum():.0f}")

    # Iterate and display combined list
    for _, row in combined_df.iterrows():
        st.markdown("---")
        if row['type'] == 'BOT_TRADE':
            # --- Render a Bot Trade Card ---
            with st.container():
                pnl = row.get('pnl_percent', np.nan)
                success = row.get('success', False)

                if pd.notna(pnl):
                    status = "CLOSED"
                    pnl_str = f"{pnl:+.2f}%"
                    color = "#4CC9F0" if pnl > 0 else "#F72585"
                    status_emoji = "✅" if pnl > 0 else "🔻"
                    badge_color = "🟢" if pnl > 0 else "🔴"
                elif success:
                    status = "LIVE"
                    pnl_str = f"${row.get('position_size_usdt', 0):.0f}"
                    color = "#7209B7"
                    status_emoji = "⚡"
                    badge_color = "🔵"
                else:
                    status = "FAILED"
                    pnl_str = row.get('error_message', 'N/A')[:30]
                    color = "#F72585"
                    status_emoji = "❌"
                    badge_color = "🔴"

                action_emoji = "🟢" if row['action'] == 'BUY' else "🔴"
                timestamp = pd.to_datetime(row['timestamp']).strftime('%m/%d %H:%M:%S')
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{badge_color} {row['symbol']} (BOT TRADE)**")
                    st.caption(f"{action_emoji} {row['action']} • {timestamp}")
                    
                with col2:
                    st.markdown(f"<span style='color: {color}; font-weight: 600;'>{status_emoji} {pnl_str}</span>", unsafe_allow_html=True)
                    st.caption(status)
                    
                with col3:
                    st.metric("Size", f"${row.get('position_size_usdt', 0):.0f}")
                    
                with col4:
                    confidence = row.get('confidence', 0)
                    confidence_color = "#4CC9F0" if confidence > 70 else "#7209B7" if confidence > 50 else "#F72585"
                    st.markdown(f"<span style='color: {confidence_color}; font-weight: 600;'>Conf: {confidence:.0f}%</span>", unsafe_allow_html=True)

                # Expandable details
                with st.expander("🔍 View Trade Analysis", expanded=False):
                    display_trade_analysis(row)

def display_trade_analysis(trade_data):
    """Display detailed analysis for a single trade with ALL raw metrics including technical indicators"""
    st.markdown("---") # Add divider inside the expander
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🎯 Strategy Scores")
        scores = [
            ("Trend Score", trade_data.get('trend_score', 0)),
            ("Mean Reversion", trade_data.get('mr_score', 0)),
            ("Breakout Score", trade_data.get('breakout_score', 0)),
            ("ML Score", trade_data.get('ml_score', 0)),
            ("MTF Score", trade_data.get('mtf_score', 0)),
        ]
        
        for name, score in scores:
            if pd.notna(score):
                progress = min(score / 100, 1.0) if score > 0 else 0
                st.progress(progress, text=f"{name}: {score:.1f}")
    
    with col2:
        st.markdown("##### ⚡ Risk Management")
        risk_cols = st.columns(2)
        with risk_cols[0]:
            st.metric("Stop Loss", f"${trade_data.get('stop_loss', 0):.4f}")
            st.metric("Position Size", f"${trade_data.get('position_size_usdt', 0):.2f}")
        with risk_cols[1]:
            st.metric("Take Profit", f"${trade_data.get('take_profit', 0):.4f}")
            st.metric("R/R Ratio", f"{trade_data.get('risk_reward_ratio', 0):.1f}:1")
    
    st.markdown("##### 📊 Final Decision Metrics")
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    with comp_col1:
        st.metric("Composite Score", f"{trade_data.get('composite_score', 0):.2f}")
    with comp_col2:
        st.metric("Confidence", f"{trade_data.get('confidence', 0):.1f}%")
    with comp_col3:
        # --- closure_event_id display removed ---
        pass

    st.markdown("##### 🔍 Data Availability")
    tech_fields_present = []
    ml_fields_present = []
    
    for key in trade_data.keys():
        if 'technical' in key.lower() or 'indicator' in key.lower():
            tech_fields_present.append(key)
        if 'ml' in key.lower() or 'prediction' in key.lower():
            ml_fields_present.append(key)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Technical Fields Found:", tech_fields_present if tech_fields_present else "None")
    with col2:
        st.write("ML Fields Found:", ml_fields_present if ml_fields_present else "None")
    
    st.markdown("##### 📋 Complete Raw Trade Metrics (All Data)")
    
    trade_metrics = {}
    
    for key, value in trade_data.items():
        trade_metrics[key] = value
    
    json_fields_to_parse = ['technical_indicators_json', 'ml_prediction_details']
    
    for json_field in json_fields_to_parse:
        if json_field in trade_data and trade_data[json_field]:
            try:
                parsed_data = json.loads(trade_data[json_field])
                trade_metrics[f"{json_field}_parsed"] = parsed_data
                st.success(f"✅ Successfully parsed {json_field}")
            except (json.JSONDecodeError, TypeError) as e:
                st.warning(f"⚠️ Could not parse {json_field}: {e}")
                if isinstance(trade_data[json_field], str):
                    trade_metrics[f"{json_field}_raw"] = trade_data[json_field]
    
    try:
        json_str = json.dumps(trade_metrics, indent=2, default=str)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**All Raw Trade Data (JSON):**")
        
        with col2:
            if st.button("📋 Copy JSON", key=f"copy_{trade_data.get('symbol', 'unknown')}_{trade_data.get('timestamp', 'unknown')}"):
                st.text_area("Copy this JSON:", json_str, height=300, key=f"json_copy_{trade_data.get('symbol', 'unknown')}")
                st.success("✅ Select and copy the JSON above!")
        
        st.code(json_str, language='json')
        
    except Exception as e:
        st.error(f"Error creating JSON: {e}")
        st.write("Raw data structure:")
        st.write(trade_metrics)
    
    if 'technical_indicators_json_parsed' in trade_metrics:
        with st.expander("📊 Technical Indicators (Parsed)", expanded=True):
            tech_data = trade_metrics['technical_indicators_json_parsed']
            
            if isinstance(tech_data, dict):
                st.json(tech_data)
                
                if 'rsi' in tech_data:
                    st.metric("RSI", f"{tech_data.get('rsi', 0):.2f}")
                if 'macd' in tech_data:
                    st.metric("MACD", f"{tech_data.get('macd', 0):.4f}")
                if 'volume' in tech_data:
                    st.metric("Volume", f"{tech_data.get('volume', 0):.0f}")
            else:
                st.write(tech_data)
    
    if 'ml_prediction_details_parsed' in trade_metrics:
        with st.expander("🤖 ML Prediction Details (Parsed)", expanded=True):
            ml_data = trade_metrics['ml_prediction_details_parsed']
            st.json(ml_data)
    
    if 'technical_indicators_json_raw' in trade_metrics:
        with st.expander("📊 Technical Indicators (Raw JSON String)"):
            st.text(trade_metrics['technical_indicators_json_raw'])
    
    if 'ml_prediction_details_raw' in trade_metrics:
        with st.expander("🤖 ML Prediction Details (Raw JSON String)"):
            st.text(trade_metrics['ml_prediction_details_raw'])

# =============================================
# 🚀 MAIN APPLICATION
# =============================================

def main():
    # Configure page (MUST be the first Streamlit command)
    st.set_page_config(
        page_title="Quantum Trading Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styling
    apply_custom_styling()

    # Header with status
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🚀 Quantum Trading Dashboard")
        st.markdown("### *AI-Powered Market Intelligence*")
    with col2:
        st.metric("System Status", "🟢 LIVE")
    with col3:
        st.metric("Refresh Rate", "10s")

    st.markdown("---")

    try:
        db = TradingDatabase(db_path="trading_data.db")
    except Exception as e:
        st.error("❌ **CRITICAL:** Failed to connect to trading database")
        st.error(f"Ensure the database file exists and the bot is running: `{e}`")
        return

    # Main dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Overview", "💰 Positions", "🧠 AI Models", "📋 Trade History"])
    
    with tab1:
        display_pnl_chart(db)
        display_performance_stats(db)
        
    with tab2:
        display_open_positions_api()
        
    with tab3:
        display_ml_insights(db)
        
    with tab4:
        display_trade_intelligence(db)

    # Footer
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.markdown("🛡️ *Quantum Trading System • Real-time AI Analysis • Professional Grade*")
    with footer_col2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    # Auto-refresh every 10 seconds
    time.sleep(10)
    st.rerun()

if __name__ == "__main__":
    main()