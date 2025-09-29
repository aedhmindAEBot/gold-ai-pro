
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Gold AI Pro", page_icon="📈", layout="wide")

st.title("🚀 Gold AI Pro — Real-Time Signal & Backtesting")
st.markdown("Built by you and Qwen — powered by real gold data")

# --- UPLOAD CSV ---
uploaded_file = st.file_uploader("Upload your gold_data.csv", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload 'gold_data.csv' from Investing.com")
    st.stop()

df = pd.read_csv(uploaded_file)
if 'Date' not in df.columns or 'Price' not in df.columns:
    if 'Date' in df.columns and 'Close' in df.columns:
        df = df.rename(columns={'Close': 'Price'})
    elif 'date' in df.columns and 'close' in df.columns:
        df = df.rename(columns={'close': 'Price', 'date': 'Date'})
    else:
        st.error("Could not find Date and Price columns")
        st.stop()

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df = df.dropna(subset=['Price'])

if len(df) < 50:
    st.error(f"Only {len(df)} days available. Need at least 50.")
    st.stop()

# --- INDICATORS ---
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Price'])
df['MA20'] = df['Price'].rolling(20).mean()
df['MA50'] = df['Price'].rolling(50).mean()
df['Trend'] = np.where(df['MA20'] > df['MA50'], 1, 0)
df = df.dropna()

# --- CURRENT SIGNAL ---
latest = df.iloc[-1]
price = float(latest['Price'])
rsi = float(latest['RSI'])
trend = int(latest['Trend'])

signal = "🟢 STRONG BUY" if (trend == 1 and rsi < 70) else \
         "🔴 SELL" if rsi > 70 else "⚪ HOLD"

# --- BACKTESTING ---
def backtest_strategy(df):
    signals = []
    returns = []
    position = 0
    entry_price = 0
    
    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday = df.iloc[i-1]
        
        if yesterday['Trend'] == 1 and yesterday['RSI'] < 70 and position == 0:
            position = 1
            entry_price = today['Price']
            signals.append(('buy', today.name, today['Price']))
        elif yesterday['RSI'] > 70 and position == 1:
            position = 0
            exit_price = today['Price']
            returns.append((exit_price - entry_price) / entry_price)
            signals.append(('sell', today.name, today['Price']))
    
    if len(returns) == 0:
        return {'win_rate': 0, 'total_return': 0, 'trades': 0}, signals
    
    win_rate = np.mean(np.array(returns) > 0)
    total_return = np.sum(returns)
    
    return {
        'win_rate': win_rate,
        'total_return': total_return,
        'trades': len(returns),
        'avg_return': np.mean(returns)
    }, signals

stats, trade_signals = backtest_strategy(df)

# --- DISPLAY ---
col1, col2 = st.columns(2)
with col1:
    st.metric("📅 Latest Date", latest.name.strftime('%Y-%m-%d'))
    st.metric("💰 Gold Price", f"${price:,.2f}")
    st.metric("📊 Signal", signal)
    st.metric("📈 RSI", f"{rsi:.1f}")
    st.metric("📈 Trend", "Bullish" if trend else "Bearish")

with col2:
    st.metric(" Trades", stats['trades'])
    st.metric(" Win Rate", f"{stats['win_rate']:.1%}")
    st.metric(" Total Return", f"{stats['total_return']:.1%}")

# --- CHARTS ---
st.subheader("📈 Strategy Visualization")
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Gold Price & Signals', 'RSI'),
    row_heights=[0.7, 0.3]
)

# Price chart
fig.add_trace(go.Scatter(x=df.index, y=df['Price'], name='Gold', line=dict(color='gold')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='red')), row=1, col=1)

# Buy/Sell markers
buy_dates = [s[1] for s in trade_signals if s[0] == 'buy']
buy_prices = [s[2] for s in trade_signals if s[0] == 'buy']
sell_dates = [s[1] for s in trade_signals if s[0] == 'sell']
sell_prices = [s[2] for s in trade_signals if s[0] == 'sell']

if buy_dates:
    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                           marker=dict(color='green', size=8, symbol='triangle-up'),
                           name='Buy'), row=1, col=1)
if sell_dates:
    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers',
                           marker=dict(color='red', size=8, symbol='triangle-down'),
                           name='Sell'), row=1, col=1)

# RSI chart
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=600, title_text="Gold AI Pro — Strategy Visualization", showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# --- SAVE TO CSV ---
if st.button("💾 Save Results"):
    output_df = pd.DataFrame([{
        'Date': latest.name.strftime('%Y-%m-%d'),
        'Gold_Price': round(price, 2),
        'Signal': signal,
        'RSI': round(rsi, 1),
        'Win_Rate': round(stats['win_rate'], 3),
        'Total_Return': round(stats['total_return'], 3)
    }])
    output_df.to_csv('gold_ai_pro.csv', index=False)
    st.success("✅ Saved to 'gold_ai_pro.csv'")
