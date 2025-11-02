# stock_screener_streamlit.py
import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import streamlit as st



# -------------- CONFIG --------------


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "INTC", "JPM", "LUPIN.NS", "HCLTECH.NS", "SBIN.NS", "V"
]

HISTORY_PERIOD = "1y"
YFINANCE_INTERVAL = "1d"

# -------------- HELPERS / INDICATORS --------------

def fetch_yahoo_data(tickers, period=HISTORY_PERIOD, interval=YFINANCE_INTERVAL):
    out = {}
    for t in tickers:
        try:
            yf_t = yf.Ticker(t)
            df = yf_t.history(period=period, interval=interval, auto_adjust=False)
            if df.empty:
                df = yf.download(t, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                st.warning(f"No data for {t}")
                continue
            df = df.rename(columns=lambda c: c.capitalize())
            df.index.name = "Datetime"
            out[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            out[t].attrs["fetched_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            st.error(f"Error fetching {t}: {e}")
    return out

def compute_indicators(df):
    df = df.copy()
    df = df.dropna(subset=["Close"]).copy()
    
    sma20 = SMAIndicator(df["Close"], window=20, fillna=True).sma_indicator()
    sma50 = SMAIndicator(df["Close"], window=50, fillna=True).sma_indicator()
    ema12 = EMAIndicator(df["Close"], window=12, fillna=True).ema_indicator()
    ema26 = EMAIndicator(df["Close"], window=26, fillna=True).ema_indicator()
    ema20 = EMAIndicator(df["Close"], window=20, fillna=True).ema_indicator()
    ema34 = EMAIndicator(df["Close"], window=34, fillna=True).ema_indicator()
    ema50 = EMAIndicator(df["Close"], window=50, fillna=True).ema_indicator()
    
    macd_obj = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    macd = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    macd_hist = macd_obj.macd_diff()
    
    rsi14 = RSIIndicator(df["Close"], window=14, fillna=True).rsi()
    
    bb = BollingerBands(df["Close"], window=20, window_dev=2, fillna=True)
    bb_h = bb.bollinger_hband()
    bb_l = bb.bollinger_lband()
    bb_mavg = bb.bollinger_mavg()

    # Assign all indicators
    df["SMA20"] = sma20
    df["SMA50"] = sma50
    df["EMA12"] = ema12
    df["EMA26"] = ema26
    df["EMA20"] = ema20
    df["EMA34"] = ema34
    df["EMA50"] = ema50
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    df["RSI14"] = rsi14
    df["BB_H"] = bb_h
    df["BB_MAVG"] = bb_mavg
    df["BB_L"] = bb_l

    df["SMA20_gt_SMA50"] = (df["SMA20"] > df["SMA50"]).astype(int)
    df["Close_gt_BB_H"] = (df["Close"] > df["BB_H"]).astype(int)
    df["Close_lt_BB_L"] = (df["Close"] < df["BB_L"]).astype(int)
    
    # EMA ordering rule check (used in recommendation logic)
    df["EMA20_lt_EMA34_lt_EMA50"] = ((df["EMA50"] > df["EMA34"]) & (df["EMA34"] > df["EMA20"])).astype(int)
    
    # VWAP calculation
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["MVWAP"] = df["VWAP"].rolling(window=20, min_periods=1).mean()
    
    return df

def summarize_latest(df, ticker):
    last = df.iloc[-1].copy()
    result = {
        "ticker": ticker,
        "timestamp": last.name.isoformat() if hasattr(last.name, "isoformat") else str(last.name),
        "close": float(last["Close"]),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "volume": int(last["Volume"]),
        "SMA20": float(last.get("SMA20", np.nan)),
        "SMA50": float(last.get("SMA50", np.nan)),
        "EMA12": float(last.get("EMA12", np.nan)),
        "EMA26": float(last.get("EMA26", np.nan)),
        "EMA20": float(last.get("EMA20", np.nan)),
        "EMA34": float(last.get("EMA34", np.nan)),
        "EMA50": float(last.get("EMA50", np.nan)),
        "MACD": float(last.get("MACD", np.nan)),
        "MACD_signal": float(last.get("MACD_signal", np.nan)),
        "MACD_hist": float(last.get("MACD_hist", np.nan)),
        "RSI14": float(last.get("RSI14", np.nan)),
        "BB_H": float(last.get("BB_H", np.nan)),
        "BB_MAVG": float(last.get("BB_MAVG", np.nan)),
        "BB_L": float(last.get("BB_L", np.nan)),
        "SMA20_gt_SMA50": int(last.get("SMA20_gt_SMA50", 0)),
        "Close_gt_BB_H": int(last.get("Close_gt_BB_H", 0)),
        "Close_lt_BB_L": int(last.get("Close_lt_BB_L", 0)),
        "EMA20_lt_EMA34_lt_EMA50": int(last.get("EMA20_lt_EMA34_lt_EMA50", 0)),
        "VWAP": float(last.get("VWAP", np.nan)),
        "MVWAP": float(last.get("MVWAP", np.nan)),
    }
    return result

def compute_recommendation_from_indicators(ind_df):
    if ind_df is None or ind_df.empty:
        return "N/A"

    last = ind_df.iloc[-1]

    # 1) EMA Bullish Rule Check: If Close < EMA20 < EMA34 < EMA50
    # Note: The user requested: If Close price < EMA20 < EMA34 < EMA50 then Bullish
    ema_bull_condition = (last["Close"] < last["EMA20"]) and \
                         (last["EMA20"] < last["EMA34"]) and \
                         (last["EMA34"] < last["EMA50"])

    # 2) MACD interpretation (based on image reference)
    macd = float(last["MACD"])
    macd_sig = float(last["MACD_signal"])
    if macd > macd_sig:
        macd_trend = "Strong Buy" if macd > 0 else "Weak Buy"
    elif macd < macd_sig:
        macd_trend = "Strong Sell" if macd < 0 else "Weak Sell"
    else:
        macd_trend = "Neutral"

    # 3) VWAP relation (based on image reference)
    close = float(last["Close"])
    vwap = float(last["VWAP"]) if not np.isnan(last["VWAP"]) else np.nan
    if not np.isnan(vwap):
        vwap_rel = "Bullish" if close > vwap else "Bearish"
    else:
        vwap_rel = "Neutral"

    # 4) BB oversold/overbought (based on image reference)
    bb_h = float(last["BB_H"])
    bb_l = float(last["BB_L"])
    if not (np.isnan(bb_h) or np.isnan(bb_l)):
        if close < bb_l:
            bb_signal = "Oversold"
        elif close > bb_h:
            bb_signal = "Overbought"
        else:
            bb_signal = "Neutral"
    else:
        bb_signal = "Neutral"

    # Simple scoring system (Tune these weights if needed)
    score = 0
    if ema_bull_condition:
        score += 3  # Giving the user's explicit rule high weight
    if macd_trend in ["Strong Buy", "Weak Buy"]:
        score += 2
    if vwap_rel == "Bullish":
        score += 1
    if bb_signal == "Oversold":
        score += 1
    
    # Decision thresholds
    if score >= 5:
        rec = "BUY"
    elif score >= 2:
        rec = "HOLD"
    else:
        rec = "SELL"

    return rec

def refresh_all(tickers_list):
    st.info(f"Fetching data for {len(tickers_list)} unique tickers...")
    start = time.time()
    fetched = fetch_yahoo_data(tickers_list)
    summaries = []
    for tick, df in fetched.items():
        try:
            ind_df = compute_indicators(df)
            summary = summarize_latest(ind_df, tick)
            # Compute derived recommendation
            rec = compute_recommendation_from_indicators(ind_df)
            summary["rec"] = rec
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error computing indicators for {tick}: {e}")

    if summaries:
        df_snap = pd.DataFrame(summaries).set_index("ticker")
        df_snap.sort_index(inplace=True)
        
        elapsed = time.time() - start
        st.success(f"Refresh finished in {elapsed:.1f}s, {len(df_snap)} tickers processed.")
        return df_snap
    else:
        st.warning("No summaries produced.")
        return pd.DataFrame()

# -------------- STREAMLIT APP --------------

st.title("Simple Stock Screener")

st.sidebar.header("Settings")

# --- 1. Predefined/Selected Tickers ---
selected_tickers = st.sidebar.multiselect(
    "Select standard tickers", 
    TICKERS, 
    default=TICKERS
)

# --- 2. Custom Ticker Input ---
custom_input = st.sidebar.text_input(
    "Additional Tickers (Comma/Space Separated)", 
    value="",
    help="Enter tickers like TSLA, XYZ, 123.NS, etc."
)

# --- 3. Refresh Button Logic ---
refresh_button = st.sidebar.button("Refresh Data")

# Load existing snapshot if available

snapshot_df = pd.DataFrame()

if refresh_button:
    # Process custom input
    custom_tickers = []
    if custom_input:
        # Split by common delimiters (comma or space) and clean up strings
        raw_list = custom_input.replace(',', ' ').split()
        custom_tickers = [t.strip().upper() for t in raw_list if t.strip()]
        
    # Combine predefined and custom, ensuring uniqueness
    tickers_to_fetch_set = set(selected_tickers) | set(custom_tickers)
    tickers_to_fetch = sorted(list(tickers_to_fetch_set))
    
    if tickers_to_fetch:
        snapshot_df = refresh_all(tickers_to_fetch)
    else:
        st.warning("Please select tickers or enter custom tickers to refresh.")

if not snapshot_df.empty:
    st.subheader("Latest Screener Snapshot")
    
    # Display columns, ensuring 'rec' is visible
    cols_to_show = ['rec', 'close', 'volume', 'VWAP', 'MACD', 'RSI14', 'EMA20', 'EMA50']
    # Ensure all requested columns exist before trying to display them
    existing_cols = [c for c in cols_to_show if c in snapshot_df.columns]
    
    st.dataframe(snapshot_df[existing_cols])

    # Filter by ticker input
    ticker_filter = st.text_input("Filter by ticker (leave empty to show all)").upper().strip()
    if ticker_filter:
        if ticker_filter in snapshot_df.index:
            st.write(f"--- Details for {ticker_filter} ---")
            st.write(snapshot_df.loc[[ticker_filter]])
        else:
            st.warning(f"Ticker {ticker_filter} not found in snapshot.")

    # Download CSV
    csv_bytes = snapshot_df.to_csv().encode()
    st.download_button(
        label="Download CSV snapshot",
        data=csv_bytes,
        
       
    )
else:
    st.info("No snapshot data available. Click 'Refresh Data' to fetch latest data.")