# app.py
import socket, ssl, time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import streamlit as st

# ---------------- CONFIG ----------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "INTC", "JPM", "LUPIN.NS",
    "HCLTECH.NS", "SBIN.NS", "V"
]
HISTORY_PERIOD   = "1y"
YFINANCE_INTERVAL = "1d"

# ---------------- NETWORK PROBE ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_reachable(host: str = "query2.finance.yahoo.com") -> bool:
    try:
        ip = socket.gethostbyname(host)
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
            s.settimeout(3)
            s.connect((ip, 443))
        return True
    except Exception:
        return False

YAHOO_OK = yahoo_reachable()

# ---------------- DATA LAYER ----------------
def _download_yahoo(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period,
                                   interval=interval,
                                   auto_adjust=False)
    if df.empty:
        df = yf.download(ticker, period=period,
                         interval=interval, progress=False)
    return df

def _download_stooq(ticker: str, period: str) -> pd.DataFrame:
    # Stooq only supports daily data; adjust ticker suffix for US (.US)
    if "." not in ticker:
        ticker = f"{ticker}.US"
    try:
        df = pdr.DataReader(ticker, "stooq",
                            start=pd.Timestamp.utcnow() - pd.Timedelta(period))
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_data(tickers, period=HISTORY_PERIOD, interval=YFINANCE_INTERVAL):
    out = {}
    for t in tickers:
        try:
            if YAHOO_OK:
                df = _download_yahoo(t, period, interval)
            else:
                df = pd.DataFrame()  # force fallback
            if df.empty:
                df = _download_stooq(t, period)
            if df.empty:
                st.warning(f"No data for {t}")
                continue
            df = df.rename(columns=str.capitalize)
            df.index.name = "Datetime"
            out[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            out[t].attrs["fetched_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            st.error(f"Error fetching {t}: {e}")
    return out

# ---------------- INDICATORS ----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Close"]).copy()

    df["SMA20"] = SMAIndicator(df["Close"], 20, True).sma_indicator()
    df["SMA50"] = SMAIndicator(df["Close"], 50, True).sma_indicator()
    df["EMA12"] = EMAIndicator(df["Close"], 12, True).ema_indicator()
    df["EMA26"] = EMAIndicator(df["Close"], 26, True).ema_indicator()
    df["EMA20"] = EMAIndicator(df["Close"], 20, True).ema_indicator()
    df["EMA34"] = EMAIndicator(df["Close"], 34, True).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Close"], 50, True).ema_indicator()

    macd_obj      = MACD(df["Close"], 26, 12, 9)
    df["MACD"]    = macd_obj.macd()
    df["MACD_signal"] = macd_obj.macd_signal()
    df["MACD_hist"]   = macd_obj.macd_diff()

    df["RSI14"] = RSIIndicator(df["Close"], 14, True).rsi()

    bb = BollingerBands(df["Close"], 20, 2, True)
    df["BB_H"]    = bb.bollinger_hband()
    df["BB_L"]    = bb.bollinger_lband()
    df["BB_MAVG"] = bb.bollinger_mavg()

    df["SMA20_gt_SMA50"]          = (df["SMA20"] > df["SMA50"]).astype(int)
    df["Close_gt_BB_H"]           = (df["Close"] > df["BB_H"]).astype(int)
    df["Close_lt_BB_L"]           = (df["Close"] < df["BB_L"]).astype(int)
    df["EMA20_lt_EMA34_lt_EMA50"] = ((df["EMA50"] > df["EMA34"]) &
                                     (df["EMA34"] > df["EMA20"])).astype(int)

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"]  = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["MVWAP"] = df["VWAP"].rolling(20, min_periods=1).mean()

    return df

def summarize_latest(df: pd.DataFrame, ticker: str) -> dict:
    last = df.iloc[-1]
    return {
        "ticker": ticker,
        "timestamp": str(last.name),
        "close":   float(last["Close"]),
        "open":    float(last["Open"]),
        "high":    float(last["High"]),
        "low":     float(last["Low"]),
        "volume":  int(last["Volume"]),
        "SMA20":   float(last["SMA20"]),
        "SMA50":   float(last["SMA50"]),
        "EMA12":   float(last["EMA12"]),
        "EMA26":   float(last["EMA26"]),
        "EMA20":   float(last["EMA20"]),
        "EMA34":   float(last["EMA34"]),
        "EMA50":   float(last["EMA50"]),
        "MACD":    float(last["MACD"]),
        "MACD_signal": float(last["MACD_signal"]),
        "MACD_hist":   float(last["MACD_hist"]),
        "RSI14":   float(last["RSI14"]),
        "BB_H":    float(last["BB_H"]),
        "BB_MAVG": float(last["BB_MAVG"]),
        "BB_L":    float(last["BB_L"]),
        "SMA20_gt_SMA50": int(last["SMA20_gt_SMA50"]),
        "Close_gt_BB_H":  int(last["Close_gt_BB_H"]),
        "Close_lt_BB_L":  int(last["Close_lt_BB_L"]),
        "EMA20_lt_EMA34_lt_EMA50": int(last["EMA20_lt_EMA34_lt_EMA50"]),
        "VWAP":  float(last["VWAP"]),
        "MVWAP": float(last["MVWAP"]),
    }

def recommendation(ind_df: pd.DataFrame) -> str:
    if ind_df.empty:
        return "N/A"
    last = ind_df.iloc[-1]
    ema_ok = last["Close"] < last["EMA20"] < last["EMA34"] < last["EMA50"]

    macd, macd_sig = float(last["MACD"]), float(last["MACD_signal"])
    if macd > macd_sig:
        macd_score = 2 if macd > 0 else 1
    elif macd < macd_sig:
        macd_score = -2 if macd < 0 else -1
    else:
        macd_score = 0

    vwap_score = 1 if last["Close"] > last["VWAP"] else 0
    bb_score   = 1 if last["Close"] < last["BB_L"] else 0

    score = (3 if ema_ok else 0) + macd_score + vwap_score + bb_score
    if score >= 4:
        return "BUY"
    if score >= 1:
        return "HOLD"
    return "SELL"

def refresh_all(tickers):
    st.toast(f"Fetching data for {len(tickers)} tickers…", icon="🔄")
    start = time.time()
    fetched = fetch_data(tickers)
    records = []
    for t, df in fetched.items():
        try:
            ind = compute_indicators(df)
            rec  = recommendation(ind)
            snap = summarize_latest(ind, t)
            snap["rec"] = rec
            records.append(snap)
        except Exception as e:
            st.error(f"{t}: indicator error – {e}")

    if not records:
        st.warning("No summaries produced.")
        return pd.DataFrame()

    out = pd.DataFrame(records).set_index("ticker").sort_index()
    st.success(f"Done in {time.time()-start:.1f}s – {len(out)} tickers")
    return out

# ---------------- STREAMLIT UI ----------------
st.title("🧐 Simple Stock Screener")

with st.sidebar:
    st.markdown("### Settings")
    sel = st.multiselect("Standard tickers", TICKERS, default=TICKERS)
    extra = st.text_input("Extra tickers (comma / space separated)")
    if extra:
        extra_list = [s.strip().upper() for s in extra.replace(",", " ").split()]
        sel = sorted(set(sel) | set(extra_list))
    go = st.button("🔄  Refresh now")

if not YAHOO_OK:
    st.info("Yahoo endpoint not reachable from this server – using Stooq fallback.")

snapshot = pd.DataFrame()
if go and sel:
    snapshot = refresh_all(sel)

if not snapshot.empty:
    st.subheader("Latest snapshot")
    view_cols = ["rec", "close", "volume", "VWAP", "MACD",
                 "RSI14", "EMA20", "EMA50"]
    st.dataframe(snapshot[view_cols])

    filt = st.text_input("Filter by ticker (e.g. AAPL)").upper().strip()
    if filt:
        if filt in snapshot.index:
            st.write(snapshot.loc[[filt]])
        else:
            st.warning(f"{filt} not in table.")

    st.download_button("Download CSV", snapshot.to_csv().encode(),
                       file_name="screener_snapshot.csv")
else:
    st.info("Press “Refresh now” to pull data.")
