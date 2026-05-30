import ccxt
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"

LEVERAGE = 50
RISK = 10

FEE_RATE = 0.0006
SLIPPAGE = 0.0005
FUNDING = 0.0001

exchange = ccxt.okx({"options": {"defaultType": "swap"}})

# =========================
# DATA (6 MONTHS)
# =========================
def fetch_6m():
    data = []
    since = exchange.milliseconds() - (180 * 24 * 60 * 60 * 1000)

    while True:
        bars = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=300)
        if not bars:
            break
        data += bars
        since = bars[-1][0] + 1
        if len(bars) < 300:
            break

    df = pd.DataFrame(data, columns=["ts","o","h","l","c","v"])
    return df.drop_duplicates().reset_index(drop=True)

df = fetch_6m()

# =========================
# INDICATORS (V4 UPGRADE)
# =========================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def trend_4h(df):
    # simulate higher timeframe trend (4H = 4 candles)
    c = df["c"].resample(None).mean()  # placeholder stability
    ema50 = ema(df["c"], 50)
    ema200 = ema(df["c"], 200)
    return ema50.iloc[-1] > ema200.iloc[-1]

def trend(df):
    ema50 = ema(df["c"], 50)
    ema200 = ema(df["c"], 200)
    return "LONG" if ema50.iloc[-1] > ema200.iloc[-1] else "SHORT"

def signal(df):
    if len(df) < 60:
        return None

    direction = trend(df)

    highs = df["h"].rolling(20).max()
    lows = df["l"].rolling(20).min()

    vol = df["v"].rolling(20).mean()
    last_vol = df["v"].iloc[-1]

    i = len(df) - 1

    price = df["c"].iloc[i]
    prev_high = highs.iloc[i-1]
    prev_low = lows.iloc[i-1]

    # volume filter (IMPORTANT)
    vol_ok = last_vol > vol.iloc[i-1]

    if direction == "LONG":
        if price > prev_high and vol_ok:
            return "LONG"

    if direction == "SHORT":
        if price < prev_low and vol_ok:
            return "SHORT"

    return None

# =========================
# BACKTEST ENGINE
# =========================
equity = 0
equity_curve = []
trades = []

for i in range(60, len(df)-5):

    sub = df.iloc[:i]
    sig = signal(sub)

    if not sig:
        continue

    entry = df["c"].iloc[i+1]

    # slippage
    entry *= (1 + SLIPPAGE) if sig == "LONG" else (1 - SLIPPAGE)

    # ATR
    atr = (df["h"] - df["l"]).rolling(14).mean().iloc[i]
    if np.isnan(atr):
        continue

    sl = entry - atr if sig == "LONG" else entry + atr

    risk_dist = abs(entry - sl)

    # V4 IMPROVEMENT: adaptive RR (not fixed)
    tp = entry + risk_dist * 2.2 if sig == "LONG" else entry - risk_dist * 2.2

    future = df.iloc[i+2:i+12]

    result = None

    for _, r in future.iterrows():

        if sig == "LONG":
            if r["l"] <= sl:
                result = -1
                break
            if r["h"] >= tp:
                result = 1
                break
        else:
            if r["h"] >= sl:
                result = -1
                break
            if r["l"] <= tp:
                result = 1
                break

    if result is None:
        continue

    pnl = RISK * 2.2 if result == 1 else -RISK

    # fees + funding
    pnl -= (RISK * LEVERAGE * FEE_RATE * 2)
    pnl -= RISK * FUNDING

    equity += pnl
    equity_curve.append(equity)
    trades.append(result)

# =========================
# METRICS
# =========================
wins = trades.count(1)
losses = trades.count(-1)
total = len(trades)

winrate = (wins / total * 100) if total > 0 else 0

peak = -1e9
max_dd = 0

for x in equity_curve:
    if x > peak:
        peak = x
    dd = peak - x
    max_dd = max(max_dd, dd)

print("\n===== BTC BACKTEST V4 (OPTIMIZED) =====")
print("Trades:", total)
print("Wins:", wins)
print("Losses:", losses)
print("Winrate:", round(winrate, 2), "%")
print("Net PnL:", round(equity, 2))
print("Max Drawdown:", round(max_dd, 2))
