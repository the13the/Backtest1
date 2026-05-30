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

exchange = ccxt.okx({
    "options": {"defaultType": "swap"}
})

# =========================
# 6 MONTH DATA FETCH
# =========================
def fetch_6_months():
    ohlcv = []
    since = exchange.milliseconds() - (180 * 24 * 60 * 60 * 1000)

    while True:
        bars = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=300
        )

        if not bars:
            break

        ohlcv += bars
        since = bars[-1][0] + 1

        if len(bars) < 300:
            break

    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    return df

df = fetch_6_months()
df = df.drop_duplicates().reset_index(drop=True)

# =========================
# INDICATORS
# =========================
def trend(df):
    ma50 = df["c"].rolling(10).mean()
    ma200 = df["c"].rolling(20).mean()
    return "LONG" if ma50.iloc[-1] > ma200.iloc[-1] else "SHORT"


def signal(df):
    if len(df) < 30:
        return None

    direction = trend(df)

    highs = df["h"].rolling(10).max()
    lows = df["l"].rolling(10).min()

    i = len(df) - 1

    if direction == "LONG":
        if df["c"].iloc[i] > highs.iloc[i-1]:
            return "LONG"

    if direction == "SHORT":
        if df["c"].iloc[i] < lows.iloc[i-1]:
            return "SHORT"

    return None


# =========================
# BACKTEST ENGINE
# =========================
equity = 0
equity_curve = []
trades = []

for i in range(30, len(df) - 5):

    sub = df.iloc[:i]
    sig = signal(sub)

    if not sig:
        continue

    entry = df["c"].iloc[i+1]

    # slippage
    if sig == "LONG":
        entry *= (1 + SLIPPAGE)
    else:
        entry *= (1 - SLIPPAGE)

    atr = (df["h"] - df["l"]).rolling(14).mean().iloc[i]
    if np.isnan(atr):
        continue

    sl = entry - atr if sig == "LONG" else entry + atr
    risk_dist = abs(entry - sl)
    tp = entry + risk_dist * 1.8 if sig == "LONG" else entry - risk_dist * 1.8

    future = df.iloc[i+2:i+10]

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

    # PnL
    if result == 1:
        pnl = RISK * 1.8
    else:
        pnl = -RISK

    # fees
    pnl -= (RISK * LEVERAGE * FEE_RATE * 2)

    # funding
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
    if dd > max_dd:
        max_dd = dd

print("\n===== BTC BACKTEST V3 (6 MONTH) =====")
print("Symbol:", SYMBOL)
print("Timeframe:", TIMEFRAME)
print("-----------------------------------")
print("Trades:", total)
print("Wins:", wins)
print("Losses:", losses)
print("Winrate:", round(winrate, 2), "%")
print("Net PnL ($):", round(equity, 2))
print("Max Drawdown ($):", round(max_dd, 2))
