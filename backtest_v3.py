import ccxt
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"

RISK = 10
FEE = 0.0006
SLIPPAGE = 0.0005

exchange = ccxt.okx({"options": {"defaultType": "swap"}})

# =========================
# DATA (6 MONTHS)
# =========================
def fetch_data():
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

df = fetch_data()

# =========================
# LIQUIDITY SWEEP LOGIC
# =========================
def signal(df):
    if len(df) < 30:
        return None

    i = len(df) - 2

    prev_high = df["h"].iloc[i-20:i].max()
    prev_low = df["l"].iloc[i-20:i].min()

    current = df.iloc[i]

    # BUY SIDE LIQUIDITY SWEEP (false breakout up → short)
    if current["h"] > prev_high and current["c"] < prev_high:
        return "SHORT"

    # SELL SIDE LIQUIDITY SWEEP (false breakdown → long)
    if current["l"] < prev_low and current["c"] > prev_low:
        return "LONG"

    return None

# =========================
# BACKTEST
# =========================
equity = 0
equity_curve = []
trades = []

for i in range(30, len(df)-5):

    sub = df.iloc[:i]
    sig = signal(sub)

    if not sig:
        continue

    entry = df["c"].iloc[i+1]

    # SL / TP (structure based)
    lookback_high = df["h"].iloc[i-10:i].max()
    lookback_low = df["l"].iloc[i-10:i].min()

    if sig == "LONG":
        sl = lookback_low
        risk_dist = abs(entry - sl)
        tp = entry + risk_dist * 2
    else:
        sl = lookback_high
        risk_dist = abs(sl - entry)
        tp = entry - risk_dist * 2

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

    pnl = RISK * 2 if result == 1 else -RISK

    # fees + slippage
    pnl -= RISK * FEE * 2 * 50
    pnl -= RISK * SLIPPAGE

    equity += pnl
    equity_curve.append(equity)
    trades.append(result)

# =========================
# RESULTS
# =========================
wins = trades.count(1)
losses = trades.count(-1)
total = len(trades)

winrate = (wins / total * 100) if total else 0

peak = -1e9
max_dd = 0

for x in equity_curve:
    peak = max(peak, x)
    max_dd = max(max_dd, peak - x)

print("\n===== BTC V5 LIQUIDITY SWEEP =====")
print("Trades:", total)
print("Wins:", wins)
print("Losses:", losses)
print("Winrate:", round(winrate, 2), "%")
print("Net PnL:", round(equity, 2))
print("Max Drawdown:", round(max_dd, 2))
