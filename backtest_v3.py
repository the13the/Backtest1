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

FEE_RATE = 0.0006       # taker fee
SLIPPAGE = 0.0005       # market slippage
FUNDING = 0.0001        # simplified funding

# =========================
# EXCHANGE + DATA
# =========================
exchange = ccxt.okx({
    "options": {"defaultType": "swap"}
})

bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)

df = pd.DataFrame(bars, columns=["ts","o","h","l","c","v"])
df = df.tail(48).reset_index(drop=True)  # last ~2 days

# =========================
# INDICATORS
# =========================
def trend(df):
    ma50 = df["c"].rolling(10).mean()
    ma200 = df["c"].rolling(20).mean()
    return "LONG" if ma50.iloc[-1] > ma200.iloc[-1] else "SHORT"


def signal(df):
    if len(df) < 20:
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
# BACKTEST
# =========================
equity = 0
equity_curve = []
trades = []

for i in range(20, len(df) - 5):

    sub = df.iloc[:i]
    sig = signal(sub)

    if not sig:
        continue

    # ENTRY = NEXT CANDLE (REALISTIC FIX)
    entry = df["c"].iloc[i+1]

    if sig == "LONG":
        entry *= (1 + SLIPPAGE)
    else:
        entry *= (1 - SLIPPAGE)

    # ATR STOP
    atr = (df["h"] - df["l"]).rolling(14).mean().iloc[i]
    if np.isnan(atr):
        continue

    sl = entry - atr if sig == "LONG" else entry + atr
    risk_dist = abs(entry - sl)
    tp = entry + risk_dist * 1.8 if sig == "LONG" else entry - risk_dist * 1.8

    # FUTURE PRICE ACTION
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

    # FEES (entry + exit * leverage effect)
    pnl -= (RISK * LEVERAGE * FEE_RATE * 2)

    # FUNDING COST
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

print("\n===== REAL BTC BACKTEST V3 =====")
print("Symbol:", SYMBOL)
print("Timeframe: 1H (Last ~2 days)")
print("----------------------------")
print("Trades:", total)
print("Wins:", wins)
print("Losses:", losses)
print("Winrate:", round(winrate, 2), "%")
print("Net PnL ($):", round(equity, 2))
print("Max Drawdown ($):", round(max_dd, 2))
