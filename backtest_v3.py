import ccxt
import pandas as pd
import numpy as np

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"

RISK_USD = 10
RR = 1.8

exchange = ccxt.okx({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})


def fetch():
    all_data = []

    since = exchange.milliseconds() - (365 * 24 * 60 * 60 * 1000)

    while True:

        bars = exchange.fetch_ohlcv(
            SYMBOL,
            TIMEFRAME,
            since=since,
            limit=100
        )

        if not bars:
            break

        all_data += bars

        since = bars[-1][0] + 1

        print("Loaded:", len(all_data))

        if len(bars) < 100:
            break

    df = pd.DataFrame(
        all_data,
        columns=["ts","o","h","l","c","v"]
    )

    return df


def atr(df, period=14):

    hl = df["h"] - df["l"]
    hc = abs(df["h"] - df["c"].shift())
    lc = abs(df["l"] - df["c"].shift())

    tr = pd.concat(
        [hl, hc, lc],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def trend(df, i):

    ma50 = df["c"].rolling(50).mean()
    ma200 = df["c"].rolling(200).mean()

    if ma50.iloc[i] > ma200.iloc[i]:
        return "LONG"

    return "SHORT"


def run_test(df, stop_lookback):

    trades = []

    for i in range(220, len(df)-1):

        direction = trend(df, i)

        highs = df["h"].rolling(20).max()
        lows = df["l"].rolling(20).min()

        close = df["c"].iloc[i]

        last_high = highs.iloc[i-1]
        last_low = lows.iloc[i-1]

        signal = None

        if direction == "LONG":

            if close > last_high:
                signal = "LONG"

        else:

            if close < last_low:
                signal = "SHORT"

        if signal is None:
            continue

        entry = df["c"].iloc[i+1]

        if signal == "LONG":

            sl = (
                df["l"]
                .rolling(stop_lookback)
                .min()
                .iloc[i]
            )

            risk_dist = entry - sl

            if risk_dist <= 0:
                continue

            tp = entry + risk_dist * RR

        else:

            sl = (
                df["h"]
                .rolling(stop_lookback)
                .max()
                .iloc[i]
            )

            risk_dist = sl - entry

            if risk_dist <= 0:
                continue

            tp = entry - risk_dist * RR

        qty = RISK_USD / risk_dist

        result = None

        for j in range(i+1, min(i+50, len(df))):

            high = df["h"].iloc[j]
            low = df["l"].iloc[j]

            if signal == "LONG":

                if low <= sl:
                    result = -RISK_USD
                    break

                if high >= tp:
                    result = RISK_USD * RR
                    break

            else:

                if high >= sl:
                    result = -RISK_USD
                    break

                if low <= tp:
                    result = RISK_USD * RR
                    break

        if result is not None:

            trades.append({
                "side": signal,
                "pnl": result
            })

    return trades


def report(name, trades):

    wins = len([x for x in trades if x["pnl"] > 0])
    losses = len([x for x in trades if x["pnl"] < 0])

    total = wins + losses

    wr = 0

    if total > 0:
        wr = wins / total * 100

    pnl = sum(x["pnl"] for x in trades)

    longs = [x for x in trades if x["side"] == "LONG"]
    shorts = [x for x in trades if x["side"] == "SHORT"]

    long_wins = len([x for x in longs if x["pnl"] > 0])
    short_wins = len([x for x in shorts if x["pnl"] > 0])

    long_wr = (
        long_wins / len(longs) * 100
        if longs else 0
    )

    short_wr = (
        short_wins / len(shorts) * 100
        if shorts else 0
    )

    print()
    print("====================================")
    print(name)
    print("====================================")
    print("Trades:", total)
    print("Wins:", wins)
    print("Losses:", losses)
    print("WR:", round(wr,2), "%")
    print("LONG WR:", round(long_wr,2), "%")
    print("SHORT WR:", round(short_wr,2), "%")
    print("Net PnL:", round(pnl,2))
    print()


print("Loading BTC 12 month data...")

df = fetch()

for stop in [5,7,10]:

    trades = run_test(df, stop)

    report(
        f"STOP LOOKBACK {stop}",
        trades
    )
