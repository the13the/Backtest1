import ccxt
import pandas as pd
import numpy as np

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"

RISK_USD = 10
RR = 1.8
STOP_LOOKBACK = 10

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


def run_test(df, tp_mode):

    trades = []

    atr_series = atr(df)

    for i in range(220, len(df)-1):

        direction = trend(df, i)

        highs = df["h"].rolling(20).max()
        lows = df["l"].rolling(20).min()

        close = df["c"].iloc[i]

        signal = None

        if direction == "LONG":

            if close > highs.iloc[i-1]:
                signal = "LONG"

        else:

            if close < lows.iloc[i-1]:
                signal = "SHORT"

        if signal is None:
            continue

        entry = df["c"].iloc[i+1]

        if signal == "LONG":

            sl = (
                df["l"]
                .rolling(STOP_LOOKBACK)
                .min()
                .iloc[i]
            )

            risk_dist = entry - sl

            if risk_dist <= 0:
                continue

            if tp_mode == "RR":
                tp = entry + risk_dist * RR

            elif tp_mode == "SWING":
                tp = (
                    df["h"]
                    .rolling(20)
                    .max()
                    .iloc[i]
                )

            elif tp_mode == "ATR":
                tp = entry + (
                    atr_series.iloc[i] * 3
                )

        else:

            sl = (
                df["h"]
                .rolling(STOP_LOOKBACK)
                .max()
                .iloc[i]
            )

            risk_dist = sl - entry

            if risk_dist <= 0:
                continue

            if tp_mode == "RR":
                tp = entry - risk_dist * RR

            elif tp_mode == "SWING":
                tp = (
                    df["l"]
                    .rolling(20)
                    .min()
                    .iloc[i]
                )

            elif tp_mode == "ATR":
                tp = entry - (
                    atr_series.iloc[i] * 3
                )

        result = None

        for j in range(i+1, min(i+80, len(df))):

            high = df["h"].iloc[j]
            low = df["l"].iloc[j]

            if signal == "LONG":

                if low <= sl:
                    result = -RISK_USD
                    break

                if high >= tp:
                    reward = (
                        abs(tp-entry)
                        / risk_dist
                    )
                    result = reward * RISK_USD
                    break

            else:

                if high >= sl:
                    result = -RISK_USD
                    break

                if low <= tp:
                    reward = (
                        abs(entry-tp)
                        / risk_dist
                    )
                    result = reward * RISK_USD
                    break

        if result is not None:

            trades.append({
                "side": signal,
                "pnl": result
            })

    return trades


def report(name, trades):

    wins = len([
        x for x in trades
        if x["pnl"] > 0
    ])

    losses = len([
        x for x in trades
        if x["pnl"] < 0
    ])

    total = wins + losses

    wr = (
        wins / total * 100
        if total else 0
    )

    pnl = sum(
        x["pnl"]
        for x in trades
    )

    longs = [
        x for x in trades
        if x["side"] == "LONG"
    ]

    shorts = [
        x for x in trades
        if x["side"] == "SHORT"
    ]

    long_wr = (
        len([x for x in longs if x["pnl"] > 0])
        / len(longs) * 100
        if longs else 0
    )

    short_wr = (
        len([x for x in shorts if x["pnl"] > 0])
        / len(shorts) * 100
        if shorts else 0
    )

    print()
    print("="*40)
    print(name)
    print("="*40)
    print("Trades:", total)
    print("WR:", round(wr,2), "%")
    print("LONG WR:", round(long_wr,2), "%")
    print("SHORT WR:", round(short_wr,2), "%")
    print("Net PnL:", round(pnl,2))


print("Loading BTC data...")

df = fetch()

for mode in ["RR", "SWING", "ATR"]:

    trades = run_test(
        df,
        mode
    )

    report(
        f"TP MODE = {mode}",
        trades
    )
