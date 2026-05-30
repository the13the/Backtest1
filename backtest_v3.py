"""
BTC/USDT 15m Backtest — btc_15m.csv'den okur
    pip install pandas numpy
    python btc_backtest_15m.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

LEVERAGE     = 50
RISK_USD     = 100
SAFE_USD     = 50
RR           = 1.8
START_EQUITY = 1000

df = pd.read_csv("btc_15m.csv")
df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
df["ts"] = df["ts"].astype(int)
df.sort_values("ts", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Yüklendi: {len(df)} bar — "
      f"{datetime.fromtimestamp(df['ts'].iloc[0]/1000).strftime('%Y-%m-%d')} → "
      f"{datetime.fromtimestamp(df['ts'].iloc[-1]/1000).strftime('%Y-%m-%d')}\n")

def calc_atr(df, period=14):
    hl = df["h"] - df["l"]
    hc = (df["h"] - df["c"].shift()).abs()
    lc = (df["l"] - df["c"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

df["ema50"]  = df["c"].ewm(span=50,  adjust=False).mean()
df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()
df["atr"]    = calc_atr(df)
df["hi20"]   = df["h"].rolling(20).max().shift(1)
df["lo20"]   = df["l"].rolling(20).min().shift(1)
df["hi10"]   = df["h"].rolling(10).max().shift(1)
df["lo10"]   = df["l"].rolling(10).min().shift(1)

trades, in_trade, trade = [], False, None

for i in range(201, len(df) - 1):
    row = df.iloc[i]
    if pd.isna(row["ema50"]) or pd.isna(row["atr"]): continue

    if in_trade and trade:
        nxt = df["c"].iloc[i]
        if trade["side"] == "LONG":
            if nxt <= trade["sl"]:
                trade.update(exit=trade["sl"], result="LOSS", pnl=-trade["risk"], exit_ts=int(df["ts"].iloc[i]))
                trades.append(trade); in_trade = False; trade = None; continue
            if nxt >= trade["tp"]:
                trade.update(exit=trade["tp"], result="WIN",  pnl=trade["risk"]*RR, exit_ts=int(df["ts"].iloc[i]))
                trades.append(trade); in_trade = False; trade = None; continue
        else:
            if nxt >= trade["sl"]:
                trade.update(exit=trade["sl"], result="LOSS", pnl=-trade["risk"], exit_ts=int(df["ts"].iloc[i]))
                trades.append(trade); in_trade = False; trade = None; continue
            if nxt <= trade["tp"]:
                trade.update(exit=trade["tp"], result="WIN",  pnl=trade["risk"]*RR, exit_ts=int(df["ts"].iloc[i]))
                trades.append(trade); in_trade = False; trade = None; continue

    if in_trade: continue

    price = row["c"]
    if   row["ema50"] > row["ema200"] and price > row["ema50"]:  direction = "LONG"
    elif row["ema50"] < row["ema200"] and price < row["ema50"]:  direction = "SHORT"
    else: continue

    o, c = row["o"], row["c"]
    sig = None
    if direction == "LONG"  and o <= row["hi20"] and c > row["hi20"]: sig = "LONG"
    if direction == "SHORT" and o >= row["lo20"] and c < row["lo20"]: sig = "SHORT"
    if not sig: continue

    a = row["atr"]
    if sig == "LONG":
        swing = row["lo10"] if not pd.isna(row["lo10"]) else price - a
        sl = min(swing, price - a);  tp = price + abs(price - sl) * RR
    else:
        swing = row["hi10"] if not pd.isna(row["hi10"]) else price + a
        sl = max(swing, price + a);  tp = price - abs(price - sl) * RR

    liq_gap = price / LEVERAGE
    sl_gap  = abs(price - sl)
    risk    = SAFE_USD if liq_gap <= sl_gap * 1.3 else RISK_USD

    in_trade = True
    trade = dict(side=sig, entry=price, sl=sl, tp=tp, risk=risk,
                 entry_ts=int(df["ts"].iloc[i]), result=None, pnl=0.0, exit=None, exit_ts=None)

wins   = [t for t in trades if t["result"] == "WIN"]
losses = [t for t in trades if t["result"] == "LOSS"]
longs  = [t for t in trades if t["side"]   == "LONG"]
shorts = [t for t in trades if t["side"]   == "SHORT"]

total_pnl  = sum(t["pnl"] for t in trades)
win_rate   = len(wins) / len(trades) * 100 if trades else 0
avg_win    = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
avg_loss   = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

eq = peak = START_EQUITY
max_dd = 0
for t in trades:
    eq += t["pnl"]
    if eq > peak: peak = eq
    dd = (peak - eq) / peak * 100
    if dd > max_dd: max_dd = dd

max_cl = cl = 0
for t in trades:
    if t["result"] == "LOSS": cl += 1; max_cl = max(max_cl, cl)
    else: cl = 0

lwr = len([t for t in longs  if t["result"]=="WIN"]) / len(longs)  * 100 if longs  else 0
swr = len([t for t in shorts if t["result"]=="WIN"]) / len(shorts) * 100 if shorts else 0

sep = "─" * 50
print(sep)
print(f"  BTC/USDT 15m — BACKTEST  (risk/işlem: ${RISK_USD})")
print(sep)
print(f"  Toplam işlem       : {len(trades)}")
print(f"  Kazanma oranı      : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
print(f"  Net P&L            : {total_pnl:+.2f} $")
print(f"  Son equity         : {START_EQUITY + total_pnl:.2f} $")
print(f"  Beklenti / işlem   : {expectancy:+.2f} $")
print(f"  Ort. kazanç        : +{avg_win:.2f} $")
print(f"  Ort. kayıp         :  {avg_loss:.2f} $")
print(f"  Max drawdown       : {max_dd:.1f}%")
print(f"  Maks. ard. kayıp   : {max_cl}")
print(sep)
print(f"  LONG  : {len(longs):>3} işlem — WR {lwr:.1f}%")
print(f"  SHORT : {len(shorts):>3} işlem — WR {swr:.1f}%")
print(sep)

print("\n  Son 15 işlem:")
print(f"  {'#':>3}  {'Yön':<6}  {'Giriş':>8}  {'SL':>8}  {'TP':>8}  {'Çıkış':>8}  {'P&L':>9}  Tarih")
print("  " + "─" * 76)
for t in trades[-15:]:
    idx = trades.index(t) + 1
    et  = datetime.fromtimestamp(t["entry_ts"]/1000).strftime("%Y-%m-%d %H:%M")
    ex  = f"{t['exit']:>8.0f}" if t["exit"] else "       —"
    print(f"  {idx:>3}  {t['side']:<6}  {t['entry']:>8.0f}  {t['sl']:>8.0f}  "
          f"{t['tp']:>8.0f}  {ex}  {t['pnl']:>+9.2f}$  {et}")
print()
