"""
BTC/USDT 1H Backtest — Son 12 Ay
Strateji: EMA50/200 trend filtresi + 20-bar breakout sinyali
Veri: Binance public API (API key gerekmez)

Kullanım:
    pip install requests pandas numpy
    python btc_backtest.py
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ─── PARAMETRELER ────────────────────────────────────────────────────────────
SYMBOL      = "BTCUSDT"
INTERVAL    = "1h"
LEVERAGE    = 50
RISK_USD    = 10
SAFE_USD    = 5
RR          = 1.8
START_EQUITY = 1000
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. VERİ ──────────────────────────────────────────────────────────────────

def fetch_binance(symbol=SYMBOL, interval=INTERVAL, months=12):
    end_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = end_ts - months * 30 * 24 * 3600 * 1000
    all_bars = []
    cur = start_ts
    print("Binance'tan veri çekiliyor...")
    while cur < end_ts:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={interval}"
            f"&startTime={cur}&endTime={end_ts}&limit=1000"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        bars = r.json()
        if not bars:
            break
        all_bars.extend(bars)
        cur = bars[-1][0] + 1
        print(f"  {len(all_bars):>5} bar — {datetime.fromtimestamp(bars[-1][0]/1000).strftime('%Y-%m-%d')}")
        time.sleep(0.15)
        if len(bars) < 1000:
            break

    df = pd.DataFrame(all_bars, columns=["ts","o","h","l","c","v","ct","qv","n","tbv","tqv","x"])
    df = df[["ts","o","h","l","c","v"]].copy()
    df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
    df["ts"] = df["ts"].astype(int)
    df.reset_index(drop=True, inplace=True)
    print(f"\nToplam {len(df)} bar yüklendi.")
    print(f"Dönem: {datetime.fromtimestamp(df['ts'].iloc[0]/1000).strftime('%Y-%m-%d')} → "
          f"{datetime.fromtimestamp(df['ts'].iloc[-1]/1000).strftime('%Y-%m-%d')}\n")
    return df


# ── 2. İNDİKATÖRLER ──────────────────────────────────────────────────────────

def calc_atr(df, period=14):
    hl = df["h"] - df["l"]
    hc = (df["h"] - df["c"].shift()).abs()
    lc = (df["l"] - df["c"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── 3. STRATEJİ ───────────────────────────────────────────────────────────────

def run_backtest(df):
    df = df.copy()
    df["ema50"]  = df["c"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()
    df["atr"]    = calc_atr(df)
    df["hi20"]   = df["h"].rolling(20).max().shift(1)   # önceki bar max
    df["lo20"]   = df["l"].rolling(20).min().shift(1)   # önceki bar min
    df["hi10"]   = df["h"].rolling(10).max().shift(1)
    df["lo10"]   = df["l"].rolling(10).min().shift(1)

    trades   = []
    in_trade = False
    trade    = None

    for i in range(201, len(df) - 1):
        row = df.iloc[i]
        if pd.isna(row["ema50"]) or pd.isna(row["atr"]):
            continue

        # ── Açık pozisyon takibi ──
        if in_trade and trade:
            nxt = df["c"].iloc[i]
            if trade["side"] == "LONG":
                if nxt <= trade["sl"]:
                    trade.update(exit=trade["sl"], result="LOSS",
                                 pnl=-trade["risk"], exit_ts=df["ts"].iloc[i])
                    trades.append(trade); in_trade = False; trade = None; continue
                if nxt >= trade["tp"]:
                    trade.update(exit=trade["tp"], result="WIN",
                                 pnl=trade["risk"] * RR, exit_ts=df["ts"].iloc[i])
                    trades.append(trade); in_trade = False; trade = None; continue
            else:
                if nxt >= trade["sl"]:
                    trade.update(exit=trade["sl"], result="LOSS",
                                 pnl=-trade["risk"], exit_ts=df["ts"].iloc[i])
                    trades.append(trade); in_trade = False; trade = None; continue
                if nxt <= trade["tp"]:
                    trade.update(exit=trade["tp"], result="WIN",
                                 pnl=trade["risk"] * RR, exit_ts=df["ts"].iloc[i])
                    trades.append(trade); in_trade = False; trade = None; continue

        if in_trade:
            continue

        # ── Trend filtresi ──
        price = row["c"]
        if row["ema50"] > row["ema200"] and price > row["ema50"]:
            direction = "LONG"
        elif row["ema50"] < row["ema200"] and price < row["ema50"]:
            direction = "SHORT"
        else:
            continue

        # ── Breakout sinyali (gövde ile kıran) ──
        o, c = row["o"], row["c"]
        sig = None
        if direction == "LONG"  and o <= row["hi20"] and c > row["hi20"]:
            sig = "LONG"
        if direction == "SHORT" and o >= row["lo20"] and c < row["lo20"]:
            sig = "SHORT"
        if not sig:
            continue

        # ── SL / TP ──
        a = row["atr"]
        if sig == "LONG":
            swing = row["lo10"] if not pd.isna(row["lo10"]) else price - a
            sl    = min(swing, price - a)
            tp    = price + abs(price - sl) * RR
        else:
            swing = row["hi10"] if not pd.isna(row["hi10"]) else price + a
            sl    = max(swing, price + a)
            tp    = price - abs(price - sl) * RR

        # ── Risk modu ──
        liq_gap = price / LEVERAGE
        sl_gap  = abs(price - sl)
        risk    = SAFE_USD if liq_gap <= sl_gap * 1.3 else RISK_USD

        in_trade = True
        trade = dict(
            side=sig, entry=price, sl=sl, tp=tp,
            risk=risk, entry_ts=df["ts"].iloc[i],
            result=None, pnl=0.0, exit=None, exit_ts=None
        )

    return trades


# ── 4. SONUÇLAR ───────────────────────────────────────────────────────────────

def print_results(trades):
    if not trades:
        print("Hiç işlem bulunamadı.")
        return

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    longs  = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]

    total_pnl  = sum(t["pnl"] for t in trades)
    win_rate   = len(wins) / len(trades) * 100
    avg_win    = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss   = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Max drawdown
    eq, peak, max_dd = START_EQUITY, START_EQUITY, 0
    for t in trades:
        eq += t["pnl"]
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd

    # Konsekütif kayıp
    max_consec_loss = cur_loss = 0
    for t in trades:
        if t["result"] == "LOSS": cur_loss += 1; max_consec_loss = max(max_consec_loss, cur_loss)
        else: cur_loss = 0

    lwr = len([t for t in longs  if t["result"]=="WIN"]) / len(longs)  * 100 if longs  else 0
    swr = len([t for t in shorts if t["result"]=="WIN"]) / len(shorts) * 100 if shorts else 0

    sep = "─" * 46
    print(sep)
    print("  BTC/USDT 1H BACKTEST SONUÇLARI")
    print(sep)
    print(f"  Toplam işlem       : {len(trades)}")
    print(f"  Kazanma oranı      : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net P&L            : {total_pnl:+.2f} $  (başlangıç: {START_EQUITY}$)")
    print(f"  Son equity         : {START_EQUITY + total_pnl:.2f} $")
    print(f"  Beklenti / işlem   : {expectancy:+.2f} $")
    print(f"  Ort. kazanç        : +{avg_win:.2f} $")
    print(f"  Ort. kayıp         :  {avg_loss:.2f} $")
    print(f"  Max drawdown       : {max_dd:.1f}%")
    print(f"  Maks. ard. kayıp   : {max_consec_loss}")
    print(sep)
    print(f"  LONG  : {len(longs):>3} işlem — WR {lwr:.1f}%")
    print(f"  SHORT : {len(shorts):>3} işlem — WR {swr:.1f}%")
    print(sep)

    # Son 15 işlem
    print("\n  Son 15 işlem:")
    print(f"  {'#':>3}  {'Yön':<6}  {'Giriş':>8}  {'SL':>8}  {'TP':>8}  {'Çıkış':>8}  {'P&L':>8}  Tarih")
    print("  " + "─"*78)
    for t in trades[-15:]:
        idx   = trades.index(t) + 1
        et    = datetime.fromtimestamp(t["entry_ts"]/1000).strftime("%Y-%m-%d")
        ex    = f"{t['exit']:>8.0f}" if t["exit"] else "       —"
        pnl_s = f"{t['pnl']:+.2f}$"
        print(f"  {idx:>3}  {t['side']:<6}  {t['entry']:>8.0f}  {t['sl']:>8.0f}  "
              f"{t['tp']:>8.0f}  {ex}  {pnl_s:>8}  {et}")

    print()


# ── 5. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df     = fetch_binance()
    trades = run_backtest(df)
    print_results(trades)
