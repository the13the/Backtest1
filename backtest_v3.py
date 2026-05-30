"""
BTC/USDT — 1h Backtest v4
Sorun analizi: EMA slope filtresi sinyalleri mahvediyor.
BTC çoğunlukla yukarı trendde olduğu için SHORT sinyali zaten az,
EMA slope + MACD birlikte çok kısıtlıyor.

v4 yaklaşımı:
- EMA slope filtresi KALDIRILDI (MACD histogram yeterli)
- MACD filtresi hafifletildi: sadece histogram yönü (pozitif/negatif)
- Pullback zone 1.5 ATR
- ADX 18 (biraz daha toleranslı)
- Cooldown 6 bar
- Trailing stop devre dışı (önce saf strateji test)
- Grafik yolu düzeltildi (lokal)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import os

SYMBOL             = "BTC-USD"
TIMEFRAME          = "1h"
LEVERAGE           = 50
RISK_PER_TRADE_USD = 10
SAFE_RISK_USD      = 5
RR                 = 2.0
INITIAL_CAPITAL    = 1000.0
PULLBACK_ATR       = 1.5
COOLDOWN_BARS      = 6
ADX_MIN            = 18
RSI_LONG_MIN       = 30
RSI_LONG_MAX       = 68
RSI_SHORT_MIN      = 32
RSI_SHORT_MAX      = 70
MIN_BODY_ATR       = 0.15
USE_TRAILING       = False

def fetch_ohlcv(months=12):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    print(f"Veri: {SYMBOL} {TIMEFRAME} | {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
    chunks, chunk_end = [], end
    while chunk_end > start:
        chunk_start = max(chunk_end - timedelta(days=89), start)
        df_chunk = yf.download(SYMBOL,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=TIMEFRAME, progress=False, auto_adjust=True)
        if not df_chunk.empty:
            chunks.append(df_chunk)
        chunk_end = chunk_start
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"v"})
    df["ts"] = df.index
    df = df.reset_index(drop=True)
    print(f"{len(df)} mum yüklendi.")
    return df

def calc_atr(df, p=14):
    hl = df["h"] - df["l"]
    hc = (df["h"] - df["c"].shift()).abs()
    lc = (df["l"] - df["c"].shift()).abs()
    return pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(p).mean()

def calc_ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_macd_hist(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return ml - ml.ewm(span=sig, adjust=False).mean()

def calc_adx(df, p=14):
    h, l, c = df["h"], df["l"], df["c"]
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdm[(pdm > 0) & (pdm <= ndm)] = 0
    ndm[(ndm > 0) & (ndm <= pdm)] = 0
    tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / atr
    dx  = (100*(pdi-ndi).abs()/(pdi+ndi)).fillna(0)
    return dx.ewm(alpha=1/p, adjust=False).mean()

def generate_signals(df):
    df = df.copy()
    df["e21"]  = calc_ema(df["c"], 21)
    df["e50"]  = calc_ema(df["c"], 50)
    df["e200"] = calc_ema(df["c"], 200)
    df["atr"]  = calc_atr(df, 14)
    df["adx"]  = calc_adx(df, 14)
    df["rsi"]  = calc_rsi(df["c"], 14)
    df["mh"]   = calc_macd_hist(df["c"])
    df["swl"]  = df["l"].rolling(10).min().shift(1)
    df["swh"]  = df["h"].rolling(10).max().shift(1)

    signals, ll, ls = [], -999, -999

    for i in range(210, len(df)):
        r  = df.iloc[i]
        c, o = r["c"], r["o"]
        e21, e50, e200 = r["e21"], r["e50"], r["e200"]
        atr, adx, rsi, mh = r["atr"], r["adx"], r["rsi"], r["mh"]

        if any(pd.isna(x) for x in [e21,e50,e200,atr,adx,rsi,mh]) or atr==0:
            continue
        if adx < ADX_MIN:
            continue

        body = abs(c - o)

        # LONG
        if (e50 > e200
            and c > e200
            and c >= e50 - PULLBACK_ATR * atr
            and c <= e50 + 0.5 * atr
            and c > e21
            and c > o
            and body >= MIN_BODY_ATR * atr
            and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX
            and mh > 0                          # MACD histogram pozitif
            and (i - ll) >= COOLDOWN_BARS):
            signals.append({"idx":i,"ts":r["ts"],"direction":"LONG",
                "entry":c,"swing_low":r["swl"],"swing_high":r["swh"],"atr":atr})
            ll = i

        # SHORT
        elif (e50 < e200
            and c < e200
            and c <= e50 + PULLBACK_ATR * atr
            and c >= e50 - 0.5 * atr
            and c < e21
            and c < o
            and body >= MIN_BODY_ATR * atr
            and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
            and mh < 0                          # MACD histogram negatif
            and (i - ls) >= COOLDOWN_BARS):
            signals.append({"idx":i,"ts":r["ts"],"direction":"SHORT",
                "entry":c,"swing_low":r["swl"],"swing_high":r["swh"],"atr":atr})
            ls = i

    return signals

def smart_stop(sig):
    e, atr = sig["entry"], sig["atr"]
    if sig["direction"] == "LONG":
        sw = sig["swing_low"]
        return min(sw, e - atr) if not pd.isna(sw) else e - atr
    sw = sig["swing_high"]
    return max(sw, e + atr) if not pd.isna(sw) else e + atr

def smart_tp(entry, sl, direction):
    risk = abs(entry - sl)
    return entry + risk*RR if direction=="LONG" else entry - risk*RR

def position_size(entry, sl, direction):
    dist = abs(entry - sl)
    if dist <= 0: return 0
    liq = abs(entry/LEVERAGE)
    risk = SAFE_RISK_USD if liq <= dist*1.3 else RISK_PER_TRADE_USD
    return risk / dist

def run_backtest(df, signals):
    capital = INITIAL_CAPITAL
    trades, open_trades = [], []
    H,L,C = df["h"].values, df["l"].values, df["c"].values
    TS = df["ts"].values
    sig_map = {}
    for s in signals:
        sig_map.setdefault(s["idx"],[]).append(s)

    for i in range(len(df)):
        h, l = H[i], L[i]
        still = []
        for t in open_trades:
            sl, tp, d = t["sl"], t["tp"], t["direction"]
            hit_sl = (d=="LONG" and l<=sl) or (d=="SHORT" and h>=sl)
            hit_tp = (d=="LONG" and h>=tp) or (d=="SHORT" and l<=tp)
            if hit_sl or hit_tp:
                ex  = sl if hit_sl else tp
                pnl = ((ex-t["entry"]) if d=="LONG" else (t["entry"]-ex)) * t["qty"]
                capital += pnl
                trades.append({
                    "entry_ts":t["entry_ts"],"exit_ts":TS[i],
                    "direction":d,"entry":t["entry"],"exit":ex,
                    "sl":sl,"tp":tp,"qty":t["qty"],
                    "pnl_usd":round(pnl,4),
                    "result":"WIN" if hit_tp else "LOSS",
                    "capital":round(capital,2),
                })
            else:
                still.append(t)
        open_trades = still
        for sig in sig_map.get(i,[]):
            sl  = smart_stop(sig)
            tp  = smart_tp(sig["entry"], sl, sig["direction"])
            qty = position_size(sig["entry"], sl, sig["direction"])
            if qty > 0:
                open_trades.append({"entry_ts":sig["ts"],"direction":sig["direction"],
                    "entry":sig["entry"],"sl":sl,"tp":tp,"qty":qty})
    return trades, capital

def analyze(trades, final_capital):
    if not trades:
        print("Hiç trade yok — filtreleri daha da gevşet.")
        return None
    df_t = pd.DataFrame(trades)
    df_t["entry_ts"] = pd.to_datetime(df_t["entry_ts"])
    df_t["exit_ts"]  = pd.to_datetime(df_t["exit_ts"])
    total = len(df_t)
    wins  = (df_t["result"]=="WIN").sum()
    wr    = wins/total*100
    gp    = df_t.loc[df_t["pnl_usd"]>0,"pnl_usd"].sum()
    gl    = df_t.loc[df_t["pnl_usd"]<0,"pnl_usd"].sum()
    pf    = gp/abs(gl) if gl else float("inf")
    aw    = df_t.loc[df_t["result"]=="WIN","pnl_usd"].mean()
    al    = df_t.loc[df_t["result"]=="LOSS","pnl_usd"].mean()
    eq    = df_t["capital"].values
    dd    = ((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)*100).min()
    dp    = df_t.set_index("exit_ts")["pnl_usd"].resample("D").sum()
    sh    = (dp.mean()/dp.std()*np.sqrt(252)) if dp.std()>0 else 0
    lt    = df_t[df_t["direction"]=="LONG"]
    st    = df_t[df_t["direction"]=="SHORT"]
    lwr   = (lt["result"]=="WIN").mean()*100 if len(lt) else 0
    swr   = (st["result"]=="WIN").mean()*100 if len(st) else 0

    print("\n"+"="*52)
    print("  BACKTEST — BTC/USDT 1h EMA Pullback v4")
    print("="*52)
    print(f"  Başlangıç    : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final        : ${final_capital:,.2f}  ({(final_capital/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"  Net PnL      : ${df_t['pnl_usd'].sum():+,.2f}")
    print(f"  Trade        : {total}  (L:{len(lt)} S:{len(st)})")
    print(f"  Win Rate     : {wr:.1f}%  (Long:{lwr:.0f}% Short:{swr:.0f}%)")
    print(f"  Ort Kazanç   : ${aw:+.2f}")
    print(f"  Ort Kayıp    : ${al:+.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max Drawdown : {dd:.2f}%")
    print(f"  Sharpe       : {sh:.2f}")
    print("="*52)
    cols = ["entry_ts","direction","entry","exit","pnl_usd","result","capital"]
    print(df_t[cols].tail(15).to_string(index=False))
    return df_t

def plot_results(df_t, df_price):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("BTC/USDT 1h — EMA Pullback v4", fontsize=14, fontweight="bold")
    ax1 = axes[0]
    ax1.plot(df_t["exit_ts"], df_t["capital"], color="#00b4d8", lw=1.5)
    ax1.axhline(INITIAL_CAPITAL, color="gray", ls="--", lw=0.8, label="Başlangıç")
    ax1.fill_between(df_t["exit_ts"], INITIAL_CAPITAL, df_t["capital"],
        where=df_t["capital"]>=INITIAL_CAPITAL, alpha=0.15, color="#06d6a0")
    ax1.fill_between(df_t["exit_ts"], INITIAL_CAPITAL, df_t["capital"],
        where=df_t["capital"]<INITIAL_CAPITAL, alpha=0.15, color="#ef476f")
    ax1.set_title("Equity Curve"); ax1.set_ylabel("USD"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2 = axes[1]
    colors = ["#06d6a0" if r=="WIN" else "#ef476f" for r in df_t["result"]]
    ax2.bar(range(len(df_t)), df_t["pnl_usd"], color=colors)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_title("PnL / Trade"); ax2.set_ylabel("USD"); ax2.grid(alpha=0.3)
    ax3 = axes[2]
    ax3.plot(df_price["ts"], df_price["c"], color="#adb5bd", lw=0.5, label="BTC")
    for d_, m_, col_ in [("LONG","^","#06d6a0"),("SHORT","v","#ef476f")]:
        sub = df_t[df_t["direction"]==d_]
        ax3.scatter(sub["entry_ts"], sub["entry"], marker=m_, color=col_, s=50, zorder=5, label=d_)
    ax3.set_title("Giriş Noktaları"); ax3.set_ylabel("USD"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    plt.tight_layout()
    # Grafik lokal dizine kaydedilir
    out = "backtest_v4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGrafik kaydedildi: {out}")

if __name__ == "__main__":
    df = fetch_ohlcv(months=12)
    signals = generate_signals(df)
    print(f"{len(signals)} sinyal bulundu.")
    trades, final_cap = run_backtest(df, signals)
    df_t = analyze(trades, final_cap)
    if df_t is not None:
        plot_results(df_t, df)
