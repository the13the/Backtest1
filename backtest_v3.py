"""
BTC/USDT — 1h Backtest v3
Strateji: EMA Trend + Pullback (DENGELİ)

v2 sorunları:
- Filtreler çok sıkıydı → 5 sinyal
- Trailing stop erken kesiyordu

v3 düzeltmeleri:
- Pullback zone genişletildi (1.2 ATR)
- ADX 20'ye indirildi (ama EMA slope filtresi eklendi)
- RSI aralığı genişletildi
- MACD: sadece histogram yönü yeterli
- Trailing stop: TP'nin %50'sine ulaşınca devreye girer
- Hacim filtresi kaldırıldı (BTC için zaten yüksek)
- Mum gövdesi 0.2 ATR'ye indirildi
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# ─── PARAMETRELER ───
SYMBOL             = "BTC-USD"
TIMEFRAME          = "1h"
LEVERAGE           = 50
RISK_PER_TRADE_USD = 10
SAFE_RISK_USD      = 5
RR                 = 2.0
INITIAL_CAPITAL    = 1000.0

PULLBACK_ATR       = 1.2   # Giriş bölgesi genişliği
COOLDOWN_BARS      = 8
ADX_MIN            = 20
RSI_LONG_MIN       = 30
RSI_LONG_MAX       = 65
RSI_SHORT_MIN      = 35
RSI_SHORT_MAX      = 70
MIN_BODY_ATR       = 0.2
USE_TRAILING       = True
TRAIL_ACTIVATE_RR  = 0.5   # TP mesafesinin %50'sine ulaşınca trailing başlar
TRAILING_STOP_ATR  = 1.0

# ─── VERİ ───
def fetch_ohlcv(months=12):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    print(f"Veri: {SYMBOL} {TIMEFRAME} | {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
    all_chunks = []
    chunk_end = end
    while chunk_end > start:
        chunk_start = max(chunk_end - timedelta(days=89), start)
        df_chunk = yf.download(SYMBOL,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=TIMEFRAME, progress=False, auto_adjust=True)
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
        chunk_end = chunk_start
    df = pd.concat(all_chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"v"})
    df["ts"] = df.index
    df = df.reset_index(drop=True)
    print(f"{len(df)} mum yüklendi.")
    return df

# ─── GÖSTERGELER ───
def calc_atr(df, period=14):
    hl = df["h"] - df["l"]
    hc = (df["h"] - df["c"].shift()).abs()
    lc = (df["l"] - df["c"].shift()).abs()
    return pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(period).mean()

def calc_ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(s, period=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_macd(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig,  adjust=False).mean()
    return ml - sl  # histogram

def calc_adx(df, period=14):
    h, l, c = df["h"], df["l"], df["c"]
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdm[(pdm > 0) & (pdm <= ndm)] = 0
    ndm[(ndm > 0) & (ndm <= pdm)] = 0
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1/period, adjust=False).mean() / atr
    dx  = (100 * (pdi-ndi).abs() / (pdi+ndi)).fillna(0)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# ─── SİNYALLER ───
def generate_signals(df):
    df = df.copy()
    df["ema21"]  = calc_ema(df["c"], 21)
    df["ema50"]  = calc_ema(df["c"], 50)
    df["ema200"] = calc_ema(df["c"], 200)
    df["atr"]    = calc_atr(df, 14)
    df["adx"]    = calc_adx(df, 14)
    df["rsi"]    = calc_rsi(df["c"], 14)
    df["macd_h"] = calc_macd(df["c"])
    # EMA50 slope: 3 bar önceye göre
    df["ema50_slope"] = df["ema50"] - df["ema50"].shift(3)
    df["swing_low"]   = df["l"].rolling(10).min().shift(1)
    df["swing_high"]  = df["h"].rolling(10).max().shift(1)

    signals, last_l, last_s = [], -999, -999

    for i in range(210, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]
        c, o = r["c"], r["o"]
        e21, e50, e200 = r["ema21"], r["ema50"], r["ema200"]
        atr, adx, rsi  = r["atr"], r["adx"], r["rsi"]
        mh, pmh        = r["macd_h"], p["macd_h"]
        slope          = r["ema50_slope"]

        if any(pd.isna(x) for x in [e21, e50, e200, atr, adx, rsi, mh]) or atr == 0:
            continue
        if adx < ADX_MIN:
            continue

        body = abs(c - o)

        # ── LONG ──
        if (
            e50 > e200                              # yukarı trend
            and slope > 0                           # EMA50 yukarı eğimli
            and c > e200                            # fiyat EMA200 üstü
            and c >= e50 - PULLBACK_ATR * atr       # pullback alt sınır
            and c <= e50 + 0.5 * atr                # pullback üst sınır
            and c > e21                             # EMA21 üstü
            and c > o                               # bullish mum
            and body >= MIN_BODY_ATR * atr          # gövde filtresi
            and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX # RSI filtresi
            and mh > pmh                            # MACD histogram yükseliyor
            and (i - last_l) >= COOLDOWN_BARS
        ):
            signals.append({"idx":i,"ts":r["ts"],"direction":"LONG",
                "entry":c,"swing_low":r["swing_low"],"swing_high":r["swing_high"],"atr":atr})
            last_l = i

        # ── SHORT ──
        elif (
            e50 < e200
            and slope < 0                           # EMA50 aşağı eğimli
            and c < e200
            and c <= e50 + PULLBACK_ATR * atr
            and c >= e50 - 0.5 * atr
            and c < e21
            and c < o
            and body >= MIN_BODY_ATR * atr
            and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
            and mh < pmh                            # MACD histogram düşüyor
            and (i - last_s) >= COOLDOWN_BARS
        ):
            signals.append({"idx":i,"ts":r["ts"],"direction":"SHORT",
                "entry":c,"swing_low":r["swing_low"],"swing_high":r["swing_high"],"atr":atr})
            last_s = i

    return signals

# ─── SL/TP ───
def smart_stop(sig):
    e, atr = sig["entry"], sig["atr"]
    if sig["direction"] == "LONG":
        sw = sig["swing_low"]
        return min(sw, e - atr) if not pd.isna(sw) else e - atr
    sw = sig["swing_high"]
    return max(sw, e + atr) if not pd.isna(sw) else e + atr

def smart_tp(entry, sl, direction):
    risk = abs(entry - sl)
    return entry + risk * RR if direction == "LONG" else entry - risk * RR

def position_size(entry, sl, direction):
    dist = abs(entry - sl)
    if dist <= 0: return 0
    liq_gap = abs(entry - (entry - entry/LEVERAGE if direction=="LONG" else entry + entry/LEVERAGE))
    risk    = SAFE_RISK_USD if liq_gap <= dist * 1.3 else RISK_PER_TRADE_USD
    return risk / dist

# ─── BACKTEST ───
def run_backtest(df, signals):
    capital = INITIAL_CAPITAL
    trades, open_trades = [], []
    H = df["h"].values; L = df["l"].values
    C = df["c"].values; TS = df["ts"].values
    ATR = calc_atr(df, 14).values

    sig_map = {}
    for s in signals:
        sig_map.setdefault(s["idx"], []).append(s)

    for i in range(len(df)):
        h, l, c = H[i], L[i], C[i]
        atr = ATR[i] if not np.isnan(ATR[i]) else 0

        still_open = []
        for t in open_trades:
            sl, tp, d = t["sl"], t["tp"], t["direction"]
            entry = t["entry"]

            # Trailing: sadece TP yolunun %50'sine ulaşınca aktif
            if USE_TRAILING and atr > 0:
                tp_dist    = abs(tp - entry)
                progress   = (c - entry) if d == "LONG" else (entry - c)
                if progress >= tp_dist * TRAIL_ACTIVATE_RR:
                    if d == "LONG":
                        new_sl = c - TRAILING_STOP_ATR * atr
                        if new_sl > t["sl"]: t["sl"] = new_sl
                    else:
                        new_sl = c + TRAILING_STOP_ATR * atr
                        if new_sl < t["sl"]: t["sl"] = new_sl
                    sl = t["sl"]

            hit_sl = (d=="LONG" and l<=sl) or (d=="SHORT" and h>=sl)
            hit_tp = (d=="LONG" and h>=tp) or (d=="SHORT" and l<=tp)

            if hit_sl or hit_tp:
                ex = sl if hit_sl else tp
                pnl = ((ex-entry) if d=="LONG" else (entry-ex)) * t["qty"]
                capital += pnl
                trades.append({
                    "entry_ts": t["entry_ts"], "exit_ts": TS[i],
                    "direction": d, "entry": entry, "exit": ex,
                    "sl": sl, "tp": tp, "qty": t["qty"],
                    "pnl_usd": round(pnl,4),
                    "result": "WIN" if hit_tp else "LOSS",
                    "capital": round(capital,2),
                })
            else:
                still_open.append(t)
        open_trades = still_open

        for sig in sig_map.get(i, []):
            sl  = smart_stop(sig)
            tp  = smart_tp(sig["entry"], sl, sig["direction"])
            qty = position_size(sig["entry"], sl, sig["direction"])
            if qty > 0:
                open_trades.append({
                    "entry_ts": sig["ts"], "direction": sig["direction"],
                    "entry": sig["entry"], "sl": sl, "tp": tp, "qty": qty
                })

    return trades, capital

# ─── ANALİZ ───
def analyze(trades, final_capital):
    if not trades:
        print("Hiç trade yok.")
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

    eq   = df_t["capital"].values
    peak = np.maximum.accumulate(eq)
    dd   = ((eq-peak)/peak*100).min()

    dp   = df_t.set_index("exit_ts")["pnl_usd"].resample("D").sum()
    sh   = (dp.mean()/dp.std()*np.sqrt(252)) if dp.std()>0 else 0

    lt = df_t[df_t["direction"]=="LONG"]
    st = df_t[df_t["direction"]=="SHORT"]
    lwr = (lt["result"]=="WIN").mean()*100 if len(lt) else 0
    swr = (st["result"]=="WIN").mean()*100 if len(st) else 0

    print("\n" + "="*55)
    print("  BACKTEST — BTC/USDT 1h EMA Pullback v3")
    print("="*55)
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
    print("="*55)
    cols = ["entry_ts","direction","entry","exit","pnl_usd","result","capital"]
    print(df_t[cols].tail(15).to_string(index=False))
    return df_t

# ─── GRAFİK ───
def plot_results(df_t, df_price):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("BTC/USDT 1h — EMA Pullback v3", fontsize=14, fontweight="bold")
    ax1 = axes[0]
    ax1.plot(df_t["exit_ts"], df_t["capital"], color="#00b4d8", lw=1.5)
    ax1.axhline(INITIAL_CAPITAL, color="gray", ls="--", lw=0.8)
    ax1.fill_between(df_t["exit_ts"], INITIAL_CAPITAL, df_t["capital"],
        where=df_t["capital"]>=INITIAL_CAPITAL, alpha=0.15, color="#06d6a0")
    ax1.fill_between(df_t["exit_ts"], INITIAL_CAPITAL, df_t["capital"],
        where=df_t["capital"]<INITIAL_CAPITAL, alpha=0.15, color="#ef476f")
    ax1.set_title("Equity Curve"); ax1.set_ylabel("USD"); ax1.grid(alpha=0.3)

    ax2 = axes[1]
    colors = ["#06d6a0" if r=="WIN" else "#ef476f" for r in df_t["result"]]
    ax2.bar(range(len(df_t)), df_t["pnl_usd"], color=colors)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_title("PnL / Trade"); ax2.set_ylabel("USD"); ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df_price["ts"], df_price["c"], color="#adb5bd", lw=0.6, label="BTC")
    for dir_, marker, color in [("LONG","^","#06d6a0"),("SHORT","v","#ef476f")]:
        sub = df_t[df_t["direction"]==dir_]
        ax3.scatter(sub["entry_ts"], sub["entry"], marker=marker, color=color, s=50, zorder=5, label=dir_)
    ax3.set_title("Giriş Noktaları"); ax3.set_ylabel("USD"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = "/mnt/user-data/outputs/backtest_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGrafik: {out}")

if __name__ == "__main__":
    df = fetch_ohlcv(months=12)
    signals = generate_signals(df)
    print(f"{len(signals)} sinyal")
    trades, final_cap = run_backtest(df, signals)
    df_t = analyze(trades, final_cap)
    if df_t is not None:
        plot_results(df_t, df)
