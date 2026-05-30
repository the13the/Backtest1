"""
BTC/USDT — 1h Backtest v2
Strateji: EMA Trend + Pullback (GELİŞTİRİLMİŞ)

İyileştirmeler:
- ADX eşiği 25'e yükseltildi (güçlü trend filtresi)
- Pullback zone daraltıldı (0.6 ATR)
- RSI 35-65 arası pullback teyidi
- MACD histogram yön filtresi
- Minimum mum gövde büyüklüğü (0.3 ATR)
- Trailing stop eklendi
- Hacim filtresi (ortalama üzeri hacim)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────
# PARAMETRELER
# ─────────────────────────────────────────
SYMBOL             = "BTC-USD"
TIMEFRAME          = "1h"
LEVERAGE           = 50
RISK_PER_TRADE_USD = 10
SAFE_RISK_USD      = 5
RR                 = 2.0          # v1'den 1.8 → 2.0
INITIAL_CAPITAL    = 1000.0

PULLBACK_ATR       = 0.6          # v1: 1.0 → daraltıldı
COOLDOWN_BARS      = 10           # v1: 8 → artırıldı
ADX_MIN            = 25           # v1: 20 → güçlü trend
RSI_LONG_MAX       = 60           # Long için RSI üst sınır
RSI_LONG_MIN       = 35           # Long için RSI alt sınır
RSI_SHORT_MIN      = 40           # Short için RSI alt sınır
RSI_SHORT_MAX      = 65           # Short için RSI üst sınır
MIN_BODY_ATR       = 0.3          # Minimum mum gövdesi (ATR cinsinden)
VOLUME_FACTOR      = 1.2          # Ortalama hacmin kaç katı olmalı
TRAILING_STOP_ATR  = 1.5          # Trailing stop mesafesi (ATR)
USE_TRAILING       = True         # Trailing stop kullan?

# ─────────────────────────────────────────
# VERİ ÇEKİMİ
# ─────────────────────────────────────────

def fetch_ohlcv(months=12):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    print(f"Veri çekiliyor: {SYMBOL} {TIMEFRAME} | {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")

    all_chunks = []
    chunk_end = end
    while chunk_end > start:
        chunk_start = max(chunk_end - timedelta(days=89), start)
        df_chunk = yf.download(
            SYMBOL,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=TIMEFRAME,
            progress=False,
            auto_adjust=True,
        )
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
        chunk_end = chunk_start

    if not all_chunks:
        raise RuntimeError("Veri indirilemedi!")

    df = pd.concat(all_chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    df["ts"] = df.index
    df = df.reset_index(drop=True)
    print(f"Toplam {len(df)} mum yüklendi.")
    return df

# ─────────────────────────────────────────
# GÖSTERGELER
# ─────────────────────────────────────────

def calc_atr(df, period=14):
    hl = df["h"] - df["l"]
    hc = (df["h"] - df["c"].shift()).abs()
    lc = (df["l"] - df["c"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_adx(df, period=14):
    high  = df["h"]
    low   = df["l"]
    close = df["c"]

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm   < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm > 0)   & (plus_dm  <= minus_dm)] = 0
    minus_dm[(minus_dm > 0) & (minus_dm <= plus_dm)]  = 0

    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    atr14    = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period,  adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# ─────────────────────────────────────────
# SİNYAL ÜRETİMİ
#
# LONG koşulları:
#   1. EMA50 > EMA200  (yukarı trend)
#   2. Fiyat EMA200 üstünde
#   3. Pullback zone: EMA50 - 0.6*ATR ile EMA50 + 0.3*ATR arası
#   4. Kapanış EMA21 üstünde (momentum teyidi)
#   5. Bullish mum + gövde >= 0.3*ATR
#   6. RSI 35-60 arası (ne çok satım ne çok alım)
#   7. MACD histogram pozitif veya yükseliyor
#   8. ADX >= 25 (güçlü trend)
#   9. Hacim > 20 periyot ortalamasının 1.2 katı
#
# SHORT koşulları (tersi)
# ─────────────────────────────────────────

def generate_signals(df):
    df = df.copy()
    df["ema21"]  = calc_ema(df["c"], 21)
    df["ema50"]  = calc_ema(df["c"], 50)
    df["ema200"] = calc_ema(df["c"], 200)
    df["atr"]    = calc_atr(df, 14)
    df["adx"]    = calc_adx(df, 14)
    df["rsi"]    = calc_rsi(df["c"], 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = calc_macd(df["c"])
    df["vol_ma"] = df["v"].rolling(20).mean()

    df["swing_low"]  = df["l"].rolling(10).min().shift(1)
    df["swing_high"] = df["h"].rolling(10).max().shift(1)

    signals = []
    last_long_bar  = -999
    last_short_bar = -999

    for i in range(200, len(df)):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1]

        ema21    = row["ema21"]
        ema50    = row["ema50"]
        ema200   = row["ema200"]
        atr      = row["atr"]
        adx      = row["adx"]
        rsi      = row["rsi"]
        hist     = row["macd_hist"]
        prev_hist= prev_row["macd_hist"]
        c        = row["c"]
        o        = row["o"]
        vol      = row["v"]
        vol_ma   = row["vol_ma"]

        if pd.isna(ema21) or pd.isna(ema50) or pd.isna(ema200) or \
           pd.isna(atr) or pd.isna(adx) or pd.isna(rsi) or \
           pd.isna(hist) or pd.isna(vol_ma) or atr == 0:
            continue

        if adx < ADX_MIN:
            continue

        body     = abs(c - o)
        min_body = MIN_BODY_ATR * atr

        # Hacim filtresi
        vol_ok = vol >= vol_ma * VOLUME_FACTOR

        # ── LONG ──
        if (
            ema50 > ema200
            and c > ema200
            and c >= ema50 - PULLBACK_ATR * atr
            and c <= ema50 + 0.3 * atr
            and c > ema21
            and c > o
            and body >= min_body
            and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX
            and (hist > 0 or hist > prev_hist)   # MACD histogram pozitif veya dönüyor
            and vol_ok
            and (i - last_long_bar) >= COOLDOWN_BARS
        ):
            signals.append({
                "idx":        i,
                "ts":         row["ts"],
                "direction":  "LONG",
                "entry":      c,
                "swing_low":  row["swing_low"],
                "swing_high": row["swing_high"],
                "atr":        atr,
            })
            last_long_bar = i

        # ── SHORT ──
        elif (
            ema50 < ema200
            and c < ema200
            and c <= ema50 + PULLBACK_ATR * atr
            and c >= ema50 - 0.3 * atr
            and c < ema21
            and c < o
            and body >= min_body
            and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
            and (hist < 0 or hist < prev_hist)   # MACD histogram negatif veya düşüyor
            and vol_ok
            and (i - last_short_bar) >= COOLDOWN_BARS
        ):
            signals.append({
                "idx":        i,
                "ts":         row["ts"],
                "direction":  "SHORT",
                "entry":      c,
                "swing_low":  row["swing_low"],
                "swing_high": row["swing_high"],
                "atr":        atr,
            })
            last_short_bar = i

    return signals

# ─────────────────────────────────────────
# SL / TP
# ─────────────────────────────────────────

def smart_stop(sig):
    entry = sig["entry"]
    atr   = sig["atr"]
    if sig["direction"] == "LONG":
        swing = sig["swing_low"]
        sl = min(swing, entry - atr) if not pd.isna(swing) else entry - atr
        return sl
    swing = sig["swing_high"]
    sl = max(swing, entry + atr) if not pd.isna(swing) else entry + atr
    return sl

def smart_tp(entry, sl, direction):
    risk = abs(entry - sl)
    return entry + risk * RR if direction == "LONG" else entry - risk * RR

def estimate_liq(entry, direction):
    move = entry / LEVERAGE
    return (entry - move) if direction == "LONG" else (entry + move)

def risk_amount(entry, sl, direction):
    liq_gap = abs(entry - estimate_liq(entry, direction))
    sl_gap  = abs(entry - sl)
    return SAFE_RISK_USD if liq_gap <= sl_gap * 1.3 else RISK_PER_TRADE_USD

def position_size(entry, sl, direction):
    dist = abs(entry - sl)
    return 0 if dist <= 0 else risk_amount(entry, sl, direction) / dist

# ─────────────────────────────────────────
# BACKTEST MOTORU — HEDGE MODE + TRAILING STOP
# ─────────────────────────────────────────

def run_backtest(df, signals):
    capital     = INITIAL_CAPITAL
    trades      = []
    open_trades = []

    high_arr = df["h"].values
    low_arr  = df["l"].values
    close_arr= df["c"].values
    atr_arr  = calc_atr(df, 14).values
    ts_arr   = df["ts"].values

    sig_map = {}
    for s in signals:
        sig_map.setdefault(s["idx"], []).append(s)

    for i in range(len(df)):
        h   = high_arr[i]
        l   = low_arr[i]
        c   = close_arr[i]
        atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0

        still_open = []
        for t in open_trades:
            sl = t["sl"]
            tp = t["tp"]
            d  = t["direction"]

            # Trailing stop güncelle
            if USE_TRAILING and atr > 0:
                if d == "LONG":
                    new_trail = c - TRAILING_STOP_ATR * atr
                    if new_trail > sl:
                        t["sl"] = new_trail
                        sl = t["sl"]
                else:
                    new_trail = c + TRAILING_STOP_ATR * atr
                    if new_trail < sl:
                        t["sl"] = new_trail
                        sl = t["sl"]

            hit_sl = (d == "LONG"  and l <= sl) or (d == "SHORT" and h >= sl)
            hit_tp = (d == "LONG"  and h >= tp) or (d == "SHORT" and l <= tp)

            if hit_sl or hit_tp:
                exit_price   = sl if hit_sl else tp
                pnl_per_unit = (exit_price - t["entry"]) if d == "LONG" \
                               else (t["entry"] - exit_price)
                pnl_usd  = pnl_per_unit * t["qty"]
                capital += pnl_usd
                trades.append({
                    "entry_ts":  t["entry_ts"],
                    "exit_ts":   ts_arr[i],
                    "direction": d,
                    "entry":     t["entry"],
                    "exit":      exit_price,
                    "sl":        sl,
                    "tp":        tp,
                    "qty":       t["qty"],
                    "pnl_usd":   round(pnl_usd, 4),
                    "result":    "WIN" if hit_tp else "LOSS",
                    "capital":   round(capital, 2),
                })
            else:
                still_open.append(t)

        open_trades = still_open

        if i in sig_map:
            for sig in sig_map[i]:
                entry = sig["entry"]
                sl    = smart_stop(sig)
                tp    = smart_tp(entry, sl, sig["direction"])
                qty   = position_size(entry, sl, sig["direction"])

                if qty > 0 and abs(entry - sl) > 0:
                    open_trades.append({
                        "entry_ts":  sig["ts"],
                        "direction": sig["direction"],
                        "entry":     entry,
                        "sl":        sl,
                        "tp":        tp,
                        "qty":       qty,
                    })

    return trades, capital

# ─────────────────────────────────────────
# ANALİZ
# ─────────────────────────────────────────

def analyze(trades, final_capital):
    if not trades:
        print("Hiç trade yok.")
        return None

    df_t = pd.DataFrame(trades)
    df_t["entry_ts"] = pd.to_datetime(df_t["entry_ts"])
    df_t["exit_ts"]  = pd.to_datetime(df_t["exit_ts"])

    total    = len(df_t)
    wins     = (df_t["result"] == "WIN").sum()
    losses   = (df_t["result"] == "LOSS").sum()
    win_rate = wins / total * 100

    gross_profit  = df_t.loc[df_t["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss    = df_t.loc[df_t["pnl_usd"] < 0, "pnl_usd"].sum()
    net_pnl       = df_t["pnl_usd"].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    avg_win       = df_t.loc[df_t["result"] == "WIN",  "pnl_usd"].mean()
    avg_loss      = df_t.loc[df_t["result"] == "LOSS", "pnl_usd"].mean()

    equity = df_t["capital"].values
    peak   = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak * 100).min()

    daily_pnl = df_t.set_index("exit_ts")["pnl_usd"].resample("D").sum()
    sharpe    = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

    long_t   = df_t[df_t["direction"] == "LONG"]
    short_t  = df_t[df_t["direction"] == "SHORT"]
    long_wr  = (long_t["result"]  == "WIN").mean() * 100 if len(long_t)  > 0 else 0
    short_wr = (short_t["result"] == "WIN").mean() * 100 if len(short_t) > 0 else 0

    print("\n" + "="*58)
    print("   BACKTEST SONUÇLARI — BTC/USDT 1h (PULLBACK v2)")
    print("="*58)
    print(f"  Başlangıç Sermaye   : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Sermaye       : ${final_capital:,.2f}")
    print(f"  Net PnL             : ${net_pnl:+,.2f}  ({(final_capital/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"  Toplam Trade        : {total}")
    print(f"  Kazanan             : {wins}  ({win_rate:.1f}%)")
    print(f"  Kaybeden            : {losses}")
    print(f"  Long  Win Rate      : {long_wr:.1f}%  ({len(long_t)} trade)")
    print(f"  Short Win Rate      : {short_wr:.1f}%  ({len(short_t)} trade)")
    print(f"  Ort. Kazanç         : ${avg_win:+.2f}")
    print(f"  Ort. Kayıp          : ${avg_loss:+.2f}")
    print(f"  Profit Factor       : {profit_factor:.2f}")
    print(f"  Max Drawdown        : {max_dd:.2f}%")
    print(f"  Sharpe (yıllık)     : {sharpe:.2f}")
    print(f"\n  [Filtreler]")
    print(f"  ADX Eşiği           : {ADX_MIN}")
    print(f"  Pullback ATR        : {PULLBACK_ATR}")
    print(f"  RSI Aralığı (Long)  : {RSI_LONG_MIN}-{RSI_LONG_MAX}")
    print(f"  RSI Aralığı (Short) : {RSI_SHORT_MIN}-{RSI_SHORT_MAX}")
    print(f"  Trailing Stop       : {'Açık' if USE_TRAILING else 'Kapalı'} ({TRAILING_STOP_ATR} ATR)")
    print(f"  Min Gövde           : {MIN_BODY_ATR} ATR")
    print(f"  Hacim Filtresi      : {VOLUME_FACTOR}x ortalama")
    print("="*58)

    cols = ["entry_ts", "direction", "entry", "exit", "pnl_usd", "result", "capital"]
    print("\n-- Son 10 Trade --")
    print(df_t[cols].tail(10).to_string(index=False))

    return df_t

# ─────────────────────────────────────────
# GRAFİK
# ─────────────────────────────────────────

def plot_results(df_t, df_price):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("BTC/USDT 1h — EMA Pullback v2 (RSI + MACD + Trailing)", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(df_t["exit_ts"], df_t["capital"], color="#00b4d8", linewidth=1.5)
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8, label="Başlangıç")
    ax1.fill_between(df_t["exit_ts"], INITIAL_CAPITAL, df_t["capital"],
                     where=df_t["capital"] >= INITIAL_CAPITAL, alpha=0.15, color="#06d6a0")
    ax1.fill_between(df_t["exit_ts"], INITIAL_CAPITAL, df_t["capital"],
                     where=df_t["capital"] < INITIAL_CAPITAL, alpha=0.15, color="#ef476f")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Sermaye (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = ["#06d6a0" if r == "WIN" else "#ef476f" for r in df_t["result"]]
    ax2.bar(range(len(df_t)), df_t["pnl_usd"], color=colors, width=0.8)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_title("Trade Başına PnL (USD)")
    ax2.set_ylabel("PnL (USD)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df_price["ts"], df_price["c"], color="#adb5bd", linewidth=0.6, label="BTC Fiyat")
    longs  = df_t[df_t["direction"] == "LONG"]
    shorts = df_t[df_t["direction"] == "SHORT"]
    wins_l = longs[longs["result"] == "WIN"]
    loss_l = longs[longs["result"] == "LOSS"]
    wins_s = shorts[shorts["result"] == "WIN"]
    loss_s = shorts[shorts["result"] == "LOSS"]
    ax3.scatter(wins_l["entry_ts"],  wins_l["entry"],  marker="^", color="#06d6a0", s=60, zorder=5, label="Long WIN")
    ax3.scatter(loss_l["entry_ts"],  loss_l["entry"],  marker="^", color="#ef476f", s=60, zorder=5, label="Long LOSS")
    ax3.scatter(wins_s["entry_ts"],  wins_s["entry"],  marker="v", color="#06d6a0", s=60, zorder=5, label="Short WIN")
    ax3.scatter(loss_s["entry_ts"],  loss_s["entry"],  marker="v", color="#ef476f", s=60, zorder=5, label="Short LOSS")
    ax3.set_title("BTC Fiyat + Giriş Noktaları")
    ax3.set_ylabel("Fiyat (USD)")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_result_v2.png", dpi=150, bbox_inches="tight")
    print("\nGrafik kaydedildi: backtest_result_v2.png")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_ohlcv(months=12)

    print("Sinyaller hesaplanıyor...")
    signals = generate_signals(df)
    print(f"Toplam {len(signals)} sinyal bulundu.")

    print("Backtest çalıştırılıyor...")
    trades, final_cap = run_backtest(df, signals)

    df_trades = analyze(trades, final_cap)

    if df_trades is not None and len(df_trades) > 0:
        plot_results(df_trades, df)
