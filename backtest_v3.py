"""
BTC/USDT — 15 Dakikalık Backtest
Veri kaynagi: yfinance (GitHub Actions ile uyumlu, kisitlama yok)
Son 6 ay verisi, orijinal strateji mantigi korundu.

Kurulum:
    pip install yfinance pandas numpy matplotlib

Calistirma:
    python backtest_v3.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GitHub Actions: ekransiz ortam icin
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────
# PARAMETRELER
# ─────────────────────────────────────────
SYMBOL             = "BTC-USD"
TIMEFRAME          = "15m"
LEVERAGE           = 50
RISK_PER_TRADE_USD = 10
SAFE_RISK_USD      = 5
RR                 = 1.8
INITIAL_CAPITAL    = 1000.0

# ─────────────────────────────────────────
# VERI CEKIMI
# ─────────────────────────────────────────

def fetch_ohlcv(months=6):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    print(f"Veri cekiliyor: {SYMBOL} {TIMEFRAME} | {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")

    # yfinance 15m icin max 60 gun destekler, parcali cekilmeli
    all_chunks = []
    chunk_end = end
    while chunk_end > start:
        chunk_start = max(chunk_end - timedelta(days=58), start)
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

    # Sutun adlarini standartlastir
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    df["ts"] = df.index
    df = df.reset_index(drop=True)

    print(f"Toplam {len(df)} mum yuklendi.")
    return df

# ─────────────────────────────────────────
# GOSTERGELER
# ─────────────────────────────────────────

def calc_atr(df, period=14):
    hl = df["h"] - df["l"]
    hc = (df["h"] - df["c"].shift()).abs()
    lc = (df["l"] - df["c"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ─────────────────────────────────────────
# STRATEJI SINYALI
# ─────────────────────────────────────────

def generate_signals(df):
    df = df.copy()
    df["ema50"]      = calc_ema(df["c"], 50)
    df["ema200"]     = calc_ema(df["c"], 200)
    df["atr"]        = calc_atr(df, 14)
    df["roll_high"]  = df["h"].rolling(20).max().shift(1)
    df["roll_low"]   = df["l"].rolling(20).min().shift(1)
    df["swing_low"]  = df["l"].rolling(10).min().shift(2)
    df["swing_high"] = df["h"].rolling(10).max().shift(2)

    signals = []
    for i in range(200, len(df)):
        row   = df.iloc[i]
        ema50 = row["ema50"]
        ema200= row["ema200"]
        price = row["c"]

        if pd.isna(ema50) or pd.isna(ema200):
            continue

        if   ema50 > ema200 and price > ema50:  direction = "LONG"
        elif ema50 < ema200 and price < ema50:  direction = "SHORT"
        else: continue

        o         = row["o"]
        c         = row["c"]
        last_high = row["roll_high"]
        last_low  = row["roll_low"]
        atr_val   = row["atr"]

        if pd.isna(last_high) or pd.isna(last_low) or pd.isna(atr_val):
            continue

        sig = None
        if direction == "LONG"  and o <= last_high and c > last_high: sig = "LONG"
        if direction == "SHORT" and o >= last_low  and c < last_low:  sig = "SHORT"

        if sig:
            signals.append({
                "idx":        i,
                "ts":         row["ts"],
                "direction":  sig,
                "entry":      c,
                "swing_low":  row["swing_low"],
                "swing_high": row["swing_high"],
                "atr":        atr_val,
            })

    return signals

# ─────────────────────────────────────────
# SL / TP
# ─────────────────────────────────────────

def smart_stop(sig):
    entry = sig["entry"]
    atr   = sig["atr"]
    if sig["direction"] == "LONG":
        swing = sig["swing_low"]
        return min(swing, entry - atr) if not pd.isna(swing) else entry - atr
    swing = sig["swing_high"]
    return max(swing, entry + atr) if not pd.isna(swing) else entry + atr

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
# BACKTEST MOTORU
# ─────────────────────────────────────────

def run_backtest(df, signals):
    capital    = INITIAL_CAPITAL
    trades     = []
    open_trade = None
    used_ts    = set()

    high_arr = df["h"].values
    low_arr  = df["l"].values
    ts_arr   = df["ts"].values
    sig_map  = {s["idx"]: s for s in signals}

    for i in range(len(df)):
        if open_trade:
            h  = high_arr[i]
            l  = low_arr[i]
            sl = open_trade["sl"]
            tp = open_trade["tp"]
            d  = open_trade["direction"]

            hit_sl = (d == "LONG"  and l <= sl) or (d == "SHORT" and h >= sl)
            hit_tp = (d == "LONG"  and h >= tp) or (d == "SHORT" and l <= tp)

            if hit_sl or hit_tp:
                exit_price   = sl if hit_sl else tp
                pnl_per_unit = (exit_price - open_trade["entry"]) if d == "LONG" \
                               else (open_trade["entry"] - exit_price)
                pnl_usd  = pnl_per_unit * open_trade["qty"]
                capital += pnl_usd
                trades.append({
                    "entry_ts":  open_trade["entry_ts"],
                    "exit_ts":   ts_arr[i],
                    "direction": d,
                    "entry":     open_trade["entry"],
                    "exit":      exit_price,
                    "sl":        sl,
                    "tp":        tp,
                    "qty":       open_trade["qty"],
                    "pnl_usd":   round(pnl_usd, 4),
                    "result":    "WIN" if hit_tp else "LOSS",
                    "capital":   round(capital, 2),
                })
                open_trade = None

        if i in sig_map and open_trade is None:
            sig = sig_map[i]
            if sig["ts"] not in used_ts:
                entry = sig["entry"]
                sl    = smart_stop(sig)
                tp    = smart_tp(entry, sl, sig["direction"])
                qty   = position_size(entry, sl, sig["direction"])
                if qty > 0 and sl != entry:
                    open_trade = {
                        "entry_ts":  sig["ts"],
                        "direction": sig["direction"],
                        "entry":     entry,
                        "sl":        sl,
                        "tp":        tp,
                        "qty":       qty,
                    }
                    used_ts.add(sig["ts"])

    return trades, capital

# ─────────────────────────────────────────
# ANALIZ
# ─────────────────────────────────────────

def analyze(trades, final_capital):
    if not trades:
        print("Hic trade yok.")
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

    print("\n" + "="*52)
    print("      BACKTEST SONUCLARI — BTC/USDT 15m")
    print("="*52)
    print(f"  Baslangic Sermaye  : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Sermaye      : ${final_capital:,.2f}")
    print(f"  Net PnL            : ${net_pnl:+,.2f}  ({(final_capital/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"  Toplam Trade       : {total}")
    print(f"  Kazanan            : {wins}  ({win_rate:.1f}%)")
    print(f"  Kaybeden           : {losses}")
    print(f"  Ort. Kazanc        : ${avg_win:+.2f}")
    print(f"  Ort. Kayip         : ${avg_loss:+.2f}")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print(f"  Max Drawdown       : {max_dd:.2f}%")
    print(f"  Sharpe (yillik)    : {sharpe:.2f}")
    print("="*52)

    cols = ["entry_ts", "direction", "entry", "exit", "pnl_usd", "result", "capital"]
    print("\n-- Son 10 Trade --")
    print(df_t[cols].tail(10).to_string(index=False))

    return df_t

# ─────────────────────────────────────────
# GRAFIK
# ─────────────────────────────────────────

def plot_results(df_t, df_price):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("BTC/USDT 15m — Backtest Sonuclari", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(df_t["exit_ts"], df_t["capital"], color="#00b4d8", linewidth=1.5)
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8, label="Baslangic")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Sermaye (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = ["#06d6a0" if r == "WIN" else "#ef476f" for r in df_t["result"]]
    ax2.bar(range(len(df_t)), df_t["pnl_usd"], color=colors, width=0.8)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_title("Trade Basina PnL (USD)")
    ax2.set_ylabel("PnL (USD)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df_price["ts"], df_price["c"], color="#adb5bd", linewidth=0.6, label="BTC Fiyat")
    longs  = df_t[df_t["direction"] == "LONG"]
    shorts = df_t[df_t["direction"] == "SHORT"]
    ax3.scatter(longs["entry_ts"],  longs["entry"],  marker="^", color="#06d6a0", s=40, zorder=5, label="Long")
    ax3.scatter(shorts["entry_ts"], shorts["entry"], marker="v", color="#ef476f", s=40, zorder=5, label="Short")
    ax3.set_title("BTC Fiyat + Giris Noktalari")
    ax3.set_ylabel("Fiyat (USD)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
    print("\nGrafik kaydedildi: backtest_result.png")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_ohlcv(months=6)

    print("Sinyaller hesaplaniyor...")
    signals = generate_signals(df)
    print(f"Toplam {len(signals)} sinyal bulundu.")

    print("Backtest calistirilıyor...")
    trades, final_cap = run_backtest(df, signals)

    df_trades = analyze(trades, final_cap)

    if df_trades is not None and len(df_trades) > 0:
        plot_results(df_trades, df)
