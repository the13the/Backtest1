"""
BTC/USDT — Bybit 15 Dakikalık Backtest
Son 6 ay verisi, orijinal strateji mantığı korundu.
(Binance ABD lokasyonlarında 451 kısıtlaması verdiği için Bybit kullanılıyor)

Kurulum:
    pip install ccxt pandas numpy matplotlib

Çalıştırma:
    python backtest_v3.py
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────
# PARAMETRELER (orijinal botla aynı)
# ─────────────────────────────────────────
SYMBOL        = "BTC/USDT"
TIMEFRAME     = "15m"
LEVERAGE      = 50
RISK_PER_TRADE_USD = 10
SAFE_RISK_USD = 5
RR            = 1.8
INITIAL_CAPITAL = 1000.0   # Başlangıç bakiyesi (USD)

# ─────────────────────────────────────────
# VERİ ÇEKİMİ
# ─────────────────────────────────────────

def fetch_kucoin_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, months=6):
    """Binance'ten son 6 aylık 15 dakikalık mum verisini çeker."""
    exchange = ccxt.kucoin({"enableRateLimit": True})

    since_dt = datetime.now(timezone.utc) - timedelta(days=months * 30)
    since_ms  = int(since_dt.timestamp() * 1000)

    print(f"Veri çekiliyor: {symbol} {timeframe} | Başlangıç: {since_dt.strftime('%Y-%m-%d')}")

    all_bars = []
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 60_000:
            break
        since_ms = last_ts + 1
        print(f"  {len(all_bars)} mum indirildi...", end="\r")

    df = pd.DataFrame(all_bars, columns=["ts", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    print(f"\nToplam {len(df)} mum yüklendi.")
    return df


# ─────────────────────────────────────────
# GÖSTERGELEr (orijinal botla birebir)
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
# STRATEJİ SİNYALİ (orijinal mantık)
# ─────────────────────────────────────────

def generate_signals(df):
    """
    Orijinal strateji:
      Trend: EMA50 > EMA200 VE kapanış EMA50 üzerinde → LONG
             EMA50 < EMA200 VE kapanış EMA50 altında  → SHORT
      Sinyal: open breakout bölgesinde, close kesin dışarıda (gövde kırılımı)
    """
    df = df.copy()

    df["ema50"]  = calc_ema(df["c"], 50)
    df["ema200"] = calc_ema(df["c"], 200)
    df["atr"]    = calc_atr(df, 14)

    # 20 bar yüksek/düşük (önceki bara kadar — shift(1) ile lookahead önlenir)
    df["roll_high"] = df["h"].rolling(20).max().shift(1)
    df["roll_low"]  = df["l"].rolling(20).min().shift(1)

    # Swing SL: 10 bar min/max (shift(2) → son kapanan barın önceki swingı)
    df["swing_low"]  = df["l"].rolling(10).min().shift(2)
    df["swing_high"] = df["h"].rolling(10).max().shift(2)

    signals = []

    for i in range(200, len(df)):
        row = df.iloc[i]

        ema50  = row["ema50"]
        ema200 = row["ema200"]
        price  = row["c"]

        if pd.isna(ema50) or pd.isna(ema200):
            continue

        # Trend yönü
        if ema50 > ema200 and price > ema50:
            direction = "LONG"
        elif ema50 < ema200 and price < ema50:
            direction = "SHORT"
        else:
            continue   # Belirsiz/chop

        o = row["o"]
        c = row["c"]
        last_high = row["roll_high"]
        last_low  = row["roll_low"]
        atr_val   = row["atr"]

        if pd.isna(last_high) or pd.isna(last_low) or pd.isna(atr_val):
            continue

        sig = None
        if direction == "LONG"  and o <= last_high and c > last_high:
            sig = "LONG"
        elif direction == "SHORT" and o >= last_low  and c < last_low:
            sig = "SHORT"

        if sig:
            signals.append({
                "idx":       i,
                "ts":        row["ts"],
                "direction": sig,
                "entry":     c,
                "swing_low": row["swing_low"],
                "swing_high":row["swing_high"],
                "atr":       atr_val,
            })

    return signals


# ─────────────────────────────────────────
# SL / TP HESABI (orijinal botla aynı)
# ─────────────────────────────────────────

def smart_stop(sig):
    entry = sig["entry"]
    atr   = sig["atr"]
    if sig["direction"] == "LONG":
        swing = sig["swing_low"]
        return min(swing, entry - atr) if not pd.isna(swing) else entry - atr
    else:
        swing = sig["swing_high"]
        return max(swing, entry + atr) if not pd.isna(swing) else entry + atr


def smart_tp(entry, sl, direction):
    risk = abs(entry - sl)
    return entry + risk * RR if direction == "LONG" else entry - risk * RR


def estimate_liq(entry, direction):
    move = entry / LEVERAGE
    return (entry - move) if direction == "LONG" else (entry + move)


def risk_amount(entry, sl, direction):
    liq     = estimate_liq(entry, direction)
    sl_gap  = abs(entry - sl)
    liq_gap = abs(entry - liq)
    return SAFE_RISK_USD if liq_gap <= sl_gap * 1.3 else RISK_PER_TRADE_USD


def position_size(entry, sl, direction):
    dist = abs(entry - sl)
    if dist <= 0:
        return 0
    risk = risk_amount(entry, sl, direction)
    return risk / dist


# ─────────────────────────────────────────
# BACKTEST MOTORU
# ─────────────────────────────────────────

def run_backtest(df, signals):
    capital   = INITIAL_CAPITAL
    trades    = []
    open_trade = None
    used_candle_ts = set()  # Aynı muma iki kez girmemek için

    price_arr = df["c"].values
    high_arr  = df["h"].values
    low_arr   = df["l"].values
    ts_arr    = df["ts"].values

    sig_map = {s["idx"]: s for s in signals}

    i = 0
    while i < len(df):
        # Açık trade takibi
        if open_trade:
            h = high_arr[i]
            l = low_arr[i]
            sl = open_trade["sl"]
            tp = open_trade["tp"]
            direction = open_trade["direction"]

            hit_sl = (direction == "LONG"  and l <= sl) or \
                     (direction == "SHORT" and h >= sl)
            hit_tp = (direction == "LONG"  and h >= tp) or \
                     (direction == "SHORT" and l <= tp)

            # Her ikisi aynı mumda: hangisi önce? yaklaşım → SL öncelikli (muhafazakar)
            if hit_sl or hit_tp:
                exit_price  = sl if hit_sl else tp
                exit_ts     = ts_arr[i]
                pnl_per_unit = (exit_price - open_trade["entry"]) if direction == "LONG" \
                               else (open_trade["entry"] - exit_price)
                pnl_usd = pnl_per_unit * open_trade["qty"]
                capital += pnl_usd

                trades.append({
                    "entry_ts":    open_trade["entry_ts"],
                    "exit_ts":     exit_ts,
                    "direction":   direction,
                    "entry":       open_trade["entry"],
                    "exit":        exit_price,
                    "sl":          sl,
                    "tp":          tp,
                    "qty":         open_trade["qty"],
                    "pnl_usd":     round(pnl_usd, 4),
                    "result":      "WIN" if hit_tp else "LOSS",
                    "capital":     round(capital, 2),
                })
                open_trade = None

        # Yeni sinyal var mı?
        if i in sig_map and open_trade is None:
            sig = sig_map[i]
            if sig["ts"] not in used_candle_ts:
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
                    used_candle_ts.add(sig["ts"])

        i += 1

    return trades, capital


# ─────────────────────────────────────────
# SONUÇ ANALİZİ
# ─────────────────────────────────────────

def analyze(trades, final_capital):
    if not trades:
        print("Hiç trade yok.")
        return

    df_t = pd.DataFrame(trades)
    df_t["entry_ts"] = pd.to_datetime(df_t["entry_ts"])
    df_t["exit_ts"]  = pd.to_datetime(df_t["exit_ts"])

    total   = len(df_t)
    wins    = (df_t["result"] == "WIN").sum()
    losses  = (df_t["result"] == "LOSS").sum()
    win_rate = wins / total * 100

    gross_profit = df_t.loc[df_t["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss   = df_t.loc[df_t["pnl_usd"] < 0, "pnl_usd"].sum()
    net_pnl      = df_t["pnl_usd"].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = df_t.loc[df_t["result"] == "WIN",  "pnl_usd"].mean()
    avg_loss = df_t.loc[df_t["result"] == "LOSS", "pnl_usd"].mean()

    # Max Drawdown
    equity = df_t["capital"].values
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe (günlük PnL baz)
    daily_pnl = df_t.set_index("exit_ts")["pnl_usd"].resample("D").sum()
    sharpe    = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

    print("\n" + "="*52)
    print("          BACKTEST SONUÇLARI — BTC/USDT 15m")
    print("="*52)
    print(f"  Başlangıç Sermaye  : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Sermaye      : ${final_capital:,.2f}")
    print(f"  Net PnL            : ${net_pnl:+,.2f}  ({(final_capital/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"  Toplam Trade       : {total}")
    print(f"  Kazanan            : {wins}  ({win_rate:.1f}%)")
    print(f"  Kaybeden           : {losses}")
    print(f"  Ort. Kazanç        : ${avg_win:+.2f}")
    print(f"  Ort. Kayıp         : ${avg_loss:+.2f}")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print(f"  Max Drawdown       : {max_dd:.2f}%")
    print(f"  Sharpe (yıllık)    : {sharpe:.2f}")
    print("="*52)

    # Son 10 trade
    print("\n── Son 10 Trade ──")
    cols = ["entry_ts", "direction", "entry", "exit", "pnl_usd", "result", "capital"]
    print(df_t[cols].tail(10).to_string(index=False))

    return df_t


# ─────────────────────────────────────────
# GRAFİK
# ─────────────────────────────────────────

def plot_results(df_t, df_price):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("BTC/USDT 15m — Backtest Sonuçları", fontsize=14, fontweight="bold")

    # 1) Equity Curve
    ax1 = axes[0]
    ax1.plot(df_t["exit_ts"], df_t["capital"], color="#00b4d8", linewidth=1.5)
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8, label="Başlangıç")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Sermaye (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2) Trade PnL Bar
    ax2 = axes[1]
    colors = ["#06d6a0" if r == "WIN" else "#ef476f" for r in df_t["result"]]
    ax2.bar(range(len(df_t)), df_t["pnl_usd"], color=colors, width=0.8)
    ax2.axhline(0, color="white", linewidth=0.5)
    ax2.set_title("Trade Başına PnL (USD)")
    ax2.set_ylabel("PnL (USD)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    # 3) BTC Fiyat + Giriş noktaları
    ax3 = axes[2]
    price_plot = df_price.set_index("ts")["c"]
    ax3.plot(price_plot.index, price_plot.values, color="#adb5bd", linewidth=0.6, label="BTC Fiyat")

    longs  = df_t[df_t["direction"] == "LONG"]
    shorts = df_t[df_t["direction"] == "SHORT"]
    wins_  = df_t[df_t["result"] == "WIN"]
    losses_= df_t[df_t["result"] == "LOSS"]

    ax3.scatter(longs["entry_ts"],  longs["entry"],  marker="^", color="#06d6a0", s=40, zorder=5, label="Long Giriş")
    ax3.scatter(shorts["entry_ts"], shorts["entry"], marker="v", color="#ef476f", s=40, zorder=5, label="Short Giriş")
    ax3.set_title("BTC Fiyat + Giriş Noktaları")
    ax3.set_ylabel("Fiyat (USD)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
    print("\nGrafik kaydedildi: backtest_result.png")
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    # 1) Veri çek
    df = fetch_kucoin_ohlcv(months=6)

    # 2) Sinyaller üret
    print("Sinyaller hesaplanıyor...")
    signals = generate_signals(df)
    print(f"Toplam {len(signals)} sinyal bulundu.")

    # 3) Backtest çalıştır
    print("Backtest çalışıyor...")
    trades, final_cap = run_backtest(df, signals)

    # 4) Analiz
    df_trades = analyze(trades, final_cap)

    # 5) Grafik
    if df_trades is not None and len(df_trades) > 0:
        plot_results(df_trades, df)
