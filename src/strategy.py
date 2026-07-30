"""
Strategi Trading:
1. OVERNIGHT  — Beli sore (menjelang penutupan), jual pagi (saat/sesudah opening)
2. INTRADAY   — Beli pagi (saat opening), jual sore (menjelang penutupan)

Termasuk:
- Harga beli & jual yang disarankan
- Stop loss & target profit
- Win rate historis berdasarkan kondisi serupa
- Analisis gap historis
- Skor kelayakan tiap strategi
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from src.patterns import CandlePattern, detect_patterns, pattern_score


@dataclass
class PriceLevel:
    entry: float = 0.0
    target_1: float = 0.0   # target konservatif
    target_2: float = 0.0   # target optimis
    stop_loss: float = 0.0
    risk_reward: float = 0.0
    potential_profit_pct: float = 0.0
    potential_loss_pct: float = 0.0


@dataclass
class StrategyResult:
    # Overnight
    overnight_score: int = 0
    overnight_feasible: bool = False
    overnight_price: PriceLevel = field(default_factory=PriceLevel)
    overnight_signals: list[str] = field(default_factory=list)
    overnight_warnings: list[str] = field(default_factory=list)
    overnight_win_rate: float = 0.0
    overnight_avg_gap: float = 0.0
    overnight_gap_freq: float = 0.0  # % hari gap up dari total

    # Intraday
    intraday_score: int = 0
    intraday_feasible: bool = False
    intraday_price: PriceLevel = field(default_factory=PriceLevel)
    intraday_signals: list[str] = field(default_factory=list)
    intraday_warnings: list[str] = field(default_factory=list)
    intraday_win_rate: float = 0.0
    intraday_avg_range_pct: float = 0.0  # rata-rata range intraday

    # Pattern
    candle_patterns: list[CandlePattern] = field(default_factory=list)
    pattern_score: int = 0

    # Historical
    hist_bullish_days_pct: float = 0.0   # % hari bullish dalam 30 hari terakhir
    hist_avg_daily_return: float = 0.0
    hist_best_return: float = 0.0
    hist_worst_return: float = 0.0
    hist_volatility: float = 0.0


def _calc_historical_stats(df: pd.DataFrame) -> dict:
    """Hitung statistik historis 30-90 hari."""
    close = df["close"]
    daily_ret = close.pct_change().dropna() * 100

    last_30 = daily_ret.tail(30)
    stats = {
        "bullish_days_pct": (last_30 > 0).sum() / len(last_30) * 100,
        "avg_daily_return": float(last_30.mean()),
        "best_return": float(last_30.max()),
        "worst_return": float(last_30.min()),
        "volatility": float(last_30.std()),
        "avg_range_pct": float(((df["high"] - df["low"]) / df["close"]).tail(30).mean() * 100),
    }
    return stats


def _calc_gap_stats(df: pd.DataFrame) -> dict:
    """Hitung statistik gap harian (open vs close sebelumnya)."""
    gaps = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100).dropna()
    gap_up = gaps[gaps > 0.5]   # gap up > 0.5%
    gap_down = gaps[gaps < -0.5]

    return {
        "gap_up_freq": len(gap_up) / len(gaps) * 100,
        "gap_down_freq": len(gap_down) / len(gaps) * 100,
        "avg_gap_up": float(gap_up.mean()) if len(gap_up) > 0 else 0.0,
        "avg_gap_down": float(gap_down.mean()) if len(gap_down) > 0 else 0.0,
        "last_gap": float(gaps.iloc[-1]) if len(gaps) > 0 else 0.0,
    }


def _calc_historical_win_rate(df: pd.DataFrame, strategy: str) -> float:
    """
    Estimasi win rate historis:
    - overnight: beli di close, cek apakah open besok lebih tinggi
    - intraday: beli di open, cek apakah close > open (hari sama)
    """
    wins = 0
    total = min(60, len(df) - 1)

    for i in range(1, total + 1):
        if strategy == "overnight":
            # Beli di close[i-1], jual di open[i]
            entry = float(df["close"].iloc[-i - 1])
            exit_ = float(df["open"].iloc[-i])
            if exit_ > entry * 1.002:  # profit > 0.2% setelah biaya
                wins += 1
        elif strategy == "intraday":
            # Beli di open[i], jual di close[i]
            entry = float(df["open"].iloc[-i])
            exit_ = float(df["close"].iloc[-i])
            if exit_ > entry * 1.003:  # profit > 0.3% setelah biaya
                wins += 1

    return (wins / total * 100) if total > 0 else 0.0


def _overnight_price_levels(df: pd.DataFrame, atr: float) -> PriceLevel:
    """Hitung harga entry/target/stoploss untuk strategi overnight."""
    last_close = float(df["close"].iloc[-1])
    last_high  = float(df["high"].iloc[-1])
    last_low   = float(df["low"].iloc[-1])

    # Entry: harga penutupan atau sedikit di bawah (beli menjelang close)
    entry = last_close

    # Target 1: Resistance terdekat atau close + 1.5x ATR
    # Gunakan pivot resistance sederhana dari 20 hari terakhir
    recent_highs = df["high"].tail(20)
    res_levels = sorted([h for h in recent_highs if h > last_close])
    target_1 = res_levels[0] if res_levels else last_close + 1.5 * atr
    target_2 = res_levels[1] if len(res_levels) > 1 else last_close + 2.5 * atr

    # Stop loss: support terdekat atau close - 1x ATR
    recent_lows = df["low"].tail(20)
    sup_levels = sorted([l for l in recent_lows if l < last_close], reverse=True)
    stop_loss = sup_levels[0] if sup_levels else last_close - atr

    pot_profit = target_1 - entry
    pot_loss   = entry - stop_loss
    rr = pot_profit / pot_loss if pot_loss > 0 else 0

    return PriceLevel(
        entry=entry,
        target_1=target_1,
        target_2=target_2,
        stop_loss=stop_loss,
        risk_reward=rr,
        potential_profit_pct=(pot_profit / entry) * 100,
        potential_loss_pct=(pot_loss / entry) * 100,
    )


def _intraday_price_levels(df: pd.DataFrame, atr: float) -> PriceLevel:
    """Hitung harga entry/target/stoploss untuk strategi intraday."""
    last_close = float(df["close"].iloc[-1])
    last_high  = float(df["high"].iloc[-1])

    # Entry: estimasi harga buka besok (close ± sedikit gap)
    # Gunakan rata-rata gap historis sebagai estimasi
    avg_gap = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)).tail(10).mean()
    est_open = last_close * (1 + avg_gap)

    entry = round(est_open, 0)

    # Target: entry + 1.5x ATR (konservatif) dan entry + 2.5x ATR (optimis)
    target_1 = entry + 1.5 * atr
    target_2 = entry + 2.5 * atr

    # Stop loss: entry - 1x ATR atau di bawah VWAP estimasi
    stop_loss = entry - atr

    pot_profit = target_1 - entry
    pot_loss   = entry - stop_loss
    rr = pot_profit / pot_loss if pot_loss > 0 else 0

    return PriceLevel(
        entry=entry,
        target_1=target_1,
        target_2=target_2,
        stop_loss=stop_loss,
        risk_reward=rr,
        potential_profit_pct=(pot_profit / entry) * 100,
        potential_loss_pct=(pot_loss / entry) * 100,
    )


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low  = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def analyze_strategy(df: pd.DataFrame) -> StrategyResult:
    result = StrategyResult()

    if df is None or len(df) < 30:
        return result

    close  = df["close"]
    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    last_open  = float(open_.iloc[-1])
    last_high  = float(high.iloc[-1])
    last_low   = float(low.iloc[-1])
    last_vol   = float(volume.iloc[-1])
    vol_avg    = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio  = last_vol / vol_avg if vol_avg > 0 else 1.0
    atr_val    = _atr(df)

    # Statistik historis
    hist = _calc_historical_stats(df)
    gap  = _calc_gap_stats(df)

    result.hist_bullish_days_pct  = hist["bullish_days_pct"]
    result.hist_avg_daily_return  = hist["avg_daily_return"]
    result.hist_best_return       = hist["best_return"]
    result.hist_worst_return      = hist["worst_return"]
    result.hist_volatility        = hist["volatility"]
    result.intraday_avg_range_pct = hist["avg_range_pct"]
    result.overnight_avg_gap      = gap["avg_gap_up"]
    result.overnight_gap_freq     = gap["gap_up_freq"]

    # Win rate historis
    result.overnight_win_rate = _calc_historical_win_rate(df, "overnight")
    result.intraday_win_rate  = _calc_historical_win_rate(df, "intraday")

    # Candlestick patterns
    result.candle_patterns = detect_patterns(df)
    result.pattern_score   = pattern_score(result.candle_patterns)

    # ─── ANALISIS OVERNIGHT ───────────────────────────────────────
    on_score = 0
    on_sig   = []
    on_warn  = []

    # Syarat utama: penutupan kuat (close mendekati high)
    close_position = (last_close - last_low) / (last_high - last_low) if (last_high - last_low) > 0 else 0.5
    if close_position >= 0.75:
        on_score += 3
        on_sig.append(f"Penutupan kuat di {close_position*100:.0f}% dari range hari ini — bullish close")
    elif close_position >= 0.55:
        on_score += 1
        on_sig.append("Penutupan di atas tengah range — cukup bullish")
    elif close_position < 0.35:
        on_score -= 2
        on_warn.append("Penutupan lemah — seller masih dominan saat close")

    # Historis gap up
    if gap["gap_up_freq"] >= 50:
        on_score += 2
        on_sig.append(f"Historis: {gap['gap_up_freq']:.0f}% hari gap up rata-rata +{gap['avg_gap_up']:.1f}%")
    elif gap["gap_up_freq"] >= 35:
        on_score += 1
        on_sig.append(f"Historis gap up cukup sering ({gap['gap_up_freq']:.0f}% hari)")

    # Win rate overnight
    if result.overnight_win_rate >= 60:
        on_score += 2
        on_sig.append(f"Win rate overnight historis tinggi: {result.overnight_win_rate:.0f}%")
    elif result.overnight_win_rate >= 50:
        on_score += 1
        on_sig.append(f"Win rate overnight: {result.overnight_win_rate:.0f}%")
    elif result.overnight_win_rate < 40:
        on_score -= 2
        on_warn.append(f"Win rate overnight rendah: {result.overnight_win_rate:.0f}%")

    # Volume sore (konfirmasi akumulasi menjelang close)
    if vol_ratio >= 1.5:
        on_score += 2
        on_sig.append(f"Volume tinggi menjelang close ({vol_ratio:.1f}x) — institutional buying")
    elif vol_ratio >= 1.0:
        on_score += 1

    # Candle patterns
    if result.pattern_score >= 2:
        on_score += 2
        bullish_patterns = [p.name for p in result.candle_patterns if p.signal == "BULLISH"]
        on_sig.append(f"Pola candle bullish: {', '.join(bullish_patterns)}")
    elif result.pattern_score <= -2:
        on_score -= 2
        on_warn.append("Pola candle bearish terdeteksi")

    # Tren intraday (apakah hari ini bullish)
    if last_close > last_open:
        on_score += 1
        on_sig.append(f"Hari ini bullish (+{((last_close-last_open)/last_open*100):.1f}% dari open)")
    else:
        on_warn.append("Hari ini merah — risiko lanjutan turun")

    # % hari bullish dalam 30 hari
    if hist["bullish_days_pct"] >= 60:
        on_score += 1
        on_sig.append(f"Historis: {hist['bullish_days_pct']:.0f}% hari bullish dalam 30 hari terakhir")

    result.overnight_score    = on_score
    result.overnight_feasible = on_score >= 6
    result.overnight_signals  = on_sig
    result.overnight_warnings = on_warn
    result.overnight_price    = _overnight_price_levels(df, atr_val)

    # ─── ANALISIS INTRADAY ────────────────────────────────────────
    id_score = 0
    id_sig   = []
    id_warn  = []

    # Volatilitas harian cukup untuk profit intraday
    avg_range = hist["avg_range_pct"]
    if avg_range >= 3.0:
        id_score += 3
        id_sig.append(f"Volatilitas intraday tinggi (rata-rata {avg_range:.1f}% per hari) — peluang profit besar")
    elif avg_range >= 1.5:
        id_score += 2
        id_sig.append(f"Volatilitas intraday cukup ({avg_range:.1f}% per hari)")
    elif avg_range < 1.0:
        id_score -= 1
        id_warn.append(f"Volatilitas rendah ({avg_range:.1f}%) — sulit profit intraday")

    # Win rate intraday
    if result.intraday_win_rate >= 60:
        id_score += 3
        id_sig.append(f"Win rate intraday historis tinggi: {result.intraday_win_rate:.0f}%")
    elif result.intraday_win_rate >= 50:
        id_score += 1
        id_sig.append(f"Win rate intraday: {result.intraday_win_rate:.0f}%")
    elif result.intraday_win_rate < 40:
        id_score -= 2
        id_warn.append(f"Win rate intraday rendah: {result.intraday_win_rate:.0f}%")

    # Volume (likuiditas = mudah masuk/keluar)
    if vol_ratio >= 1.5:
        id_score += 2
        id_sig.append(f"Likuiditas tinggi (volume {vol_ratio:.1f}x rata-rata) — mudah entry/exit")
    elif vol_ratio < 0.5:
        id_score -= 2
        id_warn.append(f"Likuiditas rendah ({vol_ratio:.1f}x) — risiko slippage saat jual")

    # Tren harian konsisten
    last_5 = close.tail(5)
    up_days = sum(1 for i in range(1, len(last_5)) if last_5.iloc[i] > last_5.iloc[i-1])
    if up_days >= 4:
        id_score += 2
        id_sig.append(f"{up_days} dari 5 hari terakhir bullish — momentum kuat")
    elif up_days >= 3:
        id_score += 1
        id_sig.append(f"{up_days} dari 5 hari terakhir bullish")
    elif up_days <= 1:
        id_score -= 1
        id_warn.append("Tren 5 hari terakhir lemah")

    # Rata-rata return harian positif
    if hist["avg_daily_return"] > 0.3:
        id_score += 1
        id_sig.append(f"Rata-rata return harian: +{hist['avg_daily_return']:.2f}% dalam 30 hari")
    elif hist["avg_daily_return"] < -0.3:
        id_score -= 1
        id_warn.append(f"Rata-rata return harian negatif: {hist['avg_daily_return']:.2f}%")

    # Pattern candle
    if result.pattern_score >= 2:
        id_score += 1
        id_sig.append("Pola candle mendukung bullish untuk besok")
    elif result.pattern_score <= -2:
        id_score -= 1

    result.intraday_score    = id_score
    result.intraday_feasible = id_score >= 6
    result.intraday_signals  = id_sig
    result.intraday_warnings = id_warn
    result.intraday_price    = _intraday_price_levels(df, atr_val)

    return result
