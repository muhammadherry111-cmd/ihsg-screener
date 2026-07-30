import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from src.bandarmologi import BandarSignal, analyze_bandar
from src.forecasting import ForecastResult, forecast
from src.strategy import StrategyResult, analyze_strategy
from src.swing import SwingResult, analyze_swing


@dataclass
class TechnicalSignal:
    ticker: str
    last_price: float
    change_pct: float
    score: int
    signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Indikator teknikal
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    ema9: float = 0.0
    ema21: float = 0.0
    sma50: float = 0.0
    bb_upper: float = 0.0
    bb_mid: float = 0.0
    bb_lower: float = 0.0
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    volume_ratio: float = 0.0

    # Bandarmologi, Forecast, Strategy & Swing
    bandar: BandarSignal = field(default_factory=BandarSignal)
    fc: ForecastResult = field(default_factory=ForecastResult)
    strat: StrategyResult = field(default_factory=StrategyResult)
    swing: SwingResult = field(default_factory=SwingResult)

    recommendation: str = "NETRAL"
    total_score: int = 0


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = _sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + (std_dev * std)
    lower = mid - (std_dev * std)
    return upper, mid, lower


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k, d


def analyze(ticker: str, df: pd.DataFrame) -> "TechnicalSignal | None":
    if df is None or len(df) < 30:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    change_pct = ((last_close - prev_close) / prev_close) * 100

    # --- Indikator Teknikal ---
    rsi_series = _rsi(close)
    macd_line, signal_line, histogram = _macd(close)
    ema9_series = _ema(close, 9)
    ema21_series = _ema(close, 21)
    sma50_series = _sma(close, 50)
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
    stoch_k, stoch_d = _stochastic(high, low, close)
    vol_avg = volume.rolling(20).mean()

    rsi_val = float(rsi_series.iloc[-1])
    macd_val = float(macd_line.iloc[-1])
    macd_sig = float(signal_line.iloc[-1])
    macd_hist = float(histogram.iloc[-1])
    ema9_val = float(ema9_series.iloc[-1])
    ema21_val = float(ema21_series.iloc[-1])
    sma50_val = float(sma50_series.iloc[-1]) if not np.isnan(sma50_series.iloc[-1]) else 0.0
    bb_u = float(bb_upper.iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_l = float(bb_lower.iloc[-1])
    stk = float(stoch_k.iloc[-1])
    std = float(stoch_d.iloc[-1])
    vol_last = float(volume.iloc[-1])
    vol_avg_val = float(vol_avg.iloc[-1]) if not np.isnan(vol_avg.iloc[-1]) else 1.0
    vol_ratio = vol_last / vol_avg_val if vol_avg_val > 0 else 1.0

    score = 0
    signals = []
    warnings = []

    # RSI
    if rsi_val < 30:
        score += 3
        signals.append(f"RSI oversold ({rsi_val:.1f}) — potensi rebound")
    elif rsi_val < 45:
        score += 1
        signals.append(f"RSI zona recovery ({rsi_val:.1f})")
    elif rsi_val > 70:
        score -= 2
        warnings.append(f"RSI overbought ({rsi_val:.1f}) — hati-hati pembalikan")
    elif rsi_val > 60:
        score += 1
        signals.append(f"RSI momentum kuat ({rsi_val:.1f})")

    # MACD
    if macd_val > macd_sig and macd_hist > 0:
        if float(histogram.iloc[-2]) < 0:
            score += 3
            signals.append("MACD golden cross — baru bullish crossover")
        else:
            score += 2
            signals.append("MACD bullish (line di atas signal)")
    elif macd_val < macd_sig and macd_hist < 0:
        if float(histogram.iloc[-2]) > 0:
            score -= 2
            warnings.append("MACD death cross — baru bearish crossover")
        else:
            score -= 1
            warnings.append("MACD bearish")

    # EMA Trend
    if last_close > ema9_val > ema21_val:
        score += 2
        signals.append("Harga > EMA9 > EMA21 (uptrend kuat)")
    elif last_close > ema9_val:
        score += 1
        signals.append("Harga di atas EMA9")
    elif last_close < ema9_val < ema21_val:
        score -= 1
        warnings.append("Harga di bawah EMA9 & EMA21 (downtrend)")

    # SMA50
    if sma50_val > 0:
        if last_close > sma50_val:
            score += 1
            signals.append("Harga di atas SMA50")
        else:
            warnings.append("Harga di bawah SMA50")

    # Bollinger Bands
    bb_range = bb_u - bb_l
    if bb_range > 0:
        bb_pos = (last_close - bb_l) / bb_range
        if bb_pos < 0.2:
            score += 2
            signals.append("Dekat lower Bollinger Band — potential bounce")
        elif bb_pos > 0.8:
            score -= 1
            warnings.append("Dekat upper Bollinger Band — resistensi")
        elif 0.4 <= bb_pos <= 0.6:
            score += 1
            signals.append("Tengah Bollinger Band — momentum stabil")

    # Stochastic
    if stk < 20 and std < 20:
        score += 2
        signals.append(f"Stochastic oversold ({stk:.1f}) — potensi naik")
    elif stk > 80 and std > 80:
        score -= 1
        warnings.append(f"Stochastic overbought ({stk:.1f})")
    elif stk > std and stk < 80:
        score += 1
        signals.append("Stochastic bullish crossover")

    # Volume
    if vol_ratio >= 2.0:
        score += 2
        signals.append(f"Volume sangat tinggi ({vol_ratio:.1f}x) — minat besar")
    elif vol_ratio >= 1.5:
        score += 1
        signals.append(f"Volume di atas rata-rata ({vol_ratio:.1f}x)")
    elif vol_ratio < 0.5:
        score -= 1
        warnings.append(f"Volume rendah ({vol_ratio:.1f}x)")

    # Perubahan harga harian
    if change_pct > 3:
        score += 1
        signals.append(f"Momentum positif hari ini (+{change_pct:.2f}%)")
    elif change_pct < -3:
        score -= 1
        warnings.append(f"Koreksi hari ini ({change_pct:.2f}%)")

    # --- Bandarmologi ---
    bandar = analyze_bandar(df)

    # --- Forecasting ---
    fc = forecast(df)

    # --- Strategy ---
    strat = analyze_strategy(df)

    # --- Swing ---
    sw = analyze_swing(df)

    # Bonus dari forecast
    fc_bonus = 0
    if fc.trend_direction == "NAIK" and fc.confidence in ("SEDANG", "TINGGI"):
        fc_bonus += 1
    if fc.risk_reward >= 2.0:
        fc_bonus += 1
    if fc.risk_reward < 1.0:
        fc_bonus -= 1

    # Bonus dari strategy
    strat_bonus = 0
    if strat.overnight_feasible:
        strat_bonus += 1
    if strat.intraday_feasible:
        strat_bonus += 1
    if strat.pattern_score >= 3:
        strat_bonus += 1
    if strat.pattern_score <= -3:
        strat_bonus -= 1

    # Total skor gabungan
    total = score + bandar.bandar_score + fc_bonus + strat_bonus

    # --- Rekomendasi ---
    if total >= 10:
        recommendation = "STRONG BUY"
    elif total >= 6:
        recommendation = "BUY"
    elif total >= 3:
        recommendation = "NETRAL+"
    elif total >= 0:
        recommendation = "NETRAL"
    elif total >= -3:
        recommendation = "NETRAL-"
    else:
        recommendation = "HINDARI"

    return TechnicalSignal(
        ticker=ticker,
        last_price=last_close,
        change_pct=change_pct,
        score=score,
        signals=signals,
        warnings=warnings,
        rsi=rsi_val,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        ema9=ema9_val,
        ema21=ema21_val,
        sma50=sma50_val,
        bb_upper=bb_u,
        bb_mid=bb_m,
        bb_lower=bb_l,
        stoch_k=stk,
        stoch_d=std,
        volume_ratio=vol_ratio,
        bandar=bandar,
        fc=fc,
        strat=strat,
        swing=sw,
        recommendation=recommendation,
        total_score=total,
    )
