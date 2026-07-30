"""
Forecasting harga saham menggunakan:
- Regresi linear (proyeksi tren)
- Support & Resistance (pivot points)
- ATR-based target & stop loss
- Fibonacci retracement
- Momentum projection
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ForecastResult:
    # Proyeksi harga
    forecast_3d: float = 0.0    # proyeksi 3 hari ke depan
    forecast_5d: float = 0.0    # proyeksi 5 hari ke depan
    trend_direction: str = "SIDEWAYS"  # NAIK / TURUN / SIDEWAYS
    confidence: str = "RENDAH"  # RENDAH / SEDANG / TINGGI

    # Support & Resistance
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    support_1: float = 0.0
    support_2: float = 0.0

    # Target & Stop Loss
    target_price: float = 0.0
    stop_loss: float = 0.0
    risk_reward: float = 0.0

    # Fibonacci
    fib_382: float = 0.0
    fib_500: float = 0.0
    fib_618: float = 0.0

    # ATR
    atr: float = 0.0
    atr_pct: float = 0.0

    signals: list[str] = field(default_factory=list)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _find_pivots(high: pd.Series, low: pd.Series, window: int = 5):
    """Temukan swing high dan swing low sebagai resistance/support."""
    pivot_highs = []
    pivot_lows = []

    for i in range(window, len(high) - window):
        h_window = high.iloc[i - window:i + window + 1]
        l_window = low.iloc[i - window:i + window + 1]
        if high.iloc[i] == h_window.max():
            pivot_highs.append(float(high.iloc[i]))
        if low.iloc[i] == l_window.min():
            pivot_lows.append(float(low.iloc[i]))

    return sorted(set(pivot_highs), reverse=True), sorted(set(pivot_lows), reverse=True)


def _linear_regression_forecast(close: pd.Series, days_ahead: int = 5) -> tuple[float, float, str]:
    """
    Regresi linear pada N candle terakhir, proyeksikan ke depan.
    Return: (forecast_value, r_squared, direction)
    """
    period = min(30, len(close))
    y = close.iloc[-period:].values.astype(float)
    x = np.arange(len(y))

    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Proyeksi ke depan
    future_x = len(y) - 1 + days_ahead
    forecast = slope * future_x + intercept

    direction = "NAIK" if slope > 0 else ("TURUN" if slope < 0 else "SIDEWAYS")
    return float(forecast), float(r2), direction


def _fibonacci_levels(swing_low: float, swing_high: float) -> tuple[float, float, float]:
    diff = swing_high - swing_low
    fib_382 = swing_high - 0.382 * diff
    fib_500 = swing_high - 0.500 * diff
    fib_618 = swing_high - 0.618 * diff
    return fib_382, fib_500, fib_618


def forecast(df: pd.DataFrame) -> ForecastResult:
    result = ForecastResult()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    last_close = float(close.iloc[-1])

    # --- ATR ---
    atr_series = _atr(high, low, close)
    result.atr = float(atr_series.iloc[-1])
    result.atr_pct = (result.atr / last_close) * 100

    # --- Regresi Linear ---
    fc_3d, r2_3d, dir_3d = _linear_regression_forecast(close, days_ahead=3)
    fc_5d, r2_5d, dir_5d = _linear_regression_forecast(close, days_ahead=5)
    result.forecast_3d = fc_3d
    result.forecast_5d = fc_5d
    result.trend_direction = dir_3d

    # Confidence berdasarkan R-squared
    avg_r2 = (r2_3d + r2_5d) / 2
    if avg_r2 >= 0.75:
        result.confidence = "TINGGI"
    elif avg_r2 >= 0.50:
        result.confidence = "SEDANG"
    else:
        result.confidence = "RENDAH"

    # --- Support & Resistance dari Pivot ---
    pivot_highs, pivot_lows = _find_pivots(high, low, window=5)

    res_levels = [h for h in pivot_highs if h > last_close]
    sup_levels = [l for l in pivot_lows if l < last_close]

    result.resistance_1 = res_levels[0] if len(res_levels) > 0 else last_close * 1.03
    result.resistance_2 = res_levels[1] if len(res_levels) > 1 else last_close * 1.06
    result.support_1 = sup_levels[0] if len(sup_levels) > 0 else last_close * 0.97
    result.support_2 = sup_levels[1] if len(sup_levels) > 1 else last_close * 0.94

    # --- Fibonacci ---
    swing_low = float(low.iloc[-30:].min())
    swing_high = float(high.iloc[-30:].max())
    result.fib_382, result.fib_500, result.fib_618 = _fibonacci_levels(swing_low, swing_high)

    # --- Target Price & Stop Loss ---
    if result.trend_direction == "NAIK":
        result.target_price = min(result.resistance_1, last_close + 2 * result.atr)
        result.stop_loss = max(result.support_1, last_close - 1.5 * result.atr)
    else:
        result.target_price = last_close + result.atr
        result.stop_loss = last_close - result.atr

    # Risk/Reward Ratio
    potential_gain = result.target_price - last_close
    potential_loss = last_close - result.stop_loss
    result.risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0

    # --- Sinyal Forecast ---
    change_3d_pct = ((result.forecast_3d - last_close) / last_close) * 100
    change_5d_pct = ((result.forecast_5d - last_close) / last_close) * 100

    if result.trend_direction == "NAIK":
        result.signals.append(
            f"Tren naik — proyeksi 3 hari: Rp {result.forecast_3d:,.0f} ({change_3d_pct:+.1f}%)"
        )
        result.signals.append(
            f"Proyeksi 5 hari: Rp {result.forecast_5d:,.0f} ({change_5d_pct:+.1f}%)"
        )
    elif result.trend_direction == "TURUN":
        result.signals.append(
            f"Tren turun — proyeksi 3 hari: Rp {result.forecast_3d:,.0f} ({change_3d_pct:+.1f}%)"
        )

    if result.risk_reward >= 2:
        result.signals.append(f"Risk/Reward sangat baik: 1:{result.risk_reward:.1f}")
    elif result.risk_reward >= 1.5:
        result.signals.append(f"Risk/Reward baik: 1:{result.risk_reward:.1f}")

    if result.confidence == "TINGGI":
        result.signals.append(f"Kepercayaan forecast TINGGI (R²={avg_r2:.2f})")

    # Harga mendekati Fibonacci support
    if abs(last_close - result.fib_618) / last_close < 0.02:
        result.signals.append(f"Harga mendekati support Fibonacci 61.8% (Rp {result.fib_618:,.0f})")
    elif abs(last_close - result.fib_500) / last_close < 0.02:
        result.signals.append(f"Harga mendekati support Fibonacci 50% (Rp {result.fib_500:,.0f})")

    return result
