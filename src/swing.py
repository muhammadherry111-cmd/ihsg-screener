import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SwingResult:
    forecast_10d: float = 0.0
    forecast_20d: float = 0.0
    trend_medium: str = "SIDEWAYS"
    trend_strength: str = "LEMAH"
    ema20: float = 0.0
    ema50: float = 0.0
    swing_entry: float = 0.0
    swing_target_1: float = 0.0
    swing_target_2: float = 0.0
    swing_stop: float = 0.0
    swing_rr: float = 0.0
    hold_days_est: int = 7
    swing_resistance: float = 0.0
    swing_support: float = 0.0
    weekly_change_pct: float = 0.0
    monthly_change_pct: float = 0.0
    swing_score: int = 0
    swing_feasible: bool = False
    swing_rating: str = "NETRAL"
    signals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _linreg(close: pd.Series, days_ahead: int):
    n = len(close)
    x = np.arange(n, dtype=float)
    y = close.values.astype(float)
    mask = np.isfinite(y)
    if mask.sum() < 5:
        return float(y[-1]) if np.isfinite(y[-1]) else 0.0, 0.0
    xm, ym = x[mask].mean(), y[mask].mean()
    denom = ((x[mask] - xm) ** 2).sum()
    if denom == 0:
        return float(y[-1]), 0.0
    slope = ((x[mask] - xm) * (y[mask] - ym)).sum() / denom
    intercept = ym - slope * xm
    return max(slope * (n - 1 + days_ahead) + intercept, 1.0), slope


def analyze_swing(df: pd.DataFrame) -> SwingResult:
    if df is None or len(df) < 20:
        return SwingResult()

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    n      = len(close)
    last   = float(close.iloc[-1])

    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    fc10, slope10 = _linreg(close, 10)
    fc20, _       = _linreg(close, 20)

    slope_pct = slope10 / last * 100
    if slope_pct > 0.08:
        trend_medium = "NAIK"
    elif slope_pct < -0.08:
        trend_medium = "TURUN"
    else:
        trend_medium = "SIDEWAYS"

    if last > ema20 > ema50:
        trend_strength = "KUAT"
    elif last > ema20 or last > ema50:
        trend_strength = "SEDANG"
    else:
        trend_strength = "LEMAH"

    n_range = min(60, n)
    swing_resistance = float(high.iloc[-n_range:].quantile(0.85))
    swing_support    = float(low.iloc[-n_range:].quantile(0.15))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr     = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = atr / last * 100 if last > 0 else 1.0

    swing_entry    = last
    swing_target_1 = round(min(swing_resistance * 1.005, last + atr * 3), 0)
    swing_target_2 = round(last + atr * 5, 0)
    swing_stop     = round(max(swing_support * 0.995, last - atr * 1.5), 0)

    risk     = last - swing_stop
    reward   = swing_target_1 - last
    swing_rr = round(reward / risk, 2) if risk > 0 else 0.0
    hold_est = max(3, min(25, int(8 / max(atr_pct, 0.4))))

    w5   = max(0, n - 5)
    m20  = max(0, n - 20)
    base5  = float(close.iloc[w5])
    base20 = float(close.iloc[m20])
    weekly_chg  = (last - base5)  / base5  * 100 if base5  > 0 else 0.0
    monthly_chg = (last - base20) / base20 * 100 if base20 > 0 else 0.0

    vol_ma     = float(volume.rolling(20).mean().iloc[-1])
    vol_recent = float(volume.iloc[-5:].mean())
    vol_up     = vol_recent > vol_ma * 1.05 if vol_ma > 0 else False

    score    = 0
    signals  = []
    warnings = []

    if trend_medium == "NAIK":
        if trend_strength == "KUAT":
            score += 4
            signals.append("Tren medium-term kuat (Harga > EMA20 > EMA50)")
        else:
            score += 2
            signals.append("Tren medium-term positif")
    elif trend_medium == "TURUN":
        score -= 3
        warnings.append("Tren medium-term turun — kurang ideal untuk swing beli")
    else:
        warnings.append("Tren sideways — tunggu konfirmasi breakout")

    fc10_pct = (fc10 - last) / last * 100
    if fc10_pct >= 3:
        score += 3
        signals.append(f"Proyeksi 10H (regresi linier): +{fc10_pct:.1f}%")
    elif fc10_pct >= 1.5:
        score += 2
        signals.append(f"Proyeksi 10H: +{fc10_pct:.1f}%")
    elif fc10_pct < 0:
        score -= 2
        warnings.append(f"Proyeksi 10H negatif: {fc10_pct:.1f}%")

    if swing_rr >= 2.5:
        score += 3
        signals.append(f"Risk/Reward swing sangat baik: 1:{swing_rr:.2f}")
    elif swing_rr >= 1.5:
        score += 2
        signals.append(f"Risk/Reward swing baik: 1:{swing_rr:.2f}")
    elif swing_rr < 1.0:
        score -= 2
        warnings.append(f"Risk/Reward kurang baik: 1:{swing_rr:.2f} — pertimbangkan ulang")

    if vol_up and trend_medium == "NAIK":
        score += 2
        signals.append("Volume naik mendukung tren medium-term")
    elif not vol_up and trend_medium == "NAIK":
        warnings.append("Volume lemah — kenaikan kurang terkonfirmasi")

    if monthly_chg >= 5:
        score += 2
        signals.append(f"Momentum 1 bulan positif: +{monthly_chg:.1f}%")
    elif monthly_chg <= -10:
        score -= 1
        warnings.append(f"Koreksi 1 bulan: {monthly_chg:.1f}%")

    if atr_pct < 0.8:
        warnings.append(f"Volatilitas sangat rendah (ATR {atr_pct:.2f}%/hari) — potensi swing terbatas")

    feasible = score >= 6 and trend_medium != "TURUN"

    if score >= 10:
        rating = "STRONG BUY"
    elif score >= 6:
        rating = "BUY"
    elif score >= 3:
        rating = "HOLD"
    elif score >= 0:
        rating = "WAIT"
    else:
        rating = "AVOID"

    return SwingResult(
        forecast_10d=round(fc10, 0),
        forecast_20d=round(fc20, 0),
        trend_medium=trend_medium,
        trend_strength=trend_strength,
        ema20=round(ema20, 2),
        ema50=round(ema50, 2),
        swing_entry=round(swing_entry, 0),
        swing_target_1=round(swing_target_1, 0),
        swing_target_2=round(swing_target_2, 0),
        swing_stop=round(swing_stop, 0),
        swing_rr=swing_rr,
        hold_days_est=hold_est,
        swing_resistance=round(swing_resistance, 0),
        swing_support=round(swing_support, 0),
        weekly_change_pct=round(weekly_chg, 2),
        monthly_change_pct=round(monthly_chg, 2),
        swing_score=score,
        swing_feasible=feasible,
        swing_rating=rating,
        signals=signals,
        warnings=warnings,
    )
