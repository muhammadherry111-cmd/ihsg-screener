import numpy as np
import pandas as pd
import math

def _safe(v, fallback=0.0):
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + g/(l.replace(0, np.nan)))

def _macd(s, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow)
    sl = _ema(m, sig)
    return m, sl, m - sl

def _bb(s, n=20, k=2):
    mid = s.rolling(n).mean()
    sd  = s.rolling(n).std()
    return mid + k*sd, mid, mid - k*sd

def _stoch_rsi(s, n=14, sk=3, sd=3):
    rsi = _rsi(s, n)
    mn, mx = rsi.rolling(n).min(), rsi.rolling(n).max()
    k = (rsi - mn) / (mx - mn + 1e-9) * 100
    return k.rolling(sk).mean(), k.rolling(sk).mean().rolling(sd).mean()

def compute_indicators(df):
    """Returns indicator dict or None if insufficient data."""
    if df is None or len(df) < 26:
        return None
    c = df['close']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(0, index=c.index)

    rsi_s  = _rsi(c)
    ml, ms, mh = _macd(c)
    e9, e20, e50, e200 = _ema(c,9), _ema(c,20), _ema(c,50), _ema(c,200)
    bbu, bbm, bbl = _bb(c)
    sk, sd = _stoch_rsi(c)
    atr_s = c.rolling(14).std() * 1.414  # approx ATR without high/low

    # VWAP proxy
    if vol.sum() > 0:
        vwap = (c * vol).cumsum() / vol.cumsum()
    else:
        vwap = c.copy()

    vol_avg = vol.rolling(20).mean()
    v_ratio = (vol.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0

    last = _safe(c.iloc[-1])
    n = lambda s: _safe(s.iloc[-1])
    p = lambda s: _safe(s.iloc[-2]) if len(s) > 1 else _safe(s.iloc[-1])

    return {
        'last': last,
        'rsi': n(rsi_s), 'rsi_prev': p(rsi_s),
        'macd': n(ml), 'macd_sig': n(ms), 'macd_hist': n(mh),
        'macd_prev': p(ml), 'macd_sig_prev': p(ms),
        'ema9': n(e9), 'ema20': n(e20), 'ema50': n(e50), 'ema200': n(e200),
        'bb_upper': n(bbu), 'bb_mid': n(bbm), 'bb_lower': n(bbl),
        'stoch_k': n(sk), 'stoch_d': n(sd), 'stoch_k_prev': p(sk),
        'atr': n(atr_s), 'atr_pct': (n(atr_s) / last * 100) if last else 0,
        'vwap': n(vwap),
        'vol_ratio': _safe(v_ratio, 1.0),
        'support':    _safe(c.iloc[-20:].min()),
        'resistance': _safe(c.iloc[-20:].max()),
    }

def crypto_score(coin, ind):
    """Score -20 to +20. Returns (score, signals, warnings)."""
    score = 0
    signals, warnings = [], []
    last = ind['last']
    rsi = ind['rsi']

    # RSI
    if 55 <= rsi <= 70:   score += 3; signals.append(f"RSI {rsi:.0f} momentum bullish")
    elif 45 <= rsi < 55:  score += 1; signals.append(f"RSI {rsi:.0f} netral bullish")
    elif rsi > 80:        score -= 3; warnings.append(f"RSI {rsi:.0f} overbought")
    elif rsi < 30:        score -= 3; warnings.append(f"RSI {rsi:.0f} oversold/bearish")
    elif 30 <= rsi < 45:  score += 1; signals.append(f"RSI {rsi:.0f} potential bounce")

    # MACD cross
    m, ms_ = ind['macd'], ind['macd_sig']
    mp, msp = ind['macd_prev'], ind['macd_sig_prev']
    if m > ms_:
        if mp <= msp: score += 3; signals.append("MACD golden cross – bullish crossover")
        else:         score += 1; signals.append("MACD bullish")
    else:
        if mp >= msp: score -= 2; warnings.append("MACD death cross – bearish")
        else:         score -= 1

    # EMA trend
    above = sum([last > ind['ema9'], last > ind['ema20'], last > ind['ema50'], last > ind['ema200']])
    if above == 4:   score += 4; signals.append("Di atas semua EMA – uptrend kuat")
    elif above >= 2: score += 2; signals.append("Di atas EMA20/50 – bullish")
    elif above <= 1: score -= 3; warnings.append("Di bawah mayoritas EMA – downtrend")
    if ind['ema9'] > ind['ema20'] > ind['ema50']:
        score += 1; signals.append("EMA alignment bullish (9>20>50)")

    # Bollinger
    bbu, bbl = ind['bb_upper'], ind['bb_lower']
    bb_pos = (last - bbl) / (bbu - bbl) if bbu > bbl else 0.5
    if 0.5 <= bb_pos <= 0.85: score += 2; signals.append(f"BB position {bb_pos*100:.0f}% – zona bullish")
    elif bb_pos > 0.95:       score -= 1; warnings.append("Menyentuh BB upper")
    elif bb_pos < 0.15:       score -= 2; warnings.append("Dekat BB lower")

    # Volume
    vr = ind['vol_ratio']
    if vr >= 3:   score += 3; signals.append(f"Volume {vr:.1f}x rata-rata – konfirmasi kuat")
    elif vr >= 2: score += 2; signals.append(f"Volume {vr:.1f}x rata-rata")
    elif vr >= 1.5: score += 1
    elif vr < 0.6:  score -= 2; warnings.append("Volume sangat rendah")

    # Stoch RSI
    sk = ind['stoch_k']
    sk_p = ind['stoch_k_prev']
    if sk < 20 and sk > sk_p: score += 2; signals.append("Stoch RSI oversold + turning up")
    elif 40 <= sk <= 70:       score += 1
    elif sk > 85:              score -= 1; warnings.append("Stoch RSI overbought")

    # Price momentum 24h
    chg = coin.get('price_change_percentage_24h', 0) or 0
    if 2 <= chg <= 10:   score += 2; signals.append(f"+{chg:.1f}% 24h momentum positif")
    elif chg > 15:        score -= 1; warnings.append(f"+{chg:.1f}% 24h – sudah naik banyak")
    elif chg < -12:       score -= 3; warnings.append(f"{chg:.1f}% 24h – tekanan jual besar")

    return score, signals, warnings

def whale_score(coin, ind):
    """Whale Score 0–100. Returns (score, signals)."""
    score = 50
    sigs = []
    last = ind['last']
    vr = ind['vol_ratio']

    # Volume spike = proxy whale activity
    if vr >= 5:   score += 25; sigs.append(f"Volume {vr:.1f}x – Strong whale movement")
    elif vr >= 3: score += 15; sigs.append(f"Volume {vr:.1f}x – Whale aktif")
    elif vr >= 2: score += 8;  sigs.append(f"Volume {vr:.1f}x – Elevated activity")
    elif vr < 0.5: score -= 15; sigs.append("Volume sangat sepi – whale tidak aktif")

    # Price vs VWAP
    vwap = ind['vwap']
    if vwap > 0:
        vwap_d = (last - vwap) / vwap * 100
        if vwap_d > 3:   score += 10; sigs.append(f"Harga {vwap_d:.1f}% di atas VWAP – whale support")
        elif vwap_d < -5: score -= 12; sigs.append(f"Harga di bawah VWAP – kemungkinan distribusi")

    # Market cap tier
    mcap = coin.get('market_cap', 0) or 0
    if 0 < mcap < 100_000_000:
        score += 10; sigs.append("Small cap <$100M – rentan akumulasi whale")
    elif mcap < 1_000_000_000:
        score += 5; sigs.append("Mid cap – whale bisa gerakkan harga")

    # 24h change + volume combo
    chg = coin.get('price_change_percentage_24h', 0) or 0
    if chg > 5 and vr > 2:
        score += 8; sigs.append(f"+{chg:.1f}% + volume spike – pump whale terdeteksi")
    elif chg < -10:
        score -= 15; sigs.append("Turun tajam – kemungkinan distribusi whale")

    # 7d trend context
    chg7 = coin.get('price_change_percentage_7d_in_currency', 0) or 0
    if 5 < chg7 <= 20: score += 5; sigs.append("Uptrend 7d sehat – whale masih hold")
    elif chg7 > 30:    score -= 5; sigs.append("Sudah naik >30% 7d – hati-hati exit whale")

    return max(0, min(100, score)), sigs

def entry_targets(coin, ind):
    last = ind['last']
    atr = max(ind['atr'], last * 0.01)
    sup = ind['support']
    vwap = ind['vwap']
    ema20 = ind['ema20']

    entry_base = last
    sl = max(sup * 0.99, last - atr * 2)
    sl_pct = (last - sl) / last * 100

    risk = last - sl
    tp1 = last + risk * 1.5
    tp2 = last + risk * 3.0
    tp3 = last + risk * 5.0
    tp4 = last + risk * 8.0

    rr = risk * 1.5 / risk if risk > 0 else 1.5

    return {
        'entry': round(entry_base, 8),
        'buy_low': round(min(vwap, ema20, last) * 0.995, 8),
        'buy_high': round(last * 1.005, 8),
        'sl': round(sl, 8),
        'sl_pct': round(sl_pct, 2),
        'tp1': round(tp1, 8), 'tp1_pct': round((tp1 - last)/last*100, 2),
        'tp2': round(tp2, 8), 'tp2_pct': round((tp2 - last)/last*100, 2),
        'tp3': round(tp3, 8), 'tp3_pct': round((tp3 - last)/last*100, 2),
        'tp4': round(tp4, 8), 'tp4_pct': round((tp4 - last)/last*100, 2),
        'rr': round(rr, 2),
    }

def recommendation(score, ws):
    combined = score + (ws - 50) / 10
    if combined >= 14:  return 'STRONG BUY', '#00d97e'
    if combined >= 9:   return 'BUY',         '#22d3ee'
    if combined >= 4:   return 'WATCH',        '#ffcc00'
    if combined >= 0:   return 'NETRAL',       '#7fa8c9'
    if combined >= -5:  return 'CAUTION',      '#fb923c'
    return 'HINDARI', '#ff3d5a'

def strategy_label(score, ind):
    if score >= 12:  return 'Aggressive Long – entry sekarang'
    if score >= 8:   return 'Long – beli bertahap di pullback'
    if score >= 4:   return 'Swing – tunggu konfirmasi'
    if score >= 0:   return 'Watch – belum ada signal kuat'
    return 'Avoid – trend bearish'

def probability(score, ws):
    base = 50 + score * 2.5 + (ws - 50) * 0.3
    return round(max(5, min(95, base)), 1)
