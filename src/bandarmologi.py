"""
Bandarmologi: Deteksi aktivitas smart money berdasarkan Wyckoff Method + indikator volume.

Fase Wyckoff: ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN

Indikator: OBV (trend & divergence), MFI, Chaikin A/D, VPT, VWAP, Volume Spike
Trap: Fake Breakout · Upthrust · No Demand
Skor: 0–100 weighted multi-indikator (focus swing trading, data delay)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BandarSignal:
    # ── Indikator utama ──────────────────────────────────────────
    obv_trend: str = "NETRAL"
    obv_divergence: str = "TIDAK ADA"   # BULLISH / BEARISH / TIDAK ADA
    mfi: float = 50.0
    mfi_status: str = "NETRAL"
    ad_trend: str = "NETRAL"            # AKUMULASI / DISTRIBUSI / NETRAL
    vpt_trend: str = "NETRAL"           # BULLISH / BEARISH / NETRAL
    vwap: float = 0.0
    vwap_position: str = "NETRAL"       # DI ATAS / DI BAWAH / NETRAL

    # ── Volume spike ─────────────────────────────────────────────
    volume_spike: bool = False
    volume_spike_ratio: float = 0.0     # vol_terakhir / avg20

    # ── Pola & Wyckoff ───────────────────────────────────────────
    pola: str = "TIDAK ADA"             # AKUMULASI / MARKUP / DISTRIBUSI / BREAKOUT
    wyckoff_phase: str = "UNKNOWN"      # ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / UNKNOWN

    # ── Trap detection ───────────────────────────────────────────
    trap_detected: str = "TIDAK ADA"    # FAKE BREAKOUT / UPTHRUST / NO DEMAND / TIDAK ADA
    trap_warning: str = ""

    # ── Skor bandar ──────────────────────────────────────────────
    bandar_score: int = 0               # raw (backward compat dengan web/app.py)
    bandar_score_pct: float = 0.0       # weighted 0–100 (display utama)

    # ── Rekomendasi ──────────────────────────────────────────────
    recommendation: str = "PANTAU"      # ENTRY / PANTAU / HINDARI

    # ── Aktivitas & tier bandar ──────────────────────────────────
    bandar_list: list = field(default_factory=list)
    avg_bandar_entry: float = 0.0
    accum_days: int = 0
    accum_start_date: str = "—"
    current_vs_entry: float = 0.0

    signals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ─────────────────────────── Indikator dasar ────────────────────────────────

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
          volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0)
    neg = rmf.where(tp < tp.shift(1), 0)
    mfr = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _chaikin_ad(high: pd.Series, low: pd.Series,
                close: pd.Series, volume: pd.Series) -> pd.Series:
    hl = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl
    return (clv * volume).cumsum()


def _vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (close.pct_change() * volume).cumsum()


def _vwap(high: pd.Series, low: pd.Series,
          close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).rolling(period).sum() / volume.rolling(period).sum()


def _slope(series: pd.Series, period: int = 10) -> float:
    y = series.iloc[-period:].dropna().values
    if len(y) < max(3, period // 2):
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def _slope_normalized(series: pd.Series, period: int = 10) -> float:
    """Slope dinormalisasi oleh rata-rata nilai (unit-free)."""
    y = series.iloc[-period:].dropna().values
    if len(y) < 3 or y.mean() == 0:
        return 0.0
    return float(np.polyfit(np.arange(len(y)), y, 1)[0]) / abs(y.mean())


# ─────────────────────────── Deteksi Wyckoff ────────────────────────────────

def _wyckoff_phase(close: pd.Series, high: pd.Series, low: pd.Series,
                   volume: pd.Series, obv: pd.Series) -> str:
    """
    Deteksi fase Wyckoff berdasarkan kombinasi arah harga, OBV, dan volume.
    Multi-period: short (10) + medium (20) untuk konfirmasi.
    """
    price_slope_s  = _slope_normalized(close, 10)
    price_slope_m  = _slope_normalized(close, 20)
    obv_slope_s    = _slope_normalized(obv,   10)
    obv_slope_m    = _slope_normalized(obv,   20)

    vol_avg20 = volume.rolling(20).mean().iloc[-1]
    vol_avg5  = volume.iloc[-5:].mean()
    vol_recent_ratio = vol_avg5 / vol_avg20 if vol_avg20 > 0 else 1.0

    # Range harga 20 hari (relatif)
    high20 = float(high.iloc[-20:].max())
    low20  = float(low.iloc[-20:].min())
    price_range_pct = (high20 - low20) / low20 * 100 if low20 > 0 else 0

    # Apakah harga di area atas 20 hari (distribusi zone)
    last_close = float(close.iloc[-1])
    at_top = last_close >= low20 + 0.7 * (high20 - low20)
    at_bottom = last_close <= low20 + 0.3 * (high20 - low20)

    price_up   = price_slope_s > 0.002 or price_slope_m > 0.002
    price_down = price_slope_s < -0.002 or price_slope_m < -0.002
    price_side = not price_up and not price_down

    obv_up   = obv_slope_s > 0 and obv_slope_m > 0
    obv_down = obv_slope_s < 0 and obv_slope_m < 0

    # Markup: harga naik + OBV naik + volume konfirmasi
    if price_up and obv_up and vol_recent_ratio >= 1.1:
        return "MARKUP"

    # Accumulation: harga sideways/sedikit turun + OBV naik (divergence)
    if (price_side or (price_down and at_bottom)) and obv_up:
        return "ACCUMULATION"

    # Distribution: harga sideways/sedikit naik di area atas + OBV turun
    if (price_side or price_up) and obv_down and at_top:
        return "DISTRIBUTION"

    # Markdown: harga turun + OBV turun
    if price_down and obv_down:
        return "MARKDOWN"

    # Markup tanpa OBV (konfirmasi lemah)
    if price_up and vol_recent_ratio >= 1.3:
        return "MARKUP"

    # Accumulation tanpa OBV divergence (volume naik tapi harga flat)
    if price_side and vol_recent_ratio >= 1.2:
        return "ACCUMULATION"

    return "UNKNOWN"


# ─────────────────────────── Deteksi Trap ───────────────────────────────────

def _detect_trap(df: pd.DataFrame) -> tuple[str, str]:
    """
    Deteksi pola trap:
    1. Fake Breakout — high tembus resistance tapi close kembali di bawah
    2. Upthrust (Wyckoff) — spike dengan upper shadow besar + volume + reversal
    3. No Demand — rally pada volume sangat rendah (tidak didukung bandar)

    Return: (trap_type, warning_message)
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    open_  = df['open']
    volume = df['volume']

    if len(df) < 10:
        return "TIDAK ADA", ""

    last_close = float(close.iloc[-1])
    last_high  = float(high.iloc[-1])
    last_low   = float(low.iloc[-1])
    last_open  = float(open_.iloc[-1])
    prev_close = float(close.iloc[-2])
    last_vol   = float(volume.iloc[-1])
    vol_avg20  = float(volume.rolling(20).mean().iloc[-1]) or 1.0

    day_range     = last_high - last_low
    upper_shadow  = last_high - max(last_close, last_open)
    vol_ratio     = last_vol / vol_avg20
    price_chg_pct = (last_close - prev_close) / prev_close * 100 if prev_close > 0 else 0

    # Resistance: max high 20 hari sebelum candle terakhir
    recent_high_20 = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.iloc[:-1].max())

    # 1. Fake Breakout
    if (last_high > recent_high_20 * 1.005 and          # high tembus resistance
            last_close < recent_high_20 and              # close kembali di bawah
            price_chg_pct < 5.0):                        # bukan ARA hari ini
        return (
            "FAKE BREAKOUT",
            f"⚠️ Fake Breakout: High ({last_high:,.0f}) nembus resistance "
            f"({recent_high_20:,.0f}) tapi close ({last_close:,.0f}) balik ke bawah"
        )

    # 2. Upthrust (Wyckoff): candle dengan upper shadow dominan + volume + bearish close
    if (day_range > 0
            and upper_shadow / day_range >= 0.50        # upper shadow > 50% range
            and last_close < last_open                   # bearish candle
            and vol_ratio >= 1.5                         # volume tinggi
            and last_high >= recent_high_20 * 0.98):    # di area resistance
        return (
            "UPTHRUST",
            f"⚠️ Upthrust: Spike ke {last_high:,.0f} tapi ditolak keras "
            f"(upper shadow {upper_shadow/day_range*100:.0f}% range) — distribusi bandar"
        )

    # 3. No Demand: harga naik tapi volume sangat rendah
    if (price_chg_pct >= 0.5           # ada kenaikan harga
            and vol_ratio < 0.60        # volume sangat rendah
            and last_close > last_open):# candle bullish
        return (
            "NO DEMAND",
            f"⚠️ No Demand: Harga naik {price_chg_pct:.1f}% tapi volume "
            f"hanya {vol_ratio:.2f}× avg — rally tidak didukung bandar"
        )

    return "TIDAK ADA", ""


# ─────────────────────────── Volume Spike ───────────────────────────────────

def _volume_spike(volume: pd.Series) -> tuple[bool, float]:
    """Deteksi lonjakan volume hari terakhir vs rata-rata 20 hari."""
    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    if vol_avg20 <= 0:
        return False, 0.0
    ratio = float(volume.iloc[-1]) / vol_avg20
    return ratio >= 2.0, round(ratio, 2)


# ─────────────────────────── OBV Divergence (multi-period) ──────────────────

def _obv_divergence(obv: pd.Series, close: pd.Series) -> str:
    """
    Deteksi divergence OBV vs harga pada dua jendela waktu:
    - Short: 10 candle
    - Medium: 20 candle
    Konsisten di kedua periode = sinyal lebih valid.
    """
    slope_obv_s   = _slope_normalized(obv,   10)
    slope_price_s = _slope_normalized(close, 10)
    slope_obv_m   = _slope_normalized(obv,   20)
    slope_price_m = _slope_normalized(close, 20)

    obv_up_s   = slope_obv_s > 0.001
    obv_dn_s   = slope_obv_s < -0.001
    price_up_s = slope_price_s > 0.001
    price_dn_s = slope_price_s < -0.001

    obv_up_m   = slope_obv_m > 0.001
    price_dn_m = slope_price_m < -0.001
    obv_dn_m   = slope_obv_m < -0.001
    price_up_m = slope_price_m > 0.001

    # BULLISH: OBV naik saat harga turun/flat (2 konfirmasi)
    bullish_count = int(obv_up_s and price_dn_s) + int(obv_up_m and price_dn_m)
    if bullish_count >= 1:
        return "BULLISH"

    # BEARISH: OBV turun saat harga naik/flat
    bearish_count = int(obv_dn_s and price_up_s) + int(obv_dn_m and price_up_m)
    if bearish_count >= 1:
        return "BEARISH"

    return "TIDAK ADA"


# ─────────────────────────── Pola Bandar ────────────────────────────────────

def _classify_pattern(df: pd.DataFrame, obv_trend: str, vol_ratio: float) -> tuple[str, list, list]:
    """
    Klasifikasi pola berdasarkan Wyckoff + volume:
    AKUMULASI — Harga sideways sempit + volume naik secara bertahap
    MARKUP    — Candle besar bullish + volume meledak
    DISTRIBUSI — Harga stagnan di area atas + volume besar
    BREAKOUT  — Volume > 3× setelah konsolidasi
    """
    close  = df['close']
    high   = df['high']
    volume = df['volume']

    last_close = float(close.iloc[-1])
    last_vol   = float(volume.iloc[-1])
    vol_avg20  = float(volume.rolling(20).mean().iloc[-1]) or 1.0
    sma50      = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())

    last_n    = min(20, len(df))
    recent_c  = df['close'].iloc[-last_n:]
    range_pct = (recent_c.max() - recent_c.min()) / recent_c.min() * 100 if recent_c.min() > 0 else 0
    body_pct  = abs(last_close - float(close.iloc[-2])) / float(close.iloc[-2]) * 100 if float(close.iloc[-2]) > 0 else 0
    bullish   = last_close > float(close.iloc[-2])

    signals  = []
    warnings = []

    # Breakout: volume > 3× setelah konsolidasi ≥ 10 candle
    consol = df['close'].iloc[-15:-1]
    consol_range = (consol.max() - consol.min()) / consol.min() * 100 if len(consol) > 0 and consol.min() > 0 else 99
    if (last_vol >= vol_avg20 * 3.0 and body_pct > 1.5 and bullish and consol_range < 8):
        signals.append(f"Pola BREAKOUT: Volume {last_vol/vol_avg20:.1f}× setelah konsolidasi {consol_range:.1f}% — potensi kenaikan besar")
        return "BREAKOUT", signals, warnings

    # Markup: candle besar bullish + volume ≥ 2×
    if bullish and body_pct > 2.0 and last_vol >= vol_avg20 * 2.0:
        signals.append(f"Pola MARKUP: Candle bullish {body_pct:.1f}% + volume {last_vol/vol_avg20:.1f}× — bandar dorong harga")
        return "MARKUP", signals, warnings

    # Distribusi: harga di atas SMA50, range sempit, volume besar tapi tidak naik
    if (range_pct < 5 and last_close > sma50 and last_vol >= vol_avg20 * 1.5
            and not bullish):
        warnings.append(f"Pola DISTRIBUSI: Volume tinggi ({last_vol/vol_avg20:.1f}×) di area atas tapi harga stagnan — bandar lepas")
        return "DISTRIBUSI", signals, warnings

    # Akumulasi: harga sideways + volume meningkat + OBV mendukung
    if range_pct < 6 and last_vol >= vol_avg20 * 1.2 and obv_trend == "NAIK":
        signals.append(f"Pola AKUMULASI: Harga sideways {range_pct:.1f}% + volume meningkat — bandar kumpulkan perlahan")
        return "AKUMULASI", signals, warnings

    # Akumulasi tanpa OBV (konfirmasi tunggal)
    if range_pct < 5 and last_vol >= vol_avg20 * 1.3:
        signals.append(f"Pola AKUMULASI: Konsolidasi ketat {range_pct:.1f}% + volume {last_vol/vol_avg20:.1f}×")
        return "AKUMULASI", signals, warnings

    return "TIDAK ADA", signals, warnings


# ─────────────────────────── Akumulasi & Tier Bandar ────────────────────────

def _detect_accum_start(obv: pd.Series, lookback: int = 45) -> int:
    """Cari berapa hari lalu OBV mulai naik konsisten (titik awal akumulasi)."""
    n = min(lookback, len(obv) - 1)
    for i in range(5, n):
        window = obv.iloc[-(i + 5):-i] if i + 5 <= len(obv) else obv.iloc[:5]
        if len(window) < 3:
            continue
        slope = np.polyfit(np.arange(len(window)), window.values, 1)[0]
        if slope < 0:
            return i
    return n


def _analyze_bandar_activity(df: pd.DataFrame) -> dict:
    """Deteksi tier bandar berdasarkan volume spike dan estimasi harga entry (VWAP akumulasi)."""
    close  = df['close']
    volume = df['volume']

    last_price = float(close.iloc[-1])
    vol_avg20  = volume.rolling(20).mean()
    lookback   = min(60, len(df))
    recent     = df.iloc[-lookback:]
    vol_avg_r  = vol_avg20.iloc[-lookback:]

    # Estimasi durasi akumulasi dari OBV
    obv_full   = _obv(close, volume)
    accum_days = max(_detect_accum_start(obv_full, lookback), 1)
    accum_df   = df.iloc[-accum_days:]

    # VWAP periode akumulasi = estimasi avg entry bandar
    tp_acc     = (accum_df['high'] + accum_df['low'] + accum_df['close']) / 3
    vol_sum    = accum_df['volume'].sum()
    avg_entry  = float((tp_acc * accum_df['volume']).sum() / vol_sum) if vol_sum > 0 else last_price

    try:
        accum_start_date = df.index[-accum_days].strftime('%d %b %Y')
    except Exception:
        accum_start_date = '—'

    big_sessions, med_sessions, slow_sessions = [], [], []

    for i in range(len(recent)):
        row   = recent.iloc[i]
        avg_v = float(vol_avg_r.iloc[i]) if i < len(vol_avg_r) else 0
        if avg_v <= 0:
            continue
        ratio     = float(row['volume']) / avg_v
        tp        = (float(row['high']) + float(row['low']) + float(row['close'])) / 3
        day_range = (float(row['high']) - float(row['low'])) / float(row['low']) * 100 if float(row['low']) > 0 else 0
        try:
            date_str = recent.index[i].strftime('%d/%m')
        except Exception:
            date_str = '—'
        entry = {'date': date_str, 'price': tp, 'vol_ratio': round(ratio, 1)}
        if ratio >= 3:
            big_sessions.append(entry)
        elif ratio >= 2:
            med_sessions.append(entry)
        elif ratio >= 1.2 and day_range < 3.5:
            slow_sessions.append(entry)

    def _avg_price(s):
        return sum(x['price'] for x in s) / len(s) if s else 0.0

    bandar_list = []
    for tier in [
        {'key': 'big',  'sessions': big_sessions,  'label': 'Bandar Besar',       'icon': '🏦', 'dist': 20, 'hold': 5},
        {'key': 'med',  'sessions': med_sessions,  'label': 'Bandar Menengah',    'icon': '💼', 'dist': 15, 'hold': 3},
        {'key': 'slow', 'sessions': slow_sessions, 'label': 'Akumulasi Bertahap', 'icon': '📦', 'dist': 30, 'hold': 10},
    ]:
        ss = tier['sessions']
        if not ss:
            continue
        ap  = _avg_price(ss)
        pnl = (last_price - ap) / ap * 100 if ap > 0 else 0.0
        bandar_list.append({
            'category':   tier['label'],
            'icon':       tier['icon'],
            'avg_entry':  round(ap),
            'sessions':   len(ss),
            'pnl_pct':    round(pnl, 1),
            'status':     'DISTRIBUSI' if pnl >= tier['dist'] else ('HOLD' if pnl >= tier['hold'] else 'AKUMULASI'),
            'last_date':  ss[-1]['date'],
            'first_date': ss[0]['date'],
        })

    return {
        'bandar_list':      bandar_list,
        'avg_bandar_entry': round(avg_entry),
        'accum_days':       accum_days,
        'accum_start_date': accum_start_date,
        'current_vs_entry': round((last_price - avg_entry) / avg_entry * 100, 1) if avg_entry > 0 else 0.0,
    }


# ─────────────────────────── Weighted Score (0–100) ─────────────────────────

def _compute_score_pct(components: dict) -> float:
    """
    Hitung skor 0–100 dari komponen indikator.
    Setiap komponen bernilai 0.0–1.0 (bearish=0, netral=0.5, bullish=1.0).
    Bobot total = 1.0.
    """
    weights = {
        'obv':     0.20,   # OBV trend & divergence
        'mfi':     0.18,   # Money Flow Index
        'ad':      0.15,   # Chaikin A/D
        'vpt':     0.10,   # VPT
        'vwap':    0.10,   # VWAP position
        'volume':  0.12,   # Volume spike
        'pattern': 0.15,   # Pola Wyckoff
    }
    total = sum(weights[k] * components.get(k, 0.5) for k in weights)
    return round(min(100.0, max(0.0, total * 100)), 1)


# ─────────────────────────── Main Analyze ───────────────────────────────────

def analyze_bandar(df: pd.DataFrame) -> BandarSignal:
    """Analisis lengkap aktivitas bandar — Wyckoff Method + multi-indikator."""
    sig = BandarSignal()

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    last_close = float(close.iloc[-1])

    # Komponen skor (0.0 = sangat bearish, 0.5 = netral, 1.0 = sangat bullish)
    score_components: dict[str, float] = {}

    # ── OBV ─────────────────────────────────────────────────────
    obv = _obv(close, volume)
    obv_slope     = _slope_normalized(obv, 10)
    price_slope   = _slope_normalized(close, 10)
    sig.obv_trend = "NAIK" if obv_slope > 0.001 else ("TURUN" if obv_slope < -0.001 else "NETRAL")

    sig.obv_divergence = _obv_divergence(obv, close)

    if sig.obv_divergence == "BULLISH":
        score_components['obv'] = 1.0
        sig.bandar_score += 3
        sig.signals.append("OBV bullish divergence — smart money akumulasi diam-diam saat harga turun")
    elif sig.obv_divergence == "BEARISH":
        score_components['obv'] = 0.0
        sig.bandar_score -= 3
        sig.warnings.append("OBV bearish divergence — bandar distribusi saat harga masih naik")
    elif sig.obv_trend == "NAIK" and price_slope > 0:
        score_components['obv'] = 0.78
        sig.bandar_score += 1
        sig.signals.append("OBV naik searah harga — konfirmasi tren bullish")
    elif sig.obv_trend == "NAIK":
        score_components['obv'] = 0.65
    elif sig.obv_trend == "TURUN":
        score_components['obv'] = 0.25
        sig.bandar_score -= 1
    else:
        score_components['obv'] = 0.50

    # ── MFI ─────────────────────────────────────────────────────
    mfi_s   = _mfi(high, low, close, volume)
    sig.mfi = float(mfi_s.iloc[-1]) if not np.isnan(mfi_s.iloc[-1]) else 50.0

    if sig.mfi < 20:
        sig.mfi_status = "SANGAT OVERSOLD"
        score_components['mfi'] = 1.0
        sig.bandar_score += 3
        sig.signals.append(f"MFI sangat oversold ({sig.mfi:.1f}) — potensi reversal kuat")
    elif sig.mfi < 40:
        sig.mfi_status = "OVERSOLD"
        score_components['mfi'] = 0.82
        sig.bandar_score += 2
        sig.signals.append(f"MFI oversold ({sig.mfi:.1f}) — tekanan jual mereda, peluang masuk")
    elif sig.mfi < 60:
        sig.mfi_status = "NETRAL"
        score_components['mfi'] = 0.50
    elif sig.mfi < 80:
        sig.mfi_status = "KUAT"
        score_components['mfi'] = 0.68
        sig.bandar_score += 1
        sig.signals.append(f"MFI kuat ({sig.mfi:.1f}) — aliran dana masuk positif")
    else:
        sig.mfi_status = "OVERBOUGHT"
        score_components['mfi'] = 0.15
        sig.bandar_score -= 2
        sig.warnings.append(f"MFI overbought ({sig.mfi:.1f}) — bandar mungkin sedang distribusi")

    # ── Chaikin A/D ─────────────────────────────────────────────
    ad       = _chaikin_ad(high, low, close, volume)
    ad_slope = _slope_normalized(ad, 10)

    if ad_slope > 0.001:
        sig.ad_trend = "AKUMULASI"
        score_components['ad'] = 1.0
        sig.bandar_score += 2
        sig.signals.append("A/D Line naik — tekanan akumulasi dominan (uang masuk)")
    elif ad_slope < -0.001:
        sig.ad_trend = "DISTRIBUSI"
        score_components['ad'] = 0.0
        sig.bandar_score -= 1
        sig.warnings.append("A/D Line turun — tekanan distribusi dominan (uang keluar)")
    else:
        sig.ad_trend = "NETRAL"
        score_components['ad'] = 0.50

    # ── VPT ─────────────────────────────────────────────────────
    vpt      = _vpt(close, volume)
    vpt_slope = _slope_normalized(vpt, 10)

    if vpt_slope > 0.001:
        sig.vpt_trend = "BULLISH"
        score_components['vpt'] = 1.0
        sig.bandar_score += 1
        sig.signals.append("VPT positif — volume mendukung kenaikan harga")
    elif vpt_slope < -0.001:
        sig.vpt_trend = "BEARISH"
        score_components['vpt'] = 0.0
        sig.bandar_score -= 1
        sig.warnings.append("VPT negatif — volume mendukung penurunan harga")
    else:
        sig.vpt_trend = "NETRAL"
        score_components['vpt'] = 0.50

    # ── VWAP ────────────────────────────────────────────────────
    vwap_s  = _vwap(high, low, close, volume)
    sig.vwap = float(vwap_s.iloc[-1]) if not np.isnan(vwap_s.iloc[-1]) else last_close

    if last_close > sig.vwap:
        sig.vwap_position = "DI ATAS"
        score_components['vwap'] = 1.0
        sig.bandar_score += 1
        sig.signals.append(f"Harga ({last_close:,.0f}) di atas VWAP ({sig.vwap:,.0f}) — buyer control")
    elif last_close < sig.vwap * 0.99:
        sig.vwap_position = "DI BAWAH"
        score_components['vwap'] = 0.15
        sig.warnings.append(f"Harga ({last_close:,.0f}) di bawah VWAP ({sig.vwap:,.0f}) — seller control")
    else:
        sig.vwap_position = "NETRAL"
        score_components['vwap'] = 0.50

    # ── Volume Spike ────────────────────────────────────────────
    spike, spike_ratio = _volume_spike(volume)
    sig.volume_spike       = spike
    sig.volume_spike_ratio = spike_ratio

    if spike_ratio >= 4.0:
        score_components['volume'] = 1.0
        sig.bandar_score += 4
        sig.signals.append(f"Volume meledak {spike_ratio:.1f}× — aksi institusi sangat besar")
    elif spike_ratio >= 3.0:
        score_components['volume'] = 0.90
        sig.bandar_score += 3
        sig.signals.append(f"Volume sangat tinggi {spike_ratio:.1f}× — akumulasi/breakout nyata")
    elif spike_ratio >= 2.0:
        score_components['volume'] = 0.75
        sig.bandar_score += 2
        sig.signals.append(f"Volume tinggi {spike_ratio:.1f}× rata-rata 20H")
    elif spike_ratio >= 1.5:
        score_components['volume'] = 0.65
        sig.bandar_score += 1
        sig.signals.append(f"Volume di atas normal {spike_ratio:.1f}×")
    elif spike_ratio < 0.60:
        score_components['volume'] = 0.10
        sig.bandar_score -= 2
        sig.warnings.append(f"Volume sangat rendah {spike_ratio:.1f}× — likuiditas tipis, hindari")
    elif spike_ratio < 0.85:
        score_components['volume'] = 0.30
    else:
        score_components['volume'] = 0.50

    # ── Wyckoff Phase ───────────────────────────────────────────
    sig.wyckoff_phase = _wyckoff_phase(close, high, low, volume, obv)

    # ── Pola Bandar ─────────────────────────────────────────────
    pola, pola_signals, pola_warnings = _classify_pattern(df, sig.obv_trend, spike_ratio)
    sig.pola = pola
    sig.signals.extend(pola_signals)
    sig.warnings.extend(pola_warnings)

    if pola == "BREAKOUT":
        score_components['pattern'] = 1.0
        sig.bandar_score += 4
    elif pola == "MARKUP":
        score_components['pattern'] = 0.85
        sig.bandar_score += 3
    elif pola == "AKUMULASI":
        score_components['pattern'] = 0.72
        sig.bandar_score += 2
    elif pola == "DISTRIBUSI":
        score_components['pattern'] = 0.0
        sig.bandar_score -= 2
    else:
        score_components['pattern'] = 0.40

    # Koreksi negatif jika Wyckoff = DISTRIBUTION / MARKDOWN
    if sig.wyckoff_phase == "DISTRIBUTION":
        score_components['pattern'] = min(score_components['pattern'], 0.25)
        sig.warnings.append("Fase Wyckoff DISTRIBUTION — waspadai puncak distribusi bandar")
    elif sig.wyckoff_phase == "MARKDOWN":
        for k in score_components:
            score_components[k] = min(score_components[k], 0.35)
        sig.warnings.append("Fase Wyckoff MARKDOWN — tren turun aktif, hindari entry baru")
    elif sig.wyckoff_phase == "ACCUMULATION":
        sig.signals.append("Fase Wyckoff ACCUMULATION — titik optimal masuk sebelum markup")
    elif sig.wyckoff_phase == "MARKUP":
        sig.signals.append("Fase Wyckoff MARKUP — tren naik aktif, ikuti momentum")

    # ── Trap Detection ──────────────────────────────────────────
    sig.trap_detected, sig.trap_warning = _detect_trap(df)

    if sig.trap_detected != "TIDAK ADA":
        # Trap mengurangi semua komponen skor
        for k in score_components:
            score_components[k] = min(score_components[k], 0.30)
        sig.bandar_score -= 4
        sig.warnings.append(sig.trap_warning)

    # ── Weighted Score 0–100 ────────────────────────────────────
    sig.bandar_score_pct = _compute_score_pct(score_components)

    # ── Analisis Aktivitas & Tier Bandar ────────────────────────
    activity = _analyze_bandar_activity(df)
    sig.bandar_list      = activity['bandar_list']
    sig.avg_bandar_entry = activity['avg_bandar_entry']
    sig.accum_days       = activity['accum_days']
    sig.accum_start_date = activity['accum_start_date']
    sig.current_vs_entry = activity['current_vs_entry']

    # Entry bandar jauh di bawah harga = bandar sudah profit besar → distribusi risiko
    if sig.current_vs_entry > 30 and sig.pola not in ('BREAKOUT',):
        sig.warnings.append(
            f"Harga sudah {sig.current_vs_entry:.1f}% di atas avg entry bandar "
            f"({sig.avg_bandar_entry:,.0f}) — waspadai distribusi"
        )
        sig.bandar_score_pct = min(sig.bandar_score_pct, 55.0)

    # ── Rekomendasi ─────────────────────────────────────────────
    if (sig.bandar_score_pct >= 68
            and sig.trap_detected == "TIDAK ADA"
            and sig.pola in ("AKUMULASI", "BREAKOUT", "MARKUP")
            and sig.wyckoff_phase not in ("DISTRIBUTION", "MARKDOWN")):
        sig.recommendation = "ENTRY"
        sig.signals.append(
            f"REKOMENDASI ENTRY — skor {sig.bandar_score_pct:.0f}/100, "
            f"pola {sig.pola}, fase {sig.wyckoff_phase}"
        )
    elif (sig.bandar_score_pct < 35
            or sig.trap_detected != "TIDAK ADA"
            or sig.pola == "DISTRIBUSI"
            or sig.wyckoff_phase in ("DISTRIBUTION", "MARKDOWN")):
        sig.recommendation = "HINDARI"
        sig.warnings.append(
            f"REKOMENDASI HINDARI — skor {sig.bandar_score_pct:.0f}/100"
            + (f", trap: {sig.trap_detected}" if sig.trap_detected != "TIDAK ADA" else "")
        )
    else:
        sig.recommendation = "PANTAU"

    return sig
