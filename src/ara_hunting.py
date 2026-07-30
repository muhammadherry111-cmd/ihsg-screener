"""
ARA Hunting — mencari saham yang berpotensi mencapai Auto Reject Atas (ARA)
di Bursa Efek Indonesia (BEI/IDX).

Batas ARA/ARB IDX:
  Harga <  Rp200      : ±35%
  Harga Rp200-<Rp5000 : ±25%
  Harga ≥ Rp5000      : ±20%

Strategi yang dideteksi:
  1. Saham sedang consecutive ARA (rocket mode)
  2. Saham baru breakout menuju ARA hari ini
  3. Recovery dari ARB (akumulasi pasca jatuh, berpotensi balik ARA)
  4. Pre-ARA: tekanan beli kuat, volume meledak, harga mendekati limit atas
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ARAHuntingResult:
    ara_limit_pct: float = 0.0       # batas ARA hari ini (%)
    arb_limit_pct: float = 0.0       # batas ARB hari ini (%)
    change_pct: float = 0.0          # perubahan harga hari ini (%)

    # Status hari ini
    is_ara_today: bool = False        # sudah kena ARA hari ini
    is_arb_today: bool = False        # sudah kena ARB hari ini
    pct_to_ara: float = 0.0          # jarak ke ARA (%), makin kecil makin dekat

    # Pola consecutive
    consecutive_ara: int = 0          # berapa hari berturut-turut ARA
    consecutive_arb: int = 0          # berapa hari berturut-turut ARB (sebelum sekarang)
    arb_recovery_day: int = 0         # hari ke-N setelah keluar ARB streak

    # Volume & tekanan beli
    volume_ratio: float = 0.0         # volume hari ini vs rata-rata 20 hari
    volume_spike_3d: float = 0.0      # rata-rata volume 3 hari vs 20 hari
    buy_pressure: float = 0.0         # (close - low) / (high - low)

    # Candlestick ARA-oriented
    body_pct: float = 0.0             # ukuran body candle vs range total (%)
    upper_shadow_pct: float = 0.0     # upper shadow vs range (kecil = baik untuk ARA)
    candle_bullish: bool = False       # close > open

    # Skor & keputusan
    ara_score: int = 0
    ara_mode: str = "TIDAK"           # ROCKET / PRE-ARA / RECOVERY / TIDAK
    signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _ara_limit(price: float) -> float:
    """Kembalikan batas ARA (%) berdasarkan harga terakhir."""
    if price < 200:
        return 35.0
    elif price < 5000:
        return 25.0
    else:
        return 20.0


def _count_consecutive_ara(close: pd.Series, prices: pd.Series) -> int:
    """Hitung berapa hari terakhir berturut-turut kena ARA."""
    count = 0
    for i in range(len(close) - 1, 0, -1):
        prev = float(close.iloc[i - 1])
        curr = float(close.iloc[i])
        if prev <= 0:
            break
        limit = _ara_limit(prev)
        chg = (curr - prev) / prev * 100
        if chg >= limit * 0.95:   # toleransi 5% dari batas (misal kena 23.9% dianggap ARA)
            count += 1
        else:
            break
    return count


def _count_consecutive_arb(close: pd.Series, end_idx: int) -> int:
    """Hitung panjang ARB streak yang berakhir sebelum end_idx."""
    count = 0
    for i in range(end_idx, 0, -1):
        prev = float(close.iloc[i - 1])
        curr = float(close.iloc[i])
        if prev <= 0:
            break
        limit = _ara_limit(prev)
        chg = (curr - prev) / prev * 100
        if chg <= -limit * 0.95:
            count += 1
        else:
            break
    return count


def _arb_recovery_day(close: pd.Series) -> tuple[int, int]:
    """
    Cari panjang ARB streak terakhir dan berapa hari sudah berlalu sejak ARB selesai.
    Kembalikan (arb_streak_length, recovery_day).
    """
    n = len(close)
    # Cari hari terakhir masih ARB
    last_arb_idx = -1
    for i in range(n - 1, 0, -1):
        prev = float(close.iloc[i - 1])
        curr = float(close.iloc[i])
        if prev <= 0:
            continue
        limit = _ara_limit(prev)
        chg = (curr - prev) / prev * 100
        if chg <= -limit * 0.95:
            last_arb_idx = i
            break

    if last_arb_idx == -1:
        return 0, 0

    recovery_day = (n - 1) - last_arb_idx
    arb_streak = _count_consecutive_arb(close, last_arb_idx)
    return arb_streak, recovery_day


def analyze_ara_hunting(df: pd.DataFrame) -> ARAHuntingResult:
    result = ARAHuntingResult()

    if df is None or len(df) < 10:
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

    if prev_close <= 0 or last_close <= 0:
        return result

    ara_lim = _ara_limit(prev_close)
    arb_lim = ara_lim  # simetris di IDX
    change_pct = (last_close - prev_close) / prev_close * 100

    result.ara_limit_pct = ara_lim
    result.arb_limit_pct = arb_lim
    result.change_pct    = round(change_pct, 2)

    # ── Status hari ini ──────────────────────────────────────────
    result.is_ara_today = change_pct >= ara_lim * 0.95
    result.is_arb_today = change_pct <= -arb_lim * 0.95
    result.pct_to_ara   = round(ara_lim - change_pct, 2)

    # ── Consecutive ARA / ARB ────────────────────────────────────
    result.consecutive_ara = _count_consecutive_ara(close, close)
    arb_streak, rec_day    = _arb_recovery_day(close)
    result.consecutive_arb = arb_streak
    result.arb_recovery_day = rec_day

    # ── Volume ──────────────────────────────────────────────────
    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    vol_last  = float(volume.iloc[-1])
    vol_3d    = float(volume.tail(3).mean())
    result.volume_ratio    = round(vol_last / vol_avg20, 2) if vol_avg20 > 0 else 1.0
    result.volume_spike_3d = round(vol_3d  / vol_avg20, 2) if vol_avg20 > 0 else 1.0

    # ── Tekanan beli & candle ───────────────────────────────────
    day_range = last_high - last_low
    if day_range > 0:
        result.buy_pressure        = round((last_close - last_low) / day_range, 3)
        result.body_pct            = round(abs(last_close - last_open) / day_range * 100, 1)
        result.upper_shadow_pct    = round((last_high - max(last_close, last_open)) / day_range * 100, 1)
    result.candle_bullish = last_close > last_open

    # ── Scoring ─────────────────────────────────────────────────
    score    = 0
    signals  = []
    warnings = []

    # --- ROCKET MODE: consecutive ARA ---
    if result.consecutive_ara >= 3:
        score += 8
        signals.append(f"Rocket ARA: {result.consecutive_ara} hari berturut-turut kena ARA!")
    elif result.consecutive_ara == 2:
        score += 5
        signals.append("2 hari berturut ARA — momentum sangat kuat")
    elif result.is_ara_today:
        score += 3
        signals.append(f"ARA hari ini (+{change_pct:.1f}%) — pantau pembukaan besok")

    # --- PRE-ARA: hampir kena limit atas ---
    if not result.is_ara_today and result.pct_to_ara <= 5.0:
        score += 4
        signals.append(f"Harga sangat dekat ARA, tinggal {result.pct_to_ara:.1f}% lagi")
    elif not result.is_ara_today and result.pct_to_ara <= 10.0:
        score += 2
        signals.append(f"Mendekati batas ARA, sisa {result.pct_to_ara:.1f}%")

    # --- RECOVERY dari ARB ---
    if arb_streak >= 3 and 1 <= rec_day <= 5:
        score += 5
        signals.append(
            f"Recovery pasca {arb_streak} hari ARB berturut (hari ke-{rec_day} recovery) — "
            "akumulasi bandar sering terjadi di fase ini"
        )
    elif arb_streak >= 2 and 1 <= rec_day <= 3:
        score += 3
        signals.append(f"Keluar dari {arb_streak}x ARB, hari ke-{rec_day} recovery")

    # --- Volume spike ---
    if result.volume_ratio >= 5.0:
        score += 4
        signals.append(f"Volume meledak {result.volume_ratio:.1f}x — aksi luar biasa")
    elif result.volume_ratio >= 3.0:
        score += 3
        signals.append(f"Volume sangat tinggi {result.volume_ratio:.1f}x rata-rata 20H")
    elif result.volume_ratio >= 2.0:
        score += 2
        signals.append(f"Volume tinggi {result.volume_ratio:.1f}x rata-rata")
    elif result.volume_ratio < 0.5:
        score -= 2
        warnings.append(f"Volume sangat rendah {result.volume_ratio:.1f}x — likuiditas tipis")

    # Volume spike 3 hari berturut (tanda akumulasi)
    if result.volume_spike_3d >= 2.5 and not result.is_ara_today:
        score += 2
        signals.append(f"Rata-rata volume 3 hari = {result.volume_spike_3d:.1f}x — akumulasi aktif")

    # --- Tekanan beli (buy pressure) ---
    if result.buy_pressure >= 0.85:
        score += 3
        signals.append(f"Tekanan beli ekstrem — close di {result.buy_pressure*100:.0f}% atas range hari ini")
    elif result.buy_pressure >= 0.70:
        score += 2
        signals.append(f"Tekanan beli kuat ({result.buy_pressure*100:.0f}%)")
    elif result.buy_pressure < 0.30:
        score -= 2
        warnings.append("Tekanan jual dominan — close di bagian bawah range")

    # --- Candle bullish & body besar ---
    if result.candle_bullish and result.body_pct >= 70:
        score += 2
        signals.append(f"Candle bullish badan besar ({result.body_pct:.0f}% range) — buyer kuat")
    elif result.candle_bullish and result.body_pct >= 50:
        score += 1
        signals.append("Candle bullish solid")

    # Upper shadow kecil (tidak ditolak di atas)
    if result.upper_shadow_pct <= 10 and result.candle_bullish:
        score += 1
        signals.append("Upper shadow minim — tidak ada penolakan di atas")
    elif result.upper_shadow_pct >= 40:
        score -= 1
        warnings.append(f"Upper shadow besar ({result.upper_shadow_pct:.0f}%) — ada penolakan di harga tinggi")

    # --- Tren harga 5 hari ---
    if len(close) >= 6:
        chg5 = (last_close - float(close.iloc[-6])) / float(close.iloc[-6]) * 100
        if chg5 >= ara_lim * 0.8:
            score += 3
            signals.append(f"Kenaikan 5 hari = +{chg5:.1f}% — hampir setara ARA dalam seminggu")
        elif chg5 >= 10:
            score += 2
            signals.append(f"Kenaikan 5 hari: +{chg5:.1f}%")
        elif chg5 <= -15:
            score -= 2
            warnings.append(f"Penurunan 5 hari: {chg5:.1f}% — masih dalam tekanan jual")

    # --- Mode & rating ──────────────────────────────────────────
    if result.consecutive_ara >= 2:
        ara_mode = "ROCKET"
    elif score >= 10:
        ara_mode = "PRE-ARA" if not result.is_ara_today else "ROCKET"
    elif arb_streak >= 2 and rec_day <= 5 and score >= 6:
        ara_mode = "RECOVERY"
    elif score >= 6:
        ara_mode = "PRE-ARA"
    else:
        ara_mode = "TIDAK"

    result.ara_score  = score
    result.ara_mode   = ara_mode
    result.signals    = signals
    result.warnings   = warnings
    return result
