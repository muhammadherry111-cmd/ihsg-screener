"""
Candlestick Pattern Recognition
Deteksi pola candle yang relevan untuk prediksi arah harga.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CandlePattern:
    name: str
    signal: str      # BULLISH / BEARISH / NETRAL
    strength: int    # 1 (lemah) - 3 (kuat)
    description: str


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _candle_range(h: float, l: float) -> float:
    return h - l


def detect_patterns(df: pd.DataFrame) -> list[CandlePattern]:
    """Deteksi pola candlestick dari data OHLCV terbaru."""
    patterns = []

    if len(df) < 5:
        return patterns

    # Ambil 5 candle terakhir
    c5 = df.iloc[-5:]
    o1, h1, l1, c1 = float(c5["open"].iloc[-1]), float(c5["high"].iloc[-1]), float(c5["low"].iloc[-1]), float(c5["close"].iloc[-1])
    o2, h2, l2, c2 = float(c5["open"].iloc[-2]), float(c5["high"].iloc[-2]), float(c5["low"].iloc[-2]), float(c5["close"].iloc[-2])
    o3, h3, l3, c3 = float(c5["open"].iloc[-3]), float(c5["high"].iloc[-3]), float(c5["low"].iloc[-3]), float(c5["close"].iloc[-3])

    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    upper1 = _upper_wick(o1, h1, c1)
    lower1 = _lower_wick(o1, l1, c1)
    range1 = _candle_range(h1, l1)
    avg_body = df["close"].diff().abs().tail(14).mean()

    # --- CANDLE TERAKHIR (1 candle) ---

    # Hammer: tubuh kecil di atas, ekor bawah panjang (>= 2x tubuh), sedikit/tidak ada ekor atas
    if (range1 > 0 and lower1 >= 2 * body1 and upper1 <= 0.3 * body1
            and body1 < 0.4 * range1 and c2 > o2):  # dalam downtrend
        patterns.append(CandlePattern(
            name="Hammer",
            signal="BULLISH",
            strength=2,
            description="Ekor bawah panjang — tekanan jual tertolak, potensi reversal naik"
        ))

    # Inverted Hammer: tubuh kecil di bawah, ekor atas panjang
    if (range1 > 0 and upper1 >= 2 * body1 and lower1 <= 0.3 * body1
            and body1 < 0.4 * range1):
        patterns.append(CandlePattern(
            name="Inverted Hammer",
            signal="BULLISH",
            strength=1,
            description="Ekor atas panjang setelah turun — pembeli mulai mengambil alih"
        ))

    # Shooting Star: sama seperti inverted hammer tapi di uptrend = bearish
    if (range1 > 0 and upper1 >= 2 * body1 and lower1 <= 0.3 * body1
            and body1 < 0.4 * range1 and c2 > o2):
        patterns.append(CandlePattern(
            name="Shooting Star",
            signal="BEARISH",
            strength=2,
            description="Ekor atas panjang di area tinggi — potensi pembalikan turun"
        ))

    # Doji: tubuh sangat kecil
    if range1 > 0 and body1 <= 0.05 * range1:
        doji_type = "Bullish" if c2 < o2 else "Bearish"
        patterns.append(CandlePattern(
            name=f"Doji ({doji_type})",
            signal="BULLISH" if c2 < o2 else "NETRAL",
            strength=1,
            description="Candle ragu-ragu — keseimbangan buyer/seller, tunggu konfirmasi"
        ))

    # Marubozu Bullish: tubuh besar, hampir tidak ada ekor
    if (range1 > 0 and body1 >= 0.85 * range1 and c1 > o1
            and body1 > avg_body * 1.5):
        patterns.append(CandlePattern(
            name="Bullish Marubozu",
            signal="BULLISH",
            strength=3,
            description="Candle besar bullish tanpa ekor — buyer dominan sepanjang hari"
        ))

    # Marubozu Bearish
    if (range1 > 0 and body1 >= 0.85 * range1 and c1 < o1
            and body1 > avg_body * 1.5):
        patterns.append(CandlePattern(
            name="Bearish Marubozu",
            signal="BEARISH",
            strength=3,
            description="Candle besar bearish tanpa ekor — seller dominan sepanjang hari"
        ))

    # --- POLA 2 CANDLE ---

    # Bullish Engulfing: candle merah diikuti candle hijau yang lebih besar
    if (c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2 and body1 > body2):
        patterns.append(CandlePattern(
            name="Bullish Engulfing",
            signal="BULLISH",
            strength=3,
            description="Candle hijau menelan candle merah sebelumnya — momentum beli kuat"
        ))

    # Bearish Engulfing
    if (c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2 and body1 > body2):
        patterns.append(CandlePattern(
            name="Bearish Engulfing",
            signal="BEARISH",
            strength=3,
            description="Candle merah menelan candle hijau — momentum jual kuat"
        ))

    # Tweezer Bottom: dua candle dengan low hampir sama
    if (c2 < o2 and c1 > o1 and abs(l1 - l2) / max(l1, l2) < 0.005):
        patterns.append(CandlePattern(
            name="Tweezer Bottom",
            signal="BULLISH",
            strength=2,
            description="Dua candle dengan low sama — support kuat, potensi reversal"
        ))

    # Piercing Line: candle merah diikuti candle hijau yang menutup di atas 50% body candle merah
    if (c2 < o2 and c1 > o1 and o1 < c2 and c1 > (o2 + c2) / 2 and c1 < o2):
        patterns.append(CandlePattern(
            name="Piercing Line",
            signal="BULLISH",
            strength=2,
            description="Candle hijau menembus 50% body merah — pembalikan bullish"
        ))

    # --- POLA 3 CANDLE ---

    # Morning Star: merah besar, doji/kecil, hijau besar
    if (c3 < o3 and body2 < 0.3 * body1 and c1 > o1
            and c1 > (o3 + c3) / 2 and body1 > avg_body):
        patterns.append(CandlePattern(
            name="Morning Star",
            signal="BULLISH",
            strength=3,
            description="Tiga candle reversal — sinyal pembalikan uptrend yang kuat"
        ))

    # Evening Star: hijau besar, doji/kecil, merah besar
    if (c3 > o3 and body2 < 0.3 * body1 and c1 < o1
            and c1 < (o3 + c3) / 2 and body1 > avg_body):
        patterns.append(CandlePattern(
            name="Evening Star",
            signal="BEARISH",
            strength=3,
            description="Tiga candle distribusi — sinyal pembalikan downtrend yang kuat"
        ))

    # Three White Soldiers: tiga candle hijau berturut-turut dengan penutupan semakin tinggi
    if (c3 > o3 and c2 > o2 and c1 > o1
            and c1 > c2 > c3 and o2 > o3 and o1 > o2
            and body1 > avg_body * 0.8 and body2 > avg_body * 0.8):
        patterns.append(CandlePattern(
            name="Three White Soldiers",
            signal="BULLISH",
            strength=3,
            description="Tiga candle hijau berturut naik — tren bullish sangat kuat"
        ))

    # Three Black Crows
    if (c3 < o3 and c2 < o2 and c1 < o1
            and c1 < c2 < c3 and o2 < o3 and o1 < o2
            and body1 > avg_body * 0.8):
        patterns.append(CandlePattern(
            name="Three Black Crows",
            signal="BEARISH",
            strength=3,
            description="Tiga candle merah berturut turun — tren bearish sangat kuat"
        ))

    return patterns


def pattern_score(patterns: list[CandlePattern]) -> int:
    """Hitung skor bersih dari semua pola candle yang terdeteksi."""
    score = 0
    for p in patterns:
        if p.signal == "BULLISH":
            score += p.strength
        elif p.signal == "BEARISH":
            score -= p.strength
    return score
