"""
Analisis posisi bandar: Institutional Holders (yfinance) + Volume Profile (OHLCV).

Dua lapisan:
  1. Global Institutions — data dari yfinance (Vanguard, BlackRock, Fidelity, dll)
  2. Smart Money Lokal   — deteksi dari analisis volume OHLCV
"""

import math
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


# ── Kategori institusi ──────────────────────────────────────────────────────
_TIER_MAP = {
    # ETF & Index Fund besar
    'VANGUARD':   ('ETF / Index Fund', '🌐'),
    'ISHARES':    ('ETF / Index Fund', '🌐'),
    'BLACKROCK':  ('Hedge Fund Global', '🏦'),
    'STATE STREET':('ETF / Index Fund', '🌐'),
    'FIDELITY':   ('Mutual Fund', '💼'),
    'DIMENSIONAL':('Quant Fund', '📊'),
    'NORGES':     ('Sovereign Wealth', '🏛️'),
    'TEMASEK':    ('Sovereign Wealth', '🏛️'),
    'GIC':        ('Sovereign Wealth', '🏛️'),
    'GQG':        ('Hedge Fund Global', '🏦'),
    'SCHRODERS':  ('Asset Manager',    '💼'),
    'ABERDEEN':   ('Asset Manager',    '💼'),
    'FRANKLIN':   ('Mutual Fund',      '💼'),
    'JPMORGAN':   ('Investment Bank',  '🏦'),
    'GOLDMAN':    ('Investment Bank',  '🏦'),
    'MORGAN STANLEY': ('Investment Bank', '🏦'),
    'DEUTSCHE':   ('Investment Bank',  '🏦'),
    'LAZARD':     ('Asset Manager',    '💼'),
    'T. ROWE':    ('Asset Manager',    '💼'),
    'PRICE (T.ROWE)': ('Asset Manager','💼'),
    'EUROPACIFIC': ('Mutual Fund',     '💼'),
    'ADVISORS':   ('Hedge Fund',       '🏦'),
    'SCHWAB':     ('Broker/ETF',       '🌐'),
}

def _classify(name: str):
    upper = name.upper()
    for key, val in _TIER_MAP.items():
        if key in upper:
            return val
    return ('Institusi Lain', '🏢')


def _safe(v, fallback=0.0):
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback


def _shorten_name(name: str, maxlen: int = 38) -> str:
    """Potong nama institusi agar tidak terlalu panjang."""
    # Hapus prefix fund description yang panjang
    for sep in ['-', '—']:
        if sep in name:
            parts = name.split(sep)
            name = parts[-1].strip() if len(parts[-1].strip()) > 6 else parts[0].strip()
            break
    return (name[:maxlen] + '…') if len(name) > maxlen else name


def fetch_broker_data(ticker: str, period_days: int = 30) -> dict:
    """
    Gabungkan data institutional holders (yfinance) dengan volume analysis.

    Returns dict siap kirim ke frontend.
    """
    code = ticker.upper().replace('.JK', '')
    yf_ticker = yf.Ticker(code + '.JK')

    # ── 1. Institutional & Mutual Fund Holders ─────────────────────────────
    institutions = []
    try:
        ih  = yf_ticker.institutional_holders
        mfh = yf_ticker.mutualfund_holders

        for df, src in [(mfh, 'Mutual Fund'), (ih, 'Institusi')]:
            if df is None or len(df) == 0:
                continue
            for _, row in df.iterrows():
                shares = _safe(row.get('Shares', 0))
                value  = _safe(row.get('Value',  0))
                if shares <= 0:
                    continue
                avg_entry = round(value / shares) if value > 0 else 0
                pct_held   = _safe(row.get('pctHeld',   0)) * 100
                pct_change = _safe(row.get('pctChange',  0)) * 100

                raw_name = str(row.get('Holder', 'Unknown'))
                short    = _shorten_name(raw_name)
                category, icon = _classify(raw_name)

                if pct_change > 2:   status = 'AKUMULASI'
                elif pct_change < -2: status = 'DISTRIBUSI'
                else:                status = 'HOLD'

                try:
                    dr = str(row.get('Date Reported', ''))
                    date_str = datetime.strptime(dr[:10], '%Y-%m-%d').strftime('%d %b %Y') if dr else '—'
                except Exception:
                    date_str = '—'

                institutions.append({
                    'name':          short,
                    'full_name':     raw_name,
                    'category':      category,
                    'icon':          icon,
                    'source':        src,
                    'pct_held':      round(pct_held, 3),
                    'shares':        int(shares),
                    'avg_entry':     avg_entry,
                    'pct_change':    round(pct_change, 2),
                    'status':        status,
                    'date_reported': date_str,
                })
    except Exception:
        pass

    # Sort: terbesar holding dulu
    institutions.sort(key=lambda x: x['pct_held'], reverse=True)

    # ── 2. Ringkasan Institusional ─────────────────────────────────────────
    accum_count = sum(1 for i in institutions if i['status'] == 'AKUMULASI')
    distrib_count = sum(1 for i in institutions if i['status'] == 'DISTRIBUSI')
    hold_count    = sum(1 for i in institutions if i['status'] == 'HOLD')

    # Weighted avg entry berdasarkan jumlah shares
    total_shares_w = sum(i['shares'] for i in institutions if i['avg_entry'] > 0)
    avg_entry_weighted = 0
    if total_shares_w > 0:
        avg_entry_weighted = round(sum(
            i['avg_entry'] * i['shares'] for i in institutions if i['avg_entry'] > 0
        ) / total_shares_w)

    total_pct = round(sum(i['pct_held'] for i in institutions), 2)

    net_flow = accum_count - distrib_count
    if net_flow >= 2:      inst_sentiment = 'BULLISH'
    elif net_flow >= 0:    inst_sentiment = 'NETRAL'
    else:                  inst_sentiment = 'BEARISH'

    inst_summary = {
        'total_pct':       total_pct,
        'total_holders':   len(institutions),
        'accum_count':     accum_count,
        'distrib_count':   distrib_count,
        'hold_count':      hold_count,
        'avg_entry_weighted': avg_entry_weighted,
        'sentiment':       inst_sentiment,
    }

    # ── 3. Volume / Smart Money Lokal ──────────────────────────────────────
    from src.fetcher import fetch_stock_data
    from src.bandarmologi import _obv, _detect_accum_start

    df = fetch_stock_data(code, period_days=max(period_days * 2, 90))
    vol_analysis = _build_volume_analysis(df, period_days) if df is not None else {}

    # ── 4. Harga sekarang ──────────────────────────────────────────────────
    current_price = 0
    if df is not None and len(df) > 0:
        current_price = float(df['close'].iloc[-1])

    # ── 5. Rekomendasi Entry ───────────────────────────────────────────────
    combined = _build_recommendation(
        current_price,
        avg_entry_weighted,
        vol_analysis.get('avg_bandar_entry', 0),
        inst_sentiment,
    )

    # ── 6. Timeline chart data (net pctChange per institusi per periode) ───
    timeline = _build_timeline(institutions)

    return {
        'ticker':           code,
        'current_price':    current_price,
        'period_days':      period_days,
        'institutions':     institutions[:15],   # maks 15 ditampilkan
        'inst_summary':     inst_summary,
        'volume_analysis':  vol_analysis,
        'combined':         combined,
        'timeline':         timeline,
    }


def _build_volume_analysis(df, period_days: int) -> dict:
    """Volume profile untuk smart money lokal."""
    if df is None or len(df) < 10:
        return {}

    from src.bandarmologi import _obv, _detect_accum_start

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    last   = float(close.iloc[-1])

    vol_avg20 = volume.rolling(20).mean()
    lookback  = min(period_days * 2, 90, len(df))
    recent    = df.iloc[-lookback:]
    vol_avg_r = vol_avg20.iloc[-lookback:]

    obv_full   = _obv(close, volume)
    accum_days = _detect_accum_start(obv_full, close, lookback)
    accum_days = max(accum_days, 1)
    accum_df   = df.iloc[-accum_days:]

    typical_acc = (accum_df['high'] + accum_df['low'] + accum_df['close']) / 3
    vol_sum_acc = accum_df['volume'].sum()
    avg_entry   = float((typical_acc * accum_df['volume']).sum() / vol_sum_acc) if vol_sum_acc > 0 else last

    try:
        start_date = df.index[-accum_days].strftime('%d %b %Y')
    except Exception:
        start_date = '—'

    # Tier bandar dari volume spike
    big = []; med = []; slow = []
    for i in range(len(recent)):
        row  = recent.iloc[i]
        avg_v = float(vol_avg_r.iloc[i]) if i < len(vol_avg_r) else 0
        if avg_v <= 0:
            continue
        ratio = float(row['volume']) / avg_v
        tp    = (float(row['high']) + float(row['low']) + float(row['close'])) / 3
        dr    = (float(row['high']) - float(row['low'])) / float(row['low']) * 100 if float(row['low']) > 0 else 0
        try:
            ds = recent.index[i].strftime('%d/%m')
        except Exception:
            ds = '—'
        entry = {'date': ds, 'price': tp, 'vol_ratio': round(ratio, 1)}
        if ratio >= 3:
            big.append(entry)
        elif ratio >= 2:
            med.append(entry)
        elif ratio >= 1.2 and dr < 3.5:
            slow.append(entry)

    def _avg_p(lst):
        return sum(s['price'] for s in lst) / len(lst) if lst else 0.0

    bandar_tiers = []
    for tier in [
        ('🏦', 'Smart Money Besar', big,  20, 8),
        ('💼', 'Smart Money Menengah', med, 15, 5),
        ('📦', 'Akumulasi Bertahap', slow, 25, 10),
    ]:
        icon, label, lst, dist_thr, hold_thr = tier
        if not lst:
            continue
        ap  = _avg_p(lst)
        pnl = (last - ap) / ap * 100 if ap > 0 else 0.0
        status = 'DISTRIBUSI' if pnl >= dist_thr else ('HOLD' if pnl >= hold_thr else 'AKUMULASI')
        bandar_tiers.append({
            'icon': icon, 'label': label,
            'avg_entry': round(ap),
            'sessions': len(lst),
            'pnl_pct': round(pnl, 1),
            'status': status,
            'first_date': lst[0]['date'],
            'last_date': lst[-1]['date'],
        })

    return {
        'avg_bandar_entry':  round(avg_entry),
        'accum_days':        accum_days,
        'accum_start_date':  start_date,
        'current_vs_entry':  round((last - avg_entry) / avg_entry * 100, 1) if avg_entry > 0 else 0.0,
        'bandar_tiers':      bandar_tiers,
    }


def _build_recommendation(current: float, inst_entry: float, vol_entry: float, sentiment: str) -> dict:
    """Hitung zona aman entry dan rekomendasi."""
    if inst_entry > 0 and vol_entry > 0:
        avg_entry = round((inst_entry * 0.6 + vol_entry * 0.4))
    elif inst_entry > 0:
        avg_entry = inst_entry
    elif vol_entry > 0:
        avg_entry = vol_entry
    else:
        return {}

    if avg_entry <= 0 or current <= 0:
        return {}

    pct_vs_entry = (current - avg_entry) / avg_entry * 100

    # Zona
    safe_max    = round(avg_entry * 1.10)   # sampai +10% di atas avg entry bandar = aman
    caution_max = round(avg_entry * 1.20)   # +10–20% = hati-hati
    danger_min  = round(avg_entry * 1.20)   # > +20%  = berbahaya

    # Target distribusi bandar (estimasi mereka jual)
    distrib_target = round(avg_entry * 1.30)

    if current < avg_entry:
        zone = 'SANGAT AMAN'
        zone_desc = 'Harga di bawah avg entry bandar — bandar rugi jika jual sekarang, mustahil guyur'
        zone_cls  = 'very-safe'
        rec_action = f'Entry sekarang sangat aman. Target minimal Rp {round(avg_entry * 1.15):,}'
    elif current <= safe_max:
        zone = 'AMAN'
        zone_desc = f'Bandar baru untung {pct_vs_entry:.1f}% — terlalu kecil untuk distribusi'
        zone_cls  = 'safe'
        rec_action = f'Entry masih aman di Rp {round(current * 0.98):,} – Rp {round(current * 1.02):,}'
    elif current <= caution_max:
        zone = 'HATI-HATI'
        zone_desc = f'Bandar untung {pct_vs_entry:.1f}% — mulai pertimbangkan distribusi bertahap'
        zone_cls  = 'caution'
        rec_action = f'Tunggu koreksi ke Rp {round(avg_entry * 1.05):,} sebelum entry. Jika sudah punya, hold'
    else:
        zone = 'BERISIKO TINGGI'
        zone_desc = f'Bandar untung {pct_vs_entry:.1f}% — zona distribusi aktif, risiko guyur sangat tinggi'
        zone_cls  = 'danger'
        rec_action = f'Hindari entry baru. Jika punya, pertimbangkan ambil profit di atas Rp {round(current):,}'

    return {
        'avg_entry':       avg_entry,
        'current_price':   round(current),
        'pct_vs_entry':    round(pct_vs_entry, 1),
        'zone':            zone,
        'zone_desc':       zone_desc,
        'zone_cls':        zone_cls,
        'rec_action':      rec_action,
        'safe_entry_min':  round(avg_entry * 0.95),
        'safe_entry_max':  safe_max,
        'caution_price':   caution_max,
        'danger_price':    danger_min,
        'distrib_target':  distrib_target,
        'inst_entry':      inst_entry,
        'vol_entry':       vol_entry,
    }


def _build_timeline(institutions: list) -> list:
    """Buat data timeline akumulasi/distribusi per institusi."""
    items = []
    for inst in institutions:
        chg = inst['pct_change']
        items.append({
            'name':    inst['name'],
            'icon':    inst['icon'],
            'change':  chg,
            'status':  inst['status'],
            'date':    inst['date_reported'],
        })
    return sorted(items, key=lambda x: x['change'], reverse=True)[:12]
