#!/usr/bin/env python3
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
from src.fetcher import (
    get_idx_stock_list, fetch_stock_data, refresh_stock_list,
    get_stock_count, fetch_stocks_parallel, clear_ohlcv_cache
)
from src.analyzer import analyze

app = Flask(__name__)


def _last_scalar(series):
    """Ambil elemen terakhir sebuah kolom sebagai float skalar.
    Kadang data dari yfinance untuk ticker tertentu punya kolom terduplikasi
    sehingga df['volume'] bisa jadi DataFrame, bukan Series — ambil kolom
    pertama agar tetap dapat skalar, bukan error 'cannot convert series to float'.
    """
    v = series.iloc[-1]
    if hasattr(v, 'iloc'):
        v = v.iloc[0]
    return float(v)


def _f(v, decimals=2, fallback=0.0):
    """Bulatkan float dan ganti NaN/Inf dengan fallback agar JSON valid."""
    try:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return fallback
        return round(v, decimals)
    except (TypeError, ValueError):
        return fallback


def sig_to_dict(sig):
    return {
        'ticker': sig.ticker,
        'last_price': _f(sig.last_price, 0),
        'change_pct': _f(sig.change_pct),
        'score': sig.score,
        'total_score': sig.total_score,
        'recommendation': sig.recommendation,
        'rsi': _f(sig.rsi),
        'macd': _f(sig.macd, 4),
        'macd_signal': _f(sig.macd_signal, 4),
        'macd_hist': _f(sig.macd_hist, 4),
        'ema9': _f(sig.ema9),
        'ema21': _f(sig.ema21),
        'sma50': _f(sig.sma50),
        'bb_upper': _f(sig.bb_upper),
        'bb_mid': _f(sig.bb_mid),
        'bb_lower': _f(sig.bb_lower),
        'stoch_k': _f(sig.stoch_k),
        'stoch_d': _f(sig.stoch_d),
        'volume_ratio': _f(sig.volume_ratio),
        'signals': sig.signals,
        'warnings': sig.warnings,
        'bandar': {
            'mfi': _f(sig.bandar.mfi),
            'mfi_status': sig.bandar.mfi_status,
            'obv_trend': sig.bandar.obv_trend,
            'obv_divergence': sig.bandar.obv_divergence,
            'ad_trend': sig.bandar.ad_trend,
            'vpt_trend': sig.bandar.vpt_trend,
            'vwap': _f(sig.bandar.vwap, 0),
            'vwap_position': sig.bandar.vwap_position,
            'volume_spike': sig.bandar.volume_spike,
            'volume_spike_ratio': _f(sig.bandar.volume_spike_ratio),
            'pola': sig.bandar.pola,
            'wyckoff_phase': sig.bandar.wyckoff_phase,
            'trap_detected': sig.bandar.trap_detected,
            'trap_warning': sig.bandar.trap_warning,
            'bandar_score': sig.bandar.bandar_score,
            'bandar_score_pct': _f(sig.bandar.bandar_score_pct, 1),
            'recommendation': sig.bandar.recommendation,
            'bandar_list': sig.bandar.bandar_list,
            'avg_bandar_entry': _f(sig.bandar.avg_bandar_entry, 0),
            'accum_days': sig.bandar.accum_days,
            'accum_start_date': sig.bandar.accum_start_date,
            'current_vs_entry': _f(sig.bandar.current_vs_entry),
            'signals': sig.bandar.signals,
            'warnings': sig.bandar.warnings,
        },
        'fc': {
            'forecast_3d': _f(sig.fc.forecast_3d, 0),
            'forecast_5d': _f(sig.fc.forecast_5d, 0),
            'trend_direction': sig.fc.trend_direction,
            'confidence': sig.fc.confidence,
            'resistance_1': _f(sig.fc.resistance_1, 0),
            'resistance_2': _f(sig.fc.resistance_2, 0),
            'support_1': _f(sig.fc.support_1, 0),
            'support_2': _f(sig.fc.support_2, 0),
            'target_price': _f(sig.fc.target_price, 0),
            'stop_loss': _f(sig.fc.stop_loss, 0),
            'risk_reward': _f(sig.fc.risk_reward),
            'atr': _f(sig.fc.atr, 0),
            'atr_pct': _f(sig.fc.atr_pct),
            'fib_382': _f(sig.fc.fib_382, 0),
            'fib_500': _f(sig.fc.fib_500, 0),
            'fib_618': _f(sig.fc.fib_618, 0),
            'signals': sig.fc.signals,
        },
        'swing': {
            'forecast_10d': _f(sig.swing.forecast_10d, 0),
            'forecast_20d': _f(sig.swing.forecast_20d, 0),
            'trend_medium': sig.swing.trend_medium,
            'trend_strength': sig.swing.trend_strength,
            'ema20': _f(sig.swing.ema20),
            'ema50': _f(sig.swing.ema50),
            'swing_entry': _f(sig.swing.swing_entry, 0),
            'swing_target_1': _f(sig.swing.swing_target_1, 0),
            'swing_target_2': _f(sig.swing.swing_target_2, 0),
            'swing_stop': _f(sig.swing.swing_stop, 0),
            'swing_rr': _f(sig.swing.swing_rr),
            'hold_days_est': sig.swing.hold_days_est,
            'swing_resistance': _f(sig.swing.swing_resistance, 0),
            'swing_support': _f(sig.swing.swing_support, 0),
            'weekly_change_pct': _f(sig.swing.weekly_change_pct),
            'monthly_change_pct': _f(sig.swing.monthly_change_pct),
            'swing_score': sig.swing.swing_score,
            'swing_feasible': sig.swing.swing_feasible,
            'swing_rating': sig.swing.swing_rating,
            'signals': sig.swing.signals,
            'warnings': sig.swing.warnings,
        },
        'strat': {
            'overnight_score': sig.strat.overnight_score,
            'overnight_feasible': sig.strat.overnight_feasible,
            'overnight_win_rate': _f(sig.strat.overnight_win_rate, 1),
            'overnight_gap_freq': _f(sig.strat.overnight_gap_freq, 1),
            'overnight_avg_gap': _f(sig.strat.overnight_avg_gap),
            'overnight_signals': sig.strat.overnight_signals,
            'overnight_warnings': sig.strat.overnight_warnings,
            'overnight_price': {
                'entry': _f(sig.strat.overnight_price.entry, 0),
                'target_1': _f(sig.strat.overnight_price.target_1, 0),
                'target_2': _f(sig.strat.overnight_price.target_2, 0),
                'stop_loss': _f(sig.strat.overnight_price.stop_loss, 0),
                'risk_reward': _f(sig.strat.overnight_price.risk_reward),
                'potential_profit_pct': _f(sig.strat.overnight_price.potential_profit_pct),
                'potential_loss_pct': _f(sig.strat.overnight_price.potential_loss_pct),
            },
            'intraday_score': sig.strat.intraday_score,
            'intraday_feasible': sig.strat.intraday_feasible,
            'intraday_win_rate': _f(sig.strat.intraday_win_rate, 1),
            'intraday_avg_range_pct': _f(sig.strat.intraday_avg_range_pct),
            'intraday_signals': sig.strat.intraday_signals,
            'intraday_warnings': sig.strat.intraday_warnings,
            'intraday_price': {
                'entry': _f(sig.strat.intraday_price.entry, 0),
                'target_1': _f(sig.strat.intraday_price.target_1, 0),
                'target_2': _f(sig.strat.intraday_price.target_2, 0),
                'stop_loss': _f(sig.strat.intraday_price.stop_loss, 0),
                'risk_reward': _f(sig.strat.intraday_price.risk_reward),
                'potential_profit_pct': _f(sig.strat.intraday_price.potential_profit_pct),
                'potential_loss_pct': _f(sig.strat.intraday_price.potential_loss_pct),
            },
            'candle_patterns': [
                {'name': p.name, 'signal': p.signal, 'strength': p.strength, 'description': p.description}
                for p in sig.strat.candle_patterns
            ],
            'pattern_score': sig.strat.pattern_score,
            'hist_bullish_days_pct': _f(sig.strat.hist_bullish_days_pct, 1),
            'hist_avg_daily_return': _f(sig.strat.hist_avg_daily_return),
            'hist_best_return': _f(sig.strat.hist_best_return),
            'hist_worst_return': _f(sig.strat.hist_worst_return),
            'hist_volatility': _f(sig.strat.hist_volatility),
        }
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/screen/stream')
def screen_stream():
    max_stocks = request.args.get('max', 9999, type=int)
    max_workers = min(max(5, request.args.get('workers', 20, type=int)), 50)
    min_volume = request.args.get('min_volume', 1000, type=int)

    def generate():
        try:
            tickers = get_idx_stock_list()[:max_stocks]
            total = len(tickers)

            yield f"data: {json.dumps({'type':'phase','phase':'download','total':total})}\n\n"
            downloaded = fetch_stocks_parallel(tickers, max_workers=max_workers)

            yield f"data: {json.dumps({'type':'phase','phase':'analyze','downloaded':len(downloaded),'total':total})}\n\n"
            results = []
            for i, (ticker, df) in enumerate(downloaded.items()):
                try:
                    if _last_scalar(df['volume']) < min_volume:
                        continue
                    yield f"data: {json.dumps({'type':'progress','current':i+1,'total':len(downloaded),'ticker':ticker})}\n\n"
                    sig = analyze(ticker, df)
                    if sig is None:
                        continue
                    d = sig_to_dict(sig)
                    results.append(d)
                    yield f"data: {json.dumps({'type':'result','data':d})}\n\n"
                except Exception:
                    continue
            results.sort(key=lambda x: x['total_score'], reverse=True)
            yield f"data: {json.dumps({'type':'complete','total_tickers':total,'analyzed':len(results)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


LIQUID_STOCKS = [
    "BBCA","BBRI","BMRI","TLKM","ASII","GOTO","BUKA","ANTM","ADRO","INDF",
    "KLBF","UNVR","ICBP","BSDE","PWON","PGAS","MEDC","PTBA","SMGR","TPIA",
    "TBIG","MNCN","EMTK","EXCL","ISAT","BRPT","BYAN","MDKA","AMMN","SIDO",
    "ACES","MAPI","ERAA","TOWR","JSMR","WSKT","WIKA","WTON","AKRA","INCO"
]

def scalping_score(d):
    """Hitung skor khusus untuk scalping: prioritas volume, volatilitas, momentum bersih."""
    s = 0
    vr = d['volume_ratio']
    if vr >= 3:   s += 4
    elif vr >= 2: s += 3
    elif vr >= 1.5: s += 2
    elif vr >= 1: s += 1
    else:         s -= 2

    atr = d['fc']['atr_pct']
    if atr >= 3:   s += 3
    elif atr >= 2: s += 2
    elif atr >= 1.5: s += 1
    elif atr < 0.8: s -= 1

    rsi = d['rsi']
    if 45 <= rsi <= 65:   s += 3
    elif 35 <= rsi <= 72: s += 1
    elif rsi > 75 or rsi < 28: s -= 2

    if d['macd'] > d['macd_signal']: s += 2

    mfi = d['bandar']['mfi']
    if 40 <= mfi <= 70: s += 2
    elif mfi > 85 or mfi < 20: s -= 1

    chg = d['change_pct']
    if 0.5 <= chg <= 5:   s += 2
    elif chg > 5:         s += 1
    elif chg < -3:        s -= 2

    if d['stoch_k'] < 80 and d['stoch_d'] < 80: s += 1

    if d['bandar']['pola'] in ('BREAKOUT', 'MARKING UP'): s += 3
    elif d['bandar']['pola'] == 'AKUMULASI': s += 1

    return s


@app.route('/api/generate', methods=['POST'])
def generate_report():
    """Generate laporan rekomendasi scalping + bullish dari data yang dikirim atau scan cepat."""
    payload   = request.json or {}
    sent_data = payload.get('results', [])

    if sent_data:
        all_data = sent_data
    else:
        # Scan cepat 40 saham paling likuid
        all_data = []
        for tk in LIQUID_STOCKS:
            df = fetch_stock_data(tk)
            if df is None: continue
            sig = analyze(tk, df)
            if sig: all_data.append(sig_to_dict(sig))

    if not all_data:
        return jsonify({'error': 'Tidak ada data untuk dianalisis'}), 400

    # Tambahkan scalping_score ke tiap emiten
    for d in all_data:
        d['scalping_score'] = scalping_score(d)

    # Kategori
    scalping   = sorted(all_data, key=lambda x: x['scalping_score'], reverse=True)[:5]
    bullish    = [r for r in sorted(all_data, key=lambda x: x['total_score'], reverse=True)
                  if r['recommendation'] in ('STRONG BUY', 'BUY')][:5]
    overnight  = [r for r in all_data if r['strat']['overnight_feasible']]
    overnight  = sorted(overnight, key=lambda x: x['strat']['overnight_win_rate'], reverse=True)[:5]
    intraday   = [r for r in all_data if r['strat']['intraday_feasible']]
    intraday   = sorted(intraday,  key=lambda x: x['strat']['intraday_win_rate'],  reverse=True)[:5]

    from datetime import datetime
    return jsonify({
        'generated_at': datetime.now().strftime('%d %B %Y %H:%M WIB'),
        'total_analyzed': len(all_data),
        'scalping': scalping,
        'bullish': bullish,
        'overnight': overnight,
        'intraday': intraday,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_watch():
    tickers = [t.upper().strip() for t in (request.json or {}).get('tickers', []) if t.strip()]
    if not tickers:
        return jsonify({'error': 'Tidak ada ticker'}), 400
    results = []
    for ticker in tickers[:30]:
        df = fetch_stock_data(ticker)
        if df is None:
            continue
        sig = analyze(ticker, df)
        if sig:
            results.append(sig_to_dict(sig))
    results.sort(key=lambda x: x['total_score'], reverse=True)
    return jsonify({'results': results, 'total_tickers': len(tickers), 'analyzed': len(results)})


@app.route('/api/chart/<ticker>')
def chart(ticker):
    df = fetch_stock_data(ticker.upper(), period_days=90)
    if df is None:
        return jsonify({'error': 'Data tidak tersedia'}), 404
    ema9  = df['close'].ewm(span=9,  adjust=False).mean()
    ema21 = df['close'].ewm(span=21, adjust=False).mean()
    def safe(v, decimals=2):
        try:
            f = float(v)
            return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, decimals)
        except (TypeError, ValueError):
            return 0.0

    return jsonify({
        'dates':  [d.strftime('%d/%m') for d in df.index],
        'close':  [safe(v) for v in df['close']],
        'high':   [safe(v) for v in df['high']],
        'low':    [safe(v) for v in df['low']],
        'open':   [safe(v) for v in df['open']],
        'volume': [int(safe(v, 0)) for v in df['volume']],
        'ema9':   [safe(v) for v in ema9],
        'ema21':  [safe(v) for v in ema21],
    })


@app.route('/api/fundamental/<ticker>')
def fundamental(ticker):
    from src.fundamental import get_fundamentals
    fd = get_fundamentals(ticker.upper())
    if not fd.available:
        return jsonify({'available': False, 'message': 'Data fundamental tidak tersedia'})
    return jsonify({
        'available': True,
        'company_name': fd.company_name,
        'sector': fd.sector,
        'industry': fd.industry,
        'market_cap': _f(fd.market_cap, 0),
        'current_price': _f(fd.current_price, 0),
        'pe_ratio': _f(fd.pe_ratio, 2),
        'pb_ratio': _f(fd.pb_ratio, 2),
        'eps': _f(fd.eps, 2),
        'book_value': _f(fd.book_value, 2),
        'roe': _f(fd.roe, 2),
        'roa': _f(fd.roa, 2),
        'profit_margin': _f(fd.profit_margin, 2),
        'gross_margin': _f(fd.gross_margin, 2),
        'operating_margin': _f(fd.operating_margin, 2),
        'revenue_growth': _f(fd.revenue_growth, 2),
        'earnings_growth': _f(fd.earnings_growth, 2),
        'debt_to_equity': _f(fd.debt_to_equity, 2),
        'current_ratio': _f(fd.current_ratio, 2),
        'dividend_yield': _f(fd.dividend_yield, 2),
        'dividend_rate': _f(fd.dividend_rate, 2),
        'week52_high': _f(fd.week52_high, 0),
        'week52_low': _f(fd.week52_low, 0),
        # Neraca & Arus Kas
        'total_assets': _f(fd.total_assets, 0),
        'total_liabilities': _f(fd.total_liabilities, 0),
        'total_equity': _f(fd.total_equity, 0),
        'cash_and_equivalents': _f(fd.cash_and_equivalents, 0),
        'long_term_debt': _f(fd.long_term_debt, 0),
        'total_revenue': _f(fd.total_revenue, 0),
        'net_income': _f(fd.net_income, 0),
        'operating_cash_flow': _f(fd.operating_cash_flow, 0),
        'free_cash_flow': _f(fd.free_cash_flow, 0),
        'shares_outstanding': _f(fd.shares_outstanding, 0),
        'employees': fd.employees,
        # Harga Wajar
        'graham_number': _f(fd.graham_number, 0),
        'pe_fair_value': _f(fd.pe_fair_value, 0),
        'peg_fair_value': _f(fd.peg_fair_value, 0),
        'dcf_fair_value': _f(fd.dcf_fair_value, 0),
        'fair_value_avg': _f(fd.fair_value_avg, 0),
        'margin_of_safety': _f(fd.margin_of_safety, 1),
        'ideal_buy_price': _f(fd.ideal_buy_price, 0),
        'max_buy_price': _f(fd.max_buy_price, 0),
        'valuation_label': fd.valuation_label,
        # Outlook
        'future_outlook': fd.future_outlook,
        'outlook_score': fd.outlook_score,
        'outlook_reasons': fd.outlook_reasons,
        'outlook_risks': fd.outlook_risks,
        'investment_score': fd.investment_score,
        'investment_rating': fd.investment_rating,
        'signals': fd.signals,
        'warnings': fd.warnings,
    })


def _pre_ara_volume_bonus(df):
    """Deteksi volume building up 3-5 hari terakhir (sinyal pre-ARA)."""
    bonus = 0
    try:
        vol = df['volume']
        if len(vol) < 10:
            return 0
        avg3  = vol.iloc[-3:].mean()
        avg10 = vol.iloc[-13:-3].mean()
        if avg10 > 0:
            if avg3 >= avg10 * 2.0: bonus += 3
            elif avg3 >= avg10 * 1.5: bonus += 2
            elif avg3 >= avg10 * 1.2: bonus += 1

        # Higher lows pattern (konsolidasi bullish)
        closes = df['close'].iloc[-5:].values
        if len(closes) >= 4:
            highs = [max(closes[i-1], closes[i]) for i in range(1, len(closes))]
            lows  = [min(closes[i-1], closes[i]) for i in range(1, len(closes))]
            if all(lows[i] >= lows[i-1] * 0.995 for i in range(1, len(lows))):
                bonus += 2
            elif all(highs[i] >= highs[i-1] for i in range(1, len(highs))):
                bonus += 1
    except Exception:
        pass
    return bonus


def _analyze_resistance_behavior(df, resistance, last_price):
    """
    Deteksi false breakout dan hitung berapa kali resistance sudah ditest.

    False breakout: harga high tembus resistance tapi close kembali di bawahnya.
    Multiple resistance tests meningkatkan probabilitas breakout nyata berikutnya.
    Volume konfirmasi menentukan kualitas breakout saat ini.
    """
    result = {
        'false_breakout_count': 0.0,
        'resistance_tests': 0,
        'breakout_status': 'NONE',   # NONE / APPROACHING / CONFIRMED / FALSE_RISK
        'vol_confirmation': False,
        'penalty': 0,
        'bonus': 0,
        'warning': None,
    }
    if df is None or resistance <= 0 or last_price <= 0:
        return result
    try:
        lookback = min(30, len(df) - 1)
        if lookback < 5:
            return result

        highs   = df['high'].iloc[-lookback:].values
        closes  = df['close'].iloc[-lookback:].values
        volumes = df['volume'].iloc[-lookback:].values

        # Baseline volume dari periode sebelum window
        pre = df['volume'].iloc[max(0, len(df) - lookback - 15):max(1, len(df) - lookback)]
        vol_avg = pre.mean() if len(pre) >= 3 else df['volume'].mean()
        if not vol_avg or vol_avg <= 0:
            vol_avg = df['volume'].mean()

        zone = 0.02   # ±2% = zona sentuh resistance

        # Analisis candle historis (kecuali candle terakhir)
        fb_score = 0.0
        for i in range(len(highs) - 1):
            h, c, v = highs[i], closes[i], volumes[i]
            vr = v / vol_avg if vol_avg > 0 else 1.0
            if h >= resistance * (1 - zone):
                result['resistance_tests'] += 1
                # High tembus tapi close di bawah resistance = false breakout
                if h > resistance and c < resistance:
                    if vr < 1.2:
                        fb_score += 1.0   # volume rendah = false breakout murni
                    elif vr < 1.8:
                        fb_score += 0.7
                    else:
                        fb_score += 0.4   # ada volume tapi tetap gagal

        result['false_breakout_count'] = round(fb_score, 1)

        # Analisis candle terkini
        lh = highs[-1]
        lc = closes[-1]
        lv = volumes[-1]
        cvr = lv / vol_avg if vol_avg > 0 else 1.0

        if lc > resistance:
            result['vol_confirmation'] = cvr >= 1.5
            result['breakout_status'] = 'CONFIRMED' if cvr >= 1.5 else 'FALSE_RISK'
        elif lh > resistance and lc < resistance:
            result['breakout_status'] = 'FALSE_RISK'
        elif resistance > lc and (resistance - lc) / lc <= zone:
            result['breakout_status'] = 'APPROACHING'

        # Penalty untuk false breakout
        fb = result['false_breakout_count']
        if fb >= 2:
            result['penalty'] = 4
            result['warning'] = f"⚠️ {int(fb)}× False Breakout – waspadai jebakan"
        elif fb >= 1:
            result['penalty'] = 2
            result['warning'] = "⚠️ False Breakout terdeteksi"
        elif fb >= 0.5:
            result['penalty'] = 1

        if result['breakout_status'] == 'FALSE_RISK':
            result['penalty'] += 2
            result['warning'] = (result['warning'] or '') + (' — ' if result['warning'] else '') + "❌ Breakout tanpa konfirmasi volume"

        # Bonus untuk multiple resistance tests (setup makin valid)
        rt = result['resistance_tests']
        st = result['breakout_status']
        if st in ('APPROACHING', 'CONFIRMED', 'NONE'):
            if rt >= 3:
                result['bonus'] = 3
            elif rt >= 2:
                result['bonus'] = 2
            elif rt >= 1:
                result['bonus'] = 1

        # Bonus ekstra untuk breakout terkonfirmasi volume
        if st == 'CONFIRMED' and result['vol_confirmation']:
            result['bonus'] += 2

    except Exception:
        pass
    return result


def ara_score(d, df=None):
    """Hitung skor potensi ARA (Auto Rejection Atas). Fokus deteksi SEBELUM ARA terjadi."""
    s = 0
    vr = d['volume_ratio']
    if vr >= 5:     s += 5
    elif vr >= 3:   s += 4
    elif vr >= 2:   s += 2
    elif vr >= 1.5: s += 1
    else:           s -= 1

    # Bobot change_pct dikurangi — kita ingin deteksi sebelum harga sudah naik besar
    chg = d['change_pct']
    if chg >= 5:    s += 2   # sudah naik banyak hari ini, bukan pre-ARA
    elif chg >= 3:  s += 3
    elif chg >= 1:  s += 2
    elif chg >= 0:  s += 1   # flat tapi akumulasi diam = sinyal bagus
    elif chg < -3:  s -= 3

    pola = d['bandar']['pola']
    if pola == 'BREAKOUT':     s += 4
    elif pola == 'MARKING UP': s += 3
    elif pola == 'AKUMULASI':  s += 3  # dinaikkan: akumulasi sebelum ARA sangat relevan
    elif pola == 'DISTRIBUSI': s -= 3

    # RSI sweet spot: 45-68 = setup pre-ARA terbaik (ada ruang naik, momentum membangun)
    rsi = d['rsi']
    if 45 <= rsi <= 68: s += 3
    elif 68 < rsi <= 78: s += 2
    elif rsi > 78:       s += 1
    elif 35 <= rsi < 45: s += 1  # reversal setup
    elif rsi < 35:       s -= 1

    # MACD bullish crossover
    if d['macd'] > d['macd_signal']: s += 2
    # MACD histogram positif dan tumbuh = momentum membangun
    if d.get('macd_hist', 0) > 0: s += 1

    mfi = d['bandar']['mfi']
    if mfi >= 60:   s += 2
    elif mfi >= 50: s += 1
    elif mfi < 30:  s -= 1

    last = d['last_price']
    if last > 0:
        fc3_pct = (d['fc']['forecast_3d'] - last) / last * 100
        if fc3_pct >= 5:   s += 3
        elif fc3_pct >= 3: s += 2
        elif fc3_pct >= 1: s += 1

    if d['total_score'] >= 10: s += 2
    elif d['total_score'] >= 6: s += 1

    if last > 0 and d['ema9'] > 0 and d['ema21'] > 0:
        if last > d['ema9'] > d['ema21']: s += 2
        elif last > d['ema9']:            s += 1

    atr = d['fc']['atr_pct']
    if atr >= 3:    s += 2
    elif atr >= 2:  s += 1

    # Pre-ARA bonus: OBV trend naik = akumulasi diam-diam
    if d['bandar']['obv_trend'] == 'NAIK': s += 2
    elif d['bandar']['obv_trend'] == 'TURUN': s -= 1

    # Pre-ARA bonus: harga dekat resistance = siap breakout
    r1 = d['fc']['resistance_1']
    if last > 0 and r1 > 0:
        dist_r = (r1 - last) / last * 100
        if 0 < dist_r <= 2:    s += 3  # sangat dekat resistance
        elif 0 < dist_r <= 5:  s += 2  # dekat resistance
        elif dist_r < 0:       s += 1  # sudah di atas resistance (breakout)

    # Pre-ARA bonus: VWAP dan swing trend
    if d['bandar']['vwap_position'] == 'DI ATAS': s += 1
    if d.get('swing', {}).get('trend_medium') == 'NAIK': s += 1

    # Pre-ARA bonus: volume buildup dari df (deteksi akumulasi multi-hari)
    if df is not None:
        s += _pre_ara_volume_bonus(df)

        # Analisis false breakout & resistance behavior
        rb = _analyze_resistance_behavior(df, r1, last)
        s -= rb['penalty']
        s += rb['bonus']
        d['breakout_analysis'] = {
            'false_breakout_count': rb['false_breakout_count'],
            'resistance_tests':     rb['resistance_tests'],
            'breakout_status':      rb['breakout_status'],
            'vol_confirmation':     rb['vol_confirmation'],
            'warning':              rb['warning'],
        }

        # Deteksi momentum bullish langsung (tanpa fase tes resistance lebih dulu)
        try:
            cl = df['close'].iloc[-4:].values
            vl = df['volume'].iloc[-4:].values
            va = df['volume'].iloc[-24:-4].mean() if len(df) > 8 else df['volume'].mean()
            if len(cl) >= 3 and va > 0:
                green_streak = all(cl[i] > cl[i - 1] for i in range(1, len(cl)))
                if green_streak and vl[-1] >= va * 1.5:
                    s += 2   # bullish langsung + volume konfirmasi
                elif green_streak:
                    s += 1   # bullish langsung tanpa volume besar
        except Exception:
            pass
    else:
        d['breakout_analysis'] = {}

    return s


def ara_limit_pct(price):
    if price < 200:    return 35
    elif price <= 5000: return 25
    else:               return 20


def ara_validity(d, df):
    """
    ARA Validity Score (0–100): seberapa aman entry tanpa risiko di-guyur bandar.

    10 kriteria @ max 10 poin = 100 total.
    Fokus utama: apakah bandar MASIH akumulasi atau sudah distribusi?

    Score   Label
    80-100  SANGAT VALID  — bandar masih butuh dorong harga, aman entry
    65-79   VALID         — entry dengan stop loss ketat di -3% s/d -5%
    50-64   RAGU-RAGU     — pantau 1-2 hari lagi sebelum masuk
    <50     TIDAK VALID   — risiko tinggi di-guyur, hindari
    """
    score   = 0
    reasons = []
    risks   = []

    b   = d.get('bandar', {})
    rsi = d.get('rsi', 50)
    chg = d.get('change_pct', 0)
    vr  = d.get('volume_ratio', 1.0)

    cve        = b.get('current_vs_entry', 0)
    wyckoff    = b.get('wyckoff_phase', 'UNKNOWN')
    pola       = b.get('pola', 'TIDAK ADA')
    trap       = b.get('trap_detected', 'TIDAK ADA')
    obv_trend  = b.get('obv_trend', 'NETRAL')
    obv_div    = b.get('obv_divergence', 'TIDAK ADA')
    mfi        = b.get('mfi', 50)
    ad_trend   = b.get('ad_trend', 'NETRAL')
    accum_days = b.get('accum_days', 0)

    # ── 1. Bandar Profitability (0-10) ───────────────────────────
    # Semakin kecil jarak harga ke entry bandar = bandar belum profit besar
    # = bandar MASIH butuh dorong harga lebih tinggi = aman entry
    if cve < 5:
        score += 10
        reasons.append(f"Harga hanya {cve:.1f}% di atas entry bandar — bandar baru mulai markup, belum distribusi")
    elif cve < 15:
        score += 8
        reasons.append(f"Bandar profit {cve:.1f}% dari entry — masih wajar, belum tanda distribusi")
    elif cve < 25:
        score += 5
    elif cve < 40:
        score += 2
        risks.append(f"Harga sudah {cve:.1f}% di atas entry bandar — mulai waspadai distribusi")
    else:
        risks.append(f"Harga {cve:.1f}% di atas entry bandar — bandar sudah profit besar, risiko dump tinggi")

    # ── 2. Wyckoff Phase (0-10) ──────────────────────────────────
    # Fase Wyckoff = posisi saham dalam siklus bandar
    if wyckoff == 'ACCUMULATION':
        score += 10
        reasons.append("Fase Wyckoff ACCUMULATION — titik entry terbaik, harga belum markup")
    elif wyckoff == 'MARKUP':
        score += 7
        reasons.append("Fase Wyckoff MARKUP — tren naik aktif, ikuti momentum")
    elif wyckoff == 'UNKNOWN':
        score += 4
    elif wyckoff == 'DISTRIBUTION':
        risks.append("Fase Wyckoff DISTRIBUTION — bandar sedang jual ke retail, JANGAN BELI")
    elif wyckoff == 'MARKDOWN':
        risks.append("Fase Wyckoff MARKDOWN — tren turun aktif, tidak ada alasan masuk")

    # ── 3. Bandar Pattern (0-10) ─────────────────────────────────
    if pola == 'AKUMULASI':
        score += 10
        reasons.append("Pola AKUMULASI — bandar kumpulkan saham, harga belum naik = entry ideal")
    elif pola == 'BREAKOUT':
        score += 9
        reasons.append("Pola BREAKOUT — bandar dorong harga keluar konsolidasi")
    elif pola in ('MARKUP', 'MARKING UP'):
        score += 6
        reasons.append("Pola MARKUP — bandar aktif dorong harga")
    elif pola == 'DISTRIBUSI':
        risks.append("Pola DISTRIBUSI — bandar jual saham ke retail, HINDARI")
    else:
        score += 3

    # ── 4. Trap Safety (0-10) ────────────────────────────────────
    # Trap = bandar buat ilusi naik untuk menarik retail lalu dump
    if trap == 'TIDAK ADA':
        score += 10
        reasons.append("Tidak ada sinyal trap — pergerakan harga genuine")
    elif trap == 'FAKE BREAKOUT':
        risks.append("FAKE BREAKOUT: High tembus resistance tapi close balik ke bawah — bandar jual ke retail yang FOMO")
    elif trap == 'UPTHRUST':
        risks.append("UPTHRUST (Wyckoff): Spike tajam langsung dibalik — bandar distribusi ke pembeli panik")
    elif trap == 'NO DEMAND':
        score += 2
        risks.append("NO DEMAND: Harga naik tapi volume rendah — rally semu, tidak ada institusi di belakangnya")

    # ── 5. OBV Quality (0-10) ────────────────────────────────────
    # OBV = rekam jejak volume. Bullish divergence = bandar beli diam-diam
    if obv_div == 'BULLISH':
        score += 10
        reasons.append("OBV Bullish Divergence — bandar akumulasi diam-diam saat harga tertekan")
    elif obv_trend == 'NAIK':
        score += 7
        reasons.append("OBV naik — volume beli mendukung kenaikan harga")
    elif obv_trend == 'NETRAL':
        score += 4
    elif obv_div == 'BEARISH':
        risks.append("OBV Bearish Divergence — bandar jual diam-diam saat harga masih tinggi")
    else:
        score += 2
        risks.append("OBV turun — volume tidak mendukung kenaikan")

    # ── 6. MFI Zone (0-10) ───────────────────────────────────────
    # MFI 40-65 = sweet spot: uang masuk, belum overbought
    if 40 <= mfi <= 65:
        score += 10
        reasons.append(f"MFI {mfi:.0f} di zona optimal — momentum membangun, belum overbought")
    elif 20 <= mfi < 40:
        score += 8
        reasons.append(f"MFI {mfi:.0f} oversold — tekanan jual mereda, potensi reversal")
    elif mfi < 20:
        score += 6
        reasons.append(f"MFI {mfi:.0f} sangat oversold — tunggu konfirmasi balik naik")
    elif 65 < mfi <= 75:
        score += 5
    elif 75 < mfi <= 85:
        score += 2
        risks.append(f"MFI {mfi:.0f} mendekati overbought — aliran dana mulai melambat")
    else:
        risks.append(f"MFI {mfi:.0f} overbought — uang sudah banyak masuk, risiko kehabisan buyer")

    # ── 7. Volume Consistency (0-10) ─────────────────────────────
    # Volume naik 3 hari berturut = akumulasi nyata, bukan manipulasi 1 hari
    vol_pts = 3
    if df is not None and len(df) >= 5:
        try:
            va20   = float(df['volume'].rolling(20).mean().iloc[-1]) or 1.0
            ratios = [float(df['volume'].iloc[i]) / va20 for i in [-3, -2, -1]]
            konsisten_naik = ratios[2] >= ratios[1] >= ratios[0] * 0.85
            semua_atas     = all(r > 1.2 for r in ratios)
            if semua_atas and konsisten_naik:
                vol_pts = 10
                reasons.append(f"Volume 3 hari konsisten di atas rata-rata — akumulasi genuine bukan spike satu hari")
            elif ratios[2] > 1.5 and ratios[1] > 1.2:
                vol_pts = 7
                reasons.append(f"Volume 2 hari tinggi berturut — minat beli meningkat")
            elif ratios[2] > 1.2:
                vol_pts = 4
            elif ratios[2] < 0.6:
                vol_pts = 0
                risks.append(f"Volume hari ini sangat rendah ({ratios[2]:.1f}×) — tidak ada minat beli nyata")
        except Exception:
            pass
    score += vol_pts

    # ── 8. Momentum Not Exhausted (0-10) ─────────────────────────
    # Saham yang sudah naik terlalu banyak hari ini = berisiko exhaustion
    # Retail yang kejar harga di puncak = target dump bandar
    if chg < 5 and rsi < 60:
        score += 10
        reasons.append(f"Momentum belum panas (naik {chg:.1f}%, RSI {rsi:.0f}) — ruang naik masih besar")
    elif chg < 8 and rsi < 65:
        score += 8
    elif chg < 12 and rsi < 70:
        score += 5
    elif chg < 18 and rsi < 75:
        score += 2
        risks.append(f"Kenaikan {chg:.1f}% hari ini cukup besar — hindari kejar harga, tunggu koreksi kecil")
    else:
        risks.append(f"Kenaikan {chg:.1f}% + RSI {rsi:.0f} — kemungkinan exhaustion, beli = jebakan puncak")

    # ── 9. Buy Pressure / Candle Quality (0-10) ──────────────────
    # Close dekat high = buyer masih dominan, belum ada penolakan di atas
    bp_pts = 4
    if df is not None and len(df) >= 1:
        try:
            lh = float(df['high'].iloc[-1])
            ll = float(df['low'].iloc[-1])
            lc = float(df['close'].iloc[-1])
            lo = float(df['open'].iloc[-1])
            dr = lh - ll
            if dr > 0:
                bp = (lc - ll) / dr
                br = abs(lc - lo) / dr
                if bp >= 0.85 and br >= 0.6:
                    bp_pts = 10
                    reasons.append(f"Buy pressure {bp*100:.0f}% + candle bullish solid — buyer full control")
                elif bp >= 0.70:
                    bp_pts = 7
                    reasons.append(f"Close di {bp*100:.0f}% atas range — tekanan beli dominan")
                elif bp >= 0.50:
                    bp_pts = 5
                elif bp < 0.30:
                    bp_pts = 1
                    risks.append(f"Close di area bawah range ({bp*100:.0f}%) — seller lebih kuat dari buyer")
        except Exception:
            pass
    score += bp_pts

    # ── 10. A/D Confirmation (0-10) ──────────────────────────────
    # Chaikin A/D: konfirmasi independen arah aliran uang
    if ad_trend == 'AKUMULASI':
        score += 10
        reasons.append("Chaikin A/D Line naik — konfirmasi independen uang masuk secara konsisten")
    elif ad_trend == 'NETRAL':
        score += 5
    else:
        risks.append("Chaikin A/D Line turun — uang keluar dari saham ini secara konsisten")

    # ── Bonus: Durasi akumulasi panjang ──────────────────────────
    if accum_days >= 10:
        score = min(100, score + 3)
        reasons.append(f"Akumulasi {accum_days} hari — bandar sabar kumpulkan, breakout biasanya kuat")
    elif accum_days >= 5:
        score = min(100, score + 1)

    # ── Penalti keras ─────────────────────────────────────────────
    # Trap + distribusi = langsung cap maksimum 35 (tidak valid)
    if trap != 'TIDAK ADA' and pola == 'DISTRIBUSI':
        score = min(score, 25)
    elif trap != 'TIDAK ADA':
        score = min(score, 38)
    elif wyckoff in ('DISTRIBUTION', 'MARKDOWN') and pola == 'DISTRIBUSI':
        score = min(score, 30)

    score = max(0, min(100, score))

    if score >= 80:   label = "SANGAT VALID"
    elif score >= 65: label = "VALID"
    elif score >= 50: label = "RAGU-RAGU"
    else:             label = "TIDAK VALID"

    return {
        'score':   score,
        'label':   label,
        'reasons': reasons[:3],
        'risks':   risks[:3],
    }


def pre_ara_screen(d, df):
    """
    Filter ketat 8 kriteria PRE-ARA dengan probabilitas lanjut naik.
    Semua kriteria harus terpenuhi agar is_candidate = True.
    """
    WEIGHTS = {
        'three_candle_up': 15,
        'vol2d_ok':        15,
        'strong_close':    15,
        'range_ok':        15,
        'body_ok':         10,
        'value_ok':        10,
        'no_exhaustion':   10,
        'break_high20':    10,
    }
    result = {k: False for k in WEIGHTS}
    result.update({'probability': 0.0, 'criteria_met': 0, 'is_candidate': False,
                   'value_transaksi_m': 0.0, 'high20': 0.0, 'pct_to_ara': 0.0})

    if df is None or len(df) < 22:
        return result

    try:
        close  = df['close']
        high   = df['high']
        low    = df['low']
        open_  = df['open']
        volume = df['volume']

        last_close = float(close.iloc[-1])
        last_high  = float(high.iloc[-1])
        last_low   = float(low.iloc[-1])
        last_open  = float(open_.iloc[-1])

        # 1. Close 3 candle terakhir naik (C0 > C1 > C2)
        result['three_candle_up'] = (
            last_close > float(close.iloc[-2]) > float(close.iloc[-3])
        )

        # 2. Volume 2 candle terakhir > 1.5x avg volume 20 hari
        vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
        if vol_avg20 > 0:
            result['vol2d_ok'] = (
                float(volume.iloc[-1]) > 1.5 * vol_avg20 and
                float(volume.iloc[-2]) > 1.5 * vol_avg20
            )

        # 3. Close terakhir >= 97% dari high (strong close)
        if last_high > 0:
            result['strong_close'] = last_close >= 0.97 * last_high

        # 4. Body candle / range >= 0.6
        day_range = last_high - last_low
        if day_range > 0:
            result['body_ok'] = abs(last_close - last_open) / day_range >= 0.6

        # 5. Jarak ke harga ARA di range 3%-12%
        ara_lim  = ara_limit_pct(last_close)
        chg_pct  = d['change_pct']
        pct_to_ara = ara_lim - chg_pct
        result['pct_to_ara'] = round(pct_to_ara, 2)
        result['range_ok'] = 3.0 <= pct_to_ara <= 12.0

        # 6. Value transaksi >= 20M IDR
        value_m = last_close * float(volume.iloc[-1]) / 1_000_000
        result['value_transaksi_m'] = round(value_m, 1)
        result['value_ok'] = value_m >= 20.0

        # 7. Kenaikan hari ini < 20% (hindari exhaustion)
        result['no_exhaustion'] = chg_pct < 20.0

        # 8. Break high 20 hari ATAU resistance terdekat
        high20 = float(high.iloc[-21:-1].max())
        result['high20'] = round(high20, 0)
        r1 = d['fc']['resistance_1']
        result['break_high20'] = (
            last_close > high20 or
            (r1 > 0 and last_close >= r1 * 0.99)
        )

        # Probabilitas weighted (0–100%)
        met_score = sum(w for k, w in WEIGHTS.items() if result[k])
        result['criteria_met']  = sum(1 for k in WEIGHTS if result[k])
        result['probability']   = float(met_score)
        result['is_candidate']  = all(result[k] for k in WEIGHTS)

    except Exception:
        pass

    return result


@app.route('/api/stocks/status')
def stocks_status():
    """Cek status daftar saham: jumlah, sumber, dan kapan cache expire."""
    info = get_stock_count()
    stocks = get_idx_stock_list()
    info['sample'] = stocks[:10]
    return jsonify(info)


@app.route('/api/stocks/refresh', methods=['POST'])
def stocks_refresh():
    """Paksa refresh daftar saham dari IDX API, hapus cache lama."""
    stocks = refresh_stock_list()
    return jsonify({
        'success': True,
        'total': len(stocks),
        'sample': stocks[:10],
        'message': f'Berhasil memperbarui {len(stocks)} kode saham dari IDX API'
    })


@app.route('/ara')
def ara_page():
    return render_template('ara.html')


@app.route('/api/stocks/clear-cache', methods=['POST'])
def clear_cache_endpoint():
    ticker = request.json.get('ticker') if request.json else None
    clear_ohlcv_cache(ticker)
    return jsonify({'success': True, 'message': 'Cache OHLCV dibersihkan'})


@app.route('/api/ara-hunting')
def ara_hunting():
    """Stream screening ARA candidates secara paralel."""
    max_stocks = request.args.get('max', 9999, type=int)
    max_workers = min(max(5, request.args.get('workers', 20, type=int)), 50)
    min_volume = request.args.get('min_volume', 1000, type=int)

    def generate():
        try:
            tickers = get_idx_stock_list()[:max_stocks]
            total = len(tickers)

            yield f"data: {json.dumps({'type':'phase','phase':'download','total':total})}\n\n"
            downloaded = fetch_stocks_parallel(tickers, max_workers=max_workers)

            yield f"data: {json.dumps({'type':'phase','phase':'analyze','downloaded':len(downloaded),'total':total})}\n\n"
            candidates = []
            for i, (ticker, df) in enumerate(downloaded.items()):
                try:
                    if _last_scalar(df['volume']) < min_volume:
                        continue
                    yield f"data: {json.dumps({'type':'progress','current':i+1,'total':len(downloaded),'ticker':ticker})}\n\n"
                    sig = analyze(ticker, df)
                    if sig is None: continue
                    d = sig_to_dict(sig)
                    score = ara_score(d, df)
                    pre_ara  = pre_ara_screen(d, df)
                    validity = ara_validity(d, df)
                    d['pre_ara']  = pre_ara
                    d['validity'] = validity
                    if pre_ara['is_candidate']:
                        score += 3
                    elif pre_ara['criteria_met'] >= 6:
                        score += 1
                    d['ara_score'] = score
                    d['ara_limit_pct'] = ara_limit_pct(d['last_price'])
                    d['ara_target'] = round(d['last_price'] * (1 + d['ara_limit_pct'] / 100))
                    if score >= 10 or pre_ara['is_candidate']:
                        candidates.append(d)
                        yield f"data: {json.dumps({'type':'result','data':d})}\n\n"
                except Exception:
                    continue
            candidates.sort(key=lambda x: x['ara_score'], reverse=True)
            yield f"data: {json.dumps({'type':'complete','total_tickers':total,'found':len(candidates)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/broker/<ticker>')
def broker_data(ticker):
    """Data institutional holders + volume analysis per emiten."""
    period = min(max(5, request.args.get('period', 30, type=int)), 90)
    from src.broker_data import fetch_broker_data
    try:
        data = fetch_broker_data(ticker.upper(), period_days=period)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sentiment/<ticker>')
def get_sentiment(ticker):
    """Sentimen mikro & makro untuk emiten tertentu."""
    df = fetch_stock_data(ticker.upper())
    if df is None:
        return jsonify({'error': 'Data tidak tersedia'}), 404
    sig = analyze(ticker.upper(), df)
    if sig is None:
        return jsonify({'error': 'Analisis gagal'}), 500
    d = sig_to_dict(sig)

    from src.fundamental import get_fundamentals
    fd = get_fundamentals(ticker.upper())

    # ── MICRO ──
    micro_score = 0
    micro_signals = []
    micro_warnings = []

    vr = d['volume_ratio']
    if vr >= 2:
        micro_score += 2
        micro_signals.append(f"Volume {vr:.1f}× di atas rata-rata — aktivitas beli meningkat signifikan")
    elif vr >= 1.5:
        micro_score += 1
        micro_signals.append(f"Volume {vr:.1f}× di atas normal — minat beli tumbuh")
    elif vr < 0.5:
        micro_score -= 1
        micro_warnings.append(f"Volume sangat sepi ({vr:.2f}×) — likuiditas rendah")

    chg = d['change_pct']
    if chg >= 3:
        micro_score += 2
        micro_signals.append(f"Harga naik {chg:.1f}% hari ini — momentum bullish kuat")
    elif chg >= 1:
        micro_score += 1
        micro_signals.append(f"Harga naik {chg:.1f}% — momentum positif")
    elif chg < -3:
        micro_score -= 2
        micro_warnings.append(f"Harga turun {abs(chg):.1f}% — tekanan jual dominan")
    elif chg < -1:
        micro_score -= 1
        micro_warnings.append(f"Harga turun {abs(chg):.1f}% — sentimen bearish")

    pola = d['bandar']['pola']
    pola_map = {
        'BREAKOUT':    (+3, "Pola BREAKOUT — smart money masuk, potensi naik besar"),
        'MARKING UP':  (+2, "Pola MARKING UP — bandar aktif dorong harga ke atas"),
        'AKUMULASI':   (+1, "Pola AKUMULASI — smart money kumpulkan saham perlahan"),
        'DISTRIBUSI':  (-2, "Pola DISTRIBUSI — bandar terindikasi jual, waspadai penurunan"),
    }
    if pola in pola_map:
        pts, msg = pola_map[pola]
        micro_score += pts
        (micro_signals if pts > 0 else micro_warnings).append(msg)

    rsi = d['rsi']
    if 45 <= rsi <= 65:
        micro_score += 1
        micro_signals.append(f"RSI {rsi:.0f} di zona optimal — ruang naik masih tersedia")
    elif rsi < 30:
        micro_score += 2
        micro_signals.append(f"RSI {rsi:.0f} oversold — potensi rebound kuat")
    elif rsi > 75:
        micro_score -= 1
        micro_warnings.append(f"RSI {rsi:.0f} overbought — waspadai koreksi jangka pendek")

    mfi = d['bandar']['mfi']
    if mfi >= 60:
        micro_score += 1
        micro_signals.append(f"MFI {mfi:.0f} — aliran dana bersih masuk positif")
    elif mfi < 30:
        micro_score -= 1
        micro_warnings.append(f"MFI {mfi:.0f} — aliran dana bersih keluar dari saham ini")

    if d['macd'] > d['macd_signal']:
        micro_score += 1
        micro_signals.append("MACD bullish crossover — momentum naik terkonfirmasi")
    else:
        micro_warnings.append("MACD di bawah signal line — momentum masih melemah")

    if micro_score >= 5:      micro_label, micro_cls = "SANGAT BULLISH", "green"
    elif micro_score >= 2:    micro_label, micro_cls = "BULLISH", "green"
    elif micro_score >= -1:   micro_label, micro_cls = "NETRAL", "blue"
    elif micro_score >= -3:   micro_label, micro_cls = "BEARISH", "red"
    else:                     micro_label, micro_cls = "SANGAT BEARISH", "red"

    # ── MACRO ──
    macro_score = 0
    macro_signals = []
    macro_warnings = []

    bi_rate = 5.75
    inflation = 2.48

    if bi_rate <= 5.0:
        macro_score += 2
        macro_signals.append(f"BI Rate {bi_rate}% — suku bunga rendah kondusif untuk saham")
    elif bi_rate <= 6.25:
        macro_score += 1
        macro_signals.append(f"BI Rate {bi_rate}% — suku bunga moderat, masih kondusif")
    else:
        macro_score -= 1
        macro_warnings.append(f"BI Rate {bi_rate}% — suku bunga tinggi tekan valuasi & kredit")

    if inflation <= 3.0:
        macro_score += 1
        macro_signals.append(f"Inflasi {inflation}% (terkendali) — daya beli konsumen terjaga")
    elif inflation > 5.0:
        macro_score -= 1
        macro_warnings.append(f"Inflasi {inflation}% — berpotensi menekan margin perusahaan")

    macro_warnings.append("Rupiah terhadap USD perlu dimonitor — depresiasi tingkatkan biaya impor bahan baku")
    macro_signals.append("Pertumbuhan ekonomi Indonesia +5% YoY — fondasi fundamental kuat")

    sector = fd.sector if fd.available else ""
    SECTOR_MACRO = {
        'Financial Services': {
            's': ["Pertumbuhan kredit perbankan stabil mendukung NIM", "Digitalisasi layanan keuangan perluas penetrasi pasar"],
            'w': ["NPL monitor ketat seiring perlambatan global"]
        },
        'Technology': {
            's': ["Digitalisasi dan transformasi digital terus akselerasi"],
            'w': ["Valuasi tech global terkoreksi — waspadai de-rating multiple"]
        },
        'Basic Materials': {
            's': ["Permintaan nikel & mineral kritis global meningkat seiring EV boom"],
            'w': ["Volatilitas harga komoditas global mempengaruhi pendapatan"]
        },
        'Consumer Cyclical': {
            's': ["Konsumsi domestik menjadi penopang utama ekonomi Indonesia"],
            'w': ["Tekanan daya beli masyarakat dari inflasi dan nilai tukar"]
        },
        'Consumer Defensive': {
            's': ["Sektor defensif tahan banting di tengah ketidakpastian global"],
            'w': ["Margin tertekan jika harga bahan baku naik"]
        },
        'Energy': {
            's': ["Harga energi global masih supportif untuk emiten energi domestik"],
            'w': ["Transisi energi global ancam permintaan batu bara jangka panjang"]
        },
        'Real Estate': {
            's': ["Demand properti tier 1 & 2 kota besar tetap solid"],
            'w': ["Suku bunga KPR tinggi menekan daya beli segmen menengah"]
        },
        'Healthcare': {
            's': ["Spending kesehatan Indonesia tumbuh pasca pandemi"],
            'w': ["Regulasi BPJS dan harga obat jadi risiko utama farmasi"]
        },
        'Industrials': {
            's': ["Investasi infrastruktur pemerintah dorong demand sektor industri"],
            'w': ["Kenaikan upah dan biaya energi tekan margin produsen"]
        },
        'Utilities': {
            's': ["Demand listrik Indonesia tumbuh seiring industrialisasi"],
            'w': ["Tarif listrik diatur pemerintah — fleksibilitas pricing terbatas"]
        },
        'Communication Services': {
            's': ["Penetrasi internet dan mobile terus tumbuh"],
            'w': ["Persaingan ketat antar operator menekan ARPU"]
        },
    }
    sm = SECTOR_MACRO.get(sector, {'s': ["Monitor perkembangan ekonomi global & domestik untuk konteks sektoral"], 'w': []})
    macro_signals.extend(sm['s'])
    macro_warnings.extend(sm['w'])

    if macro_score >= 3:    macro_label, macro_cls = "KONDUSIF", "green"
    elif macro_score >= 1:  macro_label, macro_cls = "CUKUP KONDUSIF", "green"
    elif macro_score >= -1: macro_label, macro_cls = "NETRAL", "blue"
    else:                   macro_label, macro_cls = "TIDAK KONDUSIF", "red"

    upcoming_events = [
        {'date': '2026-04-22', 'event': 'Rapat Dewan Gubernur BI — Keputusan Suku Bunga', 'impact': 'TINGGI', 'type': 'macro'},
        {'date': '2026-04-28', 'event': 'Rapat FOMC The Fed — Fed Funds Rate Decision', 'impact': 'TINGGI', 'type': 'global'},
        {'date': '2026-05-05', 'event': 'Rilis Data Inflasi April 2026 (BPS)', 'impact': 'TINGGI', 'type': 'macro'},
        {'date': '2026-05-05', 'event': 'Rilis Data PMI Manufaktur April 2026', 'impact': 'SEDANG', 'type': 'macro'},
        {'date': '2026-05-15', 'event': 'Rilis Data Neraca Perdagangan April 2026', 'impact': 'SEDANG', 'type': 'macro'},
        {'date': '2026-05-20', 'event': 'Rapat Dewan Gubernur BI — Keputusan Suku Bunga', 'impact': 'TINGGI', 'type': 'macro'},
        {'date': '2026-06-04', 'event': 'Rilis Data GDP Q1 2026 (BPS)', 'impact': 'TINGGI', 'type': 'macro'},
        {'date': '2026-06-17', 'event': 'Rapat FOMC The Fed — Fed Funds Rate Decision', 'impact': 'TINGGI', 'type': 'global'},
        {'date': '2026-06-17', 'event': 'Rapat Dewan Gubernur BI — Keputusan Suku Bunga', 'impact': 'TINGGI', 'type': 'macro'},
    ]

    return jsonify({
        'ticker': ticker.upper(),
        'sector': sector,
        'micro': {
            'score': micro_score,
            'label': micro_label,
            'cls': micro_cls,
            'signals': micro_signals,
            'warnings': micro_warnings,
        },
        'macro': {
            'score': macro_score,
            'label': macro_label,
            'cls': macro_cls,
            'signals': macro_signals,
            'warnings': macro_warnings,
            'bi_rate': bi_rate,
            'inflation': inflation,
        },
        'upcoming_events': upcoming_events,
    })


@app.route('/bandarmology')
def bandarmology_page():
    return render_template('bandarmology.html')


@app.route('/invest-suggestion')
def invest_suggestion_page():
    return render_template('invest_suggestion.html')


@app.route('/dividen')
def dividen_page():
    return render_template('dividen.html')


@app.route('/api/bandarmology')
def bandarmology_scan():
    """Stream scan emiten dengan pola akumulasi bandar sebelum harga naik."""
    max_stocks = min(max(50, request.args.get('max', 150, type=int)), 500)

    def generate():
        try:
            tickers = get_idx_stock_list()[:max_stocks]
            candidates = []
            for i, ticker in enumerate(tickers):
                yield f"data: {json.dumps({'type':'progress','current':i+1,'total':len(tickers),'ticker':ticker})}\n\n"
                df = fetch_stock_data(ticker)
                if df is None:
                    continue
                sig = analyze(ticker, df)
                if sig is None:
                    continue
                d = sig_to_dict(sig)
                pola = d['bandar']['pola']
                if pola not in ('AKUMULASI', 'MARKUP', 'MARKING UP', 'BREAKOUT'):
                    continue
                # Trap langsung disaring
                if d['bandar']['trap_detected'] != 'TIDAK ADA':
                    continue
                # Wyckoff phase buruk = skip
                if d['bandar']['wyckoff_phase'] in ('DISTRIBUTION', 'MARKDOWN'):
                    continue

                # Composite score berbasis bandar_score_pct (0-100) + booster kontekstual
                bs_pct = d['bandar']['bandar_score_pct']   # 0–100
                bs = bs_pct                                  # base dari weighted score

                accum_days = d['bandar']['accum_days']
                if accum_days >= 10:          bs += 5
                elif accum_days >= 5:         bs += 3
                elif accum_days >= 3:         bs += 1

                if pola == 'BREAKOUT':        bs += 6
                elif pola in ('MARKUP', 'MARKING UP'): bs += 4
                elif pola == 'AKUMULASI':     bs += 2

                if d['bandar']['obv_divergence'] == 'BULLISH': bs += 4
                if d['bandar']['wyckoff_phase'] == 'ACCUMULATION': bs += 3
                if d['bandar']['wyckoff_phase'] == 'MARKUP':       bs += 2
                if d['rsi'] < 65:             bs += 1
                if d['total_score'] >= 6:     bs += 2

                d['bandar_composite_score'] = round(bs, 1)
                if bs >= 70:
                    candidates.append(d)
                    yield f"data: {json.dumps({'type':'result','data':d})}\n\n"
            candidates.sort(key=lambda x: x['bandar_composite_score'], reverse=True)
            yield f"data: {json.dumps({'type':'complete','total_tickers':len(tickers),'found':len(candidates)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/invest-suggestion')
def invest_suggestion_scan():
    """Stream scan emiten terbaik untuk investasi jangka panjang berdasarkan fundamental."""
    max_stocks = min(max(50, request.args.get('max', 100, type=int)), 300)

    def generate():
        try:
            from src.fundamental import get_fundamentals
            tickers = get_idx_stock_list()[:max_stocks]
            candidates = []
            for i, ticker in enumerate(tickers):
                yield f"data: {json.dumps({'type':'progress','current':i+1,'total':len(tickers),'ticker':ticker})}\n\n"
                fd = get_fundamentals(ticker)
                if not fd.available:
                    continue
                if fd.investment_score < 7:
                    continue
                d = {
                    'ticker': ticker,
                    'company_name': fd.company_name,
                    'sector': fd.sector,
                    'industry': fd.industry,
                    'market_cap': _f(fd.market_cap, 0),
                    'current_price': _f(fd.current_price, 0),
                    'pe_ratio': _f(fd.pe_ratio, 2),
                    'pb_ratio': _f(fd.pb_ratio, 2),
                    'roe': _f(fd.roe, 2),
                    'roa': _f(fd.roa, 2),
                    'profit_margin': _f(fd.profit_margin, 2),
                    'revenue_growth': _f(fd.revenue_growth, 2),
                    'earnings_growth': _f(fd.earnings_growth, 2),
                    'debt_to_equity': _f(fd.debt_to_equity, 2),
                    'dividend_yield': _f(fd.dividend_yield, 2),
                    'fair_value_avg': _f(fd.fair_value_avg, 0),
                    'margin_of_safety': _f(fd.margin_of_safety, 1),
                    'valuation_label': fd.valuation_label,
                    'investment_score': fd.investment_score,
                    'investment_rating': fd.investment_rating,
                    'future_outlook': fd.future_outlook,
                    'outlook_score': fd.outlook_score,
                    'signals': fd.signals[:3],
                    'total_assets': _f(fd.total_assets, 0),
                    'total_equity': _f(fd.total_equity, 0),
                    'free_cash_flow': _f(fd.free_cash_flow, 0),
                    'employees': fd.employees,
                }
                candidates.append(d)
                yield f"data: {json.dumps({'type':'result','data':d})}\n\n"
            candidates.sort(key=lambda x: (x['investment_score'], x['outlook_score']), reverse=True)
            yield f"data: {json.dumps({'type':'complete','total_tickers':len(tickers),'found':len(candidates)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/dividen')
def dividen_scan():
    """Stream scan emiten yang membagikan dividen, dikelompokkan per bulan."""
    max_stocks = min(max(50, request.args.get('max', 200, type=int)), 500)

    def generate():
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            tickers = get_idx_stock_list()[:max_stocks]
            cutoff = datetime.now() - timedelta(days=400)
            for i, ticker in enumerate(tickers):
                yield f"data: {json.dumps({'type':'progress','current':i+1,'total':len(tickers),'ticker':ticker})}\n\n"
                try:
                    t = yf.Ticker(ticker + '.JK')
                    divs = t.dividends
                    if divs is None or len(divs) == 0:
                        continue
                    recent = divs[divs.index >= cutoff.strftime('%Y-%m-%d')]
                    if len(recent) == 0:
                        continue
                    info = t.info or {}
                    dy = _f(info.get('dividendYield', 0))
                    if 0 < dy < 1:
                        dy = dy * 100
                    price = _f(info.get('currentPrice') or info.get('previousClose') or 0)
                    company = str(info.get('shortName') or info.get('longName') or ticker)
                    sector = str(info.get('sector') or '—')
                    months = []
                    for dt, amount in recent.items():
                        try:
                            m = int(str(dt)[:7].split('-')[1])
                            months.append({'month': m, 'amount': _f(float(amount), 2), 'date': str(dt)[:10]})
                        except Exception:
                            pass
                    if not months:
                        continue
                    d = {
                        'ticker': ticker,
                        'company_name': company,
                        'sector': sector,
                        'dividend_yield': dy,
                        'current_price': price,
                        'dividend_months': months,
                    }
                    yield f"data: {json.dumps({'type':'result','data':d})}\n\n"
                except Exception:
                    continue
            yield f"data: {json.dumps({'type':'complete','total_tickers':len(tickers)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ─────────────────────────────────────────────────────────────
# CRYPTO SCREENER & WHALE TRACKER
# ─────────────────────────────────────────────────────────────

@app.route('/crypto')
def crypto():
    return render_template('crypto.html')


@app.route('/api/crypto/market')
def crypto_market():
    from src.crypto_screener import market_overview
    try:
        return jsonify(market_overview())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/crypto/screen/stream')
def crypto_screen_stream():
    limit = min(max(10, request.args.get('limit', 40, type=int)), 100)

    def generate():
        try:
            from src.crypto_screener import screen_coins
            results = []

            def on_progress(current, total, symbol):
                yield_val = json.dumps({'type': 'progress', 'current': current, 'total': total, 'symbol': symbol})
                results.append(('progress', yield_val))

            # We can't yield inside callback in a generator, so collect and stream
            pass

            coins_done = []
            from src.crypto_fetcher import get_top_coins, get_coin_history, get_whale_alerts_live
            from src.crypto_analyzer import compute_indicators, crypto_score, whale_score, entry_targets, recommendation, strategy_label, probability
            import math

            def _fmt(v, d=4):
                try:
                    f = float(v)
                    return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, d)
                except Exception:
                    return 0.0

            STABLES = {'usdt','usdc','busd','dai','tusd','usdd','frax','usdp','gusd','lusd','susd','fei','mim'}
            coins = get_top_coins(limit=limit)
            if not coins:
                yield f"data: {json.dumps({'type':'error','message':'Gagal mengambil data CoinGecko'})}\n\n"
                return

            for i, coin in enumerate(coins):
                sym = coin.get('symbol', '').upper()
                yield f"data: {json.dumps({'type':'progress','current':i+1,'total':len(coins),'symbol':sym,'name':coin.get('name','')})}\n\n"

                df = get_coin_history(coin['id'], days=60)
                ind = compute_indicators(df)
                if ind is None:
                    continue

                mkt_price = coin.get('current_price', ind['last']) or ind['last']
                ind['last'] = mkt_price

                sc, sigs, warns = crypto_score(coin, ind)
                ws, wsigs = whale_score(coin, ind)
                rec, rec_col = recommendation(sc, ws)
                et = entry_targets(coin, ind)
                prob = probability(sc, ws)

                d = {
                    'id': coin['id'], 'symbol': sym, 'name': coin.get('name', ''),
                    'image': coin.get('image', ''), 'rank': coin.get('market_cap_rank', 999) or 999,
                    'price': mkt_price,
                    'chg_1h':  _fmt(coin.get('price_change_percentage_1h_in_currency', 0), 2),
                    'chg_24h': _fmt(coin.get('price_change_percentage_24h', 0), 2),
                    'chg_7d':  _fmt(coin.get('price_change_percentage_7d_in_currency', 0), 2),
                    'market_cap': coin.get('market_cap', 0) or 0,
                    'volume_24h': coin.get('total_volume', 0) or 0,
                    'score': sc, 'whale_score': ws, 'rec': rec, 'rec_color': rec_col,
                    'probability': prob, 'strategy': strategy_label(sc, ind),
                    'signals': sigs, 'warnings': warns, 'whale_signals': wsigs,
                    'entry': et,
                    'ind': {
                        'rsi': _fmt(ind['rsi'], 1), 'macd_hist': _fmt(ind['macd_hist'], 6),
                        'ema9': _fmt(ind['ema9'], 4), 'ema20': _fmt(ind['ema20'], 4),
                        'ema50': _fmt(ind['ema50'], 4), 'ema200': _fmt(ind['ema200'], 4),
                        'bb_upper': _fmt(ind['bb_upper'], 4), 'bb_lower': _fmt(ind['bb_lower'], 4),
                        'vol_ratio': _fmt(ind['vol_ratio'], 2),
                        'stoch_k': _fmt(ind['stoch_k'], 1), 'stoch_d': _fmt(ind['stoch_d'], 1),
                        'atr_pct': _fmt(ind['atr_pct'], 2), 'vwap': _fmt(ind['vwap'], 4),
                        'support': _fmt(ind['support'], 4), 'resistance': _fmt(ind['resistance'], 4),
                    }
                }
                coins_done.append(d)
                yield f"data: {json.dumps({'type':'result','data':d})}\n\n"

            coins_done.sort(key=lambda x: x['score'] + x['whale_score'] / 10, reverse=True)
            yield f"data: {json.dumps({'type':'complete','total':len(coins),'analyzed':len(coins_done),'top10':[c['symbol'] for c in coins_done[:10]]})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/crypto/whale')
def crypto_whale():
    from src.crypto_fetcher import get_whale_alerts_live
    alerts = get_whale_alerts_live(min_usd=500_000)
    return jsonify({'alerts': alerts, 'has_key': bool(__import__('os').environ.get('WHALE_ALERT_KEY', ''))})


if __name__ == '__main__':
    print("\n  IHSG Screener Web — http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
