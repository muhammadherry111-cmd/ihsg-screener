import math
from src.crypto_fetcher import get_top_coins, get_coin_history, get_global_market, get_fear_greed, get_trending, get_whale_alerts_live
from src.crypto_analyzer import compute_indicators, crypto_score, whale_score, entry_targets, recommendation, strategy_label, probability

def _fmt(v, d=4):
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, d)
    except Exception:
        return 0.0

def screen_coins(limit=40, callback=None):
    """
    Screens top `limit` coins.
    callback(i, total, symbol) called for progress updates.
    Returns list of result dicts sorted best-first.
    """
    coins = get_top_coins(limit=limit)
    if not coins:
        return []

    results = []
    for i, coin in enumerate(coins):
        sym = coin.get('symbol', '').upper()
        if callback:
            callback(i + 1, len(coins), sym)

        df = get_coin_history(coin['id'], days=60)
        ind = compute_indicators(df)
        if ind is None:
            continue

        # Use real-time market price from CoinGecko markets endpoint
        mkt_price = coin.get('current_price', ind['last']) or ind['last']
        ind['last'] = mkt_price

        sc, sigs, warns = crypto_score(coin, ind)
        ws, wsigs = whale_score(coin, ind)
        rec, rec_col = recommendation(sc, ws)
        et = entry_targets(coin, ind)
        prob = probability(sc, ws)

        results.append({
            'id':       coin['id'],
            'symbol':   sym,
            'name':     coin.get('name', ''),
            'image':    coin.get('image', ''),
            'rank':     coin.get('market_cap_rank', 999) or 999,
            'price':    mkt_price,
            'chg_1h':   _fmt(coin.get('price_change_percentage_1h_in_currency', 0), 2),
            'chg_24h':  _fmt(coin.get('price_change_percentage_24h', 0), 2),
            'chg_7d':   _fmt(coin.get('price_change_percentage_7d_in_currency', 0), 2),
            'market_cap': coin.get('market_cap', 0) or 0,
            'volume_24h': coin.get('total_volume', 0) or 0,
            'score':    sc,
            'whale_score': ws,
            'rec':      rec,
            'rec_color': rec_col,
            'probability': prob,
            'strategy': strategy_label(sc, ind),
            'signals':  sigs,
            'warnings': warns,
            'whale_signals': wsigs,
            'entry':    et,
            'ind': {
                'rsi': _fmt(ind['rsi'], 1),
                'macd': _fmt(ind['macd'], 6),
                'macd_hist': _fmt(ind['macd_hist'], 6),
                'ema9':  _fmt(ind['ema9'], 4),
                'ema20': _fmt(ind['ema20'], 4),
                'ema50': _fmt(ind['ema50'], 4),
                'ema200': _fmt(ind['ema200'], 4),
                'bb_upper': _fmt(ind['bb_upper'], 4),
                'bb_lower': _fmt(ind['bb_lower'], 4),
                'vol_ratio': _fmt(ind['vol_ratio'], 2),
                'stoch_k': _fmt(ind['stoch_k'], 1),
                'stoch_d': _fmt(ind['stoch_d'], 1),
                'atr_pct': _fmt(ind['atr_pct'], 2),
                'vwap':  _fmt(ind['vwap'], 4),
                'support': _fmt(ind['support'], 4),
                'resistance': _fmt(ind['resistance'], 4),
            }
        })

    results.sort(key=lambda x: x['score'] + x['whale_score'] / 10, reverse=True)
    return results

def market_overview():
    gd = get_global_market()
    fg = get_fear_greed()
    trending = get_trending()

    btc_dom = gd.get('market_cap_percentage', {}).get('btc', 0)
    eth_dom = gd.get('market_cap_percentage', {}).get('eth', 0)
    altseason = max(0, min(100, (50 - btc_dom) * 4 + 50))

    return {
        'total_mcap': gd.get('total_market_cap', {}).get('usd', 0),
        'total_vol': gd.get('total_volume', {}).get('usd', 0),
        'btc_dom': round(btc_dom, 1),
        'eth_dom': round(eth_dom, 1),
        'altseason': round(altseason, 0),
        'mcap_chg_24h': round(gd.get('market_cap_change_percentage_24h_usd', 0), 2),
        'active_coins': gd.get('active_cryptocurrencies', 0),
        'fear_greed': fg['value'],
        'fear_greed_label': fg['label'],
        'trending': trending,
    }
