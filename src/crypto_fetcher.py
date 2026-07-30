import requests, time, os, numpy as np, pandas as pd

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
WHALE_ALERT_KEY = os.environ.get("WHALE_ALERT_KEY", "")
_last_cg_req = 0
_CG_DELAY = 1.5  # rate limit: 40 req/min on free tier

def _cg_get(path, params=None, retries=3):
    global _last_cg_req
    wait = _CG_DELAY - (time.time() - _last_cg_req)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(retries):
        try:
            resp = requests.get(f"{COINGECKO_BASE}{path}", params=params, timeout=12)
            _last_cg_req = time.time()
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
    return None

def get_top_coins(limit=60):
    """Return top coins by volume, excluding stablecoins."""
    STABLES = {'usdt','usdc','busd','dai','tusd','usdd','frax','usdp','gusd','lusd','susd','fei','mim'}
    data = _cg_get("/coins/markets", {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": min(limit + 20, 250),
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d",
    })
    if not data:
        return []
    return [c for c in data if c.get('symbol','').lower() not in STABLES][:limit]

def get_coin_history(coin_id, days=60):
    """Return DataFrame with close, volume columns (daily resampled)."""
    data = _cg_get(f"/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": days,
    })
    if not data:
        return None
    prices  = data.get('prices', [])
    volumes = data.get('total_volumes', [])
    if len(prices) < 30:
        return None
    df = pd.DataFrame({
        'ts':     [p[0] for p in prices],
        'close':  [p[1] for p in prices],
        'volume': [v[1] for v in volumes],
    })
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    # Resample to daily if hourly data returned
    if len(df) > days * 3:
        df = df.resample('1D').agg({'close': 'last', 'volume': 'sum'}).dropna()
    return df

def get_global_market():
    data = _cg_get("/global")
    return data.get('data', {}) if data else {}

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        fg = r.json()['data'][0]
        return {'value': int(fg['value']), 'label': fg['value_classification']}
    except Exception:
        return {'value': 50, 'label': 'Neutral'}

def get_trending():
    data = _cg_get("/search/trending")
    if not data:
        return []
    return [{'symbol': c['item']['symbol'].upper(), 'name': c['item']['name'], 'rank': c['item']['market_cap_rank']}
            for c in data.get('coins', [])[:7]]

def get_whale_alerts_live(min_usd=500_000):
    if not WHALE_ALERT_KEY:
        return []
    try:
        r = requests.get("https://api.whale-alert.io/v1/transactions", params={
            "api_key": WHALE_ALERT_KEY,
            "min_value": min_usd // 1000,
            "start": int(time.time()) - 3600,
            "limit": 30,
        }, timeout=10)
        return r.json().get('transactions', [])
    except Exception:
        return []
