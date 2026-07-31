from __future__ import annotations

import os
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Cache configuration
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".stock_list_cache.json")
_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 jam
_stock_list_memory: list[str] = []  # in-memory cache untuk session ini

# ─────────────────────────────────────────────────────────────────────────────
# API Sources (dicoba berurutan hingga berhasil)
# ─────────────────────────────────────────────────────────────────────────────
_IDX_SOURCES = [
    # Sumber 1: IDX TradingSummary (primary)
    {
        "url": "https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0",
        "headers": {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.idx.co.id/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        "parser": lambda d: [x["StockCode"] for x in d.get("data", []) if x.get("StockCode")],
    },
    # Sumber 2: IDX ListedCompanies endpoint
    {
        "url": "https://www.idx.co.id/primary/ListedCompany/GetCompanyProfiles?start=0&length=9999&exchangeBoard=",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/",
        },
        "parser": lambda d: [x["StockCode"] for x in d.get("data", []) if x.get("StockCode")],
    },
    # Sumber 3: IDX StockData endpoint
    {
        "url": "https://www.idx.co.id/primary/StockData/GetStockSummary?length=9999&start=0",
        "headers": {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer": "https://www.idx.co.id/",
        },
        "parser": lambda d: [x["StockCode"] for x in d.get("data", []) if x.get("StockCode")],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive fallback list (~900+ emiten IDX yang diketahui)
# Digunakan HANYA jika semua API gagal dan cache tidak ada
# ─────────────────────────────────────────────────────────────────────────────
MAJOR_STOCKS = [
    # A
    "AALI","ABBA","ABDA","ABMM","ACES","ACST","ADCP","ADHI","ADMF","ADMG",
    "ADRO","AGII","AGRO","AGRS","AHAP","AIMS","AISA","AKKU","AKPI","AKRA",
    "AKSI","ALFA","ALII","ALKA","ALMI","ALTO","AMAG","AMFG","AMIN","AMRT",
    "ANDI","ANJT","ANTM","APEX","APIC","APII","APLN","APOL","ARCI","AREA",
    "ARGO","ARII","ARKO","ARMY","ARTA","ARTI","ARTO","ASBI","ASDM","ASII",
    "ASJT","ASMI","ASRM","ASSA","ATIC","ATPK","AUTO","AYLS",
    # B
    "BABP","BACA","BAJA","BALI","BAPA","BAPI","BATA","BAYU","BBCA","BBHI",
    "BBKP","BBMD","BBNI","BBNP","BBRI","BBTN","BBYB","BCAP","BCIC","BCIP",
    "BEKS","BELI","BELL","BESS","BEST","BFIN","BGTG","BHAT","BHIT","BIKA",
    "BIMA","BIPI","BISI","BJBR","BJTM","BKDP","BKSL","BKSW","BLTA","BLTZ",
    "BMHS","BMRI","BMSR","BNBA","BNGA","BNII","BNLI","BOBA","BOGA","BPFI",
    "BPII","BPTR","BRAM","BRMS","BRNA","BRPT","BSDE","BSIM","BSML","BSSR",
    "BTEK","BTEL","BTON","BTPN","BTPS","BUDI","BUKA","BULL","BUMI","BUVA",
    "BVIC","BYAN",
    # C
    "CAKK","CAMP","CANI","CASS","CBMF","CCSI","CEKA","CENT","CFIN","CINT",
    "CITA","CITY","CLEO","CLPI","CMNP","CMPP","CMRY","CNKO","CNTX","COWL",
    "CPRO","CRSN","CSAP","CSIS","CSMI","CTBN","CTTH",
    # D
    "DADA","DAJK","DART","DAYU","DBDH","DCII","DEAN","DFAM","DGIK","DILD",
    "DIVA","DKFT","DLTA","DMAS","DMMX","DPUM","DRMA","DSNG","DSSA","DUCK",
    "DUTI","DYAN",
    # E
    "EAST","ECII","EKAD","ELSA","ELTY","EMDE","EMTK","ENRG","EPAC","EPMT",
    "ERAA","ERTX","ESSA","ESTE","EXCL",
    # F
    "FAST","FASW","FILM","FISH","FMII","FOOD","FORE","FPNI","FREN","FUJI",
    # G
    "GAMA","GARI","GBIC","GEMS","GGRM","GHON","GJTL","GLOB","GLVA","GMTD",
    "GOLD","GOTO","GPRA","GPSO","GRIA","GSMF","GTBO","GTSI",
    # H
    "HADE","HDTX","HEAL","HERO","HKMU","HMSP","HOKI","HOMI","HRME","HRUM",
    # I
    "IATA","IBST","ICBP","ICON","IDEA","IGAR","IKAI","IKBI","IMAS","IMJS",
    "IMPC","INAF","INAI","INCI","INCO","INDF","INDS","INDX","INDY","INKP",
    "INPC","INPP","INRU","INTA","INTD","INTP","IPPE","IPCC","IPTV","IRRA",
    "ISAT","ISSP","ITIC",
    # J
    "JAWA","JBBS","JECC","JIHD","JKON","JMAS","JPFA","JRPT","JSMR","JSPT","JTPE",
    # K
    "KAEF","KARW","KBAG","KBLF","KBLM","KBRI","KDSI","KEEN","KIAS","KICI",
    "KIJA","KLBF","KMTR","KOBX","KOIN","KONI","KOPI","KOTA","KPIG","KRAS","KREN",
    # L
    "LAPD","LCGP","LEAD","LIFE","LINK","LION","LMAS","LMPI","LMSH","LPCK",
    "LPGI","LPKR","LPLI","LPPF","LSIP","LTLS","LUCK",
    # M
    "MABA","MAMI","MAPA","MAPI","MASA","MBAP","MBSS","MBTO","MCAS","MCOL",
    "MDRN","MDKA","MEDC","MERK","META","MFIN","MFMI","MGLE","MICE","MIDI",
    "MIKA","MIRA","MITI","MKPI","MLBI","MLPL","MNCN","MPPA","MPRO","MRAT",
    "MREI","MSIN","MSKY","MTLA","MTOR","MTSM","MYOH","MYOR","MYTX",
    # N
    "NASI","NATO","NELY","NFCX","NICK","NIKL","NIRO","NISP","NOBU","NRCA",
    "NSSS","NTBK","NURI","NUSA","NXFM",
    # O
    "OCAP","OILS","OKAS","OMRE",
    # P
    "PANI","PANR","PANS","PBSA","PCAR","PDPP","PEGE","PEHA","PGAS","PGLO",
    "PICO","PJAA","PKPK","PLAS","PLDT","PLIN","PMMP","PMNA","PNBS","PNGO",
    "PNIN","PNLF","PNSE","POLL","POLY","POOL","POWR","PRAS","PRDA","PRIM",
    "PSAB","PSGO","PTBA","PTIS","PTPP","PTRO","PTSP","PTSN","PUDP","PURE",
    "PURI","PWON","PYFA",
    # R
    "RAJA","RALS","RANC","RBMS","RDTX","RELI","RICY","RIGS","RIMO","ROTI","RUIS",
    # S
    "SAFE","SAME","SAMF","SATU","SCCO","SCMA","SDMU","SDRA","SEKAR","SELC",
    "SFAN","SGRO","SIDO","SILO","SIMA","SIMP","SINI","SIPD","SKBM","SKLT",
    "SMAR","SMCB","SMDR","SMGR","SMMT","SMRA","SMSM","SOBI","SOFA","SONA",
    "SOSS","SPMA","SRIL","SRTG","SSIA","SSMS","SSTM","STAR","STTP","SUGI",
    "SULI","SUPR","SURE",
    # T
    "TALF","TARA","TAXI","TBIG","TBLA","TBMS","TCID","TELE","TFAS","TGKA",
    "TGRA","TINS","TKIM","TLKM","TMAS","TMPO","TOBA","TOPS","TOWR","TPIA",
    "TPMA","TRAM","TRIM","TRIO","TRIS","TRJA","TRST","TSPC","TUGU","TURI",
    # U
    "UANG","UCID","UICS","ULTJ","UNVR","UPCL",
    # V
    "VICO","VINS","VIVA","VOKS","VRNA",
    # W
    "WAPO","WEGE","WIKA","WIIM","WIRA","WIRG","WMUU","WOOD","WSKT","WTON",
    # Y-Z
    "YPAS","YULE","ZINC","ZYRX",
    # ── Emiten baru / IPO 2022-2025 yang sering belum ada di list lama ──
    "AMMN","NICL","CUAN","MAPA","PGEO","AVIA","NCKL","BREN","CBDK","DEWA",
    "POLU","KRYA","MPXL","FUTR","MSIN","BBYB","BANK","RMKO","TRGU","INPS",
    "INET","WIFI","RGAS","KEJU","AYAM","TAYS","GULA","IOTF","BACA","BSWD",
    "BGTG","HOMI","SMKM","TPAS","STRK","AXIO","CGAS","SINI","NEST","PACK",
    "PMMP","FAPA","ITIC","SKRN","ELIT","AMMS","BHAT","OASE","ABDA","LABA",
    "PTIS","OILS","AFII","BUKA","BELI","GOOD","HILL","DADA","SQBB","BIKE",
    "SOTS","COAL","DGNS","PRAY","GTRA","DEWA","CUAN","PANI","AKSI","TRGU",
    "RUNS","IDEA","NASI","DCII","BRMS","BULL","DSSA","ESSA","LABA","SPTO",
    "HRUM","GEMS","MBAP","FIRE","ADRO","INDY","BSSR","PTBA","ITMG","MYOH",
    "SMMT","APEX","PKPK","ARII","PKPK","DOID","MCOL","HITS","BYAN","MBSS",
    "ASSA","GIAA","CMPP","IATA","JSMR","BIRD","MFIN","BFIN","CFIN","PNLF",
    "WOMF","ADMF","BPFI","HDFA","IMJS","FUJI","TELE","LINK","EXCL","ISAT",
    "FREN","TLKM","TOWR","TBIG","MNCN","EMTK","SCMA","MSKY","BMTR","VIVA",
    "ANTM","INCO","TINS","MDKA","NICL","NCKL","IFSH","PSAB","AMMN","PGEO",
    "HRME","HEAL","MIKA","SILO","PRDA","KLBF","KAEF","SIDO","INAF","MERK",
    "PEHA","PYFA","SOHO","DVLA","TSPC","KINO","UNVR","ICBP","INDF","MYOR",
    "CLEO","ROTI","SKBM","SKLT","GOOD","STTP","ULTJ","DLTA","MLBI","HMSP",
    "GGRM","WIIM","RMBA","ITIC","BSDE","SMRA","CTRA","LPKR","PWON","ASRI",
    "DUTI","JRPT","MTLA","APLN","DMAS","MKPI","KIJA","GPRA","COWL","DART",
    "GMTD","MDLN","WIKA","PTPP","ADHI","WSKT","WTON","NRCA","TOTL","SSIA",
    "ACST","DGIK","IDPR","MMLP","BNLI","BNGA","NISP","MAYA","BKSW","NOBU",
    "MEGA","ARTO","BBYB","AGRO","BCIC","BMAS","SDRA","DNAR","BABP","BACA",
    "BGTG","BANK","BBHI","BEKS","BBMD","BJBR","BJTM","BBKP","BNBA","BBNP",
    "AGRS","INPC","INPS","BPII","BBTN","BVIC","PNBS","PNIN","PNSE","PNLF",
]


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> list[str]:
    """Baca cache dari file JSON. Return list kosong jika expired atau tidak ada."""
    try:
        if not os.path.exists(_CACHE_FILE):
            return []
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
        saved_at = data.get("saved_at", 0)
        stocks = data.get("stocks", [])
        if time.time() - saved_at < _CACHE_TTL_SECONDS and len(stocks) > 100:
            age_hours = (time.time() - saved_at) / 3600
            console.print(
                f"[dim cyan]Cache saham valid ({len(stocks)} emiten, "
                f"diperbarui {age_hours:.1f} jam lalu)[/dim cyan]"
            )
            return stocks
    except Exception:
        pass
    return []


def _save_cache(stocks: list[str]) -> None:
    """Simpan daftar saham ke cache file JSON."""
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump({"saved_at": time.time(), "stocks": stocks, "count": len(stocks)}, f)
        console.print(f"[dim green]Cache disimpan: {len(stocks)} emiten[/dim green]")
    except Exception:
        pass


def _deduplicate(stocks: list[str]) -> list[str]:
    """Hapus duplikat sambil pertahankan urutan."""
    seen = set()
    result = []
    for s in stocks:
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main fetcher
# ─────────────────────────────────────────────────────────────────────────────

def get_idx_stock_list(force_refresh: bool = False) -> list[str]:
    """
    Ambil daftar saham IDX lengkap dengan strategi berlapis:

    1. In-memory cache (session ini, instant)
    2. File cache JSON (cache 24 jam, sangat cepat)
    3. IDX API primer   (idx.co.id TradingSummary)
    4. IDX API backup 1 (idx.co.id ListedCompany)
    5. IDX API backup 2 (idx.co.id StockData)
    6. Fallback internal MAJOR_STOCKS (~900+ emiten)

    Parameter:
        force_refresh: paksa ambil ulang dari API meskipun cache masih valid
    """
    global _stock_list_memory

    # 1. In-memory cache
    if not force_refresh and _stock_list_memory:
        return _stock_list_memory

    # 2. File cache
    if not force_refresh:
        cached = _load_cache()
        if cached:
            _stock_list_memory = cached
            return cached

    # 3–5. Coba semua API sources
    console.print("[cyan]Mengambil daftar saham IDX dari API...[/cyan]")
    for i, source in enumerate(_IDX_SOURCES, 1):
        try:
            resp = requests.get(
                source["url"],
                headers=source["headers"],
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                stocks = source["parser"](data)
                stocks = _deduplicate(stocks)
                if len(stocks) > 100:
                    console.print(
                        f"[green]✓ Sumber API {i}: {len(stocks)} emiten ditemukan[/green]"
                    )
                    _save_cache(stocks)
                    _stock_list_memory = stocks
                    return stocks
                else:
                    console.print(
                        f"[yellow]Sumber API {i} mengembalikan data terlalu sedikit ({len(stocks)})[/yellow]"
                    )
        except Exception as e:
            console.print(f"[yellow]Sumber API {i} gagal: {e}[/yellow]")

    # 6. Fallback ke daftar internal
    console.print(
        f"[yellow]⚠ Semua API gagal. Menggunakan daftar internal "
        f"({len(_deduplicate(MAJOR_STOCKS))} emiten)[/yellow]"
    )
    fallback = _deduplicate(MAJOR_STOCKS)
    _stock_list_memory = fallback
    return fallback


def refresh_stock_list() -> list[str]:
    """Paksa refresh daftar saham dari API, abaikan semua cache."""
    global _stock_list_memory
    _stock_list_memory = []
    return get_idx_stock_list(force_refresh=True)


def get_stock_count() -> dict:
    """Kembalikan info jumlah saham dan status cache."""
    stocks = get_idx_stock_list()
    cache_info = {}
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r") as f:
                data = json.load(f)
            saved_at = data.get("saved_at", 0)
            age_hours = (time.time() - saved_at) / 3600
            cache_info = {
                "cached": True,
                "cache_age_hours": round(age_hours, 1),
                "cache_expires_in_hours": round(max(0, 24 - age_hours), 1),
            }
    except Exception:
        cache_info = {"cached": False}

    return {
        "total": len(stocks),
        "source": "cache" if cache_info.get("cached") else "api",
        **cache_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV Fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period_days: int = 90) -> pd.DataFrame | None:
    """Download OHLCV data untuk satu emiten dari Yahoo Finance."""
    symbol = f"{ticker}.JK"
    end = datetime.now()
    start = end - timedelta(days=period_days)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]  # jaga-jaga kolom ganda dari yfinance
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


_OHLCV_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_OHLCV_CACHE_TTL = 4 * 60 * 60  # 4 jam


def fetch_stock_data_cached(ticker: str, period_days: int = 90) -> pd.DataFrame | None:
    """Fetch OHLCV dengan in-memory cache 4 jam."""
    now = time.time()
    if ticker in _OHLCV_CACHE:
        saved_at, df = _OHLCV_CACHE[ticker]
        if now - saved_at < _OHLCV_CACHE_TTL:
            return df.copy()
    df = fetch_stock_data(ticker, period_days)
    if df is not None:
        _OHLCV_CACHE[ticker] = (now, df)
    return df


def fetch_stocks_parallel(
    tickers: list[str],
    max_workers: int = 20,
    period_days: int = 90,
    progress_cb=None,
) -> dict[str, pd.DataFrame]:
    """Download data semua saham secara PARALEL menggunakan ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, pd.DataFrame] = {}
    total = len(tickers)

    def _fetch_one(t: str):
        return t, fetch_stock_data_cached(t, period_days)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            try:
                ticker, df = future.result()
            except Exception:
                ticker = futures[future]
                df = None
            done += 1
            if df is not None:
                results[ticker] = df
            if progress_cb:
                progress_cb(ticker, done, total)

    return results


def clear_ohlcv_cache(ticker: str | None = None) -> None:
    """Hapus cache OHLCV. Jika ticker=None, hapus semua."""
    global _OHLCV_CACHE
    if ticker:
        _OHLCV_CACHE.pop(ticker, None)
    else:
        _OHLCV_CACHE.clear()

