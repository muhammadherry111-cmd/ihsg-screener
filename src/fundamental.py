import math
from dataclasses import dataclass, field

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False


@dataclass
class FundamentalData:
    company_name: str = "—"
    sector: str = "—"
    industry: str = "—"
    market_cap: float = 0.0
    current_price: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    eps: float = 0.0
    book_value: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    profit_margin: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    dividend_yield: float = 0.0
    dividend_rate: float = 0.0
    week52_high: float = 0.0
    week52_low: float = 0.0

    # Neraca & Arus Kas
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    total_equity: float = 0.0
    cash_and_equivalents: float = 0.0
    long_term_debt: float = 0.0
    total_revenue: float = 0.0
    net_income: float = 0.0
    operating_cash_flow: float = 0.0
    free_cash_flow: float = 0.0
    shares_outstanding: float = 0.0
    employees: int = 0

    # Harga Wajar
    graham_number: float = 0.0
    pe_fair_value: float = 0.0
    peg_fair_value: float = 0.0
    dcf_fair_value: float = 0.0
    fair_value_avg: float = 0.0
    margin_of_safety: float = 0.0   # % diskon dari fair value (positif = murah)
    ideal_buy_price: float = 0.0    # 85% dari fair value
    max_buy_price: float = 0.0      # 95% dari fair value
    valuation_label: str = "—"      # SANGAT MURAH / MURAH / WAJAR / MAHAL / SANGAT MAHAL

    # Outlook masa depan
    future_outlook: str = "NETRAL"      # SANGAT MENJANJIKAN / MENJANJIKAN / NETRAL / BERISIKO / SANGAT BERISIKO
    outlook_score: int = 0
    outlook_reasons: list = field(default_factory=list)
    outlook_risks: list = field(default_factory=list)

    investment_score: int = 0
    investment_rating: str = "—"
    signals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    available: bool = False


def _sf(v, pct=False):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return round(x * 100, 4) if pct else x
    except (TypeError, ValueError):
        return 0.0


def _calc_fair_value(eps, book_value, earnings_growth, pe_ratio, dividend_rate, current_price):
    """Hitung harga wajar dengan beberapa metode."""
    fvs = {}

    # 1. Graham Number: sqrt(22.5 × EPS × BVPS)
    if eps > 0 and book_value > 0:
        fvs['graham'] = math.sqrt(22.5 * eps * book_value)

    # 2. P/E Konservatif (P/E 15 untuk perusahaan stabil)
    if eps > 0:
        fvs['pe_conservative'] = eps * 15

    # 3. PEG Fair Value: EPS × growth_rate (Lynch method)
    if eps > 0 and 3 <= earnings_growth <= 50:
        fvs['peg'] = eps * min(earnings_growth, 30)

    # 4. DCF sederhana: EPS × (8.5 + 2×g) / required_return
    # (modifikasi formula Graham untuk pertumbuhan)
    if eps > 0 and earnings_growth > 0:
        g = min(earnings_growth, 15)   # cap pertumbuhan 15%
        r = 9.0                         # required return 9% (risk-free + premium)
        fvs['dcf'] = eps * (8.5 + 2 * g) * 4.4 / r

    # 5. Dividend Discount (untuk saham dividen)
    if dividend_rate > 0 and earnings_growth > 0:
        g = min(earnings_growth / 100, 0.07)   # max 7% sustainable
        r = 0.10                                # required return 10%
        if r > g:
            fvs['ddm'] = dividend_rate / (r - g)

    if not fvs:
        return {}

    # Buang nilai yang sangat jauh dari current_price (outlier)
    if current_price > 0:
        filtered = {k: v for k, v in fvs.items()
                    if current_price * 0.1 < v < current_price * 10}
        if filtered:
            fvs = filtered

    return fvs


def get_fundamentals(ticker: str) -> FundamentalData:
    if not _YF:
        return FundamentalData()
    try:
        info = yf.Ticker(ticker + '.JK').info
        if not info or len(info) < 5:
            return FundamentalData()

        company_name     = str(info.get('shortName') or info.get('longName') or ticker)
        sector           = str(info.get('sector') or '—')
        industry         = str(info.get('industryDisp') or info.get('industry') or '—')
        market_cap       = _sf(info.get('marketCap'))
        current_price    = _sf(info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'))
        pe_ratio         = _sf(info.get('trailingPE') or info.get('forwardPE'))
        pb_ratio         = _sf(info.get('priceToBook'))
        eps              = _sf(info.get('trailingEps'))
        book_value       = _sf(info.get('bookValue'))
        roe              = _sf(info.get('returnOnEquity'),   pct=True)
        roa              = _sf(info.get('returnOnAssets'),   pct=True)
        profit_margin    = _sf(info.get('profitMargins'),    pct=True)
        gross_margin     = _sf(info.get('grossMargins'),     pct=True)
        operating_margin = _sf(info.get('operatingMargins'), pct=True)
        revenue_growth   = _sf(info.get('revenueGrowth'),    pct=True)
        earnings_growth  = _sf(info.get('earningsGrowth'),   pct=True)
        debt_to_equity   = _sf(info.get('debtToEquity'))
        current_ratio    = _sf(info.get('currentRatio'))
        _dy_raw          = _sf(info.get('dividendYield'))
        dividend_yield   = _dy_raw * 100 if 0 < _dy_raw < 1 else _dy_raw
        dividend_rate    = _sf(info.get('dividendRate'))
        week52_high      = _sf(info.get('fiftyTwoWeekHigh'))
        week52_low       = _sf(info.get('fiftyTwoWeekLow'))

        # Neraca & Arus Kas
        total_assets        = _sf(info.get('totalAssets'))
        total_equity        = _sf(info.get('totalStockholderEquity') or info.get('bookValue', 0) * _sf(info.get('sharesOutstanding', 0)))
        total_liabilities   = total_assets - total_equity if total_assets > 0 and total_equity > 0 else 0.0
        cash_and_equivalents = _sf(info.get('totalCash'))
        long_term_debt      = _sf(info.get('longTermDebt'))
        total_revenue       = _sf(info.get('totalRevenue'))
        net_income          = _sf(info.get('netIncomeToCommon'))
        operating_cash_flow = _sf(info.get('operatingCashflow'))
        free_cash_flow      = _sf(info.get('freeCashflow'))
        shares_outstanding  = _sf(info.get('sharesOutstanding'))
        employees           = int(info.get('fullTimeEmployees') or 0)

        # ── Harga Wajar ──────────────────────────────────────
        fv_map = _calc_fair_value(eps, book_value, earnings_growth,
                                   pe_ratio, dividend_rate, current_price)

        graham_number  = round(fv_map.get('graham', 0), 0)
        pe_fair_value  = round(fv_map.get('pe_conservative', 0), 0)
        peg_fair_value = round(fv_map.get('peg', 0), 0)
        dcf_fair_value = round(fv_map.get('dcf', 0), 0)

        # Rata-rata dari metode yang valid
        valid_fvs = [v for v in fv_map.values() if v > 0]
        fair_value_avg = round(sum(valid_fvs) / len(valid_fvs), 0) if valid_fvs else 0.0

        # Margin of safety & label valuasi
        if fair_value_avg > 0 and current_price > 0:
            mos = (fair_value_avg - current_price) / fair_value_avg * 100
            margin_of_safety = round(mos, 1)
            ratio = current_price / fair_value_avg
            if ratio <= 0.60:
                valuation_label = "SANGAT MURAH"
            elif ratio <= 0.85:
                valuation_label = "MURAH"
            elif ratio <= 1.10:
                valuation_label = "WAJAR"
            elif ratio <= 1.30:
                valuation_label = "MAHAL"
            else:
                valuation_label = "SANGAT MAHAL"
            ideal_buy_price = round(fair_value_avg * 0.85, 0)   # 15% margin of safety
            max_buy_price   = round(fair_value_avg * 0.95, 0)   # 5% margin of safety
        else:
            margin_of_safety = 0.0
            valuation_label  = "—"
            ideal_buy_price  = 0.0
            max_buy_price    = 0.0

        # ── Skor Investasi ───────────────────────────────────
        score    = 0
        signals  = []
        warnings = []

        if 0 < pe_ratio <= 10:
            score += 3; signals.append(f"P/E sangat murah ({pe_ratio:.1f}×) — berpotensi undervalued")
        elif 0 < pe_ratio <= 20:
            score += 2; signals.append(f"P/E wajar ({pe_ratio:.1f}×)")
        elif pe_ratio > 30:
            score -= 1; warnings.append(f"P/E tinggi ({pe_ratio:.1f}×) — harga sudah mahal")
        elif pe_ratio <= 0:
            score -= 2; warnings.append("P/E negatif — perusahaan sedang merugi")

        if 0 < pb_ratio < 1:
            score += 3; signals.append(f"P/B di bawah nilai buku ({pb_ratio:.2f}×) — harga murah")
        elif 0 < pb_ratio <= 3:
            score += 1; signals.append(f"P/B wajar ({pb_ratio:.2f}×)")
        elif pb_ratio > 5:
            warnings.append(f"P/B terlalu tinggi ({pb_ratio:.2f}×)")

        if roe >= 20:
            score += 3; signals.append(f"ROE sangat baik ({roe:.1f}%) — efisiensi modal tinggi")
        elif roe >= 10:
            score += 2; signals.append(f"ROE baik ({roe:.1f}%)")
        elif 0 < roe < 10:
            score += 1
        elif roe < 0:
            score -= 2; warnings.append(f"ROE negatif ({roe:.1f}%) — modal tergerus kerugian")

        if profit_margin >= 20:
            score += 2; signals.append(f"Margin laba bersih tinggi ({profit_margin:.1f}%)")
        elif profit_margin >= 5:
            score += 1; signals.append(f"Margin laba sehat ({profit_margin:.1f}%)")
        elif profit_margin < 0:
            score -= 2; warnings.append(f"Perusahaan rugi bersih (margin {profit_margin:.1f}%)")

        if revenue_growth >= 15:
            score += 2; signals.append(f"Pertumbuhan revenue kuat ({revenue_growth:.1f}%/thn)")
        elif revenue_growth >= 5:
            score += 1; signals.append(f"Revenue tumbuh positif ({revenue_growth:.1f}%/thn)")
        elif revenue_growth < -5:
            score -= 1; warnings.append(f"Revenue menurun ({revenue_growth:.1f}%/thn)")

        if earnings_growth >= 20:
            score += 2; signals.append(f"Pertumbuhan laba kuat ({earnings_growth:.1f}%/thn)")
        elif earnings_growth < -10:
            score -= 1; warnings.append(f"Laba menurun ({earnings_growth:.1f}%/thn)")

        if 0 <= debt_to_equity <= 50:
            score += 2; signals.append(f"Utang sangat rendah (DER {debt_to_equity:.1f}%) — neraca kuat")
        elif debt_to_equity <= 100:
            score += 1; signals.append(f"Rasio utang terkendali (DER {debt_to_equity:.1f}%)")
        elif debt_to_equity > 200:
            score -= 2; warnings.append(f"Utang sangat tinggi (DER {debt_to_equity:.1f}%)")
        elif debt_to_equity > 100:
            score -= 1; warnings.append(f"Rasio utang perlu diperhatikan (DER {debt_to_equity:.1f}%)")

        if current_ratio >= 2:
            score += 1; signals.append(f"Likuiditas sangat baik (CR {current_ratio:.2f}×)")
        elif 0 < current_ratio < 1:
            score -= 1; warnings.append(f"Likuiditas kurang (CR {current_ratio:.2f}×)")

        if dividend_yield >= 4:
            score += 2; signals.append(f"Dividen menarik ({dividend_yield:.2f}%) — cocok investor pasif")
        elif dividend_yield >= 2:
            score += 1; signals.append(f"Dividen stabil ({dividend_yield:.2f}%)")

        # Bonus dari valuasi
        if valuation_label == "SANGAT MURAH":
            score += 3; signals.append("Harga jauh di bawah estimasi nilai wajar (>40% diskon)")
        elif valuation_label == "MURAH":
            score += 2; signals.append(f"Harga di bawah nilai wajar ({margin_of_safety:.0f}% diskon)")
        elif valuation_label == "SANGAT MAHAL":
            score -= 2; warnings.append("Harga jauh di atas estimasi nilai wajar — risiko koreksi tinggi")
        elif valuation_label == "MAHAL":
            score -= 1; warnings.append("Harga di atas estimasi nilai wajar")

        if score >= 14:
            rating = "STRONG BUY"
        elif score >= 9:
            rating = "BUY"
        elif score >= 5:
            rating = "HOLD"
        elif score >= 0:
            rating = "WAIT"
        else:
            rating = "AVOID"

        # ── Outlook Masa Depan ───────────────────────────────
        outlook_score = 0
        outlook_reasons = []
        outlook_risks = []

        if revenue_growth >= 15:
            outlook_score += 3
            outlook_reasons.append(f"Pertumbuhan pendapatan sangat kuat ({revenue_growth:.1f}%/thn) — momentum bisnis positif")
        elif revenue_growth >= 5:
            outlook_score += 2
            outlook_reasons.append(f"Pendapatan tumbuh konsisten ({revenue_growth:.1f}%/thn)")
        elif revenue_growth < -5:
            outlook_score -= 2
            outlook_risks.append(f"Pendapatan menyusut ({revenue_growth:.1f}%/thn) — potensi penurunan bisnis")

        if earnings_growth >= 20:
            outlook_score += 3
            outlook_reasons.append(f"Laba tumbuh agresif ({earnings_growth:.1f}%/thn) — daya saing meningkat")
        elif earnings_growth >= 5:
            outlook_score += 1
            outlook_reasons.append(f"Pertumbuhan laba positif ({earnings_growth:.1f}%/thn)")
        elif earnings_growth < -10:
            outlook_score -= 2
            outlook_risks.append(f"Laba menurun tajam ({earnings_growth:.1f}%/thn) — waspada erosi profitabilitas")

        if roe >= 20:
            outlook_score += 2
            outlook_reasons.append(f"ROE tinggi ({roe:.1f}%) — manajemen modal sangat efisien, berpotensi terus tumbuh")
        elif roe >= 10:
            outlook_score += 1
            outlook_reasons.append(f"ROE sehat ({roe:.1f}%)")
        elif roe < 0:
            outlook_score -= 2
            outlook_risks.append(f"ROE negatif ({roe:.1f}%) — perusahaan menggerogoti modal sendiri")

        if 0 <= debt_to_equity <= 50:
            outlook_score += 2
            outlook_reasons.append(f"Neraca sangat bersih (DER {debt_to_equity:.0f}%) — fleksibilitas ekspansi tinggi")
        elif debt_to_equity > 200:
            outlook_score -= 3
            outlook_risks.append(f"Beban utang sangat berat (DER {debt_to_equity:.0f}%) — risiko gagal bayar di masa depan")
        elif debt_to_equity > 100:
            outlook_score -= 1
            outlook_risks.append(f"Utang perlu diperhatikan (DER {debt_to_equity:.0f}%)")

        if free_cash_flow > 0:
            outlook_score += 2
            outlook_reasons.append("Free cash flow positif — perusahaan menghasilkan kas nyata, bisa ekspansi atau bayar dividen")
        elif free_cash_flow < 0:
            outlook_score -= 1
            outlook_risks.append("Free cash flow negatif — masih membakar kas, bergantung pendanaan eksternal")

        if dividend_yield >= 4:
            outlook_score += 1
            outlook_reasons.append(f"Dividen rutin dan menarik ({dividend_yield:.1f}%) — pemegang saham mendapat imbal hasil konsisten")

        if valuation_label in ('SANGAT MURAH', 'MURAH'):
            outlook_score += 2
            outlook_reasons.append("Harga masih di bawah nilai wajar — potensi apresiasi jangka panjang besar")
        elif valuation_label == 'SANGAT MAHAL':
            outlook_score -= 2
            outlook_risks.append("Harga sudah jauh di atas nilai wajar — ekspektasi pertumbuhan sudah terlalu tinggi (priced-in)")

        if outlook_score >= 8:
            future_outlook = "SANGAT MENJANJIKAN"
        elif outlook_score >= 4:
            future_outlook = "MENJANJIKAN"
        elif outlook_score >= 0:
            future_outlook = "NETRAL"
        elif outlook_score >= -4:
            future_outlook = "BERISIKO"
        else:
            future_outlook = "SANGAT BERISIKO"

        return FundamentalData(
            company_name=company_name, sector=sector, industry=industry,
            market_cap=market_cap, current_price=current_price,
            pe_ratio=pe_ratio, pb_ratio=pb_ratio, eps=eps, book_value=book_value,
            roe=roe, roa=roa,
            profit_margin=profit_margin, gross_margin=gross_margin,
            operating_margin=operating_margin,
            revenue_growth=revenue_growth, earnings_growth=earnings_growth,
            debt_to_equity=debt_to_equity, current_ratio=current_ratio,
            dividend_yield=dividend_yield, dividend_rate=dividend_rate,
            week52_high=week52_high, week52_low=week52_low,
            total_assets=total_assets, total_liabilities=total_liabilities,
            total_equity=total_equity, cash_and_equivalents=cash_and_equivalents,
            long_term_debt=long_term_debt, total_revenue=total_revenue,
            net_income=net_income, operating_cash_flow=operating_cash_flow,
            free_cash_flow=free_cash_flow, shares_outstanding=shares_outstanding,
            employees=employees,
            graham_number=graham_number, pe_fair_value=pe_fair_value,
            peg_fair_value=peg_fair_value, dcf_fair_value=dcf_fair_value,
            fair_value_avg=fair_value_avg, margin_of_safety=margin_of_safety,
            ideal_buy_price=ideal_buy_price, max_buy_price=max_buy_price,
            valuation_label=valuation_label,
            future_outlook=future_outlook, outlook_score=outlook_score,
            outlook_reasons=outlook_reasons, outlook_risks=outlook_risks,
            investment_score=score, investment_rating=rating,
            signals=signals, warnings=warnings, available=True,
        )
    except Exception:
        return FundamentalData()
