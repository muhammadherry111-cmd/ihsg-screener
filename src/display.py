from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box
from datetime import datetime
from src.analyzer import TechnicalSignal

console = Console()

RECOMMENDATION_STYLE = {
    "STRONG BUY": "bold green",
    "BUY":        "green",
    "NETRAL+":    "cyan",
    "NETRAL":     "white",
    "NETRAL-":    "yellow",
    "HINDARI":    "red",
}

BANDAR_POLA_STYLE = {
    "BREAKOUT":    "bold green",
    "MARKING UP":  "bold green",
    "AKUMULASI":   "cyan",
    "DISTRIBUSI":  "red",
    "TIDAK ADA":   "dim",
}


def _rec_color(rec: str) -> str:
    return RECOMMENDATION_STYLE.get(rec, "white")


def print_header():
    now = datetime.now().strftime("%d %B %Y %H:%M WIB")
    title = Text(
        f"  IHSG Technical Screener + Bandarmologi + Forecasting  |  {now}  ",
        style="bold white on blue"
    )
    console.print(Panel(title, expand=True))
    console.print()


def print_summary_table(results: list[TechnicalSignal], top_n: int = 20):
    table = Table(
        title=f"Top {top_n} Emiten — Analisis Komprehensif",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("No",        justify="right",  width=4)
    table.add_column("Kode",      style="bold white", width=7)
    table.add_column("Harga",     justify="right",  width=11)
    table.add_column("Chg%",      justify="right",  width=8)
    table.add_column("Rekomendasi", justify="center", width=13)
    table.add_column("Skor",      justify="center", width=6)
    table.add_column("RSI",       justify="right",  width=6)
    table.add_column("MFI",       justify="right",  width=6)
    table.add_column("Pola Bandar", justify="center", width=13)
    table.add_column("Forecast 3H", justify="right", width=12)
    table.add_column("R/R",       justify="right",  width=6)
    table.add_column("Vol",       justify="right",  width=7)

    for i, sig in enumerate(results[:top_n], 1):
        chg_style = "green" if sig.change_pct >= 0 else "red"
        chg_str = f"+{sig.change_pct:.2f}%" if sig.change_pct >= 0 else f"{sig.change_pct:.2f}%"
        rec_style = _rec_color(sig.recommendation)
        pola = sig.bandar.pola
        pola_style = BANDAR_POLA_STYLE.get(pola, "white")

        fc_pct = ((sig.fc.forecast_3d - sig.last_price) / sig.last_price) * 100
        fc_str = f"Rp {sig.fc.forecast_3d:,.0f} ({fc_pct:+.1f}%)"
        fc_style = "green" if fc_pct > 0 else "red"

        rr_str = f"1:{sig.fc.risk_reward:.1f}" if sig.fc.risk_reward > 0 else "-"
        rr_style = "green" if sig.fc.risk_reward >= 2 else ("yellow" if sig.fc.risk_reward >= 1.5 else "dim")

        table.add_row(
            str(i),
            sig.ticker,
            f"Rp {sig.last_price:,.0f}",
            Text(chg_str, style=chg_style),
            Text(sig.recommendation, style=rec_style),
            Text(str(sig.total_score), style=rec_style),
            f"{sig.rsi:.1f}",
            f"{sig.bandar.mfi:.1f}",
            Text(pola, style=pola_style),
            Text(fc_str, style=fc_style),
            Text(rr_str, style=rr_style),
            f"{sig.volume_ratio:.1f}x",
        )

    console.print(table)
    console.print()


def print_detail(sig: TechnicalSignal):
    rec_style = _rec_color(sig.recommendation)
    chg_str = f"+{sig.change_pct:.2f}%" if sig.change_pct >= 0 else f"{sig.change_pct:.2f}%"
    chg_style = "green" if sig.change_pct >= 0 else "red"
    fc = sig.fc
    bandar = sig.bandar

    # Header
    console.print(Rule(f"[bold]{sig.ticker}[/bold]  [{chg_style}]{chg_str}[/]  [{rec_style}]{sig.recommendation}[/]  (Total Skor: {sig.total_score})", style=rec_style))
    console.print()

    # --- Indikator Teknikal ---
    console.print("[bold cyan]▶ INDIKATOR TEKNIKAL[/bold cyan]")
    ind = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ind.add_column("Label", style="dim", width=22)
    ind.add_column("Nilai")
    ind.add_row("Harga Terakhir",    f"Rp {sig.last_price:,.0f}")
    ind.add_row("RSI (14)",          f"{sig.rsi:.2f}")
    ind.add_row("MACD / Signal",     f"{sig.macd:.4f} / {sig.macd_signal:.4f}  (Hist: {sig.macd_hist:+.4f})")
    ind.add_row("EMA9 / EMA21",      f"Rp {sig.ema9:,.2f}  /  Rp {sig.ema21:,.2f}")
    ind.add_row("SMA50",             f"Rp {sig.sma50:,.2f}" if sig.sma50 else "N/A")
    ind.add_row("Bollinger Bands",   f"Upper: {sig.bb_upper:,.0f}  Mid: {sig.bb_mid:,.0f}  Lower: {sig.bb_lower:,.0f}")
    ind.add_row("Stochastic %K/%D",  f"{sig.stoch_k:.1f}  /  {sig.stoch_d:.1f}")
    ind.add_row("Volume Ratio",      f"{sig.volume_ratio:.2f}x rata-rata 20 hari")
    ind.add_row("Skor Teknikal",     str(sig.score))
    console.print(ind)
    console.print()

    # --- Bandarmologi ---
    console.print("[bold magenta]▶ BANDARMOLOGI — Aktivitas Smart Money[/bold magenta]")
    ban = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ban.add_column("Label", style="dim", width=22)
    ban.add_column("Nilai")
    ban.add_row("MFI (Money Flow Index)", f"{bandar.mfi:.2f}  [{bandar.mfi_status}]")
    ban.add_row("OBV Trend",              bandar.obv_trend)
    ban.add_row("OBV Divergence",         bandar.obv_divergence)
    ban.add_row("A/D Line",               bandar.ad_trend)
    ban.add_row("VPT Trend",              bandar.vpt_trend)
    ban.add_row("VWAP (20 hari)",         f"Rp {bandar.vwap:,.0f}  [{bandar.vwap_position}]")
    ban.add_row("Pola Bandar",            bandar.pola)
    ban.add_row("Skor Bandarmologi",      str(bandar.bandar_score))
    console.print(ban)

    if bandar.signals:
        for s in bandar.signals:
            console.print(f"  [magenta]✓[/magenta] {s}")
    if bandar.warnings:
        for w in bandar.warnings:
            console.print(f"  [red]✗[/red] {w}")
    console.print()

    # --- Forecasting ---
    fc_dir_style = "green" if fc.trend_direction == "NAIK" else ("red" if fc.trend_direction == "TURUN" else "yellow")
    conf_style = "green" if fc.confidence == "TINGGI" else ("yellow" if fc.confidence == "SEDANG" else "dim")
    fc_3d_pct = ((fc.forecast_3d - sig.last_price) / sig.last_price) * 100
    fc_5d_pct = ((fc.forecast_5d - sig.last_price) / sig.last_price) * 100

    console.print("[bold yellow]▶ FORECASTING — Proyeksi Harga[/bold yellow]")
    fct = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    fct.add_column("Label", style="dim", width=22)
    fct.add_column("Nilai")
    fct.add_row("Arah Tren",          Text(fc.trend_direction, style=fc_dir_style))
    fct.add_row("Kepercayaan",        Text(fc.confidence, style=conf_style))
    fct.add_row("Proyeksi 3 Hari",    f"Rp {fc.forecast_3d:,.0f}  ({fc_3d_pct:+.1f}%)")
    fct.add_row("Proyeksi 5 Hari",    f"Rp {fc.forecast_5d:,.0f}  ({fc_5d_pct:+.1f}%)")
    fct.add_row("ATR (Volatilitas)",  f"Rp {fc.atr:,.0f}  ({fc.atr_pct:.2f}%)")
    fct.add_row("Resistance 1",       f"Rp {fc.resistance_1:,.0f}")
    fct.add_row("Resistance 2",       f"Rp {fc.resistance_2:,.0f}")
    fct.add_row("Support 1",          f"Rp {fc.support_1:,.0f}")
    fct.add_row("Support 2",          f"Rp {fc.support_2:,.0f}")
    fct.add_row("Target Price",       f"Rp {fc.target_price:,.0f}")
    fct.add_row("Stop Loss",          f"Rp {fc.stop_loss:,.0f}")
    rr_style = "green" if fc.risk_reward >= 2 else ("yellow" if fc.risk_reward >= 1.5 else "red")
    fct.add_row("Risk/Reward",        Text(f"1 : {fc.risk_reward:.2f}", style=rr_style))
    fct.add_row("Fibonacci 38.2%",   f"Rp {fc.fib_382:,.0f}")
    fct.add_row("Fibonacci 50.0%",   f"Rp {fc.fib_500:,.0f}")
    fct.add_row("Fibonacci 61.8%",   f"Rp {fc.fib_618:,.0f}")
    console.print(fct)

    if fc.signals:
        for s in fc.signals:
            console.print(f"  [yellow]→[/yellow] {s}")
    console.print()

    # --- Strategi Trading ---
    strat = sig.strat

    # Candlestick Patterns
    if strat.candle_patterns:
        console.print("[bold white]▶ POLA CANDLESTICK TERDETEKSI[/bold white]")
        for p in strat.candle_patterns:
            p_style = "green" if p.signal == "BULLISH" else ("red" if p.signal == "BEARISH" else "yellow")
            icon = "✓" if p.signal == "BULLISH" else ("✗" if p.signal == "BEARISH" else "~")
            console.print(f"  [{p_style}]{icon}[/] [bold]{p.name}[/bold] ({p.signal}, kekuatan: {'★'*p.strength}) — {p.description}")
        console.print()

    # Statistik Historis
    console.print("[bold white]▶ STATISTIK HISTORIS (30 Hari)[/bold white]")
    hs = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    hs.add_column("Label", style="dim", width=28)
    hs.add_column("Nilai")
    hs.add_row("% Hari Bullish",         f"{strat.hist_bullish_days_pct:.0f}%")
    hs.add_row("Rata-rata Return Harian", f"{strat.hist_avg_daily_return:+.2f}%")
    hs.add_row("Return Terbaik",         f"+{strat.hist_best_return:.2f}%")
    hs.add_row("Return Terburuk",        f"{strat.hist_worst_return:.2f}%")
    hs.add_row("Volatilitas (Std Dev)",  f"{strat.hist_volatility:.2f}%")
    hs.add_row("Rata-rata Range Intraday", f"{strat.intraday_avg_range_pct:.2f}%")
    hs.add_row("Frekuensi Gap Up",       f"{strat.overnight_gap_freq:.0f}% hari  (rata-rata +{strat.overnight_avg_gap:.2f}%)")
    console.print(hs)
    console.print()

    # Overnight Strategy
    on = strat.overnight_price
    on_color = "green" if strat.overnight_feasible else ("yellow" if strat.overnight_score >= 3 else "red")
    on_label = "✅ LAYAK" if strat.overnight_feasible else ("⚠ CUKUP" if strat.overnight_score >= 3 else "❌ TIDAK DISARANKAN")
    console.print(f"[bold cyan]▶ STRATEGI 1: BELI SORE — JUAL PAGI  [{on_color}]{on_label}[/]  (Skor: {strat.overnight_score})[/bold cyan]")

    on_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    on_tbl.add_column("Label", style="dim", width=28)
    on_tbl.add_column("Nilai")
    on_tbl.add_row("⏰ Waktu Entry",    "Sore menjelang penutupan (15:30–16:00 WIB)")
    on_tbl.add_row("⏰ Waktu Exit",     "Pagi saat opening (09:00–09:30 WIB)")
    on_tbl.add_row("💰 Harga Beli",     Text(f"Rp {on.entry:,.0f}", style="bold cyan"))
    on_tbl.add_row("🎯 Target Jual 1",  Text(f"Rp {on.target_1:,.0f}  (+{on.potential_profit_pct:.1f}%)", style="bold green"))
    on_tbl.add_row("🎯 Target Jual 2",  Text(f"Rp {on.target_2:,.0f}  (optimis)", style="green"))
    on_tbl.add_row("🛑 Stop Loss",      Text(f"Rp {on.stop_loss:,.0f}  (-{on.potential_loss_pct:.1f}%)", style="bold red"))
    rr_on = "green" if on.risk_reward >= 2 else ("yellow" if on.risk_reward >= 1.5 else "red")
    on_tbl.add_row("⚖ Risk / Reward",  Text(f"1 : {on.risk_reward:.2f}", style=rr_on))
    on_tbl.add_row("📊 Win Rate Historis", f"{strat.overnight_win_rate:.0f}%")
    console.print(on_tbl)

    if strat.overnight_signals:
        for s in strat.overnight_signals:
            console.print(f"  [cyan]✓[/cyan] {s}")
    if strat.overnight_warnings:
        for w in strat.overnight_warnings:
            console.print(f"  [red]✗[/red] {w}")
    console.print()

    # Intraday Strategy
    id_ = strat.intraday_price
    id_color = "green" if strat.intraday_feasible else ("yellow" if strat.intraday_score >= 3 else "red")
    id_label = "✅ LAYAK" if strat.intraday_feasible else ("⚠ CUKUP" if strat.intraday_score >= 3 else "❌ TIDAK DISARANKAN")
    console.print(f"[bold green]▶ STRATEGI 2: BELI PAGI — JUAL SORE  [{id_color}]{id_label}[/]  (Skor: {strat.intraday_score})[/bold green]")

    id_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    id_tbl.add_column("Label", style="dim", width=28)
    id_tbl.add_column("Nilai")
    id_tbl.add_row("⏰ Waktu Entry",    "Pagi saat opening (09:00–09:30 WIB)")
    id_tbl.add_row("⏰ Waktu Exit",     "Sore menjelang penutupan (15:00–16:00 WIB)")
    id_tbl.add_row("💰 Harga Beli (est.)", Text(f"Rp {id_.entry:,.0f}", style="bold cyan"))
    id_tbl.add_row("🎯 Target Jual 1",  Text(f"Rp {id_.target_1:,.0f}  (+{id_.potential_profit_pct:.1f}%)", style="bold green"))
    id_tbl.add_row("🎯 Target Jual 2",  Text(f"Rp {id_.target_2:,.0f}  (optimis)", style="green"))
    id_tbl.add_row("🛑 Stop Loss",      Text(f"Rp {id_.stop_loss:,.0f}  (-{id_.potential_loss_pct:.1f}%)", style="bold red"))
    rr_id = "green" if id_.risk_reward >= 2 else ("yellow" if id_.risk_reward >= 1.5 else "red")
    id_tbl.add_row("⚖ Risk / Reward",  Text(f"1 : {id_.risk_reward:.2f}", style=rr_id))
    id_tbl.add_row("📊 Win Rate Historis", f"{strat.intraday_win_rate:.0f}%")
    id_tbl.add_row("📈 Avg Range Intraday", f"{strat.intraday_avg_range_pct:.2f}% per hari")
    console.print(id_tbl)

    if strat.intraday_signals:
        for s in strat.intraday_signals:
            console.print(f"  [green]✓[/green] {s}")
    if strat.intraday_warnings:
        for w in strat.intraday_warnings:
            console.print(f"  [red]✗[/red] {w}")
    console.print()

    # --- Ringkasan Sinyal ---
    console.print("[bold green]▶ SINYAL BULLISH (Teknikal)[/bold green]")
    if sig.signals:
        for s in sig.signals:
            console.print(f"  [green]✓[/green] {s}")
    else:
        console.print("  [dim](tidak ada)[/dim]")

    console.print()
    console.print("[bold red]▶ PERINGATAN[/bold red]")
    if sig.warnings:
        for w in sig.warnings:
            console.print(f"  [red]✗[/red] {w}")
    else:
        console.print("  [dim](tidak ada)[/dim]")

    console.print()
    console.rule(style="dim")
    console.print()


def print_top_picks(results: list[TechnicalSignal], n: int = 5):
    buy = [r for r in results if r.recommendation in ("STRONG BUY", "BUY")][:n]
    if not buy:
        console.print("[yellow]Tidak ada emiten dengan sinyal BUY saat ini.[/yellow]\n")
        return

    console.print(f"[bold green]★  TOP {len(buy)} PILIHAN TERBAIK UNTUK DAILY TRADE[/bold green]\n")
    for i, sig in enumerate(buy, 1):
        rec_style = _rec_color(sig.recommendation)
        chg_str = f"+{sig.change_pct:.2f}%" if sig.change_pct >= 0 else f"{sig.change_pct:.2f}%"
        chg_style = "green" if sig.change_pct >= 0 else "red"
        fc_pct = ((sig.fc.forecast_3d - sig.last_price) / sig.last_price) * 100
        pola = sig.bandar.pola
        strat = sig.strat

        on_label = "[green]✅ ON[/]" if strat.overnight_feasible else "[dim]— ON[/]"
        id_label = "[green]✅ ID[/]" if strat.intraday_feasible else "[dim]— ID[/]"

        console.print(
            f"  {i}. [bold]{sig.ticker}[/bold]  "
            f"Rp {sig.last_price:,.0f}  [{chg_style}]{chg_str}[/]  "
            f"[{rec_style}]{sig.recommendation}[/]  Skor:{sig.total_score}  "
            f"RSI:{sig.rsi:.1f}  MFI:{sig.bandar.mfi:.1f}  "
            f"Forecast 3H:{fc_pct:+.1f}%  WinON:{strat.overnight_win_rate:.0f}%  WinID:{strat.intraday_win_rate:.0f}%  "
            f"{on_label} {id_label}"
        )
        # Harga beli/jual overnight
        if strat.overnight_feasible:
            on = strat.overnight_price
            console.print(
                f"     [cyan]Overnight[/cyan]: Beli Rp {on.entry:,.0f} → "
                f"Target Rp {on.target_1:,.0f} (+{on.potential_profit_pct:.1f}%) | "
                f"Stop Rp {on.stop_loss:,.0f} | R/R 1:{on.risk_reward:.1f}"
            )
        # Harga beli/jual intraday
        if strat.intraday_feasible:
            id_ = strat.intraday_price
            console.print(
                f"     [green]Intraday[/green]: Beli Rp {id_.entry:,.0f} → "
                f"Target Rp {id_.target_1:,.0f} (+{id_.potential_profit_pct:.1f}%) | "
                f"Stop Rp {id_.stop_loss:,.0f} | R/R 1:{id_.risk_reward:.1f}"
            )
        if pola != "TIDAK ADA":
            pola_style = BANDAR_POLA_STYLE.get(pola, "white")
            console.print(f"     [dim]Bandar: [{pola_style}]{pola}[/][/dim]")
        if sig.strat.candle_patterns:
            names = ", ".join(p.name for p in sig.strat.candle_patterns[:2] if p.signal == "BULLISH")
            if names:
                console.print(f"     [dim]Candle: {names}[/dim]")
        console.print()
    console.print()


def print_stats(total: int, analyzed: int, results: list[TechnicalSignal]):
    strong_buy = sum(1 for r in results if r.recommendation == "STRONG BUY")
    buy        = sum(1 for r in results if r.recommendation == "BUY")
    netral     = sum(1 for r in results if "NETRAL" in r.recommendation)
    hindari    = sum(1 for r in results if r.recommendation == "HINDARI")

    breakout   = sum(1 for r in results if r.bandar.pola == "BREAKOUT")
    akumulasi  = sum(1 for r in results if r.bandar.pola == "AKUMULASI")
    markup     = sum(1 for r in results if r.bandar.pola == "MARKING UP")
    distribusi = sum(1 for r in results if r.bandar.pola == "DISTRIBUSI")

    overnight_ok  = sum(1 for r in results if r.strat.overnight_feasible)
    intraday_ok   = sum(1 for r in results if r.strat.intraday_feasible)

    stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats.add_column("Label", style="dim")
    stats.add_column("Nilai", style="bold")

    stats.add_row("Total emiten diproses",     str(total))
    stats.add_row("Berhasil dianalisis",        str(analyzed))
    stats.add_row("[bold green]Strong Buy[/]", str(strong_buy))
    stats.add_row("[green]Buy[/]",             str(buy))
    stats.add_row("[white]Netral[/]",          str(netral))
    stats.add_row("[red]Hindari[/]",           str(hindari))
    stats.add_row("─── Strategi ───",          "")
    stats.add_row("[cyan]Layak Overnight (Beli Sore)[/]",  str(overnight_ok))
    stats.add_row("[green]Layak Intraday (Beli Pagi)[/]",  str(intraday_ok))
    stats.add_row("─── Pola Bandar ───",       "")
    stats.add_row("[bold green]Breakout[/]",   str(breakout))
    stats.add_row("[bold green]Marking Up[/]", str(markup))
    stats.add_row("[cyan]Akumulasi[/]",        str(akumulasi))
    stats.add_row("[red]Distribusi[/]",        str(distribusi))

    console.print(Panel(stats, title="[bold]Statistik Screening[/bold]", border_style="blue"))
    console.print()
