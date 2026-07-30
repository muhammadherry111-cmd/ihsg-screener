#!/usr/bin/env python3
"""
IHSG Technical Screener
Screening emiten IHSG berdasarkan analisis teknikal untuk daily trade.
"""
import argparse
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.fetcher import get_idx_stock_list, fetch_stock_data
from src.analyzer import analyze, TechnicalSignal
from src.display import (
    print_header, print_summary_table, print_detail,
    print_top_picks, print_stats
)

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="IHSG Technical Screener - Analisis teknikal saham IDX untuk daily trade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py                          # Screening 100 emiten teratas
  python main.py --max 200               # Screening 200 emiten
  python main.py --top 10                # Tampilkan top 10 rekomendasi
  python main.py --detail BBCA TLKM      # Detail analisis emiten tertentu
  python main.py --watch BBCA TLKM ASII  # Monitor emiten pilihan
        """
    )
    parser.add_argument("--max", type=int, default=100, help="Jumlah maksimal emiten yang di-screening (default: 100)")
    parser.add_argument("--top", type=int, default=20, help="Jumlah top emiten yang ditampilkan (default: 20)")
    parser.add_argument("--detail", nargs="+", metavar="KODE", help="Tampilkan detail analisis untuk emiten tertentu")
    parser.add_argument("--watch", nargs="+", metavar="KODE", help="Analisis hanya emiten yang disebutkan")
    parser.add_argument("--min-score", type=int, default=None, help="Filter minimum skor (default: semua)")
    parser.add_argument("--buy-only", action="store_true", help="Tampilkan hanya emiten dengan sinyal BUY")
    return parser.parse_args()


def run_screening(tickers: list[str], top_n: int, min_score: int | None, buy_only: bool):
    results: list[TechnicalSignal] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Menganalisis emiten...", total=len(tickers))

        for ticker in tickers:
            progress.update(task, description=f"Menganalisis [bold]{ticker}[/bold]...", advance=1)
            df = fetch_stock_data(ticker, period_days=90)
            if df is None:
                continue
            signal = analyze(ticker, df)
            if signal is not None:
                results.append(signal)

    # Urutkan berdasarkan total skor gabungan (teknikal + bandar + forecast)
    results.sort(key=lambda x: x.total_score, reverse=True)

    # Filter jika ada
    if min_score is not None:
        results = [r for r in results if r.total_score >= min_score]
    if buy_only:
        results = [r for r in results if r.recommendation in ("STRONG BUY", "BUY")]

    return results


def run_detail(tickers: list[str]):
    for ticker in tickers:
        ticker = ticker.upper()
        console.print(f"[cyan]Mengambil data {ticker}...[/cyan]")
        df = fetch_stock_data(ticker, period_days=90)
        if df is None:
            console.print(f"[red]Data tidak tersedia untuk {ticker}[/red]")
            continue
        signal = analyze(ticker, df)
        if signal is None:
            console.print(f"[red]Analisis gagal untuk {ticker} (data tidak cukup)[/red]")
            continue
        from src.display import print_detail
        print_detail(signal)


def main():
    args = parse_args()
    print_header()

    # Mode detail: tampilkan analisis mendalam untuk emiten tertentu
    if args.detail:
        console.print("[bold]Mode: Detail Analisis[/bold]\n")
        run_detail(args.detail)
        return

    # Mode watch: analisis hanya emiten pilihan user
    if args.watch:
        tickers = [t.upper() for t in args.watch]
        console.print(f"[bold]Mode: Watch List ({len(tickers)} emiten)[/bold]\n")
    else:
        tickers = get_idx_stock_list()[:args.max]
        console.print(f"[bold]Mode: Screening {len(tickers)} emiten IHSG[/bold]\n")

    console.print("[dim]Catatan: Data diambil dari Yahoo Finance (.JK). Harga mungkin sedikit delay.[/dim]\n")

    results = run_screening(tickers, args.top, args.min_score, args.buy_only)

    if not results:
        console.print("[red]Tidak ada data yang berhasil dianalisis.[/red]")
        sys.exit(1)

    # Tampilkan hasil
    print_stats(len(tickers), len(results), results)
    print_top_picks(results, n=min(5, args.top))
    print_summary_table(results, top_n=args.top)

    # Tanya apakah ingin melihat detail
    if not args.detail and not args.buy_only:
        console.print("[dim]Gunakan [bold]--detail KODE[/bold] untuk analisis mendalam suatu emiten.[/dim]")
        console.print("[dim]Contoh: python main.py --detail BBCA TLKM ASII[/dim]\n")


if __name__ == "__main__":
    main()
