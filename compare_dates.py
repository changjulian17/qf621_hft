from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy
from src.performance import evaluate_strategy_performance
import polars as pl
import gc
import csv
import os
from datetime import datetime, timedelta
import random
import pandas_market_calendars as mcal

EXCHANGES = ["'Z'"] 
# EXCHANGES = ["'Z'", "'Q'", "'K'", "'N'"]
QU_COND_FILTER = "'R'"
VWAP_WINDOW = 500
OBI_THRESHOLD = 0
SIZE_THRESHOLD = 0
VWAP_THRESHOLD = 0

def get_random_trading_days(year, n_days, seed=42):
    nyse = mcal.get_calendar('XNYS')
    schedule = nyse.schedule(start_date=f'{year}-01-01', end_date=f'{year}-06-30')
    trading_days = schedule.index.strftime('%Y-%m-%d').tolist()
    # random.seed(seed)
    return random.sample(trading_days, n_days)

def main():
    # Load tickers
    with open("data/positive_return_tickers_v1.txt") as f:
        all_filtered = [line.strip() for line in f if line.strip()]

    batch_size = 8  # Number of tickers per batch
    year = 2023
    n_days = 12  # Number of random trading days to pick

    trading_days = get_random_trading_days(year, n_days)
    print(f"Random trading days in H1 2023: {trading_days}")

    for day in trading_days:
        print(f"\nProcessing trading day: {day}")

        batch = all_filtered[:batch_size]  # You can change batching logic if needed

        batch_results = []

        for ex in EXCHANGES:
            print(f"\nProcessing exchange: {ex}")
            df = fetch_taq_data(
                tickers=batch,
                exchanges=ex,
                quote_conds=QU_COND_FILTER,
                start_date=day,
                end_date=day,
                wrds_username='changjulian17'
            )
            stock_tickers = df["sym_root"].unique().to_list()
            for ticker in stock_tickers:
                print(f"Processing {ticker} on exchange {ex}...")
                ticker_data = df.filter(pl.col("sym_root") == ticker)
                strategy = OBIVWAPStrategy(
                    vwap_window=VWAP_WINDOW, 
                    obi_threshold=OBI_THRESHOLD, 
                    size_threshold=SIZE_THRESHOLD,
                    vwap_threshold=VWAP_THRESHOLD,
                    start_time=(9, 55), 
                    end_time=(15, 36)
                )
                ticker_data = strategy.generate_signals(ticker_data)
                backtest_data = strategy.backtest(ticker_data)
                metrics = evaluate_strategy_performance(backtest_data)
                avg_sharpe = metrics.get("Average_Sharpe")
                total_trades = backtest_data["Cumulative_Trades"][-1] if "Cumulative_Trades" in backtest_data.columns else None
                batch_results.append({
                    "date": day,
                    "ticker": ticker,
                    "exchange": ex.replace("'", ""),
                    "Total_Returns": metrics.get("Total_Returns"),
                    "Max_Drawdown": metrics.get("Max_Drawdown"),
                    "Average_Sharpe": avg_sharpe,
                    "Average_Bid_Ask_Spread": metrics.get("Average_Bid_Ask_Spread"),
                    "Cumulative_Trades": total_trades
                })
                del ticker_data, backtest_data
                gc.collect()
            del df
            gc.collect()

        # Write/appends results for each day
        if batch_results:
            fieldnames = [
                "date", "ticker", "exchange", "Total_Returns", "Max_Drawdown",
                "Average_Sharpe", "Average_Bid_Ask_Spread", "Cumulative_Trades"
            ]
            file_path = "data/dates_comparison_metrics.csv"
            write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                for row in batch_results:
                    writer.writerow(row)
            print(f"Results for {day} written to {file_path}")

    print("\nAll selected trading days processed and results saved to data/day_comparison_metrics.csv")

if __name__ == "__main__":
    main()