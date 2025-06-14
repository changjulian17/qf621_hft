from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy, MeanReversionStrategy, StrategyPortfolio
from src.performance import evaluate_strategy_performance
import polars as pl
import gc
import csv
import os
from datetime import datetime, timedelta
import random
import pandas_market_calendars as mcal

EXCHANGES = ["'Z'","'N'"] 
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

                # --- Run both strategies in a portfolio ---
                obi_strategy = OBIVWAPStrategy(
                    vwap_window=VWAP_WINDOW, 
                    obi_threshold=OBI_THRESHOLD, 
                    size_threshold=SIZE_THRESHOLD,
                    vwap_threshold=VWAP_THRESHOLD,
                    start_time=(9, 55), 
                    end_time=(15, 36)
                )
                meanrev_strategy = MeanReversionStrategy(
                    vwap_window=20,
                    deviation_threshold=0.0001,
                    volatility_window=20,
                    volume_window=20,
                    max_position=100,
                    stop_loss_pct=0.3,
                    profit_target_pct=0.6,
                    risk_per_trade=0.02,
                    min_profit_threshold=0.001,
                    start_time=(9, 55),
                    end_time=(15, 36)
                )
                portfolio = StrategyPortfolio(initial_cash=100_000)
                portfolio.add_strategy("OBIVWAP", obi_strategy, weight=0.5)
                portfolio.add_strategy("MeanReversion", meanrev_strategy, weight=0.5)
                portfolio_results = portfolio.run_backtest(ticker_data)

                obi_df = portfolio_results.get("OBIVWAP")
                meanrev_df = portfolio_results.get("MeanReversion")
                portfolio_df = portfolio_results.get("Portfolio")

                obi_metrics = evaluate_strategy_performance(obi_df)
                meanrev_metrics = evaluate_strategy_performance(meanrev_df)
                portfolio_metrics = evaluate_strategy_performance(portfolio_df)

                batch_results.append({
                    "start_date": day,
                    "end_date": day,
                    "ticker": ticker,
                    "exchange": ex.replace("'", ""),
                    "OBIVWAP_Returns": obi_metrics.get("Total_Returns"),
                    "MeanRev_Returns": meanrev_metrics.get("Total_Returns"),
                    "Portfolio_Returns": portfolio_metrics.get("Total_Returns"),
                    "OBIVWAP_Sharpe": obi_metrics.get("Average_Sharpe"),
                    "MeanRev_Sharpe": meanrev_metrics.get("Average_Sharpe"),
                    "Portfolio_Sharpe": portfolio_metrics.get("Average_Sharpe"),
                    "OBIVWAP_Avg_Spread": obi_metrics.get("Average_Bid_Ask_Spread"),
                    "MeanRev_Avg_Spread": meanrev_metrics.get("Average_Bid_Ask_Spread"),
                    "Portfolio_Avg_Spread": portfolio_metrics.get("Average_Bid_Ask_Spread"),
                    "OBIVWAP_Trades": obi_metrics.get("Cumulative_Trades"),
                    "MeanRev_Trades": meanrev_metrics.get("Cumulative_Trades"),
                    "Portfolio_Trades": portfolio_metrics.get("Cumulative_Trades"),
                })
                del ticker_data
                gc.collect()
            del df
            gc.collect()

        # Write/appends results for each day
        if batch_results:
            fieldnames = [
                "start_date", "end_date", "ticker", "exchange",
                "OBIVWAP_Returns", "MeanRev_Returns", "Portfolio_Returns",
                "OBIVWAP_Sharpe", "MeanRev_Sharpe", "Portfolio_Sharpe",
                "OBIVWAP_Avg_Spread", "MeanRev_Avg_Spread", "Portfolio_Avg_Spread",
                "OBIVWAP_Trades", "MeanRev_Trades", "Portfolio_Trades"
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