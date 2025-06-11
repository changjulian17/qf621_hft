from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy
from src.performance import evaluate_strategy_performance
import polars as pl
import random
import gc
import csv

EXCHANGES = ["'Z'", "'Q'", "'K'", "'N'"]
# EXCHANGES = ["'Z'"]
QU_COND_FILTER = "'R'"
START_DATE = '2023-05-10'
END_DATE = '2023-05-10'
START_TIME = (9, 55)
END_TIME = (15, 36)
VWAP_WINDOW = 500
OBI_THRESHOLD = 0
SIZE_THRESHOLD = 0
VWAP_THRESHOLD = 0

def extract_avg_sharpe(metrics):
    # If metrics contains a 'Sharpe' key, use it directly
    if "Sharpe" in metrics:
        return metrics["Sharpe"]
    # Otherwise, try to extract from 'Daily_Sharpe_Ratios'
    dsr = metrics.get("Daily_Sharpe_Ratios")
    if dsr is None:
        return None
    # If it's a polars DataFrame or similar, extract the value
    try:
        # If dsr is a polars DataFrame
        if hasattr(dsr, "to_pandas"):
            df = dsr.to_pandas()
            return df["Daily_Sharpe_Ratio"].mean()
        # If dsr is a string representation, try to parse the number
        elif isinstance(dsr, str):
            import re
            vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", dsr)]
            if vals:
                return sum(vals) / len(vals)
        # If dsr is a list or array
        elif hasattr(dsr, "__iter__"):
            vals = list(dsr)
            if vals and hasattr(vals[0], "get"):
                vals = [v.get("Daily_Sharpe_Ratio", 0) for v in vals]
            return sum(vals) / len(vals)
    except Exception:
        return None
    return None

def main():
    # Load tickers
    with open("data/positive_return_tickers_v1.txt") as f:
        all_filtered = [line.strip() for line in f if line.strip()]

    batch_size = 8
    num_batches = 8
    results = []

    for batch_num, batch_start in enumerate(range(0, batch_size * num_batches, batch_size), 1):
        batch = all_filtered[batch_start:batch_start + batch_size]
        if not batch:
            break
        print(f"\nProcessing batch {batch_num}: {batch}")

        batch_results = []

        for ex in EXCHANGES:
            print(f"\nProcessing exchange: {ex}")
            df = fetch_taq_data(
                tickers=batch,
                exchanges=ex,
                quote_conds=QU_COND_FILTER,
                start_date=START_DATE,
                end_date=END_DATE,
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
                    start_time=START_TIME, 
                    end_time=END_TIME
                )
                ticker_data = strategy.generate_signals(ticker_data)
                backtest_data = strategy.backtest(ticker_data)
                metrics = evaluate_strategy_performance(backtest_data)
                avg_sharpe = extract_avg_sharpe(metrics)
                total_trades = backtest_data["Cumulative_Trades"][-1] if "Cumulative_Trades" in backtest_data.columns else None
                batch_results.append({
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

        # Overwrite results after each batch run
        if batch_results:
            fieldnames = [
                "ticker", "exchange", "Total_Returns", "Max_Drawdown",
                "Average_Sharpe", "Average_Bid_Ask_Spread", "Cumulative_Trades"
            ]
            with open("data/exchange_comparison_metrics.csv", "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in batch_results:
                    writer.writerow(row)
            print(f"Batch {batch_num} results written to data/exchange_comparison_metrics.csv")

    print("\nAll batches processed and last batch results saved to data/exchange_comparison_metrics.csv")

if __name__ == "__main__":
    main()