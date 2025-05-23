from src.data_loader import load_and_filter_data
from src.strategy import OBIVWAPStrategy
from src.plot import plot_account_balance
from src.performance import evaluate_strategy_performance
import matplotlib.pyplot as plt
import polars as pl
import itertools

# Configuration Parameters
VWAP_WINDOW = 500  # Rolling window size for VWAP calculation
OBI_THRESHOLD = 0.1  # Threshold for Order Book Imbalance (OBI) signals
SIZE_THRESHOLD = 2  # Minimum size threshold for bid and ask sizes
VWAP_THRESHOLD = 0.1  # VWAP threshold for signal generation

EX_FILTER = "Q"  # Exchange filter
QU_COND_FILTER = "R"  # Quote condition filter
START_TIME = (9, 55)  # Start time for generating signals (HH, MM)
END_TIME = (15, 36)  # End time for generating signals (HH, MM)
DATA_FILE = "./data/stock_sample6.csv"

"""
Main script for running the high-frequency trading analysis.

This script orchestrates the loading of data, application of trading strategies,
performance evaluation, and visualization of results.

Modules:
    - data_loader: Handles data loading and preprocessing.
    - strategy: Implements trading strategies.
    - plot: Provides visualization utilities.
    - performance: Evaluates strategy performance.

Usage:
    Run the script directly to process stock data, apply strategies, and visualize results.
"""

if __name__ == "__main__":
    print("Loading data...")

    # Load and filter data for all tickers
    df = load_and_filter_data(
        DATA_FILE,
        ex_filter=EX_FILTER,
        qu_cond_filter=QU_COND_FILTER
    )

    # Extract unique stock tickers from the SYM_ROOT column
    stock_tickers = df["SYM_ROOT"].unique().to_list()
    print(f"Found stock tickers: {stock_tickers}")

    for ticker in stock_tickers:
        print(f"Processing {ticker}...")

        # Filter data for the specific ticker
        ticker_data = df.filter(pl.col("SYM_ROOT") == ticker)

        # Apply strategy
        strategy = OBIVWAPStrategy(
            vwap_window=VWAP_WINDOW, 
            obi_threshold=OBI_THRESHOLD, 
            size_threshold=SIZE_THRESHOLD,
            vwap_threshold=VWAP_THRESHOLD,
            start_time=START_TIME, 
            end_time=END_TIME
        )
        # ticker_data = strategy.generate_signals(ticker_data)
        # backtest_data = strategy.backtest(ticker_data)

        # Plot account balance
        # plot_account_balance(backtest_data)

        # Evaluate strategy performance
        # performance_metrics = evaluate_strategy_performance(backtest_data)

    # Parameter optimization
    # Define parameter grids
    VWAP_WINDOWS = [500]
    OBI_THRESHOLDS = [0]
    SIZE_THRESHOLDS = [0]
    VWAP_THRESHOLDS = [0]  # Add your desired search values here

    best_result = None
    best_params = None

    for vwap_window, obi_threshold, size_threshold, vwap_threshold in itertools.product(
        VWAP_WINDOWS, OBI_THRESHOLDS, SIZE_THRESHOLDS, VWAP_THRESHOLDS
    ):
        print(f"Testing VWAP_WINDOW={vwap_window}, OBI_THRESHOLD={obi_threshold}, SIZE_THRESHOLD={size_threshold}, VWAP_THRESHOLD={vwap_threshold}")
        all_metrics = []
        for ticker in stock_tickers:
            ticker_data = df.filter(pl.col("SYM_ROOT") == ticker)
            strategy = OBIVWAPStrategy(
                vwap_window=vwap_window, 
                obi_threshold=obi_threshold, 
                size_threshold=size_threshold,
                vwap_threshold=vwap_threshold,
                start_time=START_TIME, 
                end_time=END_TIME
            )
            ticker_data = strategy.generate_signals(ticker_data)
            backtest_data = strategy.backtest(ticker_data)
            metrics = evaluate_strategy_performance(backtest_data)
            all_metrics.append(metrics["Total_Returns"])  # or use another metric

        avg_metric = sum(all_metrics) / len(all_metrics)
        if (best_result is None) or (avg_metric > best_result):
            best_result = avg_metric
            best_params = (vwap_window, obi_threshold, size_threshold, vwap_threshold)

    print(f"Best params: VWAP_WINDOW={best_params[0]}, OBI_THRESHOLD={best_params[1]}, SIZE_THRESHOLD={best_params[2]}, VWAP_THRESHOLD={best_params[3]} with avg return {best_result:.2f}%")

    # Show all plots at the end
    plt.show()