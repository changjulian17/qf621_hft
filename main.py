from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy
from src.plot import plot_account_balance
from src.performance import evaluate_strategy_performance
import polars as pl
import random
# random.seed(42)

EX_FILTER = "'Q', 'T', 'N'"
QU_COND_FILTER = "'R'"
START_DATE = '2023-05-10'
END_DATE = '2023-05-11'
START_TIME = (9, 55)
END_TIME = (15, 36)
VWAP_WINDOW = 500
OBI_THRESHOLD = 0
SIZE_THRESHOLD = 0
VWAP_THRESHOLD = 0

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    with open("data/filtered_tickers.txt") as f:
        all_filtered = [line.strip() for line in f if line.strip()]

    positive_return_tickers = []

    for batch_num, batch in enumerate(chunked(all_filtered, 10), 1):
        print(f"\nProcessing batch {batch_num}: {batch}")
        df = fetch_taq_data(
            tickers=batch,
            exchanges=EX_FILTER,
            quote_conds=QU_COND_FILTER,
            start_date=START_DATE,
            end_date=END_DATE,
            wrds_username='changjulian17'
        )

        stock_tickers = df["sym_root"].unique().to_list()
        print(f"Tickers: {stock_tickers}")

        for ticker in stock_tickers:
            print(f"Processing {ticker}...")
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
            plot_account_balance(backtest_data)
            metrics = evaluate_strategy_performance(backtest_data)
            if metrics.get("Total_Returns", 0) > 0:
                positive_return_tickers.append(ticker)

    # Save tickers with positive returns to a text file
    with open("data/positive_return_tickers.txt", "w") as f:
        for t in positive_return_tickers:
            f.write(f"{t}\n")

    print("\nTickers with Total_Returns > 0 saved to data/positive_return_tickers.txt")

    """" Uncomment the following block to perform grid search """
    # VWAP_WINDOWS = [500]
    # OBI_THRESHOLDS = [0]
    # SIZE_THRESHOLDS = [0]
    # VWAP_THRESHOLDS = [0]

    # best_result = None
    # best_params = None

    # for vwap_window in VWAP_WINDOWS:
    #     for obi_threshold in OBI_THRESHOLDS:
    #         for size_threshold in SIZE_THRESHOLDS:
    #             for vwap_threshold in VWAP_THRESHOLDS:
    #                 print(f"Testing VWAP={vwap_window}, OBI={obi_threshold}, SIZE={size_threshold}, VWAP_T={vwap_threshold}")
    #                 all_metrics = []
    #                 for ticker in stock_tickers:
    #                     ticker_data = df.filter(pl.col("sym_root") == ticker)
    #                     strategy = OBIVWAPStrategy(
    #                         vwap_window=vwap_window, 
    #                         obi_threshold=obi_threshold, 
    #                         size_threshold=size_threshold,
    #                         vwap_threshold=vwap_threshold,
    #                         start_time=START_TIME, 
    #                         end_time=END_TIME
    #                     )
    #                     ticker_data = strategy.generate_signals(ticker_data)
    #                     backtest_data = strategy.backtest(ticker_data)
    #                     plot_account_balance(backtest_data)
    #                     metrics = evaluate_strategy_performance(backtest_data)
    #                     all_metrics.append(metrics["Total_Returns"])
    #                 avg_metric = sum(all_metrics) / len(all_metrics)
    #                 if (best_result is None) or (avg_metric > best_result):
    #                     best_result = avg_metric
    #                     best_params = (vwap_window, obi_threshold, size_threshold, vwap_threshold)
    # print(f"Best: VWAP={best_params[0]}, OBI={best_params[1]}, SIZE={best_params[2]}, VWAP_T={best_params[3]}, avg return {best_result:.2f}%")