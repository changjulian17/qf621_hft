from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy, StrategyPortfolio, MeanReversionStrategy
from src.plot import plot_account_balance
from src.performance import evaluate_strategy_performance
from src.logger_config import setup_logger
import polars as pl
import random
import os
import gc
import itertools
# random.seed(42)

EX_FILTER = "'Q', 'T', 'N'"
QU_COND_FILTER = "'R'"
START_DATE = '2023-08-14'
END_DATE = '2023-08-18'
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
    # Set up main logger
    logger = setup_logger()
    logger.info("Starting HFT Strategy Analysis")
    logger.info(f"Configuration: VWAP_WINDOW={VWAP_WINDOW}, OBI_THRESHOLD={OBI_THRESHOLD}, "
                f"SIZE_THRESHOLD={SIZE_THRESHOLD}, VWAP_THRESHOLD={VWAP_THRESHOLD}")

    with open("data/filtered_tickers.txt") as f:
        all_filtered = [line.strip() for line in f if line.strip()]

    positive_return_tickers = []

    for batch_num, batch in enumerate(chunked(all_filtered[:6], 2), 1): # TODO remove splice
        logger.info(f"\nProcessing batch {batch_num}: {batch}")
        df = fetch_taq_data(
            tickers=batch,
            exchanges=EX_FILTER,
            quote_conds=QU_COND_FILTER,
            start_date=START_DATE,
            end_date=END_DATE,
            wrds_username='changjulian17'
        )

        stock_tickers = df["sym_root"].unique().to_list()
        logger.info(f"Tickers: {stock_tickers}")

        portfolio = StrategyPortfolio(initial_cash=1_000_000, rebalance_threshold=0.1)
        obi_strategy = OBIVWAPStrategy(
            vwap_window=500,
            obi_window=20,
            price_impact_window=50,
            momentum_window=100,
            obi_threshold=0.1,
            size_threshold=3,
            vwap_threshold=0.1,
            volatility_window=50,
            trend_window=20,
            max_position=100,
            stop_loss_pct=0.5,
            profit_target_pct=1.0,
            risk_per_trade=0.20,
            min_profit_threshold=0.001,
            start_time=START_TIME,
            end_time=END_TIME
        )
        portfolio.add_strategy("OBI-VWAP", obi_strategy, weight=0.6)
        
        mean_rev_strategy = MeanReversionStrategy(
            vwap_window=100,
            deviation_threshold=0.002,
            volatility_window=20,
            volume_window=50,
            max_position=100,
            stop_loss_pct=0.3,
            profit_target_pct=0.6,
            risk_per_trade=0.20,
            min_profit_threshold=0.001,
            start_time=START_TIME,
            end_time=END_TIME
        )
        portfolio.add_strategy("Mean-Reversion", mean_rev_strategy, weight=0.4)
        all_results = {}

        for ticker in stock_tickers:
            logger.info(f"Processing {ticker}...")
            ticker_data = df.filter(pl.col("sym_root") == ticker)

            results = portfolio.run_backtest(ticker_data)
            all_results[ticker] = results
            
            # Log portfolio metrics
            logger.info(f"\nPortfolio Performance for {ticker}:")
            logger.info(f"Total Return: {results['Metrics']['Total_Return']:.2f}%")
            logger.info(f"Sharpe Ratio: {results['Metrics']['Sharpe_Ratio']:.4f}")
            logger.info(f"Sortino Ratio: {results['Metrics']['Sortino_Ratio']:.4f}")
            logger.info(f"Maximum Drawdown: {results['Metrics']['Max_Drawdown']:.2f}%")
            # ticker_data = strategy.generate_signals(ticker_data)
            # backtest_data = strategy.backtest(ticker_data)
            # # plot_account_balance(backtest_data)
            # metrics = evaluate_strategy_performance(backtest_data)
            if results['Metrics']['Total_Return'] > 0:
                positive_return_tickers.append(ticker)
        del df
        gc.collect()

        with open("data/positive_return_tickers.txt", "w") as f:
            for t in positive_return_tickers:
                f.write(f"{t}\n")

    logger.info("\nTickers with Total_Returns > 0 saved to data/positive_return_tickers.txt")

    # Parameter optimization
    logger.info("Starting parameter optimization...")
    # Define parameter grids
    VWAP_WINDOWS = [500]
    OBI_THRESHOLDS = [0]
    SIZE_THRESHOLDS = [0]
    VWAP_THRESHOLDS = [0]  # Add your desired search values here

    best_result = None
    best_params = None

