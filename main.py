from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy, StrategyPortfolio, MeanReversionStrategy, OBIVWAPoldStrategy
from src.plot import plot_account_balance
from src.performance import evaluate_strategy_performance
from src.logger_config import setup_logger
import polars as pl
import random
import os
import gc
import itertools
import argparse
from typing import Tuple

def parse_time(time_str: str) -> Tuple[int, int]:
    """Parse time string in HH:MM format to tuple of (hour, minute)."""
    hour, minute = map(int, time_str.split(':'))
    return (hour, minute)

def parse_arguments():
    parser = argparse.ArgumentParser(description='High-Frequency Trading Strategy Analysis')
    
    # Data filtering parameters
    parser.add_argument('--exchanges', default="'Q', 'T', 'N'", help='Exchange filter')
    parser.add_argument('--quote-cond', default="'R'", help='Quote condition filter')
    parser.add_argument('--start-date', default='2023-08-14', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2023-08-14', help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-time', default='9:55', help='Trading start time (HH:MM)')
    parser.add_argument('--end-time', default='15:36', help='Trading end time (HH:MM)')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to process')
    parser.add_argument('--wrds-username', default='shobhit999', help='WRDS username')


    
    # Portfolio parameters
    parser.add_argument('--initial-cash', type=float, default=1_000_000, help='Initial cash for portfolio')
    parser.add_argument('--rebalance-threshold', type=float, default=0.1, help='Portfolio rebalance threshold')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set up main logger
    logger = setup_logger()
    logger.info("Starting HFT Strategy Analysis")


    # Parse trading hours
    start_time = parse_time(args.start_time)
    end_time = parse_time(args.end_time)

    logger.info(f"\nProcessing tickers: {args.tickers}")
    df = fetch_taq_data(
        tickers=args.tickers,
        exchanges=args.exchanges,
        quote_conds=args.quote_cond,
        start_date=args.start_date,
        end_date=args.end_date,
        wrds_username=args.wrds_username
    )

    stock_tickers = df["sym_root"].unique().to_list()
    logger.info(f"Tickers found in data: {stock_tickers}")

    portfolio = StrategyPortfolio(initial_cash=args.initial_cash, rebalance_threshold=args.rebalance_threshold)
    obi_strategy = OBIVWAPStrategy(
        vwap_window=500,
        obi_window=20,
        price_impact_window=50,
        momentum_window=100,
        volatility_window=50,
        trend_window=20,
        max_position=100,
        stop_loss_pct=0.5,
        profit_target_pct=1.0,
        risk_per_trade=0.20,
        min_profit_threshold=0.001,
        start_time=start_time,
        end_time=end_time,
        logger=logger
    )
    portfolio.add_strategy("OBI-VWAP", obi_strategy, weight=0.5)
    
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
        start_time=start_time,
        end_time=end_time,
        logger=logger
    )
    portfolio.add_strategy("Mean-Reversion", mean_rev_strategy, weight=0.5)

    positive_return_tickers = []
    all_results = {}

    for ticker in stock_tickers:
        logger.info(f"Processing {ticker}...")
        ticker_data = df.filter(pl.col("sym_root") == ticker)
        logger.info(f"Data for {ticker} contains {ticker_data.shape[0]} records")

        # Run backtest for the portfolio
        logger.info(f"Running backtest for {ticker}...")


        results = portfolio.run_backtest(ticker_data)
        all_results[ticker] = results
        
        # Log portfolio metrics
        logger.info(f"\nPortfolio Performance for {ticker}:")
        logger.info(f"Total Return: {results['Metrics']['Total_Return']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['Metrics']['Sharpe_Ratio']:.4f}")
        logger.info(f"Sortino Ratio: {results['Metrics']['Sortino_Ratio']:.4f}")
        logger.info(f"Maximum Drawdown: {results['Metrics']['Max_Drawdown']:.2f}%")

        if results['Metrics']['Total_Return'] > 0:
            positive_return_tickers.append(ticker)

    # Append to positive return tickers file
    with open("data/positive_return_tickers.txt", "a") as f:
        for t in positive_return_tickers:
            f.append(f"{t}\n")

    del df
    gc.collect()


