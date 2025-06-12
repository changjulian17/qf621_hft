from src.data_loader import load_and_filter_data
from src.strategy import OBIVWAPStrategy, MeanReversionStrategy, StrategyPortfolio
from src.plot import plot_account_balance
from src.performance import evaluate_strategy_performance
import matplotlib.pyplot as plt
import polars as pl
import itertools
import numpy as np
import pandas as pd

# Configuration Parameters
VWAP_WINDOW = 500  # Rolling window size for VWAP calculation
OBI_THRESHOLD = 0.1  # Threshold for Order Book Imbalance (OBI) signals
SIZE_THRESHOLD = 2  # Minimum size threshold for bid and ask sizes
VWAP_THRESHOLD = 0.1  # VWAP threshold for signal generation

EX_FILTER = "Q"  # Exchange filter
QU_COND_FILTER = "R"  # Quote condition filter
START_TIME = (9, 55)  # Start time for generating signals (HH, MM)
END_TIME = (15, 36)  # End time for generating signals (HH, MM)
DATA_FILE = "./data/3_stock_trading_hrs.csv"

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

def main():
    # Load market data
    print("Loading data...")
    data = load_and_filter_data(
        DATA_FILE,
        ex_filter=EX_FILTER,
        qu_cond_filter=QU_COND_FILTER
    )
    
    # Create strategy portfolio
    portfolio = StrategyPortfolio(initial_cash=1_000_000, rebalance_threshold=0.1)
    
    # Create and add OBI-VWAP strategy
    # obi_strategy = OBIVWAPStrategy(
    #     vwap_window=500,
    #     obi_window=20,
    #     price_impact_window=50,
    #     momentum_window=100,
    #     obi_threshold=0.1,
    #     size_threshold=3,
    #     vwap_threshold=0.1,
    #     volatility_window=50,
    #     trend_window=20,
    #     max_position=100,
    #     stop_loss_pct=0.5,
    #     profit_target_pct=1.0,
    #     risk_per_trade=0.20,
    #     min_profit_threshold=0.001,
    #     start_time=START_TIME,
    #     end_time=END_TIME
    # )
    # portfolio.add_strategy("OBI-VWAP", obi_strategy, weight=0.6)
    
    # Create and add Mean Reversion strategy
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
    
    # Extract unique stock tickers
    stock_tickers = data["SYM_ROOT"].unique().to_list()
    print(f"Found stock tickers: {stock_tickers}")
    
    # Run backtest for each ticker
    all_results = {}
    for ticker in stock_tickers:
        print(f"\nProcessing {ticker}...")
        
        # Filter data for the specific ticker
        ticker_data = data.filter(pl.col("SYM_ROOT") == ticker)
        
        # Run portfolio backtest
        results = portfolio.run_backtest(ticker_data)
        all_results[ticker] = results
        
        # Print portfolio metrics
        print(f"\nPortfolio Performance for {ticker}:")
        print(f"Total Return: {results['Metrics']['Total_Return']:.2f}%")
        print(f"Sharpe Ratio: {results['Metrics']['Sharpe_Ratio']:.4f}")
        print(f"Sortino Ratio: {results['Metrics']['Sortino_Ratio']:.4f}")
        print(f"Maximum Drawdown: {results['Metrics']['Max_Drawdown']:.2f}%")
        
        # Plot portfolio value
        # plt.figure(figsize=(12, 6))
        # plt.plot(results['Portfolio']['Portfolio_Value'], label='Portfolio Value')
        # plt.title(f'Portfolio Performance - {ticker}')
        # plt.xlabel('Time')
        # plt.ylabel('Portfolio Value ($)')
        # plt.legend()
        # plt.grid(True)
        
        # # Plot individual strategy performance
        # plt.figure(figsize=(12, 6))
        # for strategy_name in ['OBI-VWAP', 'Mean-Reversion']:
        #     plt.plot(results[strategy_name]['Account_Balance'], 
        #             label=f'{strategy_name} Strategy')
        # plt.title(f'Strategy Performance Comparison - {ticker}')
        # plt.xlabel('Time')
        # plt.ylabel('Account Balance ($)')
        # plt.legend()
        # plt.grid(True)
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
