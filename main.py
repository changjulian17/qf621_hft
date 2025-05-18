from src.data_loader import load_and_filter_data
from src.strategy import OBIVWAPStrategy
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
    # Create strategy instance with 20% risk per trade
    strategy = OBIVWAPStrategy(
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
        initial_cash=100_000,
        risk_per_trade=0.20,  # Using 20% risk per trade as requested
        min_profit_threshold=0.001
    )
    
    try:
        # Load market data
        data = pl.read_csv("data/3_stock_trading_hrs.csv")
        
        # Run backtest
        results = strategy.backtest(data)
        
        # Results are already printed in the backtest method
        
    except Exception as e:
        print(f"Error running strategy: {str(e)}")

if __name__ == "__main__":
    main()

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
    # Define parameter grids with more aggressive ranges
    VWAP_WINDOWS = [100, 200, 500]  # VWAP calculation window - testing faster response
    OBI_WINDOWS = [10, 20, 30]  # Order book imbalance window
    MOMENTUM_WINDOWS = [50, 100, 200]  # Momentum calculation window
    OBI_THRESHOLDS = [0.05, 0.1, 0.15]  # Order book imbalance threshold - wider range
    STOP_LOSS_PCTS = [0.5, 1.0, 1.5]  # Stop loss percentages - wider for higher risk
    PROFIT_TARGET_PCTS = [1.0, 2.0, 3.0]  # Profit target percentages - higher targets
    RISK_PER_TRADES = [0.15, 0.20, 0.25]  # Risk per trade - testing around 20%

    best_result = None
    best_params = None
    best_sharpe = float('-inf')
    best_trades = 0

    # Test combinations of parameters
    for vwap_window, obi_window, momentum_window, obi_threshold, stop_loss, profit_target, risk_per_trade in itertools.product(
        VWAP_WINDOWS, OBI_WINDOWS, MOMENTUM_WINDOWS, OBI_THRESHOLDS,
        STOP_LOSS_PCTS, PROFIT_TARGET_PCTS, RISK_PER_TRADES
    ):
        print(f"\nTesting parameters:")
        print(f"VWAP_WINDOW={vwap_window}, OBI_WINDOW={obi_window}")
        print(f"MOMENTUM_WINDOW={momentum_window}, OBI_THRESHOLD={obi_threshold}")
        print(f"STOP_LOSS={stop_loss}%, PROFIT_TARGET={profit_target}%")
        print(f"RISK_PER_TRADE={risk_per_trade*100}%")
        
        all_metrics = []
        all_sharpe_ratios = []
        total_trades = 0
        
        for ticker in stock_tickers:
            ticker_data = df.filter(pl.col("SYM_ROOT") == ticker)
            strategy = OBIVWAPStrategy(
                vwap_window=vwap_window,
                obi_window=obi_window,
                momentum_window=momentum_window,
                obi_threshold=obi_threshold,
                stop_loss_pct=stop_loss,
                profit_target_pct=profit_target,
                risk_per_trade=risk_per_trade,
                start_time=START_TIME,
                end_time=END_TIME
            )
            ticker_data = strategy.generate_signals(ticker_data)
            backtest_data = strategy.backtest(ticker_data)
            
            # Extract returns and Sharpe ratio from backtest data
            returns = backtest_data.select(pl.col("Returns"))
            avg_return = returns.mean().item()
            std_return = returns.std().item()
            
            if std_return > 0:
                sharpe = (avg_return * 252) / (std_return * np.sqrt(252))
                all_sharpe_ratios.append(sharpe)
            
            all_metrics.append(backtest_data["Account_Balance"][-1] / backtest_data["Account_Balance"][0] - 1)
            total_trades += len(strategy.trades)

        avg_metric = sum(all_metrics) / len(all_metrics)
        avg_sharpe = sum(all_sharpe_ratios) / len(all_sharpe_ratios) if all_sharpe_ratios else float('-inf')
        avg_trades = total_trades / len(stock_tickers)
        
        print(f"Average return: {avg_metric:.2%}")
        print(f"Average Sharpe: {avg_sharpe:.4f}")
        print(f"Average trades per ticker: {avg_trades:.1f}")
        
        # Update best parameters based on Sharpe ratio and minimum trade count
        if avg_sharpe > best_sharpe and avg_trades >= 10:
            best_sharpe = avg_sharpe
            best_result = avg_metric
            best_trades = avg_trades
            best_params = (vwap_window, obi_window, momentum_window, obi_threshold,
                         stop_loss, profit_target, risk_per_trade)

    print("\nBest parameters found:")
    print(f"VWAP_WINDOW={best_params[0]}")
    print(f"OBI_WINDOW={best_params[1]}")
    print(f"MOMENTUM_WINDOW={best_params[2]}")
    print(f"OBI_THRESHOLD={best_params[3]}")
    print(f"STOP_LOSS={best_params[4]}%")
    print(f"PROFIT_TARGET={best_params[5]}%")
    print(f"RISK_PER_TRADE={best_params[6]*100}%")
    print(f"Average return: {best_result:.2%}")
    print(f"Average Sharpe: {best_sharpe:.4f}")
    print(f"Average trades per ticker: {best_trades:.1f}")

    # Show all plots at the end
    plt.show()