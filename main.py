from src.data_loader import load_and_filter_data
from src.strategy import OBIVWAPStrategy
from src.plot import plot_account_balance
from src.performance import calculate_performance_metrics  # Import the performance metrics function
from src.performance import calculate_daily_sharpe_ratio
import matplotlib.pyplot as plt
import os

# Configuration Parameters
VWAP_WINDOW = 20  # Rolling window size for VWAP calculation
OBI_THRESHOLD = 0.05  # Threshold for Order Book Imbalance (OBI) signals
EX_FILTER = "Q"  # Exchange filter
QU_COND_FILTER = "R"  # Quote condition filter

# List of stock tickers to analyze
STOCK_TICKERS = ["AAPL", "TSLA", "NVDA"]  # Example tickers
print(os.getcwd())
DATA_FILE = "./data/3_stock_trading_hrs.csv"

if __name__ == "__main__":
    for ticker in STOCK_TICKERS:
        print(f"Processing {ticker}...")
        
        # Load and filter data for the specific ticker
        df = load_and_filter_data(
            DATA_FILE,
            ex_filter=EX_FILTER,
            qu_cond_filter=QU_COND_FILTER,
            sym_root=ticker
        )
        
        # Apply strategy
        strategy = OBIVWAPStrategy(vwap_window=VWAP_WINDOW, obi_threshold=OBI_THRESHOLD)
        df = strategy.generate_signals(df)
        backtest_data = strategy.backtest(df)
        
        # Plot account balance
        plot_account_balance(backtest_data)
        
        # Calculate and print performance metrics
        calculate_performance_metrics(backtest_data["Account_Balance"].to_numpy())
        
        # Pass the Polars DataFrame to calculate_daily_sharpe_ratio
        calculate_daily_sharpe_ratio(backtest_data)
    
    # Show all plots at the end
    plt.show()