# qf621_hft

QF621_hft_analysis

Initial results are not good:

- try geting aapl or single stock and try for a long period and find out if there is any alpha with a range of OBI and VWAP
- try another strategy instead of relying on OBI which is very sensitive
  - use together with volume and VWAP?
- try across a larger range of stocks including small cap names
- analyse tracking multiple levels of bid-offer than the best one given by EX (rolling window of size?)
- working list: FTAI, ZION, XOMA

ways to recover:

- go back to using mid price instead of crossing bid offer, mix trade data to show intermediate prices are trading

## Executive Summary

This project focuses on analyzing high-frequency trading (HFT) strategies using real-world stock market data. It provides tools for data preprocessing, strategy implementation, performance evaluation, and visualization. The goal is to assess the effectiveness of various trading strategies and gain insights into their behavior under different market conditions. The project is modular, allowing users to customize and extend its functionality for their specific needs.

## Features

1. **Data Preprocessing**:

   - Load and filter stock market data from CSV files using the `Polars` library.
   - Apply filters based on exchange codes, quote conditions, and stock tickers.
   - Restrict data to trading hours, ensuring only relevant data is processed.
2. **Strategy Implementation**:

   - Implements the Order Book Imbalance (OBI) and Volume Weighted Average Price (VWAP) strategy.
   - Generates buy/sell signals based on OBI thresholds and VWAP calculations.
   - Signals are dynamically adjusted to ensure no trades occur outside specified trading hours, with parameterizable start and end times.
3. **Performance Evaluation**:

   - Backtest strategies on historical data, simulating trades and calculating account balances.
   - Compute performance metrics such as total returns, maximum drawdown, and daily Sharpe ratios.
4. **Visualization**:

   - Visualize key metrics, such as account balance over time, to gain insights into strategy performance.
5. **Parameterization**:

   - Fully configurable parameters, including trading hours, VWAP window size, and OBI thresholds, to suit various trading scenarios.

## Configurable Parameters

The following parameters can be configured in `main.py` to customize the analysis:

- **`VWAP_WINDOW`**: Rolling window size for VWAP calculation (default: `20`).
- **`OBI_THRESHOLD`**: Threshold for Order Book Imbalance (OBI) signals (default: `0.05`).
- **`EX_FILTER`**: Exchange filter to include only rows with the specified exchange code (default: `"Q"`).
- **`QU_COND_FILTER`**: Quote condition filter to include only rows with the specified quote condition (default: `"R"`).
- **`START_TIME`**: Start time for generating signals in `(HH, MM, MS)` format (default: `(9, 30, 865)`).
- **`END_TIME`**: End time for generating signals in `(HH, MM, MS)` format (default: `(16, 28, 954)`).
- **`DATA_FILE`**: Path to the CSV file containing stock market data (default: `"./data/3_stock_5_trading_days.csv"`).

These parameters allow users to adapt the analysis to different datasets, trading hours, and strategy configurations.

## Usage

1. **Setup**:

   - Ensure Python 3.11 or higher is installed.
   - Install required dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
2. **Run the Analysis**:

   - Execute the main script:
     ```bash
     python main.py
     ```
3. **Customize Parameters**:

   - Modify the configurable parameters in `main.py` to suit your requirements.

## Data

The `data/` directory contains sample CSV files for testing and analysis:
