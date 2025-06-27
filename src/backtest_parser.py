import re
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def parse_backtest_results_folder(folder_path: str) -> List[Dict]:
    """
    Recursively parses the backtest_results folder and extracts info from .parquet filenames.
    Returns a list of dictionaries with extracted info.
    """
    folder = Path(folder_path)
    results = []
    # Pattern: Strategy_Ticker_YYYY-MM-DD.parquet
    file_pattern = re.compile(r"^(?P<strategy>.+)_(?P<ticker>[A-Z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.parquet$")

    for file in folder.rglob("*.parquet"):
        name = file.name
        m = file_pattern.match(name)
        if m:
            entry = {
                'file': str(file),
                'strategy': m.group('strategy'),
                'ticker': m.group('ticker'),
                'date': m.group('date')
            }
            results.append(entry)
    return results

def parse_performance_txt(filepath: str) -> Dict[str, float]:
    """
    Parses a performance.txt file with key: value pairs.
    Returns a dictionary of metrics.
    """
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
    return metrics

def parse_results_txt(filepath: str) -> List[Dict[str, str]]:
    """
    Parses a results.txt file with CSV-style data.
    Extracts Ticker and Strategy from the file name.
    Adds a 'Month' key extracted from the 'Date' field.
    Returns a list of dictionaries (one per row).
    """
    import csv
    from pathlib import Path

    # Extract Ticker and Strategy from file name
    file_name = Path(filepath).stem
    strategy, ticker = file_name.split('_')[:2]

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        results = [row for row in reader]
        for row in results:
            row['Ticker'] = ticker
            row['Strategy'] = strategy

            # Extract month from 'Date' field
            if 'Date' in row:
                try:
                    row['Month'] = row['Date'].split('-')[1]  # Assumes 'Date' is in YYYY-MM-DD format
                except IndexError:
                    row['Month'] = None

        df = pd.DataFrame(results)
        df['Intraday Final Balance'] = pd.to_numeric(df['Intraday Final Balance'], errors='coerce')
        df['daily_return'] = df['Intraday Final Balance'].pct_change()
        df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], 0)
        df.loc[0, 'daily_return'] = (df.loc[0, 'Intraday Final Balance'] / 300_000 - 1) * 0.1  # Fix chained assignment
        df['daily_return'] = df['daily_return'] * 0.1
        df['Intraday Final Balance'] = df['daily_return'].add(1).cumprod() * 300_000
        results = df.to_dict(orient='records')
        return results



def plot_equity_curve(results: list, output_html: str):
    """
    Plots the equity curve (Date vs. Intraday Final Balance) and saves as HTML.
    """
    if not results:
        print("No results to plot.")
        return
    # Sort by date
    results_sorted = sorted(results, key=lambda x: x['Date'])
    dates = [row['Date'] for row in results_sorted]
    balances = [float(row['Intraday Final Balance']) for row in results_sorted]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=balances, mode='lines+markers', name='Equity Curve'))
    fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Intraday Final Balance')
    fig.write_html(output_html)
    print(f"Equity curve saved to {output_html}")

def plot_grouped_equity_curve(results: list, output_html: str, month: str):
    """
    Plots grouped equity curves by ticker and strategy for a single month.
    Creates multiple subplots, each dedicated to one strategy and all stocks.
    """
    if not results:
        print("No results to plot.")
        return

    # Filter results for the specified month
    results_filtered = [row for row in results if row.get('Month') == month]
    if not results_filtered:
        print(f"No results found for month {month}.")
        return

    # Group results by strategy
    grouped_data = {}
    for row in results_filtered:
        strategy = row.get('Strategy', row.get('strategy', 'Unknown'))
        ticker = row.get('Ticker', row.get('ticker', 'Unknown'))
        if strategy not in grouped_data:
            grouped_data[strategy] = {}
        if ticker not in grouped_data[strategy]:
            grouped_data[strategy][ticker] = []
        grouped_data[strategy][ticker].append(row)

    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(grouped_data), cols=1, subplot_titles=list(grouped_data.keys()))

    for idx, (strategy, ticker_data) in enumerate(grouped_data.items(), start=1):
        for ticker, data in ticker_data.items():
            data_sorted = sorted(data, key=lambda x: x['Date'])
            dates = [row['Date'] for row in data_sorted]
            balances = [float(row['Intraday Final Balance']) for row in data_sorted]
            fig.add_trace(
                go.Scatter(x=dates, y=balances, mode='lines+markers', name=ticker),
                row=idx, col=1
            )

    # Update layout
    fig.update_layout(
        title=f'Grouped Equity Curve for Month: {month}',
        xaxis_title='Date',
        yaxis_title='Intraday Final Balance',
        height=1000 * len(grouped_data),
        showlegend=True
    )

    # Save plot
    fig.write_html(output_html)
    print(f"Grouped equity curve saved to {output_html}")

def plot_strategy_equity_curves(results: list, output_folder: str):
    """
    Plots separate equity curves for each strategy and saves them as individual HTML files.
    """
    if not results:
        print("No results to plot.")
        return

    # Group results by strategy
    grouped_data = {}
    for row in results:
        strategy = row.get('Strategy', row.get('strategy', 'Unknown'))
        if strategy not in grouped_data:
            grouped_data[strategy] = []
        grouped_data[strategy].append(row)

    # Create separate plots for each strategy
    for strategy, data in grouped_data.items():
        fig = go.Figure()
        grouped_by_ticker = {}
        for row in data:
            ticker = row.get('Ticker', row.get('ticker', 'Unknown'))
            if ticker not in grouped_by_ticker:
                grouped_by_ticker[ticker] = []
            grouped_by_ticker[ticker].append(row)

        for ticker, ticker_data in grouped_by_ticker.items():
            ticker_data_sorted = sorted(ticker_data, key=lambda x: x['Date'])
            dates = [row['Date'] for row in ticker_data_sorted]
            balances = [float(row['Intraday Final Balance']) for row in ticker_data_sorted]
            fig.add_trace(go.Scatter(x=dates, y=balances, mode='lines+markers', name=ticker))

        fig.update_layout(title=f'Equity Curve for Strategy: {strategy}', xaxis_title='Date', yaxis_title='Intraday Final Balance')
        output_html = Path(output_folder) / f"{strategy}_equity_curve.html"
        fig.write_html(str(output_html))
        print(f"Equity curve for strategy {strategy} saved to {output_html}")

def plot_average_monthly_returns(performance_files: list, output_html: str):
    """
    Plots a bar chart of average monthly returns for each stock, sorted from last to first.
    This isnt useful
    """

    # Collect data from performance files
    stock_returns = []
    for pf in performance_files:
        file_name = Path(pf).stem
        strategy, ticker = file_name.split('_')[:2]
        if ticker == "LEN":
            continue  # Skip stock LEN
        performance_data = parse_performance_txt(pf)
        month = file_name.split('_')[-1]  # Extract month from filename
        res_file = Path(pf).parent / f"{strategy}_{ticker}_results_{month}.txt"
        results_data = parse_results_txt(res_file)
        df = pd.DataFrame(results_data)
        total_return = (df['Intraday Final Balance'].iloc[-1] / 300_000 - 1 ) * 100 if 'Intraday Final Balance' in df.columns else np.nan
        if 'total_return' in performance_data:
            stock_returns.append({'Ticker': ticker, 'Strategy': strategy, 'Total Return': total_return})
        # convert to results.txt parsing

    # Create DataFrame and convert Total Return to numeric
    df = pd.DataFrame(stock_returns)
    df['Total Return'] = pd.to_numeric(df['Total Return'], errors='coerce')
    df.dropna(subset=['Total Return'], inplace=True)

    # Calculate average monthly returns
    df['Monthly Return'] = df['Total Return'] 
    df_sorted = df.sort_values(by='Monthly Return', ascending=False)

    # Create bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_sorted['Ticker'], y=df_sorted['Monthly Return'], text=df_sorted['Strategy'], name='Monthly Return'))
    fig.update_layout(title='Average Monthly Returns by Stock', xaxis_title='Stock', yaxis_title='Average Monthly Return', xaxis=dict(categoryorder='total descending'))
    fig.write_html(output_html)
    print(f"Average monthly returns bar chart saved to {output_html}")

def plot_average_monthly_returns_by_strategy(performance_files: list, output_folder: str):
    """
    Plots a single bar chart for average monthly returns for each strategy.
    """
    import pandas as pd

    # Collect data from performance files
    strategy_returns = {}
    for pf in performance_files:
        file_name = Path(pf).stem
        strategy, ticker = file_name.split('_')[:2]
        if ticker == "LEN":
            continue  # Skip stock LEN
        performance_data = parse_performance_txt(pf)
        # if 'total_return' in performance_data:
        #     if strategy not in strategy_returns:
        #         strategy_returns[strategy] = []
        #     strategy_returns[strategy].append(performance_data['total_return'])
        month = file_name.split('_')[-1]  # Extract month from filename
        res_file = Path(pf).parent / f"{strategy}_{ticker}_results_{month}.txt"
        results_data = parse_results_txt(res_file)
        df = pd.DataFrame(results_data)
        # print(df['Intraday Final Balance'].iloc[-1], df['Intraday Final Balance'].iloc[0])
        total_return = (df['Intraday Final Balance'].iloc[-1] / df['Intraday Final Balance'].iloc[0] - 1 ) * 100 if 'Intraday Final Balance' in df.columns else np.nan
        if strategy not in strategy_returns:
            strategy_returns[strategy] = []
        strategy_returns[strategy].append(total_return)  # Assume annualized return, convert to monthly
        
    # Calculate average monthly returns for each strategy
    strategy_avg_monthly_returns = {
        strategy: sum(returns) / len(returns)
        for strategy, returns in strategy_returns.items()
    }

    # Create DataFrame for plotting
    df = pd.DataFrame(
        list(strategy_avg_monthly_returns.items()),
        columns=['Strategy', 'Average Monthly Return']
    )
    df_sorted = df.sort_values(by='Average Monthly Return', ascending=False)

    # Create bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_sorted['Strategy'], y=df_sorted['Average Monthly Return'], name='Monthly Return'))
    fig.update_layout(title='Average Monthly Returns by Strategy', xaxis_title='Strategy', yaxis_title='Average Monthly Return %', xaxis=dict(categoryorder='total descending'))

    # Save chart for each strategy
    output_html = Path(output_folder) / "average_monthly_returns_by_strategy.html"
    fig.write_html(str(output_html))
    print(f"Average monthly returns bar chart saved to {output_html}")

# def plot_average_monthly_returns_by_stock_and_strategy(performance_files: list, output_folder: str):
#     """
#     Plots a bar chart of average monthly returns for each stock, grouped and averaged by strategy.
#     Each bar represents the MEAN monthly return for a stock under a strategy.
#     """
#     strategy_stock_returns = {}

#     # Collect data from performance files
#     for pf in performance_files:
#         file_name = Path(pf).stem
#         strategy, ticker = file_name.split('_')[:2]
#         if ticker == "LEN":
#             continue  # Skip stock LEN
#         performance_data = parse_performance_txt(pf)
#         if 'total_return' in performance_data:
#             if strategy not in strategy_stock_returns:
#                 strategy_stock_returns[strategy] = []
#             strategy_stock_returns[strategy].append({
#                 'Ticker': ticker,
#                 'Monthly Return': performance_data['total_return'] / 12  # Assume annualized return
#             })

#     # Generate plot for each strategy
#     for strategy, records in strategy_stock_returns.items():
#         df = pd.DataFrame(records)
#         df['Monthly Return'] = pd.to_numeric(df['Monthly Return'], errors='coerce')
#         df = df.dropna(subset=['Monthly Return'])

#         # Aggregate by mean monthly return per ticker
#         mean_returns = df.groupby('Ticker', as_index=False)['Monthly Return'].mean()
#         mean_returns = mean_returns.sort_values(by='Monthly Return', ascending=False)

#         # Plot
#         fig = go.Figure(data=[
#             go.Bar(
#                 x=mean_returns['Ticker'],
#                 y=mean_returns['Monthly Return'],
#                 name='Mean Monthly Return'
#             )
#         ])
#         fig.update_layout(
#             title=f'Average Monthly Returns by Stock for Strategy: {strategy}',
#             xaxis_title='Stock',
#             yaxis_title='Average Monthly Return',
#             xaxis=dict(categoryorder='total descending'),
#         )

#         output_html = Path(output_folder) / f"{strategy}_average_monthly_returns_by_stock.html"
#         fig.write_html(str(output_html))
#         print(f"Saved plot to {output_html}")

def plot_average_sharpe_ratios_by_stock_and_strategy(performance_files: list, output_folder: str):
    """
    Plots a bar chart of average Sharpe Ratios for each stock, grouped and averaged by strategy.
    Each bar represents the MEAN Sharpe Ratio for a stock under a strategy.
    """
    strategy_stock_sharpe_ratios = {}

    # Collect data from performance files
    for pf in performance_files:
        file_name = Path(pf).stem
        strategy, ticker = file_name.split('_')[:2]
        if ticker == "LEN":
            continue  # Skip stock LEN
        performance_data = parse_performance_txt(pf)
        if 'sharpe_ratio' in performance_data:
            if strategy not in strategy_stock_sharpe_ratios:
                strategy_stock_sharpe_ratios[strategy] = []
            strategy_stock_sharpe_ratios[strategy].append({
                'Ticker': ticker,
                'Sharpe Ratio': performance_data['sharpe_ratio']
            })

    # Generate plots for each strategy
    for strategy, records in strategy_stock_sharpe_ratios.items():
        df = pd.DataFrame(records)
        df['Sharpe Ratio'] = pd.to_numeric(df['Sharpe Ratio'], errors='coerce')
        df = df.dropna(subset=['Sharpe Ratio'])

        # Aggregate by mean Sharpe Ratio per ticker
        mean_sharpes = df.groupby('Ticker', as_index=False)['Sharpe Ratio'].mean()
        mean_sharpes = mean_sharpes.sort_values(by='Sharpe Ratio', ascending=False)

        # Plot
        fig = go.Figure(data=[
            go.Bar(
                x=mean_sharpes['Ticker'],
                y=mean_sharpes['Sharpe Ratio'],
                name='Mean Sharpe Ratio'
            )
        ])
        fig.update_layout(
            title=f'Average Sharpe Ratios by Stock for Strategy: {strategy}',
            xaxis_title='Stock',
            yaxis_title='Average Sharpe Ratio',
            xaxis=dict(categoryorder='total descending'),
        )

        output_html = Path(output_folder) / f"{strategy}_average_sharpe_ratios_by_stock.html"
        fig.write_html(str(output_html))
        print(f"Saved plot to {output_html}")

def plot_intraday_sharpe_ratio_histograms(results_files: list, output_html: str):
    """
    Plots histograms of Intraday Sharpe Ratios for three strategies in one HTML dashboard.
    """
    import pandas as pd

    # Collect data from results files
    strategy_sharpe_ratios = {}
    for rf in results_files:
        file_name = Path(rf).stem
        strategy, ticker = file_name.split('_')[:2]
        if strategy not in strategy_sharpe_ratios:
            strategy_sharpe_ratios[strategy] = []
        # Read results file
        df = pd.read_csv(rf)
        if 'Intraday Sharpe Ratio' in df.columns:
            strategy_sharpe_ratios[strategy].extend(df['Intraday Sharpe Ratio'].dropna().tolist())

    # Create histograms for each strategy
    fig = go.Figure()
    for strategy, sharpe_ratios in strategy_sharpe_ratios.items():
        fig.add_trace(go.Histogram(x=sharpe_ratios, name=strategy, opacity=0.75))

    # Update layout
    fig.update_layout(
        title='Intraday Sharpe Ratio Histograms by Strategy',
        xaxis_title='Intraday Sharpe Ratio',
        yaxis_title='Frequency',
        barmode='overlay'
    )

    # Save dashboard
    fig.write_html(output_html)
    print(f"Intraday Sharpe Ratio histograms saved to {output_html}")

def plot_correlation_heatmaps_by_ticker(results_files: list, output_html: str):
    """
    Plots correlation heatmaps of strategies for each ticker as subplots.
    Each subplot represents the correlation matrix of percentage changes in Intraday Final Balance.
    """
    import pandas as pd
    import plotly.subplots as sp

    # Collect data from results files
    ticker_strategy_data = {}
    for rf in results_files:
        file_name = Path(rf).stem
        strategy, ticker = file_name.split('_')[:2]
        if ticker not in ticker_strategy_data:
            ticker_strategy_data[ticker] = {}
        # Read results file
        df = pd.read_csv(rf)
        if 'Intraday Final Balance' in df.columns:
            df['Pct Change'] = df['Intraday Final Balance'].pct_change()
            ticker_strategy_data[ticker][strategy] = df['Pct Change'].dropna()

    # Calculate number of rows needed for 3 columns
    num_tickers = len(ticker_strategy_data)
    num_rows = (num_tickers + 2) // 3  # Round up to ensure all tickers fit

    # Create subplots
    fig = sp.make_subplots(rows=num_rows, cols=3, subplot_titles=list(ticker_strategy_data.keys()))

    for idx, (ticker, strategy_data) in enumerate(ticker_strategy_data.items(), start=1):
        row = (idx - 1) // 3 + 1
        col = (idx - 1) % 3 + 1

        # Create DataFrame for correlation
        df = pd.DataFrame(strategy_data)
        correlation_matrix = df.corr()

        # Add heatmap to subplot
        heatmap = go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='BrBG',  # Updated colorscale for better aesthetics
            zmin=-1,
            zmax=1
        )
        fig.add_trace(heatmap, row=row, col=col)

    # Update layout
    fig.update_layout(
        title='Correlation Heatmaps of Strategies by Ticker',
        height=300 * num_rows,
        showlegend=False
    )

    # Save dashboard
    fig.write_html(output_html)
    print(f"Correlation heatmaps saved to {output_html}")

def analyze_and_plot_signals_for_date(parquet_files: list, date: str, ticker_filter: str, output_html: str):
    """
    Parses parquet files for two strategies on a specific date and ticker, analyzes their signals,
    and plots a comparison to understand why they are similar.
    """
    import pandas as pd

    # Collect data for the specified date and ticker
    strategy_signals = {}
    for pf in parquet_files:
        file_name = Path(pf).stem
        strategy, ticker, file_date = file_name.split('_')[:3]
        if file_date == date and ticker == ticker_filter:
            df = pd.read_parquet(pf)
            # get between the 45th and 50th percentile
            df = df[len(df) * 45 // 100: len(df) * 50 // 100]
            if 'Signal' in df.columns:
                if strategy not in strategy_signals:
                    strategy_signals[strategy] = []
                strategy_signals[strategy] = df[df['Timestamp'].dt.date == pd.to_datetime(date).date()]['Signal']

    # Create plot
    fig = go.Figure()
    for strategy, signals in strategy_signals.items():
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals,
            mode='lines+markers',
            name=strategy
        ))

    # Update layout
    fig.update_layout(
        title=f'Signal Comparison for Strategies on {date} for Ticker {ticker_filter}',
        xaxis_title='Timestamp',
        yaxis_title='Signal Value',
        legend_title='Strategy'
    )

    # Save plot
    fig.write_html(output_html)
    print(f"Signal comparison plot saved to {output_html}")

def plot_bid_ask_with_signals(parquet_files: list, date: str, stock: str, output_html: str):
    """
    Plots bid and ask prices for a given day and stock as two lines.
    Includes signals as markers for three strategies.
    """
    import pandas as pd

    # Collect data for the specified date and stock
    combined_data = None
    strategy_signals = {}
    for pf in parquet_files:
        file_name = Path(pf).stem
        strategy, ticker, file_date = file_name.split('_')[:3]
        if file_date == date and ticker == stock:
            df = pd.read_parquet(pf)
            df = clean_bid_ask_data(df)  # Clean the data using Polars
            if 'Signal' in df.columns and 'bid' in df.columns and 'ask' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Ensure Timestamp is properly parsed
                df = df[df['Timestamp'].dt.date == pd.to_datetime(date).date()]  # Filter data for the specific date
                if combined_data is None:
                    # take 45th to 50th percentile of the data
                    combined_data = df[len(df) * 45 // 100: len(df) * 50 // 100]
                if strategy not in strategy_signals:
                    strategy_signals[strategy] = df[len(df) * 45 // 100: len(df) * 50 // 100]['Signal']

    if combined_data is None:
        print("No data found for the specified date and stock.")
        return

    # Create plot
    fig = go.Figure()

    # Plot bid and ask as lines
    fig.add_trace(go.Scatter(
        x=combined_data['Timestamp'],
        y=combined_data['bid'],
        mode='lines',
        name='Bid'
    ))
    fig.add_trace(go.Scatter(
        x=combined_data['Timestamp'],
        y=combined_data['ask'],
        mode='lines',
        name='Ask'
    ))

    # Add signals as markers for each strategy
    for strategy, signals in strategy_signals.items():
        buy_signals = combined_data[signals > 0]
        sell_signals = combined_data[signals < 0]
        fig.add_trace(go.Scatter(
            x=buy_signals['Timestamp'],
            y=buy_signals['bid'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10),
            name=f'{strategy} - Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals['Timestamp'],
            y=sell_signals['ask'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10),
            name=f'{strategy} - Sell Signal'
        ))

    # Update layout
    fig.update_layout(
        title=f'Bid/Ask with Signals for {stock} on {date}',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        legend_title='Signals'
    )

    # Save plot
    fig.write_html(output_html)
    print(f"Bid/Ask plot with signals saved to {output_html}")

def plot_bid_ask_vwap_with_signals(parquet_files: list, date: str, stock: str, output_html: str):
    """
    Plots bid, ask, and VWAP prices for a given day and stock as three lines.
    Includes signals as markers for three strategies.
    """
    import pandas as pd

    # Collect data for the specified date and stock
    combined_data = None
    strategy_signals = {}
    for pf in parquet_files:
        file_name = Path(pf).stem
        strategy, ticker, file_date = file_name.split('_')[:3]
        if file_date == date and ticker == stock and strategy == 'OBI-VWAP':
            df = pd.read_parquet(pf)
            df = clean_bid_ask_data(df)  # Clean the data using Polars
            if 'Signal' in df.columns and 'bid' in df.columns and 'ask' in df.columns and 'VWAP' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Ensure Timestamp is properly parsed
                df = df[df['Timestamp'].dt.date == pd.to_datetime(date).date()]  # Filter data for the specific date
                if combined_data is None:
                    # take 45th to 50th percentile of the data
                    combined_data = df[len(df) * 40 // 100: len(df) * 50 // 100]

    for pf in parquet_files:
        file_name = Path(pf).stem
        strategy, ticker, file_date = file_name.split('_')[:3]
        if file_date == date and ticker == stock:
            df = pd.read_parquet(pf)
            df = clean_bid_ask_data(df)  # Clean the data using Polars      
            if file_date == date and ticker == stock:
                if strategy not in strategy_signals:
                    strategy_signals[strategy] = df[len(df) * 40 // 100: len(df) * 50 // 100]['Signal']

    if combined_data is None:
        print("No data found for the specified date and stock.")
        return

    # Create plot
    fig = go.Figure()

    # Plot bid, ask, and VWAP as lines
    fig.add_trace(go.Scatter(
        x=combined_data['Timestamp'],
        y=combined_data['bid'],
        mode='lines',
        name='Bid'
    ))
    fig.add_trace(go.Scatter(
        x=combined_data['Timestamp'],
        y=combined_data['ask'],
        mode='lines',
        name='Ask'
    ))
    fig.add_trace(go.Scatter(
        x=combined_data['Timestamp'],
        y=combined_data['VWAP'],
        mode='lines',
        name='VWAP'
    ))

    # Add signals as markers for each strategy
    for strategy, signals in strategy_signals.items():
        buy_signals = combined_data[signals > 0]
        sell_signals = combined_data[signals < 0]
        fig.add_trace(go.Scatter(
            x=buy_signals['Timestamp'],
            y=buy_signals['bid'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10),
            name=f'{strategy} - Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals['Timestamp'],
            y=sell_signals['ask'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10),
            name=f'{strategy} - Sell Signal'
        ))

    # Update layout
    fig.update_layout(
        title=f'Bid/Ask/VWAP with Signals for {stock} on {date}',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        legend_title='Signals'
    )

    # Save plot
    fig.write_html(output_html)
    print(f"Bid/Ask/VWAP plot with signals saved to {output_html}")

def clean_bid_ask_data(df: pd.DataFrame, rolling_window: int = 100, mad_threshold: float = 10) -> pd.DataFrame:
    """
    Cleans TAQ bid/ask data from a Parquet file.

    Args:
        parquet_path (str): Path to the Parquet file.
        rolling_window (int): Rolling window size for median/MAD filter.
        mad_threshold (float): Threshold in MAD units to detect outliers.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    
    # Drop rows with non-positive bids or asks
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]

    # Remove crossed markets (bid > ask)
    df = df[df['bid'] <= df['ask']]

    # Compute mid price
    df['mid'] = (df['bid'] + df['ask']) / 2

    # Filter out large deviations from rolling median using MAD
    rolling_median = df['mid'].rolling(window=rolling_window, min_periods=1).median()
    mad = df['mid'].rolling(window=rolling_window, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # Avoid divide-by-zero MAD
    mad = mad.replace(0, 1e-6)

    deviation = np.abs(df['mid'] - rolling_median)
    df = df[deviation < mad_threshold * mad]

    # Drop helper columns
    df = df.drop(columns=['mid'])

    return df



if __name__ == "__main__":
    import pprint
    folder = "data/backtest_results"  # adjust path as needed
    cols = {
        "Mean-Reversion": [],
        "Inverse-OBI-VWAP": [],
        "OBI-VWAP_ALL": []
    }
    cols["Mean-Reversion"] = ['bid','ask','Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'Price_Deviation', 'Volatility', 'Volume_MA', 'Volume_Ratio', 'Deviation_MA', 'Mean_Reversion_Score', 'Signal', 'Account_Balance', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']
    cols["Inverse-OBI-VWAP"] = ['bid','ask', 'Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'VWAP_STD', 'VWAP_Upper', 'VWAP_Lower', 'Rolling_Median_VWAP', 'Inverse_VWAP', 'Spread', 'Relative_Spread', 'Bid_Pressure', 'Dollar_Volume', 'Bid_Depth_Ratio', 'Ask_Depth_Ratio', 'Price_Impact', 'Rolling_Median_Volume', 'Inverted_Volume', 'Volume_Norm', 'Price_Momentum', 'Volume_Momentum', 'Mean_Reversion_Score', 'OB_RSI', 'Volatility', 'Parkinson_Vol', 'Vol_Adjusted_Vol', 'Raw_OBI', 'Price_Weighted_Volume', 'Time_Weighted_OBI', 'OBI', 'Signal', 'Account_Balance', 'Time', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']
    cols["OBI-VWAP_ALL"] = ['bid','ask','Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'VWAP_STD', 'VWAP_Upper', 'VWAP_Lower', 'Spread', 'Relative_Spread', 'Bid_Pressure', 'Dollar_Volume', 'Bid_Depth_Ratio', 'Ask_Depth_Ratio', 'Price_Impact', 'Price_Momentum', 'Volume_Momentum', 'Mean_Reversion_Score', 'OB_RSI', 'Volatility', 'Parkinson_Vol', 'Vol_Adjusted_Vol', 'Short_Trend', 'Medium_Trend', 'Long_Trend', 'Uptrend', 'Downtrend', 'High_Vol_Regime', 'Trend_Quality', 'Vol_Quality', 'Spread_Quality', 'Signal_Quality', 'Signal', 'Account_Balance', 'Time', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']                

    # Parse parquet files
    print("Parsed .parquet files:")
    parquet_results = parse_backtest_results_folder(folder)
    pprint.pprint(parquet_results)

    # Example: parse all performance and results txt files in subfolders
    from glob import glob
    import os
    perf_files = glob(os.path.join(folder, "**", "*_performance_*.txt"), recursive=True)
    results_files = glob(os.path.join(folder, "**", "*_results_*.txt"), recursive=True)

    print("\nParsed performance.txt files:")
    for pf in perf_files:
        print(f"{pf}:")
        pprint.pprint(parse_performance_txt(pf))

    print("\nParsed results.txt files:")
    for rf in results_files:
        print(f"{rf}:")
        pprint.pprint(parse_results_txt(rf))
        results = parse_results_txt(rf)
        

    # Combine all results for a single equity curve
    all_results = []
    for rf in results_files:
        all_results.extend(parse_results_txt(rf))
    if all_results:
        plot_equity_curve(all_results, os.path.join(folder, "combined_equity_curve.html"))

    # Combine all results for grouped equity curve
    all_results = []
    for rf in results_files:
        all_results.extend(parse_results_txt(rf))
    if all_results:
        month_to_plot = "08"  # Example month, adjust as needed
        plot_grouped_equity_curve(all_results, os.path.join(folder, "grouped_equity_curve.html"), month_to_plot)

    # Combine all results for strategy-specific equity curves
    all_results = []
    for rf in results_files:
        all_results.extend(parse_results_txt(rf))
    if all_results:
        plot_strategy_equity_curves(all_results, folder)

    # Create bar chart for average monthly returns
    # if perf_files:
    #     plot_average_monthly_returns(perf_files, os.path.join(folder, "average_monthly_returns.html"))

    # Create bar chart for average monthly returns by strategy
    if perf_files:
        plot_average_monthly_returns_by_strategy(perf_files, folder)

    # Create bar charts for average monthly returns by stock and strategy
    # if perf_files:
    #     plot_average_monthly_returns_by_stock_and_strategy(perf_files, folder)

    # Create bar charts for Sharpe Ratios by stock and strategy
    # if perf_files:
    #     plot_average_sharpe_ratios_by_stock_and_strategy(perf_files, folder)

    # Create histograms of Intraday Sharpe Ratios
    if results_files:
        plot_intraday_sharpe_ratio_histograms(results_files, os.path.join(folder, "intraday_sharpe_ratio_histograms.html"))

    # Create correlation heatmaps by ticker
    if results_files:
        plot_correlation_heatmaps_by_ticker(results_files, os.path.join(folder, "correlation_heatmaps_by_ticker.html"))

    # Analyze and plot signals for specific date
    if parquet_results:
        date_to_analyze = "2023-08-14"  # example date, adjust as needed
        ticker_to_analyze = "OXY"
        analyze_and_plot_signals_for_date([pr['file'] for pr in parquet_results], date_to_analyze, ticker_to_analyze, os.path.join(folder, f"signals_comparison_{date_to_analyze}.html"))

    # Plot bid and ask with signals for a specific date and stock
    if parquet_results:
        date_to_plot = "2023-08-14"  # example date, adjust as needed
        stock_to_plot = "OXY"  # example stock, adjust as needed
        plot_bid_ask_with_signals([pr['file'] for pr in parquet_results], date_to_plot, stock_to_plot, os.path.join(folder, f"bid_ask_signals_{stock_to_plot}_{date_to_plot}.html"))

    # Plot bid and ask with signals for a specific date and stock
    if parquet_results:
        date_to_plot = "2023-08-14"  # example date, adjust as needed
        stock_to_plot = "JPM"  # example stock, adjust as needed
        plot_bid_ask_with_signals([pr['file'] for pr in parquet_results], date_to_plot, stock_to_plot, os.path.join(folder, f"bid_ask_signals_{stock_to_plot}_{date_to_plot}.html"))

    # Plot bid, ask, and VWAP with signals for a specific date and stock
    if parquet_results:
        date_to_plot = "2023-07-14"  # example date, adjust as needed
        stock_to_plot = "ZION"  # example stock, adjust as needed
        plot_bid_ask_vwap_with_signals([pr['file'] for pr in parquet_results], date_to_plot, stock_to_plot, os.path.join(folder, f"bid_ask_vwap_signals_{stock_to_plot}_{date_to_plot}.html"))
