import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import polars as pl
from typing import Dict
import argparse
from pathlib import Path
import os

def create_hft_dashboard(backtest_data: pl.DataFrame, strategy_name: str = "Strategy"):
    """
    Creates a comprehensive dashboard for HFT strategy analysis using Plotly.
    
    Args:
        backtest_data (pl.DataFrame): DataFrame containing backtest results
        strategy_name (str): Name of the strategy for plot titles
    """
    pdf = backtest_data.to_pandas()
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Account Balance & Position', 'Price & VWAP Analysis',
            'Volume Analysis', 'Order Book Imbalance',
            'Strategy Signals', 'PnL Analysis',
            'Market Regime Indicators', 'Risk Metrics'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        specs=[[{"secondary_y": True}]*2]*4
    )
    
    # 1. Account Balance & Position Plot
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Account_Balance'],
                  name='Account Balance', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Position'],
                  name='Position', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # 2. Price & VWAP Analysis
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['MID_PRICE'],
                  name='Mid Price', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['VWAP'],
                  name='VWAP', line=dict(color='orange')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['VWAP_Upper'],
                  name='VWAP Upper', line=dict(color='lightgray', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['VWAP_Lower'],
                  name='VWAP Lower', line=dict(color='lightgray', dash='dash')),
        row=1, col=2
    )
    
    # 3. Volume Analysis
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Volume'],
                  name='Volume', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Volume_Norm'],
                  name='Volume Norm', line=dict(color='green')),
        row=2, col=1, secondary_y=True
    )
    
    # 4. Order Book Imbalance
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['OBI'],
                  name='OBI', line=dict(color='purple')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Bid_Pressure'],
                  name='Bid Pressure', line=dict(color='orange')),
        row=2, col=2, secondary_y=True
    )
    
    # 5. Strategy Signals
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Signal'],
                  name='Trading Signal', mode='markers',
                  marker=dict(size=10, symbol='triangle-up',
                            color=pdf['Signal'].map({1: 'green', -1: 'red', 0: 'gray'}))),
        row=3, col=1
    )
    
    # 6. PnL Analysis
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Unrealized_PnL'],
                  name='Unrealized PnL', line=dict(color='green')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Unrealized_PnL_Pct'],
                  name='Unrealized PnL %', line=dict(color='blue')),
        row=3, col=2, secondary_y=True
    )
    
    # 7. Market Regime Indicators
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Volatility'],
                  name='Volatility', line=dict(color='red')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Signal_Quality'],
                  name='Signal Quality', line=dict(color='blue')),
        row=4, col=1, secondary_y=True
    )
    
    # 8. Risk Metrics
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Drawdown'],
                  name='Drawdown', line=dict(color='red')),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        width=1600,
        title_text=f"{strategy_name} Analysis Dashboard",
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=2)
    
    return fig

def create_order_book_analysis(backtest_data: pl.DataFrame):
    """
    Creates detailed order book analysis visualizations.
    """
    pdf = backtest_data.to_pandas()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Bid-Ask Spread Analysis',
            'Order Book Depth',
            'Price Impact Analysis',
            'Order Flow Imbalance'
        ),
        vertical_spacing=0.15
    )
    
    # 1. Bid-Ask Spread
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Spread'],
                  name='Spread', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Relative_Spread'],
                  name='Relative Spread', line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. Order Book Depth
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Bid_Depth_Ratio'],
                  name='Bid Depth Ratio', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Ask_Depth_Ratio'],
                  name='Ask Depth Ratio', line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Price Impact
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Price_Impact'],
                  name='Price Impact', line=dict(color='purple')),
        row=2, col=1
    )
    
    # 4. Order Flow Imbalance
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['OBI'],
                  name='OBI', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Raw_OBI'],
                  name='Raw OBI', line=dict(color='gray', dash='dash')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text="Order Book Analysis Dashboard",
        template="plotly_white"
    )
    
    return fig

def create_performance_analysis(backtest_data: pl.DataFrame):
    """
    Creates performance analysis visualizations.
    """
    pdf = backtest_data.to_pandas()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Returns',
            'Drawdown Analysis',
            'Trade Analysis',
            'Risk Metrics'
        ),
        vertical_spacing=0.15
    )
    
    # 1. Cumulative Returns
    cumulative_returns = (1 + pdf['Returns']).cumprod()
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=cumulative_returns,
                  name='Cumulative Returns', line=dict(color='green')),
        row=1, col=1
    )
    
    # 2. Drawdown Analysis
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Drawdown'],
                  name='Drawdown', line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Trade Analysis
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Position'],
                  name='Position', line=dict(color='blue')),
        row=2, col=1
    )
    
    # 4. Risk Metrics
    fig.add_trace(
        go.Scatter(x=pdf['Timestamp'], y=pdf['Volatility'],
                  name='Volatility', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text="Performance Analysis Dashboard",
        template="plotly_white"
    )
    
    return fig

def plot_all_analyses(backtest_data: pl.DataFrame, strategy_name: str = "Strategy"):
    """
    Creates and returns all analysis plots.
    
    Args:
        backtest_data (pl.DataFrame): DataFrame containing backtest results
        strategy_name (str): Name of the strategy for plot titles
    
    Returns:
        Dict[str, go.Figure]: Dictionary containing all created figures
    """
    return {
        'main_dashboard': create_hft_dashboard(backtest_data, strategy_name),
        'order_book_analysis': create_order_book_analysis(backtest_data),
        'performance_analysis': create_performance_analysis(backtest_data)
    }

def setup_argparser():
    """
    Sets up command line argument parser for the plotting script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate HFT strategy analysis plots from backtest results CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing backtest results"
    )
    
    parser.add_argument(
        "--strategy_name",
        type=str,
        default="HFT Strategy",
        help="Name of the strategy for plot titles"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save the plot HTML files"
    )
    
    return parser

def main():
    """Main execution function"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Validate CSV path
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {args.csv_path}...")
    df = pl.read_csv(args.csv_path)
    
    # Generate plots
    print("Generating plots...")
    plots = plot_all_analyses(df, args.strategy_name)
    
    # Save plots to HTML files
    print(f"Saving plots to {args.output_dir}/...")
    for name, fig in plots.items():
        output_path = os.path.join(args.output_dir, f"{name}.html")
        fig.write_html(output_path)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    main()