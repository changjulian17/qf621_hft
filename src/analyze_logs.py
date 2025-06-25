import os
import re
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_ticker_strategy_metrics(log_content: str) -> List[Dict]:
    """
    Extract performance metrics for each ticker and strategy combination from a log file.
    """
    results = []
    current_ticker = None
    current_strategy = None
    
    # Regular expressions for extracting information
    ticker_pattern = r"Processing ([A-Z]+)\.\.\."
    strategy_pattern = r"Starting backtest for (.*?)(?:Strategy| with initial cash:)"
    metrics_patterns = {
        "Total Return": r"Total Return: ([-\d.]+)%",
        "Sharpe Ratio": r"Sharpe Ratio: ([-\d.]+)",
        "Maximum Drawdown": r"Maximum Drawdown: ([-\d.]+)%",
        "Number of Trades": r"Number of Trades: (\d+)",
        "Win Rate": r"Win Rate: ([-\d.]+)%",
        "Average Profit per Trade": r"Average Profit per Trade: \$([-\d.]+)",
        "Final Account Balance": r"Final Account Balance: \$([\d,]+\.\d+)"
    }

    for line in log_content.split('\n'):
        # Extract ticker
        ticker_match = re.search(ticker_pattern, line)
        if ticker_match:
            current_ticker = ticker_match.group(1)
            continue

        # Extract strategy
        strategy_match = re.search(strategy_pattern, line)
        if strategy_match:
            current_strategy = strategy_match.group(1)
            metrics = {"Ticker": current_ticker, "Strategy": current_strategy}
            continue

        # Extract metrics if we have both ticker and strategy
        if current_ticker and current_strategy:
            for metric_name, pattern in metrics_patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1).replace(',', ''))
                    metrics[metric_name] = value

            # If we found all metrics, add to results and reset strategy
            if len(metrics) == len(metrics_patterns) + 2:  # +2 for Ticker and Strategy
                # Clean up strategy name
                metrics["Strategy"] = metrics["Strategy"].strip()
                results.append(metrics)
                current_strategy = None

    return results

def analyze_log_files(logs_dir: str) -> pd.DataFrame:
    """
    Analyze all log files in the directory and combine results into a DataFrame.
    """
    all_results = []
    logs_path = Path(logs_dir)

    for log_file in logs_path.glob("*.log"):
        logger.info(f"Processing log file: {log_file}")
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            results = extract_ticker_strategy_metrics(log_content)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing {log_file}: {str(e)}")

    if not all_results:
        logger.warning("No results found in log files")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    
    # Sort by performance metrics
    df['Score'] = (
        df['Total Return'] * 0.3 +  # 30% weight on returns
        df['Sharpe Ratio'].clip(-10, 10) * 0.3 +  # 30% weight on risk-adjusted returns
        df['Win Rate'] * 0.2 +  # 20% weight on consistency
        df['Average Profit per Trade'] * 0.2  # 20% weight on trade profitability
    )
    
    return df

def generate_summary(df: pd.DataFrame, output_dir: str):
    """
    Generate summary CSV files with different views of the data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'strategy_performance_detailed.csv'), index=False)
    
    # Best strategy per ticker
    best_per_ticker = df.loc[df.groupby('Ticker')['Score'].idxmax()]
    best_per_ticker.to_csv(os.path.join(output_dir, 'best_strategy_per_ticker.csv'), index=False)
    
    # Best tickers per strategy
    best_per_strategy = df.loc[df.groupby('Strategy')['Score'].idxmax()]
    best_per_strategy.to_csv(os.path.join(output_dir, 'best_ticker_per_strategy.csv'), index=False)
    
    # Strategy ranking
    strategy_ranking = df.groupby('Strategy').agg({
        'Score': 'mean',
        'Total Return': 'mean',
        'Sharpe Ratio': 'mean',
        'Win Rate': 'mean',
        'Average Profit per Trade': 'mean'
    }).sort_values('Score', ascending=False)
    strategy_ranking.to_csv(os.path.join(output_dir, 'strategy_ranking.csv'))
    
    # Log summary statistics
    logger.info("\nStrategy Performance Summary:")
    logger.info("\nBest Strategies per Ticker:")
    for _, row in best_per_ticker.iterrows():
        logger.info(f"{row['Ticker']}: {row['Strategy']} (Return: {row['Total Return']:.2f}%, Sharpe: {row['Sharpe Ratio']:.2f})")
    
    logger.info("\nOverall Strategy Ranking:")
    for strategy, row in strategy_ranking.iterrows():
        logger.info(f"{strategy}: Avg Return: {row['Total Return']:.2f}%, Avg Sharpe: {row['Sharpe Ratio']:.2f}")

def main():
    logs_dir = "logs"
    output_dir = "analysis_results"
    
    logger.info("Starting log analysis...")
    df = analyze_log_files(logs_dir)
    
    if not df.empty:
        generate_summary(df, output_dir)
        logger.info(f"Analysis complete. Results saved in {output_dir}/")
    else:
        logger.error("No data found to analyze")

if __name__ == "__main__":
    main()
