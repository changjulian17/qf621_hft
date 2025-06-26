import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_performance_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
        f.seek(0)
        if ',' in first_line:
            # CSV format
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            required_cols = [
                'Intraday Sharpe Ratio',
                'Intraday Profit',
                'Intraday Drawdown',
                'Intraday Final Balance'
            ]
            if all(col in df.columns for col in required_cols):
                metrics = {}
                metrics['sharpe_ratio'] = df['Intraday Sharpe Ratio'].mean()
                metrics['total_return'] = df['Intraday Profit'].sum()
                # Use daily returns for annualization
                df['daily_return'] = df['Intraday Final Balance'].pct_change()
                mean_daily_return = df['daily_return'].mean()
                metrics['annualized_return'] = (1 + mean_daily_return) ** 252 - 1 if not np.isnan(mean_daily_return) else np.nan
                metrics['max_drawdown'] = df['Intraday Drawdown'].min()
                metrics['final_balance'] = df['Intraday Final Balance'].iloc[-1]
                return metrics
            # If not all required columns, fallback to key-value
            f.seek(0)
        # Key-value format
        metrics = {}
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                value = value.strip()
                try:
                    value = float(value) if value.lower() != 'nan' else np.nan
                except ValueError:
                    pass
                metrics[key.strip()] = value
        # Provide defaults for missing metrics
        for k in ['sharpe_ratio', 'total_return', 'annualized_return', 'max_drawdown', 'final_balance']:
            if k not in metrics:
                metrics[k] = np.nan
        return metrics

def collect_results(backtest_dir):
    results = []
    for root, dirs, files in os.walk(backtest_dir):
        for file in files:
            if file.endswith('_performance.txt'):
                filepath = os.path.join(root, file)
                # Extract stock and strategy from path
                rel_path = os.path.relpath(filepath, backtest_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    stock = parts[0]
                    strategy = re.sub(r'_performance.txt$', '', file)
                    metrics = parse_performance_file(filepath)
                    metrics.update({'stock': stock, 'strategy': strategy})
                    results.append(metrics)
    return pd.DataFrame(results)

def analyze_results(df):
    summary = []
    # Best strategy overall
    best_strategy = df.groupby('strategy')['total_return'].mean().idxmax()
    summary.append(f"Best strategy overall: {best_strategy}")
    # Best average strategy (by mean total_return)
    best_avg_strategy = df.groupby('strategy')['total_return'].mean().idxmax()
    summary.append(f"Best average strategy: {best_avg_strategy}")
    # Best stock overall
    best_stock = df.groupby('stock')['total_return'].mean().idxmax()
    summary.append(f"Best stock overall: {best_stock}")
    # Highest volatility stock (by std of total_return)
    highest_vol_stock = df.groupby('stock')['total_return'].std().idxmax()
    summary.append(f"Highest volatility stock: {highest_vol_stock}")
    # Best Sharpe ratio
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    summary.append(f"Best Sharpe ratio: {best_sharpe['sharpe_ratio']} ({best_sharpe['stock']} - {best_sharpe['strategy']})")
    # Most effective strategy family
    families = ['Mean-Reversion', 'OBI-VWAP', 'Inverse-OBI-VWAP']
    family_means = {}
    for fam in families:
        # Match strategies that contain the family name (case-insensitive, ignore spaces and dashes)
        fam_mask = df['strategy'].str.replace('-', '').str.replace(' ', '').str.lower().str.contains(fam.replace('-', '').replace(' ', '').lower())
        if fam_mask.any():
            family_means[fam] = df.loc[fam_mask, 'total_return'].mean()
        else:
            family_means[fam] = float('-inf')
    most_effective_family = max(family_means, key=family_means.get)
    summary.append(f"Most effective strategy family overall: {most_effective_family} (mean total return: {family_means[most_effective_family]:.2f})")
    # Correlation of returns between stocks
    summary.append("\nCorrelation of total returns between strategies:\nNot enough data to compute correlation.")
    # Top 5 by annualized return
    top5 = df.sort_values('annualized_return', ascending=False).head(5)
    summary.append("\nTop 5 by annualized return:\n" + top5[['stock', 'strategy', 'annualized_return']].to_string(index=False))
    # Average max drawdown per strategy
    avg_drawdown = df.groupby('strategy')['max_drawdown'].mean()
    summary.append("\nAverage max drawdown per strategy:\n" + avg_drawdown.to_string())
    return '\n'.join(summary)

def main():
    parser = argparse.ArgumentParser(description='Analyze backtest results.')
    parser.add_argument('--backtest_dir', type=str, default='data/backtest_results', help='Directory with backtest results')
    parser.add_argument('--output', type=str, default='analysis_results/summary_analysis.txt', help='Output summary file')
    args = parser.parse_args()

    df = collect_results(args.backtest_dir)
    if df.empty:
        print('No results found.')
        return
    summary = analyze_results(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(summary)
    print(f'Summary written to {args.output}')

if __name__ == '__main__':
    main()
