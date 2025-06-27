import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_performance_file(filepath):
    with open(filepath, 'r') as f:
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
    # Match files like *_performance_06.txt, *_performance_07.txt, *_performance_08.txt
    pattern = re.compile(r'_performance_(\d{2})\.txt$')
    for root, dirs, files in os.walk(backtest_dir):
        for file in files:
            match = pattern.search(file)
            if match:
                month = match.group(1)
                filepath = os.path.join(root, file)
                # Extract stock and strategy from path
                rel_path = os.path.relpath(filepath, backtest_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    stock = parts[0]
                    if stock == 'LEN':
                        continue
                    # Remove _performance_XX.txt from strategy name
                    strategy = re.sub(r'_performance_\d{2}\.txt$', '', file).rsplit('_', 1)[0]
                    metrics = parse_performance_file(filepath)
                    results_txt = read_and_parse_results(stock, strategy, month)
                    # final account balance - initial account balance
                    if not results_txt.empty:
                        metrics['final_balance'] = results_txt['final_balance'].iloc[-1] if 'final_balance' in results_txt.columns else np.nan
                        metrics['total_return'] = ((results_txt['final_balance'].iloc[-1] / 300_000) - 1) * 100 if 'final_balance' in results_txt.columns else np.nan
                    metrics.update({'stock': stock, 'strategy': strategy, 'month': month})
                    results.append(metrics)
    return pd.DataFrame(results)

def analyze_results(df):
    summary = []
    months = sorted(df['month'].unique())
    for month in months:
        month_df = df[df['month'] == month].copy()
        summary.append(f"\n=== Results for Month: {month} ===")
        # Best strategy overall for the month
        best_strategy = month_df.groupby('strategy')['total_return'].mean().idxmax()
        summary.append(f"Best strategy: {best_strategy}")
        # Best stock overall for the month
        best_stock = month_df.groupby('stock')['total_return'].mean().idxmax()
        summary.append(f"Best stock: {best_stock}")
        # Highest volatility stock (by std of total_return)
        highest_vol_stock = month_df.groupby('stock')['total_return'].std().idxmax()
        summary.append(f"Highest volatility stock: {highest_vol_stock}")
        # Best Sharpe ratio
        best_sharpe = month_df.loc[month_df['sharpe_ratio'].idxmax()]
        summary.append(f"Best Sharpe ratio: {best_sharpe['sharpe_ratio']} ({best_sharpe['stock']} - {best_sharpe['strategy']})")
        # Most effective strategy family
        families = ['Mean-Reversion', 'OBI-VWAP', 'Inverse-OBI-VWAP']
        family_means = {}
        for fam in families:
            fam_mask = month_df['strategy'].str.replace('-', '').str.replace(' ', '').str.lower().str.contains(fam.replace('-', '').replace(' ', '').lower())
            if fam_mask.any():
                family_means[fam] = month_df.loc[fam_mask, 'total_return'].mean()
            else:
                family_means[fam] = float('-inf')
        most_effective_family = max(family_means, key=family_means.get)
        summary.append(f"Most effective strategy family: {most_effective_family} (mean total return: {family_means[most_effective_family]:.2f} %)")
        # Average max drawdown per strategy
        avg_drawdown = month_df.groupby('strategy')['max_drawdown'].mean()
        summary.append("Average max drawdown per strategy:\n" + avg_drawdown.to_string())
    # Optionally, add overall summary across all months
    summary.append("\n=== Overall Summary Across All Months ===")
    best_strategy = df.groupby('strategy')['total_return'].mean().idxmax()
    summary.append(f"Best strategy overall: {best_strategy}")
    best_stock = df.groupby('stock')['total_return'].mean().idxmax()
    summary.append(f"Best stock overall: {best_stock}")
    return '\n'.join(summary)


# Date,Intraday Sharpe Ratio,Intraday Profit,Intraday Drawdown,Intraday Max Return,Intraday Final Balance
# 2023-07-03,-0.016523230972934434,-22029.019999995595,-0.18314445432335846,0.04397990032866517,277970.9800000044
# 2023-07-05,0.09417288349398402,146774.46999999922,-0.0824143644780696,0.0906106907379658,424745.4500000036
# 2023-07-06,0.01984839571200571,8805.950000002456,-0.026924356283131834,0.027508168488919127,433551.4000000061
# 2023-07-07,0.01983919847276208,6718.05999999668,-0.025830053390241803,0.02931928121996097,440269.46000000276
# 2023-07-10,0.07049206325701195,36680.04000000114,-0.02558660980707883,0.026096325318095914,476949.5000000039
# 2023-07-11,-0.013359843186954502,-3380.0999999996857,-0.023528975046401168,0.001196497827296028,473569.4000000042

def read_and_parse_results(stock, strategy, month):
    try:
        # Ensure month is an integer before formatting
        month = int(month)
        file_path = f'data/backtest_results/{stock}/{strategy}_{stock}_results_{month:02d}.txt'
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return pd.DataFrame()
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.columns = df.columns.str.strip()
        required_cols = [
            'Date', 'Intraday Sharpe Ratio', 'Intraday Profit',
            'Intraday Drawdown', 'Intraday Final Balance'
        ]
        if all(col in df.columns for col in required_cols):
            df = df[required_cols]
            df.rename(columns={
                'Intraday Sharpe Ratio': 'sharpe_ratio',
                'Intraday Profit': 'total_return',
                'Intraday Drawdown': 'max_drawdown',
                'Intraday Final Balance': 'final_balance'
            }, inplace=True)
            df['final_balance'] = pd.to_numeric(df['final_balance'], errors='coerce')
            df['daily_return'] = df['final_balance'].pct_change()
            df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], 0)
            df.loc[0, 'daily_return'] = (df.loc[0, 'final_balance'] / 300_000 - 1) * 0.1  # Fix chained assignment
            df['daily_return'] = df['daily_return'] * 0.1
            df['final_balance'] = df['daily_return'].add(1).cumprod() * 300_000
            return df
        else:
            print(f"File {file_path} does not contain all required columns.")
            return pd.DataFrame()
    except ValueError as e:
        print(f"Error processing month value: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

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
