from src.wrds_pull import fetch_taq_data
from src.strategy import StrategyPortfolio
from src.strats.obi import OBIVWAPStrategy
from src.strats.mean_reversion import MeanReversionStrategy
from src.strats.inverse_obi import  InverseOBIVWAPStrategy
from src.performance import evaluate_strategy_performance
from src.logger_config import setup_logger
import polars as pl
import random
import os
import gc
import itertools
import argparse
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta

def parse_time(time_str: str) -> Tuple[int, int]:
    """Parse time string in HH:MM format to tuple of (hour, minute)."""
    hour, minute = map(int, time_str.split(':'))
    return (hour, minute)

# write polars df to parquet file with stock and date as file name
def write_parquet(df: pl.DataFrame, stock: str, date: str, output_dir: str):
    """Write Polars DataFrame to Parquet file with stock and date as filename."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/{stock}_{date}.parquet"
    df.write_parquet(file_path)
    print(f"Data for {stock} on {date} written to {file_path}")

def check_data_exists(ticker: str, date: str, output_dir: str) -> bool:
    """Check if Parquet file for a given ticker and date exists."""
    file_path = f"{output_dir}/{ticker}/{ticker}_{date}.parquet"
    return os.path.exists(file_path)

def read_parquet_for_multiple_stocks_dates(tickers: list, start_date: str, end_date: str, output_dir: str) -> pl.DataFrame:
    """Read Parquet files for multiple stocks and dates into a single Polars DataFrame."""
    sdate = datetime.strptime(start_date, "%Y-%m-%d")
    edate = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = [sdate + timedelta(days=i) for i in range((edate - sdate).days + 1)]
    all_data = []

    # Define the expected schema for casting
    expected_schema = {
        'date': pl.String,
        'time_m': pl.String,
        'time_m_nano': pl.Int64,
        'ex': pl.String,
        'sym_root': pl.String,
        'sym_suffix': pl.String,
        'bid': pl.Float64,
        'bidsiz': pl.Int64,
        'ask': pl.Float64,
        'asksiz': pl.Int64,
        'qu_cond': pl.String,
        'qu_seqnum': pl.Int64,
        'natbbo_ind': pl.String,
        'qu_cancel': pl.String,
        'qu_source': pl.String,
        'rpi': pl.String,
        'ssr': pl.String,
        'luld_bbo_indicator': pl.String,
        'finra_bbo_ind': pl.String,
        'finra_adf_mpid': pl.String,
        'finra_adf_time': pl.String,
        'finra_adf_time_nano': pl.Int64,
        'finra_adf_mpq_ind': pl.String,
        'finra_adf_mquo_ind': pl.String,
        'sip_message_id': pl.String,
        'natl_bbo_luld': pl.String,
        'part_time': pl.String,
        'part_time_nano': pl.Int64,
        'secstat_ind': pl.String,
        'Timestamp': pl.Datetime("ns"),
    }

    for ticker in tickers:
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            print(f"Reading data for {ticker} on {date_str}")
            file_path = f"{output_dir}/{ticker}/{ticker}_{date_str}.parquet"
            if os.path.exists(file_path):
                df = pl.read_parquet(file_path)
                # Cast columns to expected types
                for col, dtype in expected_schema.items():
                    if col in df.columns:
                        try:
                            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
                        except Exception as e:
                            print(f"Warning: Could not cast column {col} in {file_path}: {e}")
                all_data.append(df)
            else:
                print(f"File not found: {file_path}")
        # No need to check schema here, as we enforce it above

    if all_data:
        return pl.concat(all_data, how="vertical")
    else:
        return pl.DataFrame()

def parse_arguments():
    parser = argparse.ArgumentParser(description='High-Frequency Trading Strategy Analysis')
    
    # Data filtering parameters
    parser.add_argument('--exchanges', default="'Q'", help='Exchange filter')
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
    # for all tickers and dates check if data exists, if not, fetch data
    for ticker in args.tickers:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date   = datetime.strptime(args.end_date, "%Y-%m-%d")
        curr_date = start_date
        chk_date = start_date
        while chk_date <= end_date:
            date_str = chk_date.strftime('%Y-%m-%d')
            if  not check_data_exists(ticker, date_str, "data/parquet"):
                # logger.info(f"Data for {ticker} on {date_str} not found. Fetching data...")
                # Fetch data for this ticker and date
                df_din = fetch_taq_data(
                    tickers=[ticker],
                    start_date=args.start_date,
                    end_date=args.end_date,
                    exchanges=args.exchanges,
                    quote_conds=args.quote_cond,
                    wrds_username=args.wrds_username
                )
                break
            else:
                a = 1
                # logger.info(f"Data for {ticker} on {date_str} already exists. Skipping fetch.")
            chk_date += timedelta(days=1)
        
        while curr_date <= end_date:
            
            date_str = curr_date.strftime('%Y-%m-%d')
            
            curr_date += timedelta(days=1)
            output_dir = f"data/parquet/{ticker}"
            file_path = f"{output_dir}/{ticker}_{date_str}.parquet"
            if not os.path.exists(file_path):
                # logger.info(f"Data for {ticker} on {date_str} not found. Fetching data...")
                df = df_din.filter(
                    (pl.col("sym_root") == ticker) &
                    (pl.col("date") == date_str)
                )
                write_parquet(df, ticker, date_str, output_dir)
            else:
                a = 1
                # logger.info(f"Data for {ticker} on {date_str} already exists. Skipping fetch.")

    # logger.info(f"Fetching data from {args.start_date} to {args.end_date} for tickers: {args.tickers}")
    # Read data for all tickers and dates into a single Polars DataFrame
    df = read_parquet_for_multiple_stocks_dates(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir="data/parquet"
    )

    if df.is_empty():
        logger.error("No data found for the specified tickers and date range.")
        exit(1)

    stock_tickers = df["sym_root"].unique().to_list()
    logger.info(f"Tickers found in data: {stock_tickers}")

    

    positive_return_tickers = []
    all_results = {}

    for ticker in stock_tickers:
        portfolio = StrategyPortfolio(initial_cash=args.initial_cash, rebalance_threshold=args.rebalance_threshold)
        obi_strategy = OBIVWAPStrategy(
            vwap_window=500,
            momentum_window=100,
            volatility_window=50,
            trend_window=20,
            max_position=100,
            start_time=start_time,
            end_time=end_time,
            logger=logger
        )
        portfolio.add_strategy("OBI-VWAP", obi_strategy, weight=0.3)
        
        mean_rev_strategy = MeanReversionStrategy(
            vwap_window=300,
            volatility_window=100,
            volume_window=50,
            max_position=100,
            start_time=start_time,
            end_time=end_time,
            logger=logger
        )
        portfolio.add_strategy("Mean-Reversion", mean_rev_strategy, weight=0.3)
        
        inverse_obi_strategy = InverseOBIVWAPStrategy(
            vwap_window=500,
            momentum_window=100,
            max_position=100,
            start_time=start_time,
            end_time=end_time,
            logger=logger
        )
        portfolio.add_strategy("Inverse-OBI-VWAP", inverse_obi_strategy, weight=0.3)
        logger.info(f"Processing {ticker}...")
        ticker_data = df.filter(pl.col("sym_root") == ticker)
        logger.info(f"Data for {ticker} contains {ticker_data.shape[0]} records")

        # Run backtest for the portfolio
        logger.info(f"Running backtest for {ticker}...")

        days = ticker_data.select(pl.col("date").unique().sort()).to_series().to_list()
        final_df_ls, final_returns_ls = portfolio.run_backtest(ticker_data, days=days)
        # print key for final_df_ls and final_returns_ls
        keys = final_df_ls.keys()
        logger.info(f"Final DataFrames keys for {ticker}: {keys}")

        all_results[ticker] = final_df_ls
        # df columns: ['date', 'time_m', 'time_m_nano', 'ex', 'sym_root', 'sym_suffix', 'bid', 'bidsiz', 'ask', 'asksiz', 'qu_cond', 'qu_seqnum', 'natbbo_ind', 'qu_cancel', 'qu_source', 'rpi', 'ssr', 'luld_bbo_indicator', 'finra_bbo_ind', 'finra_adf_mpid', 'finra_adf_time', 'finra_adf_time_nano', 'finra_adf_mpq_ind', 'finra_adf_mquo_ind', 'sip_message_id', 'natl_bbo_luld', 'part_time', 'part_time_nano', 'secstat_ind', 'Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'Price_Deviation', 'Volatility', 'Volume_MA', 'Volume_Ratio', 'Deviation_MA', 'Mean_Reversion_Score', 'Signal', 'Account_Balance', 'Time', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']
        
        # Save backtest results for each strategy
        output_dir = f"data/backtest_results/{ticker}"
        os.makedirs(output_dir, exist_ok=True)

        for name in ['OBI-VWAP', 'Mean-Reversion', 'Inverse-OBI-VWAP']:

            final_df = final_df_ls[name]
            final_results = final_returns_ls[name]
            res_df = pd.DataFrame(columns=['Date', 'Intraday Sharpe Ratio', 'Intraday Profit', 'Intraday Drawdown', 'Intraday Max Return', 'Intraday Final Balance'])
            for df_in,  day in zip(final_df, days):
                if name == 'Mean-Reversion':
                    cols = ['bid','ask','Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'Price_Deviation', 'Volatility', 'Volume_MA', 'Volume_Ratio', 'Deviation_MA', 'Mean_Reversion_Score', 'Signal', 'Account_Balance', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']
                elif name == 'Inverse-OBI-VWAP':
                    cols = ['bid','ask', 'Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'VWAP_STD', 'VWAP_Upper', 'VWAP_Lower', 'Rolling_Median_VWAP', 'Inverse_VWAP', 'Spread', 'Relative_Spread', 'Bid_Pressure', 'Dollar_Volume', 'Bid_Depth_Ratio', 'Ask_Depth_Ratio', 'Price_Impact', 'Rolling_Median_Volume', 'Inverted_Volume', 'Volume_Norm', 'Price_Momentum', 'Volume_Momentum', 'Mean_Reversion_Score', 'OB_RSI', 'Volatility', 'Parkinson_Vol', 'Vol_Adjusted_Vol', 'Raw_OBI', 'Price_Weighted_Volume', 'Time_Weighted_OBI', 'OBI', 'Signal', 'Account_Balance', 'Time', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']
                elif name == 'OBI-VWAP':
                    cols = ['bid','ask','Timestamp', 'MID_PRICE', 'Volume', 'VWAP', 'VWAP_STD', 'VWAP_Upper', 'VWAP_Lower', 'Spread', 'Relative_Spread', 'Bid_Pressure', 'Dollar_Volume', 'Bid_Depth_Ratio', 'Ask_Depth_Ratio', 'Price_Impact', 'Price_Momentum', 'Volume_Momentum', 'Mean_Reversion_Score', 'OB_RSI', 'Volatility', 'Parkinson_Vol', 'Vol_Adjusted_Vol', 'Short_Trend', 'Medium_Trend', 'Long_Trend', 'Uptrend', 'Downtrend', 'High_Vol_Regime', 'Trend_Quality', 'Vol_Quality', 'Spread_Quality', 'Signal_Quality', 'Signal', 'Account_Balance', 'Time', 'Position', 'Entry_Price', 'Position_Size', 'Trade_Marker', 'Stop_Loss_Hit', 'Take_Profit_Hit', 'Max_Hold_Time_Hit']                
                common_cols = ['Volatility', 'Max_Hold_Time_Hit', 'Timestamp', 'Trade_Marker', 'Position', 'Mean_Reversion_Score', 'Entry_Price', 'Account_Balance', 'MID_PRICE', 'Take_Profit_Hit', 'bid', 'Volume', 'ask', 'Signal', 'Position_Size', 'VWAP', 'Stop_Loss_Hit']
                df_in = df_in.select(cols)
                file_path = f"{output_dir}/{name}_{ticker}_{day}.parquet"
                df_in.write_parquet(file_path)
                logger.info(f"Backtest results for {name} strategy on {ticker} saved to {file_path}")
            
            for result, day in zip(final_results,days):
                result_file_path = f"{output_dir}/{name}_{ticker}_results.txt"
                # extract info from result dict
                intraday_sharpe_ratio = result.get("sharpe_ratio", "N/A")
                intraday_profit = result.get("profit", "N/A")
                intraday_drawdown = result.get("drawdown", "N/A")
                intraday_max_return = result.get("max_return", "N/A")
                intraday_final_balance = result.get("final_balance", "N/A")
                row =pd.DataFrame({
                    "Date": [day],
                    "Intraday Sharpe Ratio": [intraday_sharpe_ratio],
                    "Intraday Profit": [intraday_profit],
                    "Intraday Drawdown": [intraday_drawdown],
                    "Intraday Max Return": [intraday_max_return],
                    "Intraday Final Balance": [intraday_final_balance]
                })
                res_df = pd.concat([res_df, row], ignore_index=True)
            res_df.to_csv(result_file_path, index=False)
            performance_metrics = evaluate_strategy_performance(res_df, logger=logger)
            # write performance metrics to a file
            perf_file_path = f"{output_dir}/{name}_{ticker}_performance.txt"
            with open(perf_file_path, 'w') as f:
                for key, value in performance_metrics.items():
                    f.write(f"{key}: {value}\n")
            logger.info(f"Performance metrics for {name} strategy on {ticker} saved to {perf_file_path}")

    del df
    gc.collect()


