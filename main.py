from src.wrds_pull import fetch_taq_data
from src.strategy import OBIVWAPStrategy, StrategyPortfolio, MeanReversionStrategy, InverseOBIVWAPStrategy
from src.performance import evaluate_strategy_performance
from src.logger_config import setup_logger
import polars as pl
import random
import os
import gc
import itertools
import argparse
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
        df_din = fetch_taq_data(
            tickers=[ticker],
            exchanges=args.exchanges,
            quote_conds=args.quote_cond,
            start_date=args.start_date,
            end_date=args.end_date,
            wrds_username=args.wrds_username
        )
        while curr_date <= end_date:
            
            date_str = curr_date.strftime('%Y-%m-%d')
            df = df_din.filter(
                (pl.col("sym_root") == ticker) &
                (pl.col("date") == date_str)
            )
            curr_date += timedelta(days=1)
            output_dir = f"data/parquet/{ticker}"
            file_path = f"{output_dir}/{ticker}_{date_str}.parquet"
            if not os.path.exists(file_path):
                logger.info(f"Data for {ticker} on {date_str} not found. Fetching data...")
                
                write_parquet(df, ticker, date_str, output_dir)
            else:
                logger.info(f"Data for {ticker} on {date_str} already exists. Skipping fetch.")

    logger.info(f"Fetching data from {args.start_date} to {args.end_date} for tickers: {args.tickers}")
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
        # obi_strategy = OBIVWAPStrategy(
        #     vwap_window=500,
        #     momentum_window=100,
        #     volatility_window=50,
        #     trend_window=20,
        #     max_position=100,
        #     start_time=start_time,
        #     end_time=end_time,
        #     logger=logger
        # )
        # portfolio.add_strategy("OBI-VWAP", obi_strategy, weight=0.3)
        
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
        
        # inverse_obi_strategy = InverseOBIVWAPStrategy(
        #     vwap_window=500,
        #     momentum_window=100,
        #     volatility_window=50,
        #     trend_window=20,
        #     max_position=100,
        #     start_time=start_time,
        #     end_time=end_time,
        #     logger=logger
        # )
        # portfolio.add_strategy("Inverse-OBI-VWAP", inverse_obi_strategy, weight=0.3)
        logger.info(f"Processing {ticker}...")
        ticker_data = df.filter(pl.col("sym_root") == ticker)
        logger.info(f"Data for {ticker} contains {ticker_data.shape[0]} records")

        # Run backtest for the portfolio
        logger.info(f"Running backtest for {ticker}...")

        days = ticker_data.select(pl.col("date").unique().sort()).to_series().to_list()
        results = portfolio.run_backtest(ticker_data, days=days)
        all_results[ticker] = results
        
        # Save backtest results for each strategy
        date_str = args.start_date
        output_dir = f"data/backtest_results/{ticker}"
        os.makedirs(output_dir, exist_ok=True)

        # Save individual strategy results
        # df is a Polars DataFrame, convert to CSV
        # the strat names are results['OBI-VWAP'], results['Mean-Reversion'], results['Inverse-OBI-VWAP]
        # for name in ['OBI-VWAP', 'Mean-Reversion', 'Inverse-OBI-VWAP']:
        #     if name not in results:
        #         logger.warning(f"Strategy {name} not found in results for {ticker}. Skipping save.")
        #         continue
        #     strategy_results = results[name]
        #     if strategy_results is not None:
        #         strategy_df = pl.DataFrame(strategy_results)
        #         strategy_df.write_csv(f"{output_dir}/{name}_results_{date_str}.csv")
        #         logger.info(f"Saved {name} results for {ticker} to {output_dir}/{name}_results_{date_str}.csv")
        
        # Log portfolio metrics
        # logger.info(f"\nPortfolio Performance for {ticker}:")
        # logger.info(f"Total Return: {results['Metrics']['Total_Return']:.2f}%")
        # logger.info(f"Sharpe Ratio: {results['Metrics']['Sharpe_Ratio']:.4f}")
        # logger.info(f"Sortino Ratio: {results['Metrics']['Sortino_Ratio']:.4f}")
        # logger.info(f"Maximum Drawdown: {results['Metrics']['Max_Drawdown']:.2f}%")

        # if results['Metrics']['Total_Return'] > 0:
        #     positive_return_tickers.append(ticker)
        
        

    # Append to positive return tickers file
    # with open("data/positive_return_tickers.txt", "a") as f:
    #     for t in positive_return_tickers:
    #         f.append(f"{t}\n")

    del df
    gc.collect()


