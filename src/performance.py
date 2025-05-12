import polars as pl
import numpy as np

def calculate_performance_metrics(account_balance):
    """
    Calculate and print performance metrics for a given account balance series.

    Args:
        account_balance (list or np.array): Account balance over time.

    Returns:
        None
    """
    # Convert to numpy array for calculations
    account_balance = np.array(account_balance)
    
    # Calculate daily returns
    daily_returns = np.diff(account_balance) / account_balance[:-1]
    
    # Calculate metrics
    total_returns = (account_balance[-1] / account_balance[0] - 1) * 100
    drawdown = account_balance / np.maximum.accumulate(account_balance) - 1
    max_drawdown = np.min(drawdown) * 100

    # Print metrics
    print("\nPERFORMANCE STATISTICS:")
    print(f"Total returns: {total_returns}%")
    print(f"Max drawdown: {max_drawdown}%")

def calculate_daily_sharpe_ratio(backtest_data):
    """
    Resample account balance to 1-minute intervals, compute log returns, and calculate Sharpe ratio per day.

    Args:
        backtest_data (pl.DataFrame): Polars DataFrame containing "Timestamp" and "Account_Balance".

    Returns:
        pl.DataFrame: DataFrame with daily Sharpe ratios.
    """
    # Combine DATE and TIME_M into a single datetime column
    df = backtest_data.with_columns(
    (pl.col("DATE").cast(pl.Utf8) + " " + pl.col("TIME_M").cast(pl.Utf8))
    .str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f")
    .alias("Timestamp")
)

    # Resample to 1-minute intervals and forward-fill missing values
    df = df.group_by_dynamic("Timestamp", every="1m", closed="right").agg(
        pl.col("Account_Balance").last()
    ).fill_null(strategy="forward")

    # Compute log returns
    df = df.with_columns(
        (pl.col("Account_Balance") / pl.col("Account_Balance").shift(1)).log().alias("Log_Returns")
    )

    # Group by day and calculate Sharpe ratio
    daily_sharpe = (
        df.group_by_dynamic("Timestamp", every="1d")
        .agg(
            (pl.col("Log_Returns").mean() / pl.col("Log_Returns").std() * np.sqrt(390 ** 0.5)) # 390 trading minutes in a day
            .alias("Daily_Sharpe_Ratio")
        )
    )

    # Print daily Sharpe ratios
    print("\nDAILY SHARPE RATIOS:")
    print(daily_sharpe)

    return daily_sharpe