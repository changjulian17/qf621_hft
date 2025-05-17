import polars as pl
import numpy as np

def evaluate_strategy_performance(backtest_data: pl.DataFrame) -> dict:
    """
    Evaluates the performance of a trading strategy, including key metrics 
    and daily Sharpe ratios.

    Args:
        backtest_data (pl.DataFrame): 
            A Polars DataFrame containing columns "Account_Balance", "DATE", 
            and "TIME_M".

    Returns:
        dict: 
            A dictionary containing performance metrics such as total returns, 
            max drawdown, and daily Sharpe ratios.
    """
    # Ensure "Account_Balance" exists in the DataFrame
    if "Account_Balance" not in backtest_data.columns:
        raise ValueError("The DataFrame must contain an 'Account_Balance' column.")

    # Calculate total returns and max drawdown
    account_balance = backtest_data["Account_Balance"].to_numpy()
    total_returns = (account_balance[-1] / account_balance[0] - 1) * 100
    drawdown = account_balance / np.maximum.accumulate(account_balance) - 1
    max_drawdown = np.min(drawdown) * 100

    # Use the existing Timestamp column and ensure it's sorted
    df = backtest_data.sort("Timestamp")

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

    # Print performance metrics
    print("\nPERFORMANCE STATISTICS:")
    print(f"Total returns: {total_returns:.2f}%")
    print(f"Max drawdown: {max_drawdown:.2f}%")

    # Print daily Sharpe ratios
    print("\nDAILY SHARPE RATIOS:")
    print(daily_sharpe)

    # Return metrics as a dictionary
    return {
        "Total_Returns": total_returns,
        "Max_Drawdown": max_drawdown,
        "Daily_Sharpe_Ratios": daily_sharpe,
    }