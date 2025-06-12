import polars as pl
import numpy as np
from typing import Dict, Optional
import logging

def evaluate_strategy_performance(backtest_data: pl.DataFrame, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Evaluate the performance of a trading strategy based on backtest data.

    Args:
        backtest_data (pl.DataFrame): DataFrame containing backtest results
        logger (logging.Logger, optional): Logger instance for output

    Returns:
        dict: Dictionary containing performance metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

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
            (pl.col("Log_Returns").mean() / pl.col("Log_Returns").std() * np.sqrt(390 ** 0.5))
            .alias("Daily_Sharpe_Ratio")
        )
    )

    # Calculate total returns and max drawdown
    total_returns = ((df["Account_Balance"].tail(1).item() / df["Account_Balance"].head(1).item()) - 1) * 100
    
    peak = df["Account_Balance"].cum_max()
    drawdown = ((df["Account_Balance"] - peak) / peak) * 100
    max_drawdown = drawdown.min()

    # Log performance statistics
    logger.info("\nPERFORMANCE STATISTICS:")
    logger.info(f"Total returns: {total_returns:.2f}%")
    logger.info(f"Max drawdown: {max_drawdown:.2f}%")

    # Log daily Sharpe ratios
    logger.info("\nDAILY SHARPE RATIOS:")
    logger.info(str(daily_sharpe))

    # Return metrics as a dictionary
    return {
        "Total_Returns": total_returns,
        "Max_Drawdown": max_drawdown,
        "Daily_Sharpe_Ratios": daily_sharpe,
    }