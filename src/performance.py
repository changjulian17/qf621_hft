import polars as pl
import numpy as np
from typing import Dict, Optional
import logging
import re

def evaluate_strategy_performance(backtest_data: pl.DataFrame, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Evaluate the performance of a trading strategy based on backtest data.

    Args:
        backtest_data (pl.DataFrame): DataFrame containing backtest results
        logger (logging.Logger, optional): Logger instance for output

    Returns:
        dict: 
            A dictionary containing performance metrics such as total returns, 
            max drawdown, daily Sharpe ratios, average bid-ask spread, and average Sharpe.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

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
    # Calculate average bid-ask spread as a liquidity metric
    if "bid" in backtest_data.columns and "ask" in backtest_data.columns:
        avg_bid_ask_spread = (backtest_data["ask"] - backtest_data["bid"]).mean()
    else:
        avg_bid_ask_spread = None

    # Helper to extract average Sharpe
    def extract_avg_sharpe(metrics):
        if "Sharpe" in metrics:
            return metrics["Sharpe"]
        dsr = metrics.get("Daily_Sharpe_Ratios")
        if dsr is None:
            return None
        try:
            if hasattr(dsr, "to_pandas"):
                df = dsr.to_pandas()
                return df["Daily_Sharpe_Ratio"].mean()
            elif isinstance(dsr, str):
                vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", dsr)]
                if vals:
                    return sum(vals) / len(vals)
            elif hasattr(dsr, "__iter__"):
                vals = list(dsr)
                if vals and hasattr(vals[0], "get"):
                    vals = [v.get("Daily_Sharpe_Ratio", 0) for v in vals]
                return sum(vals) / len(vals)
        except Exception:
            return None
        return None

    # Print performance metrics
    ticker = backtest_data["sym_root"].unique()[0]
    print(f"\nPERFORMANCE {ticker} STATISTICS:")
    print(f"Total returns: {total_returns:.2f}%")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    if avg_bid_ask_spread is not None:
        print(f"Average bid-ask spread: {avg_bid_ask_spread:.6f}")
    else:
        print("Average bid-ask spread: N/A (bid/ask columns missing)")

    # Print daily Sharpe ratios
    print("\nDAILY SHARPE RATIOS:")
    print(daily_sharpe)

    avg_sharpe = extract_avg_sharpe({"Daily_Sharpe_Ratios": daily_sharpe})

    # Return metrics as a dictionary
    return {
        "Total_Returns": total_returns,
        "Max_Drawdown": max_drawdown,
        "Daily_Sharpe_Ratios": daily_sharpe,
        "Average_Bid_Ask_Spread": avg_bid_ask_spread,
        "Average_Sharpe": avg_sharpe,
    }