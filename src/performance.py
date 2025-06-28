import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from scipy.stats import skew, kurtosis
# data =pd.DataFrame({
#                     "Date": [day],
#                     "Intraday Sharpe Ratio": [intraday_sharpe_ratio],
#                     "Intraday Profit": [intraday_profit],
#                     "Intraday Drawdown": [intraday_drawdown],
#                     "Intraday Max Return": [intraday_max_return],
#                     "Intraday Final Balance": [intraday_final_balance]
#                 })
def evaluate_strategy_performance(data: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Evaluate the performance of a strategy based on its final balance.

    Args:
        data (pd.DataFrame): DataFrame containing the strategy's performance data, including a column for 'Intraday Final Balance'.
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Returns:
        Dict: Dictionary of performance metrics.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    metrics = {}

    if 'Intraday Final Balance' not in data.columns:
        logger.error("Intraday Final Balance column not found in the data.")
        return metrics

    final_balances = data['Intraday Final Balance'].values
    
    # Calculate returns
    returns = np.diff(final_balances)
    
    if len(returns) == 0:
        logger.warning("No returns to evaluate.")
        metrics['error'] = "No returns to evaluate"
        return metrics

    # Calculate total return
    total_return = final_balances[-1] - final_balances[0]
    metrics['total_return'] = total_return

    # Calculate annualized return (assuming daily returns)
    annualized_return = (1 + total_return)**(252/len(returns)) - 1
    metrics['annualized_return'] = annualized_return

    # Calculate Sharpe Ratio (assuming risk-free rate of 0)
    try:
        sharpe_ratio = np.mean(returns) / np.std(returns)
    except ZeroDivisionError:
        sharpe_ratio = 0
    metrics['sharpe_ratio'] = sharpe_ratio

    # Calculate Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    try:
        downside_deviation = np.std(downside_returns)
    except ZeroDivisionError:
        downside_deviation = 0
    if downside_deviation != 0:
        sortino_ratio = (np.mean(returns) - 0) / downside_deviation 
    else:
        sortino_ratio = np.nan
    metrics['sortino_ratio'] = sortino_ratio

    # Calculate maximum drawdown
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    try:
        drawdown = (cumulative_returns - peak) / peak
    except ZeroDivisionError:
        drawdown = (cumulative_returns - peak) / (peak + 1e-2)  # Avoid division by zero
        logger.error("Zero division error in drawdown calculation.")
    max_drawdown = np.min(drawdown)
    metrics['max_drawdown'] = max_drawdown

    # Calculate skewness and kurtosis
    try:
        skewness = skew(returns)
    except Exception as e:
        logger.error(f"Error calculating skewness: {e}")
        skewness = np.nan
    try:
        kurt = kurtosis(returns)
    except Exception as e:
        logger.error(f"Error calculating kurtosis: {e}")
        kurt = np.nan
    metrics['skewness'] = skewness
    metrics['kurtosis'] = kurt
    
    logger.info(f"Performance metrics: {metrics}")
    return metrics