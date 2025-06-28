import polars as pl
import logging
import numpy as np
import math
from typing import Optional
from datetime import datetime, timedelta

def log_backtest_summary(strategy_name: str, account_balance: list, 
                         logger: logging.Logger, num_ticks: int, df,  date: str) -> dict:
    """Logs summary metrics adjusted for tick-based Sharpe."""
    
    # Calculate metrics based on StrategyPortfolio implementation
    initial_balance = account_balance[0] if account_balance else 300_000
    final_balance = account_balance[-1] if account_balance else initial_balance
    avg_return = final_balance / initial_balance - 1

    # calculate minute by  minute returns
    # group by minute bins and calculate returns
    df = df.with_columns([
        (date + pl.col("Time").cast(pl.Utf8))
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.6f")
        .alias("timestamp")
    ])
    start_time = df.select(pl.col("timestamp").min()).item()
    end_time = df.select(pl.col("timestamp").max()).item()

    df = df.with_columns(
        pl.col("timestamp").dt.truncate("1m").alias("minute")
    )

    df = df.group_by("minute").agg(
        pl.col("Account_Balance").last().alias("minute_return")
    ).sort("minute")

    df = df.with_columns(
        (pl.col("minute_return") / pl.col("minute_return").shift(1) - 1).alias("returns")
    ).drop_nulls(subset=["returns"])

    if df.shape[0] > 1:
        try:
            sharpe_ratio = df["returns"].mean() / (df["returns"].std() + 0.0001)
        except ZeroDivisionError:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0


    logger.info(f"{'='*50}")
    logger.info(f"Backtest Summary for {strategy_name} : Intraday")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"Final Balance: ${final_balance:,.2f}")
    logger.info(f"Number of Ticks: {num_ticks}")
    logger.info(f"Average Return: {avg_return:.4%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    # start time from polars


    logger.info(f"Start Time: {start_time}")
    logger.info(f"End Time: {end_time}")
    logger.info(f"Average Trade Return: {df['returns'].mean():.4%}")
    logger.info(f"Max Drawdown: {df['returns'].min():.4%}")
    logger.info(f"Max Return: {df['returns'].max():.4%}")
    logger.info(f"Min Return: {df['returns'].min():.4%}")
    logger.info(f"Total Profit: ${final_balance - initial_balance:,.2f}")


    logger.info(f"{'='*50}\n")
    return {
        "sharpe_ratio": sharpe_ratio,
        "profit": final_balance - initial_balance,
        "drawdown": df["returns"].min(),
        "max_return": df["returns"].max(),
        "final_balance": final_balance,
        
    }




class StrategyPortfolio:
    """
    Manages multiple trading strategies simultaneously, each with its own portfolio.
    
    This class allows running and tracking multiple strategies in parallel, with each
    strategy maintaining its own position, cash balance, and performance metrics.
    
    Attributes:
        strategies (dict): Dictionary mapping strategy names to strategy instances
        initial_cash (float): Initial cash balance for each strategy
        portfolio_weights (dict): Dictionary mapping strategy names to their portfolio weights
        rebalance_threshold (float): Threshold for portfolio rebalancing
        last_rebalance_date (datetime): Date of last portfolio rebalancing
        
    Methods:
        add_strategy(name: str, strategy: BaseStrategy, weight: float = None):
            Adds a new strategy to the portfolio
        remove_strategy(name: str):
            Removes a strategy from the portfolio
        rebalance_portfolio():
            Rebalances the portfolio according to target weights
        run_backtest(data: pl.DataFrame) -> dict:
            Runs backtest for all strategies and returns combined results
        get_portfolio_performance() -> dict:
            Returns combined portfolio performance metrics
    """
    
    def __init__(self, initial_cash: float = 1_000_000, rebalance_threshold: float = 0.1):
        """Initialize the strategy portfolio."""
        self.strategies = {}
        self.initial_cash = initial_cash
        self.portfolio_weights = {}

        
    def add_strategy(self, name: str, strategy: 'OBIVWAPStrategy', weight: float = None):
        """Add a new strategy to the portfolio."""
        if name in self.strategies:
            raise ValueError(f"Strategy {name} already exists in portfolio")
            
        # Initialize strategy with its portion of initial cash
        if weight is not None:
            strategy_cash = self.initial_cash * weight
            strategy.cash = strategy_cash
            self.portfolio_weights[name] = weight
        else:
            # Equal weight if not specified
            strategy_cash = self.initial_cash / (len(self.strategies) + 1)
            strategy.cash = strategy_cash
            self.portfolio_weights[name] = 1.0 / (len(self.strategies) + 1)
            
        self.strategies[name] = strategy
        
            
    def remove_strategy(self, name: str):
        """Remove a strategy from the portfolio."""
        if name not in self.strategies:
            raise ValueError(f"Strategy {name} not found in portfolio")
            
        # Close positions and redistribute cash
        strategy = self.strategies[name]
        remaining_cash = strategy.cash + (strategy.position * strategy.entry_price if strategy.position != 0 else 0)
        
        del self.strategies[name]
        del self.portfolio_weights[name]
        
        # Redistribute cash to remaining strategies
        if self.strategies:
            cash_per_strategy = remaining_cash / len(self.strategies)
            for s in self.strategies.values():
                s.cash += cash_per_strategy

                
    def run_backtest(self, data: pl.DataFrame, days) -> dict:
        """Run backtest for all strategies and return combined results."""
        results = {}
        df = {}
        portfolio_values = []
        
        # Run backtest for each strategy
        for name, strategy in self.strategies.items():
            strategy_data = data.clone()
            df[name],results[name] = strategy.backtest(strategy_data, days)

        return df, results
        
