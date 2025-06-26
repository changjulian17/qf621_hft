import polars as pl
import logging
import numpy as np
import math
from typing import Optional
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from ..strategy import StrategyPortfolio, log_backtest_summary

class MeanReversionStrategy:
    """
    Implements a mean reversion strategy based on price deviations from VWAP.
    
    This strategy generates trading signals when price deviates significantly from VWAP
    and shows signs of mean reversion, using volume and volatility filters.
    
    Attributes:
        vwap_window (int): Rolling window size for VWAP calculation
        deviation_threshold (float): Threshold for price deviation from VWAP
        volatility_window (int): Window size for volatility calculation
        volume_window (int): Window size for volume analysis
        max_position (int): Maximum position size
        stop_loss_pct (float): Stop loss percentage
        profit_target_pct (float): Profit target percentage
        risk_per_trade (float): Percentage of portfolio to risk per trade
        min_profit_threshold (float): Minimum expected profit
        start_time (tuple): Earliest time for generating signals
        end_time (tuple): Latest time for generating signals
    """
    # (self, vwap_window: int = 20, deviation_threshold: float = 0.0001,
    #              volatility_window: int = 20, volume_window: int = 20,
    #              max_position: int = 100, stop_loss_pct: float = 0.3,
    #              profit_target_pct: float = 0.6, 
    #              risk_per_trade: float = 0.02, min_profit_threshold: float = 0.001,
    #              start_time: tuple = (9, 30, 865), end_time: tuple = (16, 28, 954),
    #              logger: Optional[logging.Logger] = None)
    def __init__(self, vwap_window: int = 20, 
                 volatility_window: int = 20, volume_window: int = 20,
                 max_position: int = 100, stop_loss_pct: float = 1,
                 profit_target_pct: float = 2, 
                 risk_per_trade: float = 0.02, 
                 start_time: tuple = (9, 30, 865), end_time: tuple = (16, 28, 954),
                 logger: Optional[logging.Logger] = None):
        """Initialize the mean reversion strategy."""
        self.vwap_window = vwap_window
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.risk_per_trade_pct = risk_per_trade
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.start_time = start_time
        self.end_time = end_time
        self.position_hold_time = 0
        self.max_hold_time = 100
        self.logger = logger or logging.getLogger(__name__)

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate 
        1. Mid price
        3. Volume
        2. VWAP
        3. price deviation from mid price and VWAP
        ."""

        df = df.with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("MID_PRICE"),
            (pl.col("bidsiz") + pl.col("asksiz")).alias("Volume")
        )
        
        # Calculate VWAP
        df = df.with_columns(
            (
                (pl.col("MID_PRICE") * pl.col("Volume")).rolling_sum(window_size=self.vwap_window)
                / pl.col("Volume").rolling_sum(window_size=self.vwap_window)
            ).alias("VWAP")
        )
        
        # Calculate price deviation from VWAP
        df = df.with_columns(
            ((pl.col("MID_PRICE") - pl.col("VWAP")) / pl.col("VWAP")).alias("Price_Deviation")
        )
        
        return df
        
    def calculate_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate volatility indicators including:
        1. Volatility, Price volatility (rolling mean of absolute price changes using mid price)
        2. Volume_ratio Volume trend (rolling mean and ratio) using volume in order book
        
        """
        # Price volatility
        df = df.with_columns(
            pl.col("MID_PRICE").pct_change()
            .abs()
            .rolling_mean(window_size=self.volatility_window)
            .alias("Volatility")
        )
        
        # Volume trend
        df = df.with_columns(
            pl.col("Volume").rolling_mean(window_size=self.volume_window).alias("Volume_MA"),
            (pl.col("Volume") / pl.col("Volume").rolling_mean(window_size=self.volume_window)).alias("Volume_Ratio")
        )
        
        return df
        
    def calculate_mean_reversion_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate mean reversion indicators."""
        # RSI-like indicator for price deviation
        df = df.with_columns(
            pl.col("Price_Deviation").rolling_mean(window_size=self.volatility_window).alias("Deviation_MA")
        )
        
        # Mean reversion score with a small constant added to the denominator to avoid division by zero
        df = df.with_columns(
            (
                (pl.col("Price_Deviation") - pl.col("Deviation_MA")) /
                (pl.col("Volatility") + 1e-6)
            ).alias("Mean_Reversion_Score")
        )
        
        return df
            
    def calculate_position_size(self, price: float, current_balance) -> int:
        """Calculate optimal position size based on risk and volatility."""
        # 1% of current_balance
        risk_amount = current_balance * self.risk_per_trade_pct

        
        position_size = int((risk_amount / price) / 5) 
        if position_size > self.max_position:
            # logging.warning(f"Position size {position_size} exceeds max position {self.max_position}. Capping to max position.")
            return self.max_position
        elif position_size < 1:
            # logging.warning(f"Position size {position_size} is less than 1. Setting to 1.")
            return 1
        return position_size
    
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals based on mean reversion indicators."""
        df = self.calculate_vwap(df)
        df = self.calculate_volatility(df)
        df = self.calculate_mean_reversion_score(df)

        # Generate signals with tightened conditions
        df = df.with_columns(
            pl.when(
                # Long signal conditions (tightened)
                (pl.col("Mean_Reversion_Score") < -0.5) &  # Tightened mean reversion signal
                (pl.col("Volume_Ratio") > 1.0) &  # Relaxed volume condition
                (pl.col("Volatility") > 0.001)  # Minimum volatility threshold
            )
            .then(1)
            .when(
                # Short signal conditions (tightened)
                (pl.col("Mean_Reversion_Score") > 0.5) &  # Tightened mean reversion signal
                (pl.col("Volume_Ratio") > 1.0) &  # Relaxed volume condition
                (pl.col("Volatility") > 0.001)  # Minimum volatility threshold
            )
            .then(-1)        
            .otherwise(0)
            .alias("Signal")
        )
        
        # self.logger.info signal distribution
        signal_counts = df.group_by("Signal").count()
        self.logger.info("Signal Distribution:")
        self.logger.info(signal_counts)
        
        # Filter signals by trading hours
        df = df.with_columns(
            pl.when(
                (pl.col("time_m") < pl.time(*self.start_time)) | 
                (pl.col("time_m") > pl.time(*self.end_time))
            )
            .then(pl.lit(0))
            .otherwise(pl.col("Signal"))
            .alias("Signal")
        )
        
        return df




        
    def backtest(self, df_fin: pl.DataFrame, days) -> pl.DataFrame:
        """Run backtest simulation and return performance metrics and all indicators for plotting."""
        # Calculate all indicators and add them to the DataFrame before running the backtest loop
        self.logger.info(f"Starting backtest for MeanReversionStrategy with initial cash: ${self.cash:,.2f}")
        self.cash = 300_000  # Reset cash for each backtest
        final_returns = []
        final_df = []

        for date in days:
            # Filter DataFrame for the current date
            df = df_fin.filter(pl.col("date") == date)
            self.position_hold_time = 0
            self.position = 0
            self.entry_price = 0
            
            df = self.generate_signals(df)

            # Prepare lists for tracking
            account_balance = []
            positions = []
            trades = []
            entry_prices = []
            trade_markers = []  # 1 for entry, -1 for exit, 0 for no trade
            stop_loss_hits = []
            take_profit_hits = []
            max_hold_time_hits = []
            position_sizes = []
            time_st = []
            last_signal = 0

            for i, row in enumerate(df.iter_rows(named=True)):
                # 1% of account_balance
                mid_price = row["MID_PRICE"]
                pos_size = self.calculate_position_size(mid_price, self.cash)

                signal = row["Signal"]
                time_st.append(row["time_m"])

                # Trading cost per trade
                trading_cost = 0.05

                # Default trade marker and event flags
                trade_marker = 0
                stop_loss_hit = 0
                take_profit_hit = 0
                max_hold_time_hit = 0
                
                if self.position > 0:
                    unrealized_pnl = (row["bid"] - self.entry_price) * self.position
                elif self.position < 0:
                    unrealized_pnl = (self.entry_price - row["ask"]) * abs(self.position)
                else:
                    unrealized_pnl = 0
                unrealized_pnl_pct =  unrealized_pnl / (self.entry_price * abs(self.position)) if self.position != 0 else 0

                # Process existing position
                if self.position != 0 and self.entry_price not in (None, 0):
                    self.position_hold_time += 1
                    # Max hold time
                    if self.position_hold_time >= self.max_hold_time:
                        if self.position > 0:
                            self.cash += row["bid"] * self.position - trading_cost
                            trades.append(("Max Hold Time Hit", self.position, self.entry_price, row["bid"]))
                        else:
                            self.cash -= row["ask"] * abs(self.position) + trading_cost
                            trades.append(("Max Hold Time Hit", self.position, self.entry_price, row["ask"]))
                        trade_marker = -1
                        max_hold_time_hit = 1
                        self.position = 0
                        self.entry_price = 0
                        self.position_hold_time = 0
                    # Stop loss
                    elif (self.position > 0 and unrealized_pnl_pct <= -self.stop_loss_pct/100) or \
                        (self.position < 0 and unrealized_pnl_pct >= self.stop_loss_pct/100):
                        if self.position > 0:
                            self.cash -= row["ask"] * self.position + trading_cost
                            trades.append(("Stop Loss", self.position, self.entry_price, row["ask"]))
                        else:
                            self.cash += row["bid"] * abs(self.position) - trading_cost
                            trades.append(("Stop Loss", self.position, self.entry_price, row["bid"]))
                        trade_marker = -1
                        stop_loss_hit = 1
                        self.position = 0
                        self.entry_price = 0
                        self.position_hold_time = 0
                    # Take profit
                    elif (self.position > 0) or \
                        (self.position < 0):
                        if self.position > 0:
                            self.cash += row["bid"] * self.position - trading_cost
                            trades.append(("Take Profit", self.position, self.entry_price, row["bid"]))
                        else:
                            self.cash -= row["ask"] * abs(self.position) + trading_cost
                            trades.append(("Take Profit", self.position, self.entry_price, row["ask"]))
                        trade_marker = -1
                        take_profit_hit = 1
                        self.position = 0
                        self.entry_price = 0
                        self.position_hold_time = 0

                # Process new signals
                if signal != 0 and signal != last_signal:
                    if signal == 1 and self.position <= 0:
                        # Close any existing short position
                        if self.position < 0:
                            self.cash -= row["ask"] * abs(self.position) + trading_cost
                            trades.append(("Close Short", self.position, self.entry_price, row["ask"]))
                            trade_marker = -1
                            self.position = 0
                            self.entry_price = 0
                            self.position_hold_time = 0
                        # Open long position
                        if pos_size > 0 and self.cash >= row["ask"] * pos_size + trading_cost:
                            self.cash -= row["ask"] * pos_size + trading_cost
                            self.position = pos_size
                            self.entry_price = row["ask"]
                            trades.append(("Buy", pos_size, row["ask"], None))
                            trade_marker = 1
                            self.position_hold_time = 0
                    elif signal == -1 and self.position >= 0:
                        # Close any existing long position
                        if self.position > 0:
                            self.cash += row["bid"] * self.position - trading_cost
                            trades.append(("Close Long", self.position, self.entry_price, row["bid"]))
                            trade_marker = -1
                            self.position = 0
                            self.entry_price = 0
                            self.position_hold_time = 0
                        # Open short position
                        if pos_size > 0:
                            self.cash += row["bid"] * pos_size - trading_cost
                            self.position = -pos_size
                            self.entry_price = row["bid"]
                            trades.append(("Sell", -pos_size, row["bid"], None))
                            trade_marker = 1
                            self.position_hold_time = 0
                # Close all positions if signal is 0 and you have an open position
                elif signal == 0 and self.position != 0:
                    if self.position > 0:
                        self.cash += row["bid"] * self.position - trading_cost
                        trades.append(("Close Long (Signal 0)", self.position, self.entry_price, row["bid"]))
                        trade_marker = -1
                    elif self.position < 0:
                        self.cash -= row["ask"] * abs(self.position) + trading_cost
                        trades.append(("Close Short (Signal 0)", self.position, self.entry_price, row["ask"]))
                        trade_marker = -1
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0

                # Update tracking variables
                if self.position != 0:
                    # calculate balance with current position using bid/ask price
                    if self.position > 0:
                        current_balance = self.cash + (self.position * row["bid"])
                    else:
                        current_balance = self.cash + (self.position * row["ask"])
                else:
                    current_balance = self.cash
                account_balance.append(current_balance)
                positions.append(self.position)
                trade_markers.append(trade_marker)
                stop_loss_hits.append(stop_loss_hit)
                take_profit_hits.append(take_profit_hit)
                max_hold_time_hits.append(max_hold_time_hit)
                position_sizes.append(pos_size)
                entry_prices.append(self.entry_price)
                last_signal = signal

            # --- Ensure all positions are closed at EOD ---
            if self.position != 0:
                # Use the last row's bid/ask for closing
                last_row = row
                if self.position > 0:
                    self.cash += last_row["bid"] * self.position - trading_cost
                    trades.append(("EOD Close Long", self.position, self.entry_price, last_row["bid"]))
                    trade_marker = -1
                elif self.position < 0:
                    self.cash -= last_row["ask"] * abs(self.position) + trading_cost
                    trades.append(("EOD Close Short", self.position, self.entry_price, last_row["ask"]))
                    trade_marker = -1
                self.position = 0
                self.entry_price = 0
                self.position_hold_time = 0
                # Instead of appending, update the last values in the tracking lists
                if len(account_balance) > 0:
                    account_balance[-1] = self.cash
                    positions[-1] = 0
                    trade_markers[-1] = trade_marker
                    stop_loss_hits[-1] = 0
                    take_profit_hits[-1] = 0
                    max_hold_time_hits[-1] = 0
                    position_sizes[-1] = 0
                    entry_prices[-1] = 0
                    time_st[-1] = last_row["time_m"]

            # Add all tracked metrics and indicators to DataFrame for plotting
            metrics_df = pl.DataFrame({
                "Account_Balance": [float(x) for x in account_balance],
                "Time": time_st,
                "Position": [float(x) for x in positions],
                "Entry_Price": [float(x) if x is not None else float('nan') for x in entry_prices],
                "Position_Size": [float(x) for x in position_sizes],
                "Trade_Marker": [float(x) for x in trade_markers],
                "Stop_Loss_Hit": [float(x) for x in stop_loss_hits],
                "Take_Profit_Hit": [float(x) for x in take_profit_hits],
                "Max_Hold_Time_Hit": [float(x) for x in max_hold_time_hits]
            })
            df = df.hstack(metrics_df)


            #convert Time to  datetime
            df = df.with_columns(
                pl.col("Time").str.strptime(pl.Time, format="%H:%M:%S%.6f")
            )


            # Log summary using the common format
            intraday_info = log_backtest_summary(
                strategy_name="Mean Reversion Strategy",
                account_balance=account_balance,
                logger=self.logger,
                num_ticks=len(df),
                df=df,
                date=date
            )
            final_returns.append(intraday_info)
            final_df.append(df)
        
        return final_df, final_returns


