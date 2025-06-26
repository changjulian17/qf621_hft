import polars as pl
import logging
import numpy as np
import math
from typing import Optional
from datetime import datetime, timedelta
from ..strategy import StrategyPortfolio, log_backtest_summary


class InverseOBIVWAPStrategy:
    """
    Implements an inverted Order Book Imbalance (OBI) and VWAP strategy.
    This strategy inverts the volume signal, assuming that high volume might signal
    reversals rather than continuations.
    """
    def __init__(self, vwap_window: int = 500, obi_window: int = 5000,
                 price_impact_window: int = 50, momentum_window: int = 100, volatility_window: int = 50,
                 trend_window: int = 20, max_position: int = 1000,
                 stop_loss_pct: float = 1, profit_target_pct: float = 1.0, risk_per_trade: float = 1,
                 obi_threshold: float = 0.1,
                 start_time: tuple = (9, 30, 865),
                 end_time: tuple = (16, 28, 954),
                 logger: Optional[logging.Logger] = None):
        """Enhanced initialization with microstructure indicators."""
        self.vwap_window = vwap_window
        self.price_impact_window = price_impact_window
        self.obi_window = obi_window
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.risk_per_trade = risk_per_trade
        self.obi_threshold = obi_threshold
        self.cash = 300_000
        self.position = 0
        self.entry_price = 0
        self.start_time = start_time
        self.end_time = end_time
        self.logger = logger or logging.getLogger(__name__)

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP, adaptive bands, and Inverse VWAP."""
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
        
        # Calculate volume-weighted price variance
        df = df.with_columns(
            (
                ((pl.col("MID_PRICE") - pl.col("VWAP")) ** 2 * pl.col("Volume"))
                .rolling_sum(window_size=self.vwap_window)
                / pl.col("Volume").rolling_sum(window_size=self.vwap_window)
            ).sqrt().alias("VWAP_STD")
        )
        
        # Adaptive VWAP bands based on volume profile
        df = df.with_columns(
            (pl.col("VWAP") + pl.col("VWAP_STD") * 
             (1 + pl.col("Volume") / pl.col("Volume").rolling_mean(window_size=self.vwap_window))
            ).alias("VWAP_Upper"),
            
            (pl.col("VWAP") - pl.col("VWAP_STD") * 
             (1 + pl.col("Volume") / pl.col("Volume").rolling_mean(window_size=self.vwap_window))
            ).alias("VWAP_Lower")
        )
        
        # --- Inverse VWAP Calculation ---
        # Calculate rolling median of VWAP
        df = df.with_columns(
            pl.col("VWAP").rolling_median(window_size=self.vwap_window, min_periods=1).alias("Rolling_Median_VWAP")
        )

        # Calculate inverted VWAP: median + (median - VWAP)
        df = df.with_columns(
            (pl.col("Rolling_Median_VWAP") + (pl.col("Rolling_Median_VWAP") - pl.col("VWAP")))
            .alias("Inverse_VWAP")
        )
        
        return df

    def calculate_price_impact_and_inverse_volume(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate price impact, market depth, and the inverted volume normalization."""
        # Calculate bid-ask spread
        df = df.with_columns(
            (pl.col("ask") - pl.col("bid")).alias("Spread"),
            ((pl.col("ask") - pl.col("bid")) / pl.col("MID_PRICE")).alias("Relative_Spread")
        )
        
        # Calculate order book pressure
        df = df.with_columns(
            (pl.col("bidsiz") / (pl.col("bidsiz") + pl.col("asksiz"))).alias("Bid_Pressure"),
            (pl.col("Spread") * pl.col("Volume")).alias("Dollar_Volume")
        )
        
        # Calculate market depth ratios
        df = df.with_columns(
            (pl.col("bidsiz") / pl.col("bidsiz").rolling_mean(window_size=100)).alias("Bid_Depth_Ratio"),
            (pl.col("asksiz") / pl.col("asksiz").rolling_mean(window_size=100)).alias("Ask_Depth_Ratio")
        )
        
        # Calculate price impact score
        df = df.with_columns(
            (
                pl.col("Dollar_Volume").rolling_sum(window_size=self.price_impact_window) /
                pl.col("Volume").rolling_sum(window_size=self.price_impact_window)
            ).alias("Price_Impact")
        )
        
        # --- Inverse Volume Calculation ---
        # Calculate rolling median of volume
        df = df.with_columns(
            pl.col("Volume").rolling_median(window_size=self.obi_window, min_periods=1).alias("Rolling_Median_Volume")
        )

        # Calculate inverted volume: median + (median - volume)
        df = df.with_columns(
            (pl.col("Rolling_Median_Volume") + (pl.col("Rolling_Median_Volume") - pl.col("Volume")))
            .alias("Inverted_Volume")
        )
        
        # Normalize the inverted volume for signal generation
        df = df.with_columns(
            (pl.col("Inverted_Volume") / pl.col("Inverted_Volume").rolling_mean(window_size=self.obi_window, min_periods=1))
            .fill_nan(1.0) # Fill NaNs with 1.0, assuming neutral impact
            .alias("Volume_Norm")
        )
        
        return df

    def calculate_momentum_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate various momentum and mean reversion indicators."""
        # Price momentum
        df = df.with_columns(
            pl.col("MID_PRICE").pct_change().rolling_mean(window_size=self.momentum_window).alias("Price_Momentum"),
            pl.col("Volume").pct_change().rolling_mean(window_size=self.momentum_window).alias("Volume_Momentum")
        )
        
        # Mean reversion score
        df = df.with_columns(
            ((pl.col("MID_PRICE") - pl.col("VWAP")) / pl.col("VWAP_STD")).alias("Mean_Reversion_Score")
        )
        
        # RSI-like indicator for order book pressure
        df = df.with_columns(
            pl.col("Bid_Pressure").rolling_mean(window_size=self.momentum_window).alias("OB_RSI")
        )
        
        return df

    def calculate_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enhanced volatility calculation with multiple estimators."""
        # Close-to-close volatility
        df = df.with_columns(
            pl.col("MID_PRICE").pct_change()
            .abs()
            .rolling_mean(window_size=self.volatility_window)
            .alias("Volatility")
        )
        
        # Parkinson volatility estimator
        df = df.with_columns(
            (
                (pl.col("ask").rolling_max(window_size=self.volatility_window) -
                 pl.col("bid").rolling_min(window_size=self.volatility_window)) /
                np.sqrt(4 * np.log(2))
            ).alias("Parkinson_Vol")
        )
        
        # Volume-weighted volatility
        df = df.with_columns(
            (
                pl.col("Volatility") * 
                (pl.col("Volume") / pl.col("Volume").rolling_mean(window_size=self.volatility_window))
            ).alias("Vol_Adjusted_Vol")
        )
        
        return df

    def calculate_market_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Identify market regime using multiple indicators."""
        # Trend strength
        df = df.with_columns(
            pl.col("VWAP").diff().rolling_mean(window_size=self.trend_window).alias("Short_Trend"),
            pl.col("VWAP").diff().rolling_mean(window_size=self.trend_window * 2).alias("Medium_Trend"),
            pl.col("VWAP").diff().rolling_mean(window_size=self.trend_window * 4).alias("Long_Trend")
        )
        
        # Trend alignment
        df = df.with_columns(
            (
                (pl.col("Short_Trend") > 0) &
                (pl.col("Medium_Trend") > 0) &
                (pl.col("Long_Trend") > 0)
            ).cast(pl.Int8).alias("Uptrend"),
            
            (
                (pl.col("Short_Trend") < 0) &
                (pl.col("Medium_Trend") < 0) &
                (pl.col("Long_Trend") < 0)
            ).cast(pl.Int8).alias("Downtrend")
        )
        
        # Volatility regime
        df = df.with_columns(
            (pl.col("Vol_Adjusted_Vol") > pl.col("Vol_Adjusted_Vol").rolling_mean(window_size=100))
            .cast(pl.Int8).alias("High_Vol_Regime")
        )
        
        return df

    def calculate_signal_quality(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate signal quality score based on multiple factors."""
        df = df.with_columns(
            # Trend quality
            (
                (pl.col("Uptrend") * pl.col("OB_RSI") > 0.7) |
                (pl.col("Downtrend") * (1 - pl.col("OB_RSI")) > 0.7)
            ).cast(pl.Int8).alias("Trend_Quality"),
            
            # Volatility quality
            (pl.col("Vol_Adjusted_Vol") < pl.col("Vol_Adjusted_Vol").rolling_mean(window_size=100))
            .cast(pl.Int8).alias("Vol_Quality"),
            
            # Spread quality
            (pl.col("Relative_Spread") < pl.col("Relative_Spread").rolling_mean(window_size=100))
            .cast(pl.Int8).alias("Spread_Quality")
        )
        
        # Combined signal quality score
        df = df.with_columns(
            (
                pl.col("Trend_Quality") +
                pl.col("Vol_Quality") +
                pl.col("Spread_Quality")
            ).alias("Signal_Quality")
        )
        
        return df

    def calculate_obi(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Order Book Imbalance (OBI) using volume-weighted pressure.
        
        OBI = (BidVolume - AskVolume) / (BidVolume + AskVolume)
        
        With additional weighting factors:
        1. Price-level weighting
        2. Time decay weighting
        3. Volume normalization
        """
        # Calculate basic order book metrics
        df = df.with_columns([
            # Basic volume imbalance
            ((pl.col("bidsiz") - pl.col("asksiz")) / 
            (pl.col("bidsiz") + pl.col("asksiz"))).alias("Raw_OBI"),
            
            # Price-weighted volume (gives more weight to prices closer to mid)
            ((pl.col("bidsiz") / (pl.col("MID_PRICE") - pl.col("bid")).abs().replace(0, 0.0001)) -
            (pl.col("asksiz") / (pl.col("ask") - pl.col("MID_PRICE")).abs().replace(0, 0.0001)))
            .alias("Price_Weighted_Volume"),
        ])
        
        # Calculate time-weighted OBI with exponential decay
        df = df.with_columns([
            pl.col("Raw_OBI")
            .ewm_mean(span=self.obi_window)
            .alias("Time_Weighted_OBI")
        ])
        
        # Final OBI calculation combining all factors
        df = df.with_columns([
            (
                pl.col("Time_Weighted_OBI") * 
                pl.col("Price_Weighted_Volume") * 
                pl.col("Volume_Norm")
            ).alias("OBI")
        ])
        
        return df

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals based on OBI, Inverse VWAP, and inverted volume."""
        self.logger.info("Generating signals for InverseOBIVWAPStrategy...")
        df = self.calculate_vwap(df)
        df = self.calculate_price_impact_and_inverse_volume(df)
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_volatility(df)
        df = self.calculate_obi(df)
        df = df.with_columns(
            pl.when(
                # Long signal conditions
                (pl.col("OBI") > self.obi_threshold) &
                (pl.col("MID_PRICE") < pl.col("Inverse_VWAP")) &
                (pl.col("Volume_Norm") > 1.0) # High inverted volume
            )
            .then(1)
            .when(
                # Short signal conditions
                (pl.col("OBI") < -self.obi_threshold) &
                (pl.col("MID_PRICE") > pl.col("Inverse_VWAP")) &
                (pl.col("Volume_Norm") > 1.0) # High inverted volume
            )
            .then(-1)
            .otherwise(0)
            .alias("Signal")
        )
        
        # self.logger.info signal distribution
        signal_counts = df.group_by("Signal").count()
        self.logger.info("Signal Distribution:")
        self.logger.info(str(signal_counts))
        
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

    def calculate_position_size(self, price: float, volatility: float, portfolio_value: float,
                              signal_quality: float) -> int:
        """Calculate optimal position size based on risk, volatility, and signal quality (aggressive version)."""
        if volatility == 0:
            volatility = 0.001  # Prevent division by zero
            
        # Very aggressive position sizing
        risk_amount = portfolio_value * self.risk_per_trade
        risk_per_share = price * volatility

        if risk_per_share == 0 or math.isnan(risk_per_share):
            return 0

        position_size = int(risk_amount / risk_per_share)
        return min(position_size, self.max_position)

    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enhanced backtesting with detailed performance tracking and risk management."""
        self.logger.info(f"Starting backtest for Inverse OBI VWAP Strategy with initial cash: ${self.cash:,.2f}")
        
        # Calculate all indicators and add them to the DataFrame
        df = self.generate_signals(df)
        
        account_balance = []
        positions = []
        trades = []
        entry_prices = []
        position_hold_times = []
        unrealized_pnl_values = []
        unrealized_pnl_pct_values = []
        trade_markers = []  # 1 for entry, -1 for exit, 0 for no trade
        stop_loss_hits = []
        take_profit_hits = []
        position_sizes = []
        last_signal = 0
        self.position_hold_time = 0 # Initialize position hold time
        
        for i, row in enumerate(df.iter_rows(named=True)):
            signal = row["Signal"]
            mid_price = row["MID_PRICE"]
            current_portfolio = self.cash + (self.position * mid_price if self.position != 0 else 0)
            
            # Calculate position size for this row (for plotting)
            position_size = min(
                int(current_portfolio * self.risk_per_trade / (mid_price + 1e-8)),
                self.max_position
            )
            position_sizes.append(position_size)
            
            # Default trade marker and event flags
            trade_marker = 0
            stop_loss_hit = 0
            take_profit_hit = 0
            
            # Track entry price for this row
            entry_prices.append(self.entry_price if self.position != 0 else None)
            position_hold_times.append(self.position_hold_time if self.position != 0 else 0)
            
            # Process existing position
            if self.position != 0 and self.entry_price not in (None, 0):
                unrealized_pnl = (mid_price - self.entry_price) * self.position
                unrealized_pnl_pct = unrealized_pnl / (self.entry_price * abs(self.position))
                
                # Increment position hold time
                self.position_hold_time += 1
                
                # Log position status
                
                # Check stop loss
                if (self.position > 0 and unrealized_pnl_pct <= -self.stop_loss_pct/100) or \
                   (self.position < 0 and unrealized_pnl_pct >= self.stop_loss_pct/100):
                    # Close position due to stop loss
                    if self.position > 0:
                        self.cash += row["bid"] * self.position
                        exit_price = row["bid"]
                    else:
                        self.cash -= row["ask"] * abs(self.position)
                        exit_price = row["ask"]
                    
                    trades.append(("Stop Loss", self.position, self.entry_price, exit_price))
                    trade_marker = -1
                    stop_loss_hit = 1
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
                
                # Check take profit
                elif (self.position > 0 and unrealized_pnl_pct >= self.profit_target_pct/100) or \
                     (self.position < 0 and unrealized_pnl_pct <= -self.profit_target_pct/100):
                    # Close position due to take profit
                    if self.position > 0:
                        self.cash += row["bid"] * self.position
                        exit_price = row["bid"]
                    else:
                        self.cash -= row["ask"] * abs(self.position)
                        exit_price = row["ask"]
                    
                    trades.append(("Take Profit", self.position, self.entry_price, exit_price))
                    trade_marker = -1
                    take_profit_hit = 1
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
            
            # Store unrealized PnL values for this timestep
            if self.position != 0 and self.entry_price not in (None, 0):
                unrealized_pnl_values.append(unrealized_pnl)
                unrealized_pnl_pct_values.append(unrealized_pnl_pct)
            else:
                unrealized_pnl_values.append(0)
                unrealized_pnl_pct_values.append(0)
                
            # Process new signals only if we haven't already closed due to SL/TP
            if self.position == 0 and signal != 0 and signal != last_signal:
                position_size = min(
                    int(current_portfolio * self.risk_per_trade / (mid_price + 1e-8)),
                    self.max_position
                )

                
                if signal == 1:
                    # Open long position
                    if position_size > 0 and self.cash >= row["ask"] * position_size:
                        self.cash -= row["ask"] * position_size
                        self.position = position_size
                        self.entry_price = row["ask"]
                        trades.append(("Buy", position_size, row["ask"], None))
                        trade_marker = 1
                
                elif signal == -1:
                    # Open short position
                    if position_size > 0:
                        self.cash += row["bid"] * position_size
                        self.position = -position_size
                        self.entry_price = row["bid"]
                        trades.append(("Sell", -position_size, row["bid"], None))
                        trade_marker = 1
            
            # Process signal changes when we have an existing position
            elif self.position != 0 and signal != 0 and signal != last_signal:
                if (signal == 1 and self.position < 0) or (signal == -1 and self.position > 0):
                    # Close existing position due to signal change
                    if self.position > 0:
                        self.cash += row["bid"] * self.position
                        trades.append(("Close Long (Signal Change)", self.position, self.entry_price, row["bid"]))
                    else:
                        self.cash -= row["ask"] * abs(self.position)
                        trades.append(("Close Short (Signal Change)", self.position, self.entry_price, row["ask"]))
                    
                    trade_marker = -1
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
                    
                    # Now open a new position in the opposite direction
                    position_size = min(
                        int(current_portfolio * self.risk_per_trade / (mid_price + 1e-8)),
                        self.max_position
                    )
                    
                    if signal == 1:
                        # Open long position
                        if position_size > 0 and self.cash >= row["ask"] * position_size:
                            self.cash -= row["ask"] * position_size
                            self.position = position_size
                            self.entry_price = row["ask"]
                            trades.append(("Buy", position_size, row["ask"], None))
                            trade_marker = 1
                    else:
                        # Open short position
                        if position_size > 0:
                            self.cash += row["bid"] * position_size
                            self.position = -position_size
                            self.entry_price = row["bid"]
                            trades.append(("Sell", -position_size, row["bid"], None))
                            trade_marker = 1
            
            # Close all positions if signal is 0 and you have an open position
            elif signal == 0 and self.position != 0:
                if self.position > 0:
                    self.cash += row["bid"] * self.position
                    trades.append(("Close Long (Signal 0)", self.position, self.entry_price, row["bid"]))
                elif self.position < 0:
                    self.cash -= row["ask"] * abs(self.position)
                    trades.append(("Close Short (Signal 0)", self.position, self.entry_price, row["ask"]))
                
                trade_marker = -1
                self.position = 0
                self.entry_price = 0
                self.position_hold_time = 0
            
            # Track trade events
            trade_markers.append(trade_marker)
            stop_loss_hits.append(stop_loss_hit)
            take_profit_hits.append(take_profit_hit)
            
            # Update tracking variables
            current_balance = self.cash + (self.position * mid_price if self.position != 0 else 0)
            account_balance.append(current_balance)
            positions.append(self.position)
            last_signal = signal
            
        # Add metrics columns
        df = df.with_columns([
            pl.Series("Account_Balance", [float(x) for x in account_balance]),
            pl.Series("Position", [float(x) for x in positions]),
            pl.Series("Entry_Price", [float(x) if x is not None else float('nan') for x in entry_prices]),
            pl.Series("Position_Hold_Time", [float(x) for x in position_hold_times]),
            pl.Series("Position_Size", [float(x) for x in position_sizes]),
            pl.Series("Unrealized_PnL", [float(x) for x in unrealized_pnl_values]),
            pl.Series("Unrealized_PnL_Pct", [float(x) for x in unrealized_pnl_pct_values]),
            pl.Series("Trade_Marker", [float(x) for x in trade_markers]),
            pl.Series("Stop_Loss_Hit", [float(x) for x in stop_loss_hits]),
            pl.Series("Take_Profit_Hit", [float(x) for x in take_profit_hits])
        ])
        
        # Calculate returns and metrics
        df = df.with_columns([
            pl.col("Account_Balance").pct_change().alias("Returns")
        ])
        
        df = df.with_columns(
            pl.col("Account_Balance").rolling_max(window_size=self.vwap_window).alias("Peak_Value")
        )
        
        df = df.with_columns(
            ((pl.col("Peak_Value") - pl.col("Account_Balance")) / pl.col("Peak_Value")).alias("Drawdown")
        )
        
        # Calculate performance metrics
        avg_daily_return = df.select(pl.col("Returns").mean()).item()
        std_daily_return = df.select(pl.col("Returns").std()).item()
        max_drawdown = df.select(pl.col("Drawdown").max()).item()


        # Log summary using the common format
        log_backtest_summary(
            strategy_name="Inverse OBI VWAP Strategy",
            account_balance=account_balance,
            trades=trades,
            avg_return=avg_daily_return,         # Now per-tick return
            std_return=std_daily_return,         # Now per-tick std
            max_drawdown=max_drawdown,
            logger=self.logger,
            num_ticks=len(df),                   # Total rows = ticks
            num_days = 1

        )
        return df

