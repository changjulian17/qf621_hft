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
                 stop_loss_pct: float = 1, profit_target_pct: float = 1.0, risk_per_trade: float = 0.01,
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
        self.trades = []
        self.start_time = start_time
        self.end_time = end_time
        self.position_hold_time = 0
        self.max_hold_time = 100
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

    def calculate_position_size(self, price: float, current_balance: float) -> int:
        """Calculate optimal position size based on risk and current balance."""
        # Risk-based position sizing similar to mean reversion strategy
        risk_amount = current_balance * self.risk_per_trade
        
        try:
            position_size = int((risk_amount / price) / 5) 
        except:
            position_size = 1
        if position_size > self.max_position:
            return self.max_position
        elif position_size < 1:
            return 1
        return position_size

    def calculate_position_size_legacy(self, price: float, volatility: float, portfolio_value: float,
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

    def backtest(self, df_fin: pl.DataFrame, days) -> tuple:
        """Run backtest simulation and return performance metrics and all indicators for plotting."""
        # Calculate all indicators and add them to the DataFrame before running the backtest loop
        self.logger.info(f"Starting backtest for Inverse OBI VWAP Strategy with initial cash: ${self.cash:,.2f}")
        self.cash = 300_000  # Reset cash for each backtest
        final_returns = []
        final_df = []

        for date in days:
            # Filter DataFrame for the current date
            df = df_fin.filter(pl.col("date") == date)
            if len(df) > 200_000:
                # skip every tenth row to reduce size
                df = df[::10]
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
                try:
                    unrealized_pnl_pct =  unrealized_pnl / (self.entry_price * abs(self.position)) if self.position != 0 else 0
                except ZeroDivisionError:
                    unrealized_pnl_pct = 0
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
                            self.cash += row["bid"] * self.position - trading_cost
                            trades.append(("Stop Loss", self.position, self.entry_price, row["bid"]))
                        else:
                            self.cash -= row["ask"] * abs(self.position) + trading_cost
                            trades.append(("Stop Loss", self.position, self.entry_price, row["ask"]))
                        trade_marker = -1
                        stop_loss_hit = 1
                        self.position = 0
                        self.entry_price = 0
                        self.position_hold_time = 0
                    # Take profit
                    elif (self.position > 0 and unrealized_pnl_pct >= self.profit_target_pct/100) or \
                        (self.position < 0 and unrealized_pnl_pct <= -self.profit_target_pct/100):
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

            # Convert Time to datetime
            df = df.with_columns(
                pl.col("Time").str.strptime(pl.Time, format="%H:%M:%S%.6f")
            )

            # Log summary using the common format
            intraday_info = log_backtest_summary(
                strategy_name="Inverse OBI VWAP Strategy",
                account_balance=account_balance,
                logger=self.logger,
                num_ticks=len(df),
                df=df,
                date=date
            )
            final_returns.append(intraday_info)
            final_df.append(df)
        
        return final_df, final_returns
