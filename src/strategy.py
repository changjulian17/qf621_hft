import polars as pl
import logging
import numpy as np
import math
from typing import Optional
from .logger_config import setup_logger




class OBIVWAPoldStrategy:
    """
    Implements the Order Book Imbalance (OBI) and Volume Weighted Average Price (VWAP) strategy.

    This strategy generates trading signals based on the imbalance in the order book
    and compares it to a VWAP threshold to decide buy/sell actions.

    Attributes:
        vwap_window (int): 
            Rolling window size for VWAP calculation.
        obi_threshold (float): 
            Threshold for Order Book Imbalance (OBI) signals.
        size_threshold (int): 
            Minimum size threshold for bid and ask sizes.
        start_time (tuple): 
            The earliest time (HH, MM, MS) for generating signals.
        end_time (tuple): 
            The latest time (HH, MM, MS) for generating signals.

    Methods:
        calculate_vwap(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the VWAP for the given data.
        generate_signals(data: pl.DataFrame) -> pl.DataFrame:
            Generates buy/sell signals based on OBI and VWAP.
        backtest(data: pl.DataFrame) -> pl.DataFrame:
            Simulates the strategy on historical data and returns performance metrics.
    """
    def __init__(self, vwap_window: int, obi_threshold: float, size_threshold: int = 3, 
                 vwap_threshold: float = 0.1,
                 initial_cash: float = 100_000, start_time: tuple = (9, 30, 865), 
                 end_time: tuple = (16, 28, 954)):
        """
        Initializes the OBIVWAPStrategy with the specified parameters.

        Args:
            vwap_window (int): Rolling window size for VWAP calculation.
            obi_threshold (float): Threshold for Order Book Imbalance (OBI) signals.
            size_threshold (int): Minimum size threshold for bid and ask sizes. Default is 3.
            vwap_threshold (float): VWAP threshold for signal generation. Default is 0.1.
            initial_cash (float): Initial cash balance for the strategy. Default is 100,000.
            start_time (tuple): The earliest time (HH, MM, MS) for generating signals.
            end_time (tuple): The latest time (HH, MM, MS) for generating signals.
        """
        self.vwap_window = vwap_window
        self.obi_threshold = obi_threshold
        self.size_threshold = size_threshold
        self.vwap_threshold = vwap_threshold
        self.cash = initial_cash
        self.position = 0
        self.account_balance = []
        self.start_time = start_time
        self.end_time = end_time

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the VWAP for the given data.

        Args:
            data (pl.DataFrame): 
                A Polars DataFrame containing stock market data with columns 
                for price and volume.

        Returns:
            pl.DataFrame: 
                A Polars DataFrame with an additional column for VWAP.
        """
        df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("MID_PRICE"))
        df = df.with_columns((pl.col("bidsiz") + pl.col("asksiz")).alias("Volume"))
        df = df.with_columns(
            (
                (pl.col("MID_PRICE") * pl.col("Volume")).rolling_sum(window_size=self.vwap_window)
                / pl.col("Volume").rolling_sum(window_size=self.vwap_window)
            ).alias("VWAP")
        )
        return df

    def calculate_obi(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the rolling average Order Book Imbalance (OBI) for the given data.

        Returns:
            pl.DataFrame: 
                A Polars DataFrame with an additional column for rolling OBI.
        """
        obi_raw = ((pl.col("bidsiz") - pl.col("asksiz")) / (pl.col("bidsiz") + pl.col("asksiz"))).alias("OBI_raw")
        df = df.with_columns(obi_raw)
        df = df.with_columns(
            pl.col("OBI_raw").rolling_mean(window_size=self.vwap_window).alias("OBI")
        )
        return df

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generates buy/sell signals based on OBI and VWAP.

        Args:
            df (pl.DataFrame): 
                A Polars DataFrame containing stock market data.

        Returns:
            pl.DataFrame: 
                A Polars DataFrame with an additional column for trading signals.
        """
        df = self.calculate_vwap(df)
        df = self.calculate_obi(df)
        df = df.with_columns(
            pl.when(
            (pl.col("OBI") > self.obi_threshold) & 
            (pl.max_horizontal("bidsiz", "asksiz") >= self.size_threshold) &
            (pl.col("MID_PRICE") < pl.col("VWAP") * (1 + self.vwap_threshold))
            )
            .then(1)
            .when(
            (pl.col("OBI") < -self.obi_threshold) & 
            (pl.max_horizontal("bidsiz", "asksiz") >= self.size_threshold) &
            (pl.col("MID_PRICE") > pl.col("VWAP") * (1 - self.vwap_threshold))
            )
            .then(-1)
            .otherwise(0)
            .alias("Signal")
        )

        # Ensure Signal is 0 when TIME_M is outside the allowed range
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

    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Backtests the strategy on historical data.
        
        Args:
            df (pl.DataFrame): Market data with 'Signal', 'ASK', 'BID'
        
        Returns:
            pl.DataFrame: Data with account balance over time and cumulative trades
        """
        account_balance = []
        cumulative_trades = []
        trades = 0

        for row in df.iter_rows(named=True):
            signal = row["Signal"]

            # Buy signal
            if signal == 1 and self.position == 0 and self.cash >= row["ask"] * 100:
                self.cash -= row["ask"] * 100
                self.position = 100
                trades += 1

            # Sell signal
            elif signal == -1 and self.position == 0:
                self.cash += row["bid"] * 100
                self.position = -100
                trades += 1

            # Close position
            elif signal == 0 and self.position != 0:
                if self.position > 0:
                    self.cash += row["bid"] * self.position
                else:
                    self.cash -= row["ask"] * abs(self.position)
                self.position = 0
                trades += 1

            # Mark-to-market balance
            market_price = row["ask"] if self.position > 0 else row["bid"] if self.position < 0 else 0
            account_balance.append(self.cash + self.position * market_price)
            cumulative_trades.append(trades)

        return df.with_columns([
            pl.Series("Account_Balance", account_balance, dtype=pl.Float64),
            pl.Series("Cumulative_Trades", cumulative_trades, dtype=pl.Int64)
        ])


class OBIVWAPStrategy:
    """
    Implements the Order Book Imbalance (OBI) and Volume Weighted Average Price (VWAP) strategy.

    This strategy generates trading signals based on the imbalance in the order book
    and compares it to a VWAP threshold to decide buy/sell actions.

    Attributes:
        vwap_window (int): 
            Rolling window size for VWAP calculation.
        obi_window (int): 
            Rolling window size for OBI calculation.
        price_impact_window (int): 
            Rolling window size for price impact calculation.
        momentum_window (int): 
            Rolling window size for momentum calculation.
        obi_threshold (float): 
            Threshold for Order Book Imbalance (OBI) signals.
        size_threshold (int): 
            Minimum size threshold for bid and ask sizes.
        vwap_threshold (float): 
            VWAP threshold for signal generation.
        volatility_window (int): 
            Window size for volatility calculation.
        trend_window (int): 
            Window size for trend detection.
        max_position (int): 
            Maximum position size.
        stop_loss_pct (float): 
            Stop loss percentage.
        profit_target_pct (float): 
            Profit target percentage.
        initial_cash (float): 
            Initial cash balance for the strategy.
        risk_per_trade (float): 
            Percentage of portfolio to risk per trade.
        min_profit_threshold (float): 
            Minimum expected profit.
        start_time (tuple): 
            The earliest time (HH, MM, MS) for generating signals.
        end_time (tuple): 
            The latest time (HH, MM, MS) for generating signals.

    Methods:
        calculate_vwap(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the VWAP for the given data.
        calculate_price_impact(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the price impact for the given data.
        calculate_momentum_indicators(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the momentum indicators for the given data.
        calculate_volatility(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the volatility for the given data.
        calculate_market_regime(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the market regime for the given data.
        calculate_signal_quality(data: pl.DataFrame) -> pl.DataFrame:
            Calculates the signal quality for the given data.
        generate_signals(data: pl.DataFrame) -> pl.DataFrame:
            Generates buy/sell signals based on OBI, VWAP, and volatility.
        calculate_position_size(price: float, volatility: float, portfolio_value: float,
                              signal_quality: float) -> int:
            Calculates the optimal position size based on risk and volatility.
        backtest(data: pl.DataFrame) -> pl.DataFrame:
            Simulates the strategy on historical data and returns performance metrics.
    """
    def __init__(self, vwap_window: int = 500, obi_window: int = 20,
                 price_impact_window: int = 50, momentum_window: int = 100,
                 obi_threshold: float = 0.01, size_threshold: int = 1,  # Further lowered thresholds
                 vwap_threshold: float = 0.01, volatility_window: int = 50,
                 trend_window: int = 20, max_position: int = 1000,  # Increased max position
                 stop_loss_pct: float = 0.5, profit_target_pct: float = 1.0, risk_per_trade: float = 0.4,
                 min_profit_threshold: float = 0.0001,
                 start_time: tuple = (9, 30, 865),
                 end_time: tuple = (16, 28, 954),
                 trading_cost_bps: float = 0.5,  # Trading cost in basis points
                 logger: Optional[logging.Logger] = None):
        """Enhanced initialization with microstructure indicators (aggressive version)."""
        self.vwap_window = vwap_window
        self.obi_window = obi_window
        self.price_impact_window = price_impact_window
        self.momentum_window = momentum_window
        self.obi_threshold = obi_threshold
        self.size_threshold = size_threshold
        self.vwap_threshold = vwap_threshold
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.risk_per_trade = risk_per_trade
        self.min_profit_threshold = min_profit_threshold
        self.trading_cost_bps = trading_cost_bps  # Trading cost in basis points
        self.cash = 100_000  # Fixed cash initialization
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.start_time = start_time
        self.end_time = end_time
        self.logger = logger or logging.getLogger(__name__)

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP with adaptive bands based on volume profile."""
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
        
        return df

    def calculate_price_impact(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate price impact and market depth indicators."""
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

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals with enhanced filtering and quality scores (aggressive version)."""
        # Critical debugging
        # self.logger.info(f"Input DataFrame shape: {df.shape}")
        # self.logger.info(f"Input DataFrame columns: {df.columns}")
        
        # Calculate basic metrics first
        df = df.with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("MID_PRICE"),
            (pl.col("bidsiz") + pl.col("asksiz")).alias("Volume")
        )
        
        # self.logger.info sample of raw data
        self.logger.debug("Sample of raw data:")
        sample_data = df.select(["bid", "ask", "bidsiz", "asksiz", "MID_PRICE", "Volume"]).head()
        self.logger.debug(str(sample_data))
        
        # Calculate VWAP
        df = df.with_columns(
            (
                (pl.col("MID_PRICE") * pl.col("Volume")).rolling_sum(window_size=self.vwap_window)
                / pl.col("Volume").rolling_sum(window_size=self.vwap_window)
            ).alias("VWAP")
        )
        
        # Calculate bid pressure
        df = df.with_columns(
            (pl.col("bidsiz") / (pl.col("bidsiz") + pl.col("asksiz"))).alias("Bid_Pressure")
        )
        
        # self.logger.info sample of calculated metrics
        self.logger.debug("Sample of calculated metrics:")
        metrics_sample = df.select(["MID_PRICE", "VWAP", "Bid_Pressure"]).head()
        self.logger.debug(str(metrics_sample))
        
        # Generate signals with extremely loose conditions
        df = df.with_columns(
            pl.when(
                # Long signal conditions (extremely loose)
                (pl.col("Bid_Pressure") > 0.5) &  # Just need slight buying pressure
                (pl.col("MID_PRICE") < pl.col("VWAP"))  # Price below VWAP
            )
            .then(1)
            .when(
                # Short signal conditions (extremely loose)
                (pl.col("Bid_Pressure") < 0.5) &  # Just need slight selling pressure
                (pl.col("MID_PRICE") > pl.col("VWAP"))  # Price above VWAP
            )
            .then(-1)
            .otherwise(0)
            .alias("Signal")
        )
        
        # self.logger.info signal distribution
        signal_counts = df.group_by("Signal").count()
        # self.logger.info("Signal Distribution:")
        # self.logger.info(str(signal_counts))
        
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
        self.logger.info(f"Starting backtest with initial cash: ${self.cash:,.2f}")
        
        # Generate signals first if they don't exist
        if "Signal" not in df.columns:
            df = self.generate_signals(df)
        
        account_balance = []
        positions = []
        trades = []
        last_signal = 0
        
        for i, row in enumerate(df.iter_rows(named=True)):
            signal = row["Signal"]
            # Use the appropriate price for current valuation based on position
            mid_price = row["MID_PRICE"]  # For zero position
            current_price = mid_price
            if self.position > 0:
                current_price = row["bid"]  # Value longs at bid price (what we could sell for)
            elif self.position < 0:
                current_price = row["ask"]  # Value shorts at ask price (what we'd pay to cover)
            
            current_portfolio = self.cash + (self.position * current_price if self.position != 0 else 0)
            
            # Process existing position
            if self.position != 0 and self.entry_price not in (None, 0):
                unrealized_pnl = 0
                if self.position > 0:
                    # For long positions, calculate P&L using bid (what we could sell for)
                    unrealized_pnl = (row["bid"] - self.entry_price) * self.position
                else:
                    # For short positions, calculate P&L using ask (what we'd pay to cover)
                    unrealized_pnl = (self.entry_price - row["ask"]) * abs(self.position)
                
                unrealized_pnl_pct = unrealized_pnl / (self.entry_price * abs(self.position))
                
                # Log position status
                self.logger.debug(f"Current position: {self.position}, Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:.2%})")
            
            # Process new signals
            if signal != 0 and signal != last_signal:
                position_size = min(
                    int(current_portfolio * self.risk_per_trade / (mid_price + 1e-8)),
                    self.max_position
                )
                
                # Log signal and position information
                # self.logger.info(f"Signal change at index {i}:")
                # self.logger.info(f"Signal: {signal}")
                # self.logger.info(f"Mid Price: ${mid_price:.2f}")
                # self.logger.info(f"Current Portfolio: ${current_portfolio:,.2f}")
                # self.logger.info(f"Bid Pressure: {row.get('Bid_Pressure', 'N/A')}")
                # self.logger.info(f"VWAP: {row.get('VWAP', 'N/A')}")
                
                if signal == 1 and self.position <= 0:
                    # Close any existing short position
                    if self.position < 0:
                        trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * abs(self.position) + trading_cost)
                        trades.append(("Close Short", self.position, self.entry_price, row["ask"]))
                        # self.logger.info(f"Closing short position: {abs(self.position)} shares at ${row['ask']:.2f}, cost: ${trading_cost:.2f}")
                        self.position = 0
                    
                    # Open long position
                    if position_size > 0 and self.cash >= row["ask"] * position_size:
                        trading_cost = row["ask"] * position_size * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * position_size + trading_cost)
                        self.position = position_size
                        self.entry_price = row["ask"]
                        trades.append(("Buy", position_size, row["ask"], None))
                        # self.logger.info(f"Opening long position: {position_size} shares at ${row['ask']:.2f}, cost: ${trading_cost:.2f}")
                
                elif signal == -1 and self.position >= 0:
                    # Close any existing long position
                    if self.position > 0:
                        trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * self.position - trading_cost)
                        trades.append(("Close Long", self.position, self.entry_price, row["bid"]))
                        # self.logger.info(f"Closing long position: {self.position} shares at ${row['bid']:.2f}, cost: ${trading_cost:.2f}")
                        self.position = 0
                    
                    # Open short position
                    if position_size > 0:
                        trading_cost = row["bid"] * position_size * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * position_size - trading_cost)
                        self.position = -position_size
                        self.entry_price = row["bid"]
                        trades.append(("Sell", -position_size, row["bid"], None))
                        # self.logger.info(f"Opening short position: {position_size} shares at ${row['bid']:.2f}, cost: ${trading_cost:.2f}")
            # Close all positions if signal is 0 and you have an open position
            elif signal == 0 and self.position != 0:
                if self.position > 0:
                    trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                    self.cash += (row["bid"] * self.position - trading_cost)
                    trades.append(("Close Long (Signal 0)", self.position, self.entry_price, row["bid"]))
                elif self.position < 0:
                    trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                    self.cash -= (row["ask"] * abs(self.position) + trading_cost)
                    trades.append(("Close Short (Signal 0)", self.position, self.entry_price, row["ask"]))
                self.position = 0
                self.entry_price = 0
                self.position_hold_time = 0
            
            # Update tracking variables - use appropriate prices for mark-to-market
            if self.position > 0:
                current_balance = self.cash + (self.position * row["bid"])  # Value longs at bid
            elif self.position < 0:
                current_balance = self.cash - (abs(self.position) * row["ask"])  # Value shorts at ask
            else:
                current_balance = self.cash
            
            account_balance.append(current_balance)
            positions.append(self.position)
            last_signal = signal
            
        # Calculate performance metrics
        df = df.with_columns(
            pl.Series("Account_Balance", account_balance, dtype=pl.Float64),
            pl.Series("Position", positions, dtype=pl.Int64)
        )
        
        # Calculate returns and metrics
        df = df.with_columns(
            pl.col("Account_Balance").pct_change().alias("Returns")
        )
        
        # Calculate rolling maximum for drawdown
        df = df.with_columns(
            pl.col("Account_Balance").rolling_max(window_size=len(df)).alias("Peak_Value")
        )
        
        df = df.with_columns(
            ((pl.col("Peak_Value") - pl.col("Account_Balance")) / pl.col("Peak_Value")).alias("Drawdown")
        )
        
        # Calculate performance metrics
        initial_balance = account_balance[0]
        final_balance = account_balance[-1]
        total_return = (final_balance / initial_balance - 1) * 100
        
        returns = df["Returns"].drop_nulls()
        avg_daily_return = returns.mean()
        std_daily_return = returns.std()
        risk_free_rate = 0.01 / 252  # Assuming 1% annual risk-free rate
        
        sharpe_ratio = ((avg_daily_return - risk_free_rate) / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Calculate Sortino Ratio
        downside_returns = returns.filter(returns < 0)
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = ((avg_daily_return - risk_free_rate) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        max_drawdown = df["Drawdown"].max() * 100
        
        # Log final performance
        final_balance = account_balance[-1]
        total_return = (final_balance / account_balance[0] - 1) * 100
        self.logger.info(f"OBI-VWAP Strategy Performance:")
        self.logger.info(f"Initial Balance: ${account_balance[0]:,.2f}")
        self.logger.info(f"Final Balance: ${final_balance:,.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        self.logger.info(f"Sortino Ratio: {sortino_ratio:.4f}")
        self.logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
        self.logger.info(f"Number of Trades: {len(trades)}")
        
        # Calculate win rate
        profitable_trades = 0
        total_trades = 0
        for trade in trades:
            if len(trade) >= 4 and trade[3] is not None:
                if (trade[1] > 0 and trade[3] > trade[2]) or (trade[1] < 0 and trade[3] < trade[2]):
                    profitable_trades += 1
                total_trades += 1
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        self.logger.info(f"Win Rate: {win_rate:.2%}")
        
        return df


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
        self.rebalance_threshold = rebalance_threshold
        self.last_rebalance_date = None
        self.portfolio_value = initial_cash
        self.portfolio_history = []
        
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
        print(f"Added strategy {name} with initial cash: ${strategy_cash:,.2f}")
        
        # Rebalance existing strategies
        if len(self.strategies) > 1:
            self.rebalance_portfolio()
            
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
                
        self.rebalance_portfolio()
        
    def rebalance_portfolio(self):
        """Rebalance the portfolio according to target weights."""
        total_value = sum(
            s.cash + (s.position * s.entry_price if s.position != 0 else 0)
            for s in self.strategies.values()
        )
        
        for name, strategy in self.strategies.items():
            current_value = strategy.cash + (strategy.position * strategy.entry_price if strategy.position != 0 else 0)
            target_value = total_value * self.portfolio_weights[name]
            
            # Check if rebalancing is needed
            if abs(current_value - target_value) / target_value > self.rebalance_threshold:
                # Close existing position
                if strategy.position != 0:
                    strategy.cash += strategy.position * strategy.entry_price
                    strategy.position = 0
                    strategy.entry_price = 0
                
                # Set new cash balance
                strategy.cash = target_value
                
    def run_backtest(self, data: pl.DataFrame) -> dict:
        """Run backtest for all strategies and return combined results."""
        results = {}
        portfolio_values = []
        logging.info("Starting backtest for all strategies...")
        
        # Run backtest for each strategy
        for name, strategy in self.strategies.items():
            logging.info(f"Running backtest for strategy: {name}")
            results[name] = strategy.backtest(data)

        logging.info("Backtest completed for all strategies.")

        # Efficiently aggregate Account_Balance and Position columns across all strategies
        account_balance_series_list = [
            result["Account_Balance"] for name, result in results.items() if "Account_Balance" in result.columns
        ]
        position_series_list = [
            result["Position"] for name, result in results.items() if "Position" in result.columns
        ]

        if account_balance_series_list:
            portfolio_account_balance = sum(account_balance_series_list)
        else:
            portfolio_account_balance = pl.Series([0.0] * len(data), dtype=pl.Float64)

        if position_series_list:
            portfolio_positions = sum(position_series_list)
        else:
            portfolio_positions = pl.Series([0] * len(data), dtype=pl.Int64)

        # Add portfolio value and position to results
        results["Portfolio"] = data.with_columns(
            pl.Series("Portfolio_Value", portfolio_account_balance),
            pl.Series("Account_Balance", portfolio_account_balance),
            pl.Series("Position", portfolio_positions)
)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(results["Portfolio"])
        results["Metrics"] = portfolio_metrics
        
        return results
        
    def calculate_portfolio_metrics(self, portfolio_data: pl.DataFrame) -> dict:
        """Calculate combined portfolio performance metrics."""
        returns = portfolio_data["Portfolio_Value"].pct_change()
        
        # Calculate metrics
        total_return = (portfolio_data["Portfolio_Value"][-1] / portfolio_data["Portfolio_Value"][0] - 1) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate drawdown
        peak = portfolio_data["Portfolio_Value"].cum_max()
        drawdown = (portfolio_data["Portfolio_Value"] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Calculate Sortino ratio
        downside_returns = returns.filter(returns < 0)
        downside_returns = downside_returns.fill_null(0)  # Handle NaNs
        sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        return {
            "Total_Return": total_return,
            "Sharpe_Ratio": sharpe_ratio,
            "Sortino_Ratio": sortino_ratio,
            "Max_Drawdown": max_drawdown
        }
        
    def get_portfolio_performance(self) -> dict:
        """Return current portfolio performance metrics."""
        total_value = sum(
            s.cash + (s.position * s.entry_price if s.position != 0 else 0)
            for s in self.strategies.values()
        )
        
        strategy_values = {
            name: s.cash + (s.position * s.entry_price if s.position != 0 else 0)
            for name, s in self.strategies.items()
        }
        
        return {
            "Total_Value": total_value,
            "Strategy_Values": strategy_values,
            "Weights": self.portfolio_weights
        }


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
    
    def __init__(self, vwap_window: int = 20, deviation_threshold: float = 0.0001,
                 volatility_window: int = 20, volume_window: int = 20,
                 max_position: int = 100, stop_loss_pct: float = 0.3,
                 profit_target_pct: float = 0.6, 
                 risk_per_trade: float = 0.02, min_profit_threshold: float = 0.001,
                 start_time: tuple = (9, 30, 865), end_time: tuple = (16, 28, 954),
                 trading_cost_bps: float = 0.5,  # Trading cost in basis points
                 logger: Optional[logging.Logger] = None):
        """Initialize the mean reversion strategy."""
        self.vwap_window = vwap_window
        self.deviation_threshold = deviation_threshold
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.risk_per_trade = risk_per_trade
        self.min_profit_threshold = min_profit_threshold
        self.trading_cost_bps = trading_cost_bps  # Trading cost in basis points
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.start_time = start_time
        self.cash = 100_000  # Fixed cash initialization
        self.end_time = end_time
        self.position_hold_time = 0
        self.max_hold_time = 100
        self.logger = logger or logging.getLogger(__name__)

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP and price deviation metrics."""
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
        """Calculate volatility and volume metrics."""
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
        
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals based on mean reversion indicators."""
        try:
            df = self.calculate_vwap(df)
            # self.logger.info("VWAP non-nulls:", df["VWAP"].drop_nulls().len())
            # self.logger.info("Price_Deviation non-nulls:", df["Price_Deviation"].drop_nulls().len())
            df = self.calculate_volatility(df)
            # self.logger.info("Volatility non-nulls:", df["Volatility"].drop_nulls().len())
            # self.logger.info("Volume_Ratio non-nulls:", df["Volume_Ratio"].drop_nulls().len())
            df = self.calculate_mean_reversion_score(df)
            # self.logger.info("Mean_Reversion_Score non-nulls:", df["Mean_Reversion_Score"].drop_nulls().len())
            

            # Generate signals with tightened conditions
            df = df.with_columns(
                pl.when(
                    # Long signal conditions (tightened)
                    (pl.col("Mean_Reversion_Score") < -0.5) &  # Tightened mean reversion signal
                    (pl.col("Volume_Ratio") > 1.0) &  # Relaxed volume condition
                    (pl.col("Volatility") > 0.0001)  # Minimum volatility threshold
                )
                .then(1)
                .when(
                    # Short signal conditions (tightened)
                    (pl.col("Mean_Reversion_Score") > 0.5) &  # Tightened mean reversion signal
                    (pl.col("Volume_Ratio") > 1.0) &  # Relaxed volume condition
                    (pl.col("Volatility") > 0.0001)  # Minimum volatility threshold
                )
                .then(-1)
                .otherwise(0)
                .alias("Signal")
            )
            
            # self.logger.info signal distribution
            signal_counts = df.group_by("Signal").count()
            # self.logger.info("\nSignal Distribution:")
            # self.logger.info(signal_counts)
            
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
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise e
        
    def calculate_position_size(self, price: float, volatility: float, portfolio_value: float) -> int:
        """Calculate optimal position size based on risk and volatility."""
        if volatility == 0 or math.isnan(volatility):
            return 0

        # Risk-based position sizing
        risk_amount = portfolio_value * self.risk_per_trade
        risk_per_share = price * volatility * 2

        if risk_per_share == 0 or math.isnan(risk_per_share):
            return 0

        position_size = int(risk_amount / risk_per_share)
        return min(position_size, self.max_position)
        
    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run backtest simulation and return performance metrics."""
        # Generate signals first if they don't exist
        if "Signal" not in df.columns:
            df = self.generate_signals(df)
        
        account_balance = []
        positions = []
        trades = []
        last_signal = 0
        
        for i, row in enumerate(df.iter_rows(named=True)):
            signal = row["Signal"]
            # Use bid/ask prices for mark-to-market depending on position
            market_price = row["bid"] if self.position > 0 else row["ask"] if self.position < 0 else row["MID_PRICE"]
            current_portfolio = self.cash + (self.position * market_price if self.position != 0 else 0)
            
            # Calculate position size
            volatility = row.get("Volatility", 0.01)
            
            # Process existing position
            if self.position != 0 and self.entry_price not in (None, 0):
                unrealized_pnl = 0
                if self.position > 0:
                    # For long positions, we sell at bid
                    unrealized_pnl = (row["bid"] - self.entry_price) * self.position
                else:
                    # For short positions, we buy back at ask
                    unrealized_pnl = (self.entry_price - row["ask"]) * abs(self.position)
                
                unrealized_pnl_pct = unrealized_pnl / (self.entry_price * abs(self.position))
                
                # Increment position hold time
                self.position_hold_time += 1
                
                # Check if max hold time is exceeded
                if self.position_hold_time >= self.max_hold_time:
                    # Close position due to max hold time
                    if self.position > 0:
                        trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * self.position - trading_cost)  # Sell at bid
                        trades.append(("Max Hold Time", self.position, self.entry_price, row["bid"]))
                    else:
                        trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * abs(self.position) + trading_cost)  # Buy at ask
                        trades.append(("Max Hold Time", self.position, self.entry_price, row["ask"]))
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
                
                elif (self.position > 0 and unrealized_pnl_pct <= -self.stop_loss_pct/100) or \
                     (self.position < 0 and unrealized_pnl_pct >= self.stop_loss_pct/100):
                    # Stop loss hit - use bid for long, ask for short
                    if self.position > 0:
                        trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * self.position - trading_cost)
                        trades.append(("Stop Loss", self.position, self.entry_price, row["bid"]))
                    else:
                        trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * abs(self.position) + trading_cost)
                        trades.append(("Stop Loss", self.position, self.entry_price, row["ask"]))
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
                    
                elif (self.position > 0 and unrealized_pnl_pct >= self.profit_target_pct/100) or \
                     (self.position < 0 and unrealized_pnl_pct <= -self.profit_target_pct/100):
                    # Take profit hit - use bid for long, ask for short
                    if self.position > 0:
                        trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * self.position - trading_cost)
                        trades.append(("Take Profit", self.position, self.entry_price, row["bid"]))
                    else:
                        trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * abs(self.position) + trading_cost)
                        trades.append(("Take Profit", self.position, self.entry_price, row["ask"]))
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
            
            # Process new signals
            if signal != 0 and signal != last_signal:
                position_size = self.calculate_position_size(market_price, volatility, current_portfolio)
                
                if signal == 1 and self.position <= 0:
                    # Close any existing short position
                    if self.position < 0:
                        trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * abs(self.position) + trading_cost)
                        trades.append(("Close Short", self.position, self.entry_price, row["ask"]))
                        self.position = 0
                        self.entry_price = 0
                        self.position_hold_time = 0
                    
                    # Open long position
                    if position_size > 0 and self.cash >= row["ask"] * position_size:
                        trading_cost = row["ask"] * position_size * (self.trading_cost_bps / 10000)
                        self.cash -= (row["ask"] * position_size + trading_cost)
                        self.position = position_size
                        self.entry_price = row["ask"]
                        trades.append(("Buy", position_size, row["ask"], None))
                        self.position_hold_time = 0
                        
                elif signal == -1 and self.position >= 0:
                    # Close any existing long position
                    if self.position > 0:
                        trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * self.position - trading_cost)
                        trades.append(("Close Long", self.position, self.entry_price, row["bid"]))
                        self.position = 0
                        self.entry_price = 0
                        self.position_hold_time = 0
                    
                    # Open short position
                    if position_size > 0:
                        trading_cost = row["bid"] * position_size * (self.trading_cost_bps / 10000)
                        self.cash += (row["bid"] * position_size - trading_cost)
                        self.position = -position_size
                        self.entry_price = row["bid"]
                        trades.append(("Sell", -position_size, row["bid"], None))
                        self.position_hold_time = 0
            # Close all positions if signal is 0 and you have an open position
            elif signal == 0 and self.position != 0:
                if self.position > 0:
                    trading_cost = row["bid"] * self.position * (self.trading_cost_bps / 10000)
                    self.cash += (row["bid"] * self.position - trading_cost)
                    trades.append(("Close Long (Signal 0)", self.position, self.entry_price, row["bid"]))
                elif self.position < 0:
                    trading_cost = row["ask"] * abs(self.position) * (self.trading_cost_bps / 10000)
                    self.cash -= (row["ask"] * abs(self.position) + trading_cost)
                    trades.append(("Close Short (Signal 0)", self.position, self.entry_price, row["ask"]))
                self.position = 0
                self.entry_price = 0
                self.position_hold_time = 0
            
            # Update tracking variables
            if self.position > 0:
                current_balance = self.cash + (self.position * row["bid"])  # Value longs at bid
            elif self.position < 0:
                current_balance = self.cash - (abs(self.position) * row["ask"])  # Value shorts at ask
            else:
                current_balance = self.cash
            account_balance.append(current_balance)
            positions.append(self.position)
            last_signal = signal
            
        # Calculate performance metrics
        df = df.with_columns(
            pl.Series("Account_Balance", account_balance, dtype=pl.Float64),
            pl.Series("Position", positions, dtype=pl.Int64)
        )
        
        # Calculate returns and metrics
        df = df.with_columns(
            pl.col("Account_Balance").pct_change().alias("Returns")
        )
        
        # Calculate rolling maximum for drawdown
        df = df.with_columns(
            pl.col("Account_Balance").rolling_max(window_size=len(df)).alias("Peak_Value")
        )
        
        df = df.with_columns(
            ((pl.col("Peak_Value") - pl.col("Account_Balance")) / pl.col("Peak_Value")).alias("Drawdown")
        )
        
        # Calculate performance metrics
        initial_balance = account_balance[0]
        final_balance = account_balance[-1]
        total_return = (final_balance / initial_balance - 1) * 100
        
        returns = df["Returns"].drop_nulls()
        avg_daily_return = returns.mean()
        std_daily_return = returns.std()
        risk_free_rate = 0.01 / 252  # Assuming 1% annual risk-free rate
        
        sharpe_ratio = ((avg_daily_return - risk_free_rate) / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Calculate Sortino Ratio
        downside_returns = returns.filter(returns < 0)
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = ((avg_daily_return - risk_free_rate) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        max_drawdown = df["Drawdown"].max() * 100
        
        # Log performance metrics
        self.logger.info(f"\nMean Reversion Strategy Performance:")
        self.logger.info(f"Initial Balance: ${initial_balance:,.2f}")
        self.logger.info(f"Final Balance: ${final_balance:,.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        self.logger.info(f"Sortino Ratio: {sortino_ratio:.4f}")
        self.logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
        self.logger.info(f"Number of Trades: {len(trades)}")
        
        # Calculate win rate and profit metrics
        profitable_trades = 0
        total_trades = 0
        total_profit = 0
        for trade in trades:
            if len(trade) >= 4 and trade[3] is not None:
                profit = (trade[3] - trade[2]) * trade[1] if trade[1] != 0 else 0
                total_profit += profit
                if profit > 0:
                    profitable_trades += 1
                total_trades += 1
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        self.logger.info(f"Win Rate: {win_rate:.2%}")
        self.logger.info(f"Average Profit per Trade: ${avg_profit:.2f}")
        
        return df

