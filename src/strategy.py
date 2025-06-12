import polars as pl
import numpy as np
from datetime import datetime, time
from typing import Tuple, List

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
                 end_time: tuple = (16, 28, 954)):
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
        self.cash = 100_000  # Fixed cash initialization
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.start_time = start_time
        self.end_time = end_time

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP with adaptive bands based on volume profile."""
        df = df.with_columns(
            ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID_PRICE"),
            (pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("Volume")
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
            (pl.col("ASK") - pl.col("BID")).alias("Spread"),
            ((pl.col("ASK") - pl.col("BID")) / pl.col("MID_PRICE")).alias("Relative_Spread")
        )
        
        # Calculate order book pressure
        df = df.with_columns(
            (pl.col("BIDSIZ") / (pl.col("BIDSIZ") + pl.col("ASKSIZ"))).alias("Bid_Pressure"),
            (pl.col("Spread") * pl.col("Volume")).alias("Dollar_Volume")
        )
        
        # Calculate market depth ratios
        df = df.with_columns(
            (pl.col("BIDSIZ") / pl.col("BIDSIZ").rolling_mean(window_size=100)).alias("Bid_Depth_Ratio"),
            (pl.col("ASKSIZ") / pl.col("ASKSIZ").rolling_mean(window_size=100)).alias("Ask_Depth_Ratio")
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
                (pl.col("ASK").rolling_max(window_size=self.volatility_window) -
                 pl.col("BID").rolling_min(window_size=self.volatility_window)) /
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
        print("\nCritical Debug Information:")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input DataFrame columns: {df.columns}")
        
        # Calculate basic metrics first
        df = df.with_columns(
            ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID_PRICE"),
            (pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("Volume")
        )
        
        # Print sample of raw data
        print("\nSample of raw data:")
        print(df.select(["BID", "ASK", "BIDSIZ", "ASKSIZ", "MID_PRICE", "Volume"]).head())
        
        # Calculate VWAP
        df = df.with_columns(
            (
                (pl.col("MID_PRICE") * pl.col("Volume")).rolling_sum(window_size=self.vwap_window)
                / pl.col("Volume").rolling_sum(window_size=self.vwap_window)
            ).alias("VWAP")
        )
        
        # Calculate bid pressure
        df = df.with_columns(
            (pl.col("BIDSIZ") / (pl.col("BIDSIZ") + pl.col("ASKSIZ"))).alias("Bid_Pressure")
        )
        
        # Print sample of calculated metrics
        print("\nSample of calculated metrics:")
        print(df.select(["MID_PRICE", "VWAP", "Bid_Pressure"]).head())
        
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
        
        # Print signal distribution
        signal_counts = df.group_by("Signal").count()
        print("\nSignal Distribution:")
        print(signal_counts)
        
        # Filter signals by trading hours
        df = df.with_columns(
            pl.when(
                (pl.col("TIME_M") < pl.time(*self.start_time)) | 
                (pl.col("TIME_M") > pl.time(*self.end_time))
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
        
        if risk_per_share == 0:
            return 0
            
        position_size = int(risk_amount / risk_per_share)
        return min(position_size, self.max_position)

    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enhanced backtesting with detailed performance tracking and risk management."""
        # Generate signals first if they don't exist
        if "Signal" not in df.columns:
            df = self.generate_signals(df)
        
        account_balance = []
        positions = []
        trades = []
        last_signal = 0
        daily_returns = []
        current_day = None
        
        print("\nStarting backtest...")
        print(f"Initial cash: ${self.cash:,.2f}")
        
        for i, row in enumerate(df.iter_rows(named=True)):
            signal = row["Signal"]
            mid_price = row["MID_PRICE"]
            current_portfolio = self.cash + (self.position * mid_price if self.position != 0 else 0)
            
            # Process existing position
            if self.position != 0:
                unrealized_pnl = (mid_price - self.entry_price) * self.position
                unrealized_pnl_pct = unrealized_pnl / (self.entry_price * abs(self.position))
                
                # Dynamic stop loss and take profit
                if (self.position > 0 and unrealized_pnl_pct <= -self.stop_loss_pct/100) or \
                   (self.position < 0 and unrealized_pnl_pct >= self.stop_loss_pct/100):
                    # Stop loss hit
                    self.cash += mid_price * self.position
                    trades.append(("Stop Loss", self.position, self.entry_price, mid_price))
                    self.position = 0
                    self.entry_price = 0
                    
                elif (self.position > 0 and unrealized_pnl_pct >= self.profit_target_pct/100) or \
                     (self.position < 0 and unrealized_pnl_pct <= -self.profit_target_pct/100):
                    # Take profit hit
                    self.cash += mid_price * self.position
                    trades.append(("Take Profit", self.position, self.entry_price, mid_price))
                    self.position = 0
                    self.entry_price = 0
            
            # Process new signals
            if signal != 0 and signal != last_signal:
                # Calculate position size (more aggressive)
                position_size = min(
                    int(current_portfolio * self.risk_per_trade / mid_price),
                    self.max_position
                )
                
                if signal == 1 and self.position <= 0:
                    # Close any existing short position
                    if self.position < 0:
                        self.cash -= row["ASK"] * abs(self.position)
                        trades.append(("Close Short", self.position, self.entry_price, row["ASK"]))
                        self.position = 0
                    
                    # Open long position
                    if position_size > 0 and self.cash >= row["ASK"] * position_size:
                        self.cash -= row["ASK"] * position_size
                        self.position = position_size
                        self.entry_price = row["ASK"]
                        trades.append(("Buy", position_size, row["ASK"], None))
                        print(f"Opening long position: {position_size} shares at ${row['ASK']:.2f}")
                        
                elif signal == -1 and self.position >= 0:
                    # Close any existing long position
                    if self.position > 0:
                        self.cash += row["BID"] * self.position
                        trades.append(("Close Long", self.position, self.entry_price, row["BID"]))
                        self.position = 0
                    
                    # Open short position
                    if position_size > 0:
                        self.cash += row["BID"] * position_size
                        self.position = -position_size
                        self.entry_price = row["BID"]
                        trades.append(("Sell", -position_size, row["BID"], None))
                        print(f"Opening short position: {position_size} shares at ${row['BID']:.2f}")
            
            # Update tracking variables
            current_balance = self.cash + (self.position * mid_price if self.position != 0 else 0)
            account_balance.append(current_balance)
            positions.append(self.position)
            last_signal = signal
            
            # Print trade information when signal changes
            if signal != 0 and signal != last_signal:
                print(f"\nSignal change at index {i}:")
                print(f"Signal: {signal}")
                print(f"Mid Price: ${mid_price:.2f}")
                print(f"Current Portfolio: ${current_portfolio:,.2f}")
                print(f"Bid Pressure: {row.get('Bid_Pressure', 'N/A')}")
                print(f"VWAP: {row.get('VWAP', 'N/A')}")
        
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
            pl.col("Account_Balance").rolling_max(window_size=self.vwap_window).alias("Peak_Value")
        )
        
        df = df.with_columns(
            ((pl.col("Peak_Value") - pl.col("Account_Balance")) / pl.col("Peak_Value")).alias("Drawdown")
        )
        
        # Calculate performance metrics
        avg_daily_return = df.select(pl.col("Returns").mean()).item()
        std_daily_return = df.select(pl.col("Returns").std()).item()
        risk_free_rate = 0.01 / 252  # Assuming 1% annual risk-free rate
        
        sharpe_ratio = ((avg_daily_return - risk_free_rate) / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Calculate Sortino Ratio
        downside_returns = df.filter(pl.col("Returns") < 0).select("Returns")
        downside_deviation = downside_returns.select(pl.col("Returns").std()).item() if len(downside_returns) > 0 else 0
        sortino_ratio = ((avg_daily_return - risk_free_rate) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        max_drawdown = df.select(pl.col("Drawdown").max()).item()
        
        # Print performance metrics
        print(f"\nStrategy Performance Metrics:")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Sortino Ratio: {sortino_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4%}")
        print(f"Total Return: {(account_balance[-1] / account_balance[0] - 1):.4%}")
        print(f"Number of Trades: {len(trades)}")
        
        # Calculate win rate and profit metrics
        profitable_trades = sum(1 for t in trades if 
                              (t[0] in ["Take Profit", "Close Long", "Close Short"]) and
                              ((t[1] > 0 and t[3] > t[2]) or (t[1] < 0 and t[3] < t[2])))
        win_rate = profitable_trades / len(trades) if trades else 0
        print(f"Win Rate: {win_rate:.2%}")
        
        # Calculate average profit per trade
        if trades:
            profits = [(t[3] - t[2]) * t[1] for t in trades if t[3] is not None]
            avg_profit = sum(profits) / len(profits) if profits else 0
            print(f"Average Profit per Trade: ${avg_profit:.2f}")
        
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
        
        # Run backtest for each strategy
        for name, strategy in self.strategies.items():
            strategy_data = data.clone()
            results[name] = strategy.backtest(strategy_data)
            
        # Combine results
        for i in range(len(data)):
            portfolio_value = 0
            for name, result in results.items():
                portfolio_value += result["Account_Balance"][i]
            portfolio_values.append(portfolio_value)
            
        # Add portfolio value to results
        results["Portfolio"] = data.with_columns(
            pl.Series("Portfolio_Value", portfolio_values)
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
                 start_time: tuple = (9, 30, 865), end_time: tuple = (16, 28, 954)):
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
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.start_time = start_time
        self.end_time = end_time
        self.position_hold_time = 0
        self.max_hold_time = 100

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP and price deviation metrics."""
        df = df.with_columns(
            ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID_PRICE"),
            (pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("Volume")
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
        df = self.calculate_vwap(df)
        print("VWAP non-nulls:", df["VWAP"].drop_nulls().len())
        print("Price_Deviation non-nulls:", df["Price_Deviation"].drop_nulls().len())
        df = self.calculate_volatility(df)
        print("Volatility non-nulls:", df["Volatility"].drop_nulls().len())
        print("Volume_Ratio non-nulls:", df["Volume_Ratio"].drop_nulls().len())
        df = self.calculate_mean_reversion_score(df)
        print("Mean_Reversion_Score non-nulls:", df["Mean_Reversion_Score"].drop_nulls().len())
        
        # Debug prints for min/max/range of key metrics
        print("\nKey Metrics Range:")
        print("Price_Deviation min:", df["Price_Deviation"].min())
        print("Price_Deviation max:", df["Price_Deviation"].max())
        print("Mean_Reversion_Score min:", df["Mean_Reversion_Score"].min())
        print("Mean_Reversion_Score max:", df["Mean_Reversion_Score"].max())
        print("Volume_Ratio min:", df["Volume_Ratio"].min())
        print("Volume_Ratio max:", df["Volume_Ratio"].max())
        
        # Debug prints
        print("\nMean Reversion Strategy Debug:")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input DataFrame columns: {df.columns}")
        print("\nSample of calculated metrics:")
        print(df.select(["MID_PRICE", "VWAP", "Price_Deviation", "Mean_Reversion_Score", "Volume_Ratio", "Volatility"]).head())
        
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
        
        # Print signal distribution
        signal_counts = df.group_by("Signal").count()
        print("\nSignal Distribution:")
        print(signal_counts)
        
        # Filter signals by trading hours
        df = df.with_columns(
            pl.when(
                (pl.col("TIME_M") < pl.time(*self.start_time)) | 
                (pl.col("TIME_M") > pl.time(*self.end_time))
            )
            .then(pl.lit(0))
            .otherwise(pl.col("Signal"))
            .alias("Signal")
        )
        
        return df
        
    def calculate_position_size(self, price: float, volatility: float, portfolio_value: float) -> int:
        """Calculate optimal position size based on risk and volatility."""
        if volatility == 0:
            return 0
            
        # Risk-based position sizing
        risk_amount = portfolio_value * self.risk_per_trade
        risk_per_share = price * volatility * 2
        
        if risk_per_share == 0:
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
            mid_price = row["MID_PRICE"]
            current_portfolio = self.cash + (self.position * mid_price if self.position != 0 else 0)
            
            # Calculate position size
            volatility = row.get("Volatility", 0.01)
            
            # Process existing position
            if self.position != 0:
                unrealized_pnl = (mid_price - self.entry_price) * self.position
                unrealized_pnl_pct = unrealized_pnl / (self.entry_price * abs(self.position))
                
                # Increment position hold time
                self.position_hold_time += 1
                
                # Check if max hold time is exceeded
                if self.position_hold_time >= self.max_hold_time:
                    # Close position due to max hold time
                    self.cash += mid_price * self.position
                    trades.append(("Max Hold Time", self.position, self.entry_price, mid_price))
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
                
                elif (self.position > 0 and unrealized_pnl_pct <= -self.stop_loss_pct/100) or \
                     (self.position < 0 and unrealized_pnl_pct >= self.stop_loss_pct/100):
                    # Stop loss hit
                    self.cash += mid_price * self.position
                    trades.append(("Stop Loss", self.position, self.entry_price, mid_price))
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
                    
                elif (self.position > 0 and unrealized_pnl_pct >= self.profit_target_pct/100) or \
                     (self.position < 0 and unrealized_pnl_pct <= -self.profit_target_pct/100):
                    # Take profit hit
                    self.cash += mid_price * self.position
                    trades.append(("Take Profit", self.position, self.entry_price, mid_price))
                    self.position = 0
                    self.entry_price = 0
                    self.position_hold_time = 0
            
            # Process new signals
            if signal != 0 and signal != last_signal:
                position_size = self.calculate_position_size(mid_price, volatility, current_portfolio)
                
                if signal == 1 and self.position <= 0:
                    # Close any existing short position
                    if self.position < 0:
                        self.cash -= row["ASK"] * abs(self.position)
                        trades.append(("Close Short", self.position, self.entry_price, row["ASK"]))
                        self.position = 0
                        self.position_hold_time = 0
                    
                    # Open long position
                    if position_size > 0 and self.cash >= row["ASK"] * position_size:
                        self.cash -= row["ASK"] * position_size
                        self.position = position_size
                        self.entry_price = row["ASK"]
                        trades.append(("Buy", position_size, row["ASK"], None))
                        self.position_hold_time = 0
                        
                elif signal == -1 and self.position >= 0:
                    # Close any existing long position
                    if self.position > 0:
                        self.cash += row["BID"] * self.position
                        trades.append(("Close Long", self.position, self.entry_price, row["BID"]))
                        self.position = 0
                        self.position_hold_time = 0
                    
                    # Open short position
                    if position_size > 0:
                        self.cash += row["BID"] * position_size
                        self.position = -position_size
                        self.entry_price = row["BID"]
                        trades.append(("Sell", -position_size, row["BID"], None))
                        self.position_hold_time = 0
            
            # Update tracking variables
            current_balance = self.cash + (self.position * mid_price if self.position != 0 else 0)
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
            pl.col("Account_Balance").rolling_max(window_size=self.vwap_window).alias("Peak_Value")
        )
        
        df = df.with_columns(
            ((pl.col("Peak_Value") - pl.col("Account_Balance")) / pl.col("Peak_Value")).alias("Drawdown")
        )
        
        # Calculate performance metrics
        avg_daily_return = df.select(pl.col("Returns").mean()).item()
        std_daily_return = df.select(pl.col("Returns").std()).item()
        risk_free_rate = 0.01 / 252  # Assuming 1% annual risk-free rate
        
        sharpe_ratio = ((avg_daily_return - risk_free_rate) / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Calculate Sortino Ratio
        downside_returns = df.filter(pl.col("Returns") < 0).select("Returns")
        downside_deviation = downside_returns.select(pl.col("Returns").std()).item() if len(downside_returns) > 0 else 0
        sortino_ratio = ((avg_daily_return - risk_free_rate) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        max_drawdown = df.select(pl.col("Drawdown").max()).item()
        
        # Print performance metrics
        print(f"\nMean Reversion Strategy Performance Metrics:")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Sortino Ratio: {sortino_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4%}")
        print(f"Total Return: {(account_balance[-1] / account_balance[0] - 1):.4%}")
        print(f"Number of Trades: {len(trades)}")
        
        # Calculate win rate and profit metrics
        profitable_trades = sum(1 for t in trades if 
                              (t[0] in ["Take Profit", "Close Long", "Close Short"]) and
                              ((t[1] > 0 and t[3] > t[2]) or (t[1] < 0 and t[3] < t[2])))
        win_rate = profitable_trades / len(trades) if trades else 0
        print(f"Win Rate: {win_rate:.2%}")
        
        # Calculate average profit per trade
        if trades:
            profits = [(t[3] - t[2]) * t[1] for t in trades if t[3] is not None]
            avg_profit = sum(profits) / len(profits) if profits else 0
            print(f"Average Profit per Trade: ${avg_profit:.2f}")
        
        return df

