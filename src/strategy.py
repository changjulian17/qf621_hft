import polars as pl

class OBIVWAPStrategy:
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
            pl.DataFrame: Data with account balance over time
        """
        account_balance = []

        for row in df.iter_rows(named=True):
            signal = row["Signal"]

            # Buy signal
            if signal == 1 and self.position == 0 and self.cash >= row["ask"] * 100:
                self.cash -= row["ask"] * 100
                self.position = 100

            # Sell signal
            elif signal == -1 and self.position == 0:
                self.cash += row["bid"] * 100
                self.position = -100

            # Close position
            elif signal == 0 and self.position != 0:
                if self.position > 0:
                    self.cash += row["bid"] * self.position
                else:
                    self.cash -= row["ask"] * abs(self.position)
                self.position = 0

            # Mark-to-market balance
            market_price = row["ask"] if self.position > 0 else row["bid"] if self.position < 0 else 0
            account_balance.append(self.cash + self.position * market_price)

        return df.with_columns(pl.Series("Account_Balance", account_balance, dtype=pl.Float64))

