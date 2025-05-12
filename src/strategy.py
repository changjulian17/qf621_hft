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
    def __init__(self, vwap_window: int, obi_threshold: float, initial_cash: float = 100_000, 
                 start_time: tuple = (9, 30, 865), end_time: tuple = (16, 28, 954)):
        """
        Initializes the OBIVWAPStrategy with the specified parameters.

        Args:
            vwap_window (int): Rolling window size for VWAP calculation.
            obi_threshold (float): Threshold for Order Book Imbalance (OBI) signals.
            initial_cash (float): Initial cash balance for the strategy. Default is 100,000.
            start_time (tuple): The earliest time (HH, MM, MS) for generating signals.
            end_time (tuple): The latest time (HH, MM, MS) for generating signals.
        """
        self.vwap_window = vwap_window
        self.obi_threshold = obi_threshold
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
        df = df.with_columns(((pl.col("BID") + pl.col("ASK")) / 2).alias("MID_PRICE"))
        df = df.with_columns((pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("Volume"))
        df = df.with_columns(
            (
                (pl.col("MID_PRICE") * pl.col("Volume")).rolling_sum(window_size=self.vwap_window)
                / pl.col("Volume").rolling_sum(window_size=self.vwap_window)
            ).alias("VWAP")
        )
        return df

    def calculate_obi(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            (
                (pl.col("BIDSIZ") - pl.col("ASKSIZ")) / (pl.col("BIDSIZ") + pl.col("ASKSIZ"))
            ).alias("OBI")
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
            pl.when(pl.col("OBI") > self.obi_threshold)
            .then(1)
            .when(pl.col("OBI") < -self.obi_threshold)
            .then(-1)
            .otherwise(0)
            .alias("Signal")
        )

        # Ensure Signal is 0 when TIME_M is outside the allowed range
        df = df.with_columns(
            pl.when(
                (pl.col("TIME_M") < pl.time(*self.start_time)) | 
                (pl.col("TIME_M") > pl.time(*self.end_time))
            )
            .then(0)
            .otherwise(pl.col("Signal"))
            .alias("Signal")
        )

        return df

    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Simulates the strategy on historical data and returns performance metrics.

        Args:
            data (pl.DataFrame): 
                A Polars DataFrame containing stock market data.

        Returns:
            pl.DataFrame: 
                A Polars DataFrame with columns for account balance and performance metrics.
        """
        account_balance = []
        for row in df.iter_rows(named=True):
            if row["Signal"] == 1 and self.cash >= row["ASK"] * 100 and self.position <= 1:
                self.position = 100
            elif row["Signal"] == -1 and self.position > -1:
                self.position = -100
            elif row["Signal"] == 0 and self.position != 0:
                if self.position > 0:
                    self.cash += row["BID"] * self.position
                else:
                    self.cash -= row["ASK"] * abs(self.position)
                self.position = 0
            account_balance.append(self.cash + self.position * (row["ASK"] if self.position > 0 else row["BID"]))
        df = df.with_columns(pl.Series("Account_Balance", account_balance))
        return df