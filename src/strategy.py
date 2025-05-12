import polars as pl

class OBIVWAPStrategy:
    def __init__(self, vwap_window: int, obi_threshold: float, initial_cash: float = 100_000):
        self.vwap_window = vwap_window
        self.obi_threshold = obi_threshold
        self.cash = initial_cash
        self.position = 0
        self.account_balance = []

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
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
        return df

    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        account_balance = []
        for row in df.iter_rows(named=True):
            if row["Signal"] == 1 and self.cash >= row["ASK"] * 100 and self.position <= 1:
                self.position = 100
                self.cash -= row["ASK"] * 100
            elif row["Signal"] == -1 and self.position > -1:
                self.position = -100
                self.cash += row["BID"] * 100
            elif row["Signal"] == 0 and self.position != 0:
                if self.position > 0:
                    self.cash += row["BID"] * self.position
                else:
                    self.cash -= row["ASK"] * abs(self.position)
                self.position = 0
            account_balance.append(self.cash + self.position * (row["ASK"] if self.position > 0 else row["BID"]))
        df = df.with_columns(pl.Series("Account_Balance", account_balance))
        return df