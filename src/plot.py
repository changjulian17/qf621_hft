import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl

def plot_account_balance(backtest_data: pl.DataFrame):
    backtest_data = backtest_data.with_columns(
        (pl.col("DATE").cast(pl.Utf8) + " " + pl.col("TIME_M").cast(pl.Utf8))
        .str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f")
        .alias("TIME_M")
    )
    pdf = backtest_data.to_pandas()
    for ticker in set(pdf['SYM_ROOT']):
        ticker_df = pdf[pdf['SYM_ROOT'] == ticker]
        plt.figure(figsize=(12, 6))
        plt.plot(ticker_df["TIME_M"], ticker_df["Account_Balance"])
        plt.xlabel("TIME_M")
        plt.ylabel("Account_Balance")
        plt.title(f"Account Balance for {ticker}")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        # Do not call plt.show() here