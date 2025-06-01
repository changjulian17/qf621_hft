import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl

def plot_account_balance(backtest_data: pl.DataFrame):
    """
    Plots the account balance over time.

    Args:
        account_balance (pl.DataFrame): 
            A Polars DataFrame containing time-series data for account balance.
        title (str): 
            Title of the plot. Default is "Account Balance Over Time".

    Returns:
        None
    """
    # backtest_data = backtest_data.with_columns(
    #     (pl.col("date").cast(pl.Utf8) + " " + pl.col("time_m").cast(pl.Utf8))
    #     .str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f")
    #     .alias("time_m")
    # )
    pdf = backtest_data.to_pandas()
    for ticker in set(pdf['sym_root']):
        ticker_df = pdf[pdf['sym_root'] == ticker]
        plt.figure(figsize=(12, 6))
        plt.plot(ticker_df["Timestamp"], ticker_df["Account_Balance"])
        plt.xlabel("time_m")
        plt.ylabel("Account_Balance")
        plt.title(f"Account Balance for {ticker}")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        # plt show in main