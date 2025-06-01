import polars as pl
import yfinance as yf
import random
from typing import List
from datetime import datetime as dttm

def select_random_lowcap_spy_tickers(
    start_date: str,
    sp500_csv_path: str,
    master_file_path: str,
    n_lowest: int = 100,
    n_select: int = 10,
    random_seed: int = 42
) -> List[str]:
    """
    Selects random low market cap S&P 500 tickers as of a given date that are present in the master file.

    Args:
        start_date (str): The date in 'YYYY-MM-DD' format.
        sp500_csv_path (str): Path to the S&P 500 historical components CSV.
        master_file_path (str): Path to the master file containing valid tickers (one per line or in a column named 'ticker').
        n_lowest (int): Number of lowest market cap tickers to consider.
        n_select (int): Number of random tickers to select from the lowest group.
        random_seed (int): Seed for reproducibility.

    Returns:
        List[str]: List of selected ticker symbols.
    """
    # Load S&P 500 historical components
    df = pl.read_csv(sp500_csv_path)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    start_date_dt = dttm.strptime(start_date, "%Y-%m-%d")
    df = df.filter(pl.col("date") <= start_date_dt)
    if df.is_empty():
        raise ValueError("No tickers found for or before the given start_date.")
    tickers_str = df.sort("date").select("tickers").to_series()[-1]
    tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
    print(f"Found {len(tickers)} tickers in S&P 500 as of {start_date}.")

    # Download market cap data using yfinance
    market_caps = {}
    import datetime
    prev_close_date = (datetime.datetime.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    for ticker in tickers: 
        try:
            info = yf.Ticker(ticker).info
            shares = info.get('sharesOutstanding', None)
            hist = yf.Ticker(ticker).history(start=prev_close_date, end=start_date)
            if shares is not None and not hist.empty:
                previous_close = hist['Close'].iloc[0]
                cap = shares * previous_close
                market_caps[ticker] = cap
        except Exception:
            continue
    print(f"Retrieved market cap for {len(market_caps)} tickers.")

    # Sort by market cap and take the lowest n_lowest
    lowcap_sorted = sorted(market_caps.items(), key=lambda x: x[1])
    lowcap_tickers = [t[0] for t in lowcap_sorted[:n_lowest]]

    # Load master file and filter tickers
    master = pl.read_csv(master_file_path)
    if "symbol_15" in master.columns:
        master_tickers = set(master["symbol_15"].to_list())
    else:
        master_tickers = set(master.select(pl.col(master.columns[0])).to_series().to_list())
    filtered = [t for t in lowcap_tickers if t.upper() in {x.upper() for x in master_tickers}]

    # Save all filtered tickers to a text file
    with open("data/filtered_tickers.txt", "w") as f:
        print(f"Saving {len(filtered)} filtered tickers to 'filtered_tickers.txt'.")
        for t in filtered:
            f.write(f"{t}\n")

    # Randomly select n_select tickers
    random.seed(random_seed)
    selected = random.sample(filtered, min(n_select, len(filtered)))
    print(f"Selected {len(selected)} random tickers from the filtered list.")

    return selected

# Example usage:
if __name__ == "__main__":
    tickers = select_random_lowcap_spy_tickers(
        start_date="2023-03-10",
        sp500_csv_path="data/S&P 500 Historical Components & Changes(03-10-2025).csv",
        master_file_path="data/master_file.csv",  # Update this path as needed
        n_lowest=100,
        n_select=10 
    )
    print("Selected tickers:", tickers)