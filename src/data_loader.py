import polars as pl

def load_and_filter_data(
    path: str, 
    ex_filter: str = "Q", 
    qu_cond_filter: str = "R", 
    sym_root: str = None
) -> pl.DataFrame:
    """
    Load and filter stock market data from a CSV file.

    This function reads a CSV file containing stock market data, processes it, 
    and filters the data based on specified criteria such as exchange, quote 
    condition, and stock ticker. It also ensures that the data is restricted 
    to trading hours (9:30 AM to 4:00 PM).

    Args:
        path (str): 
            Path to the CSV file containing the stock market data.
        ex_filter (str): 
            Exchange filter. Only rows with the specified exchange code 
            in the "EX" column will be included. Default is "Q".
        qu_cond_filter (str): 
            Quote condition filter. Only rows with the specified quote 
            condition in the "QU_COND" column will be included. Default is "R".
        sym_root (str, optional): 
            Stock ticker filter. If provided, only rows with the specified 
            stock ticker in the "SYM_ROOT" column will be included.

    Returns:
        pl.DataFrame: 
            A Polars DataFrame containing the filtered stock market data.

    Raises:
        FileNotFoundError: 
            If the specified file does not exist.
        ValueError: 
            If the CSV file does not contain the required columns.

    Example:
        >>> df = load_and_filter_data(
        ...     path="data/3_stock_trading_hrs.csv", 
        ...     ex_filter="Q", 
        ...     qu_cond_filter="R", 
        ...     sym_root="AAPL"
        ... )
        >>> print(df)
    """
    # Read the CSV file into a Polars DataFrame
    df = pl.read_csv(path)

    # Convert the "TIME_M" column to time format and alias it
    df = df.with_columns(
        pl.col("TIME_M").str.to_time(format="%H:%M:%S%.f").alias("TIME_M")
    )

    # Filter rows to include only those within trading hours (9:30 AM to 4:00 PM)
    df = df.filter(
        pl.col("TIME_M").is_between(pl.time(9, 30), pl.time(16, 0))
    )

    # Apply exchange filter if specified
    if ex_filter:
        df = df.filter(pl.col("EX") == ex_filter)

    # Apply quote condition filter if specified
    if qu_cond_filter:
        df = df.filter(pl.col("QU_COND") == qu_cond_filter)

    # Apply stock ticker filter if specified
    if sym_root:
        df = df.filter(pl.col("SYM_ROOT") == sym_root)

    # Combine DATE and TIME_M into a single datetime column and sort by it
    df = df.with_columns(
        (pl.col("DATE").cast(pl.Utf8) + " " + pl.col("TIME_M").cast(pl.Utf8))
        .str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f")
        .alias("Timestamp")
    ).sort("Timestamp")

    return df