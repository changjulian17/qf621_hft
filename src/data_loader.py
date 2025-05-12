import polars as pl

def load_and_filter_data(
    path: str, 
    ex_filter: str = "Q", 
    qu_cond_filter: str = "R", 
    sym_root: str = None
) -> pl.DataFrame:
    """
    Load and filter data from a CSV file.

    Args:
        path (str): Path to the CSV file.
        ex_filter (str): Exchange filter.
        qu_cond_filter (str): Quote condition filter.
        sym_root (str, optional): Stock ticker to filter by (SYM_ROOT column).

    Returns:
        pl.DataFrame: Filtered DataFrame.
    """
    df = pl.read_csv(path)
    df = df.with_columns(
        pl.col("TIME_M").str.to_time(format="%H:%M:%S%.f").alias("TIME_M")
    )
    df = df.filter(
        pl.col("TIME_M").is_between(pl.time(9, 30), pl.time(16, 0))
    )
    if ex_filter:
        df = df.filter(pl.col("EX") == ex_filter)
    if qu_cond_filter:
        df = df.filter(pl.col("QU_COND") == qu_cond_filter)
    if sym_root:
        df = df.filter(pl.col("SYM_ROOT") == sym_root)
    return df