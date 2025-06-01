import wrds
import polars as pl
import matplotlib.pyplot as plt

def fetch_taq_data(
    tickers,
    exchanges,
    quote_conds,
    start_date,
    end_date,
    start_time="09:30:00",
    end_time="16:00:00",
    wrds_username='changjulian17'
):
    print("Fetching TAQ data from WRDS...")
    db = wrds.Connection(wrds_username=wrds_username) if wrds_username else wrds.Connection()
    tickers_str = ", ".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT
            date,
            time_m,
            time_m_nano,
            ex,
            sym_root,
            sym_suffix,
            bid,
            bidsiz,
            ask,
            asksiz,
            qu_cond,
            qu_seqnum,
            natbbo_ind,
            qu_cancel,
            qu_source,
            rpi,
            ssr,
            luld_bbo_indicator,
            finra_bbo_ind,
            finra_adf_mpid,
            finra_adf_time,
            finra_adf_time_nano,
            finra_adf_mpq_ind,
            finra_adf_mquo_ind,
            sip_message_id,
            natl_bbo_luld,
            part_time,
            part_time_nano,
            secstat_ind
        FROM taqm_2023.cqm_2023
        WHERE sym_root IN ({tickers_str})
          AND ex IN ({exchanges})
          AND qu_cond IN ({quote_conds})
          AND date BETWEEN '{start_date}' AND '{end_date}'
          AND time_m BETWEEN '{start_time}' AND '{end_time}'
    """
    data = db.raw_sql(query)
    print("Data fetched successfully.")

    # Convert to Polars DataFrame first
    data = pl.from_pandas(data)

    data = data.with_columns(
        (pl.col("date").cast(pl.Utf8) + " " 
         + pl.col("time_m").cast(pl.Utf8)
         + pl.col("time_m_nano").cast(pl.Utf8))
        .str.strptime(pl.Datetime("ns"), 
                      format="%Y-%m-%d %H:%M:%S%.f")
        .alias("Timestamp")
    ).sort("Timestamp")

    return data

def main():
    # Configurable parameters
    tickers = ['FTAI', 'WLFC', 'HEES', 'AL', 'GATX', 'ALTG']
    exchanges = "'Q', 'N'"
    start_date = '2023-05-10'
    end_date = '2023-05-11'
    wrds_username = 'changjulian17'  # Set to None to use default

    data = fetch_taq_data(
        tickers=tickers,
        exchanges=exchanges,
        quote_conds="'R'",
        start_date=start_date,
        end_date=end_date,
        wrds_username=wrds_username
    )

    print(data.head())

if __name__ == "__main__":
    main()