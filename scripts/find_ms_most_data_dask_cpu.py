import pandas as pd
import pyarrow.dataset as ds
import os
import dask.dataframe as dd

from config import setup_cpu_cluster, OUTPUT_PATH

def main():
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        datapath = "/home/mpds/data/bronze/table=MSMetrics"
    
        ms_metrics = dd.read_parquet(datapath,
        split_row_groups=True,
        columns=["day", "msname", "cpu_utilization"],
        )

        g = ms_metrics.groupby("msname", "day").size().compute()

        res = g.sort_values(by=["day", "size"], ascending=[True, False])

        res.to_csv(OUTPUT_PATH + "ms_most_data.csv", index=False)
        

if __name__ == "__main__":
    main()