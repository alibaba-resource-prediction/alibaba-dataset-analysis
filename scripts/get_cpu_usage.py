import dask.dataframe as dd
from config import setup_cpu_cluster, OUTPUT_PATH

def main():
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        target_days = [0, 1, 2]
        target_ms = ['MS_63670', 'MS_53154', 'MS_15284', 'MS_66431']
        ms_metrics = dd.read_parquet(datapath,
        split_row_groups=True,
        columns=["day", "timestamp", "msname", "cpu_utilization"],
        filters=[("day", "in", target_days), ("msname", "in", target_ms)]
        )

        g = ms_metrics#.groupby("msname")

        print(g)
        
        # write to csv file with day , timestamp , msname , cpu_utilization
        g.compute().to_csv(OUTPUT_PATH + "cpu_usage.csv", index=False, columns=["day", "timestamp", "msname", "cpu_utilization"])

if __name__ == "__main__":
    main()
