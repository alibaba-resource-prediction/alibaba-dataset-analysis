from dask.distributed import Client
import dask.dataframe as dd
from config import setup_gpu_cluster
def main():

    with setup_gpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)
        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        target = 0

        df = dd.read_parquet(
            datapath,
            split_row_groups=True,
            columns=["day", "msname", "cpu_utilization"],
            filters=[("day", "==", target)],
        )

        result = df.groupby("msname").agg({"cpu_utilization": "mean"})
        print(result.compute().head())


if __name__ == "__main__":
    main()