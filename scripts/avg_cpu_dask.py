from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cudf
import dask
import dask.dataframe as dd
def main():
    #https://docs.rapids.ai/api/dask-cudf/stable/#the-dask-dataframe-api-recommended
    dask.config.set({"dataframe.backend": "cudf"})

    cluster = LocalCUDACluster(
        n_workers=2,
        device_memory_limit="22GB",
        threads_per_worker=1,
        local_directory="/tmp/dask-cuda",
    )
    client = cluster.get_client()

    cudf.set_option("io.parquet.low_memory", True)

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