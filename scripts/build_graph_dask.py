from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cudf
import dask
import dask.dataframe as dd

OUTPUT_PATH = "/home/mpds/data/analysis/"
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

    datapath = "/home/mpds/data/bronze/table=CallGraph"
    target = 0
    df = dd.read_parquet(datapath,
    split_row_groups=True,
    columns=["day","um","dm"],
    filters=[("day", "==", target)],
    )
    print(df.head())

    result = (
        df.groupby(["um", "dm"])
        .size()
        .to_frame(name="count")
        .reset_index()
    )
    result.compute().to_csv(OUTPUT_PATH + "graph_edges.csv", index=False)


if __name__ == "__main__":
    main()