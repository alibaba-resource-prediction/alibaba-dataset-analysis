import dask.dataframe as dd

from config import with_cuda_client, OUTPUT_PATH
@with_cuda_client
def main(client):
    datapath = "/home/mpds/data/bronze/table=CallGraph"
    #target = 0
    df = dd.read_parquet(datapath,
    split_row_groups=True,
    columns=["day","um","dm"],
    #filters=[("day", "==", target)],
    )
    print(df.head())

    result = (
        df.groupby(["um", "dm"])
        .size()
        .to_frame(name="count")
        .reset_index()
    )
    result.compute().to_csv(OUTPUT_PATH + "graph_edges_full.csv", index=False)


if __name__ == "__main__":
    main()