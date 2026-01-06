import dask_cudf
import cudf
import datetime as dt

cudf.set_option("io.parquet.low_memory", True)
datapath = "/home/mpds/data/bronze/table=CallGraph"
target = 0
df = dask_cudf.read_parquet(datapath,
 split_row_groups=True,
 columns=["day","traceid","rpc_id"],
 filters=[("day", "==", target)])
print(df.head())

df.groupby("traceid").agg({"rpc_id": "count"}).head()