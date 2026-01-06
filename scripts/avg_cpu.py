import dask_cudf
import cudf
import datetime as dt

cudf.set_option("io.parquet.low_memory", True)
datapath = "/home/mpds/data/bronze/table=MSMetrics"
target = 0
df = dask_cudf.read_parquet(datapath,
 split_row_groups=True,
 columns=["day","msname","cpu_utilization"],
 filters=[("day", "==", target)])
print(df.head())

print(df.groupby("msname").agg({"cpu_utilization": "mean"}).head())

# this works and runs on one gpu