from dask_cuda import LocalCUDACluster
import cudf
from distributed.deploy.local import LocalCluster

OUTPUT_PATH = "/home/mpds/data/analysis/"

def setup_gpu_cluster():
    cluster = LocalCUDACluster(
        n_workers=2,
        threads_per_worker=1,
        device_memory_limit="22GB",
        local_directory="/tmp/dask-cuda",
        dashboard_address=":42409",
    )
    cudf.set_option("io.parquet.low_memory", True)
    return cluster

def setup_cpu_cluster():
    cluster = LocalCluster(
        n_workers=48,
        memory_limit="auto",
        threads_per_worker=1,
        local_directory="/tmp/dask-cpu",
        dashboard_address=":42409",
    )
    return cluster