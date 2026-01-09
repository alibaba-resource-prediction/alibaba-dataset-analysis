from dask_cuda import LocalCUDACluster
import cudf
from distributed.deploy.local import LocalCluster
from pathlib import Path

OUTPUT_PATH = "/home/mpds/data/analysis/"

# Common data processing constants
DAYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Hour groups for incremental processing
FIRST_4_HOURS = [0, 1, 2, 3]
SECOND_4_HOURS = [4, 5, 6, 7]
THIRD_4_HOURS = [8, 9, 10, 11]
FOURTH_4_HOURS = [12, 13, 14, 15]
FIFTH_4_HOURS = [16, 17, 18, 19]
SIXTH_4_HOURS = [20, 21, 22, 23]
HOUR_GROUPS = [FIRST_4_HOURS, SECOND_4_HOURS, THIRD_4_HOURS, FOURTH_4_HOURS, FIFTH_4_HOURS, SIXTH_4_HOURS]

def get_hour_suffix(hour_group):
    """Get suffix string for hour group."""
    if hour_group == FIRST_4_HOURS:
        return "first_4_hours"
    elif hour_group == SECOND_4_HOURS:
        return "second_4_hours"
    elif hour_group == THIRD_4_HOURS:
        return "third_4_hours"
    elif hour_group == FOURTH_4_HOURS:
        return "fourth_4_hours"
    elif hour_group == FIFTH_4_HOURS:
        return "fifth_4_hours"
    elif hour_group == SIXTH_4_HOURS:
        return "sixth_4_hours"
    return "unknown"

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
        #n_workers=48,
        #memory_limit="auto",
        #threads_per_worker=1,

        ## this config i used for msmetrics_coverage.py
        n_workers=4,
        threads_per_worker=2,
        memory_limit="40GiB",

        local_directory="/tmp/dask-cpu",
        dashboard_address=":42409",
    )
    return cluster