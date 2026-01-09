"""
Compute workload statistics for top 1000 MS by calls.
Outputs per-day CSV files with stats per instance.
Processes data day-by-day to avoid memory issues.
"""
import pandas as pd
import dask.dataframe as dd
from config import setup_cpu_cluster, OUTPUT_PATH, DAYS, HOUR_GROUPS, get_hour_suffix
from pathlib import Path


def main():
    # Strict dependencies - need ranked CSV to get top 1000 MS
    ranked_dir = Path(OUTPUT_PATH) / "callgraph_ranked_ms"
    ranked_files = sorted(ranked_dir.glob("callgraph_ranked_ms_day_*.csv"))
    
    if not ranked_files:
        raise FileNotFoundError(
            f"Required ranked CSVs not found under: {ranked_dir}\n"
            "Run callgraph_rank_ms.py first."
        )
    
    # Aggregate ranked data to get top 1000 MS
    print("Aggregating ranked data to find top 1000 MS...")
    total_calls_dict = {}
    for ranked_file in ranked_files:
        df = pd.read_csv(ranked_file)
        for _, row in df.iterrows():
            dm = row["dm"]
            total_calls_dict[dm] = total_calls_dict.get(dm, 0) + row["total_calls"]
    
    # Get top 1000 MS
    ranked_list = sorted(total_calls_dict.items(), key=lambda x: x[1], reverse=True)
    top_1000_ms = [ms for ms, _ in ranked_list[:1000]]
    print(f"Found top 1000 MS (total calls range: {ranked_list[0][1]} to {ranked_list[999][1]})")
    
    # Create rank mapping
    rank_dict = {ms: idx + 1 for idx, ms in enumerate(top_1000_ms)}
    
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)
        
        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        
        # Process day by day and hour group by hour group
        for day in DAYS:
            for h in HOUR_GROUPS:
                print(f"Processing day {day} hours {h}...")
                
                # Read MSMetrics for this day and hour group
                df = dd.read_parquet(
                    datapath,
                    split_row_groups=True,
                    columns=["msname", "day", "hour", "cpu_utilization", "memory_utilization", "msinstanceid"],
                    filters=[
                        ("msname", "in", top_1000_ms),
                        ("day", "==", day),
                        ("hour", "in", h),
                    ],
                )
                
                # Clear divisions to help dask optimize
                df = df.clear_divisions()
                
                # Convert string IDs to integers and drop original columns in one step
                # All operations stay in dask until compute()
                df = df.assign(
                    msname_int=df["msname"].str.extract(r"MS_(\d+)", expand=False).astype("int32"),
                    pod_num_int=df["msinstanceid"].str.extract(r"MS_\d+_POD_(\d+)", expand=False).astype("int32")
                ).drop(columns=["msname", "msinstanceid"])
                
                # Aggregate by integer IDs (much more memory efficient)
                stats = df.groupby(["msname_int", "pod_num_int", "day"]).agg({
                    "cpu_utilization": ["mean", "max"],
                    "memory_utilization": ["mean", "max"],
                })
                
                # Compute aggregated result (should be much smaller than raw data)
                result = stats.compute()
                
                # Flatten MultiIndex columns in pandas first (small dataset, fast)
                result.columns = ["_".join(col).strip() for col in result.columns.values]
                result = result.rename(columns={
                    "cpu_utilization_mean": "cpu_mean",
                    "cpu_utilization_max": "cpu_max",
                    "memory_utilization_mean": "memory_mean",
                    "memory_utilization_max": "memory_max",
                }).reset_index()
                
                # Convert back to dask for remaining operations
                result_dd = dd.from_pandas(result, npartitions=1)
                
                # Convert integer IDs back to original string format in dask
                result_dd = result_dd.assign(
                    msname=("MS_" + result_dd["msname_int"].astype(str)),
                    msinstanceid=("MS_" + result_dd["msname_int"].astype(str) + "_POD_" + result_dd["pod_num_int"].astype(str))
                ).drop(columns=["msname_int", "pod_num_int"])
                
                # Create rank mapping as a dask series
                rank_series = result_dd["msname"].map(rank_dict)
                result_dd = result_dd.assign(rank_by_calls=rank_series)
                
                # Sort by rank, then instance in dask
                result_dd = result_dd.sort_values(["rank_by_calls", "msinstanceid"])
                
                # Compute final result and save CSV
                result = result_dd.compute()
                hour_suffix = get_hour_suffix(h)
                output_path = Path(OUTPUT_PATH) / "msmetrics_workload_stats" / f"msmetrics_workload_stats_day_{day}_{hour_suffix}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(output_path, index=False)
                
                print(f"Wrote workload stats for day {day} hours {h} to {output_path}")
                print(f"Total rows: {len(result)}")


if __name__ == "__main__":
    main()

