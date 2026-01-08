import pandas as pd
import dask.dataframe as dd
from config import setup_cpu_cluster, OUTPUT_PATH
from pathlib import Path

def main():
    ranked_dir = Path(OUTPUT_PATH) / "callgraph_ranked_ms"
    matched_files = sorted(ranked_dir.glob("callgraph_ranked_ms_day_*.csv"))
    if not matched_files:
        raise FileNotFoundError(
            f"Required ranked CSVs not found under: {ranked_dir}\n"
            "Run callgraph_rank_ms.py first."
        )

    ms_set = set()
    for ranked_file in matched_files:
        ranked_df = pd.read_csv(ranked_file, usecols=["dm"])
        ms_set.update(ranked_df["dm"].dropna().unique().tolist())
    ms_list = sorted(ms_set)
    
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)
        
        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        
        # Read MSMetrics for the ranked MS

        print(f"Reading MSMetrics for {len(ms_list)} MSs")

        days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for day in days:
            df = dd.read_parquet(
                datapath,
                split_row_groups=True,
                columns=["msname", "day", "hour","cpu_utilization"],
                filters=[
                ("msname", "in", ms_list),
                ("day", "==", day),
                ],
            )
            df = df.clear_divisions()
            # Add a count column for aggregation
            df = df.assign(count=1)
            
            # Aggregate by msname (instances aggregated)
            # Compute avg CPU utilization and count in a single groupby
            coverage = (
                df.groupby(["msname", "day", "hour"])
                .agg({
                    "cpu_utilization": "mean",
                    "count": "sum"
                })
                .reset_index()
                .rename(columns={"cpu_utilization": "avg_cpu_utilization"})
            )
            
            # Compute
            coverage_df = coverage.compute()
            
            # Save CSV
            coverage_path = Path(OUTPUT_PATH) / Path("msmetrics_coverage") / f"msmetrics_coverage_day_{day}.csv"
            
            coverage_df.to_csv(coverage_path, index=False)
            
            print(f"Wrote coverage for day {day} to {coverage_path}")

            #gc.collect()


if __name__ == "__main__":
    main()

