"""
Rank all destination microservices by total calls and unique callers.
Outputs a sorted CSV with all MS ranked.
"""
import dask.dataframe as dd
from config import (
    setup_cpu_cluster, OUTPUT_PATH, DAYS, HOUR_GROUPS,
    FIRST_4_HOURS, get_hour_suffix
)
from pathlib import Path
import time

times_log_file = "times.log"

def main():
    def log_operation_time_to_file(start, end, operation):
        with open(times_log_file, "a") as f:
            f.write(f"{operation} took {end - start} seconds\n")

    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)
        
        datapath = "/home/mpds/data/bronze/table=CallGraph"
        
        resume_from_day = 0
        resume_from_hour = FIRST_4_HOURS

        for day in DAYS:
            ### this is to resume from a specific day and hour
            if day < resume_from_day:
                continue
            for h in HOUR_GROUPS:
                if day == resume_from_day and h != resume_from_hour:
                    continue
            ### this is to resume from a specific day and hour

                start_time = time.time()
                df = dd.read_parquet(
                    datapath,
                    split_row_groups=True,
                    columns=["um", "dm"],
                    filters=[("day", "==", day), ("hour", "in", h)],
                )
                # Filter out UNKNOWN, UNAVAILABLE, USER
                df = df[~df["dm"].isin(["UNKNOWN", "UNAVAILABLE", "USER"])]
                df = df[~df["um"].isin(["UNKNOWN", "UNAVAILABLE", "USER"])]
                
                # Rank by total calls (count of all calls to each destination)
                total_calls = (
                    df.groupby("dm")
                    .size()
                    .to_frame(name="total_calls")
                    .reset_index()
                )
                
                # Count unique callers per destination
                unique_callers = (
                    df.groupby("dm")["um"]
                    .nunique()
                    .to_frame(name="unique_callers")
                    .reset_index()
                )
                
                # Merge rankings
                ranked = total_calls.merge(unique_callers, on="dm", how="outer")
                ranked = ranked.fillna({"total_calls": 0, "unique_callers": 0})
                
                # Sort by total_calls descending
                ranked = ranked.sort_values("total_calls", ascending=False)
                
                # Compute and save
                result = ranked.compute()

                log_operation_time_to_file(start_time, time.time(), f"Ranking day {day} hours {h}")

                hour_suffix = get_hour_suffix(h)
                output_path = Path(OUTPUT_PATH) / "callgraph_ranked_ms" / f"callgraph_ranked_ms_day_{day}_{hour_suffix}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

