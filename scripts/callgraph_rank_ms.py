"""
Rank all destination microservices by total calls and unique callers.
Outputs a sorted CSV with all MS ranked.
"""
import cudf
import dask
import dask.dataframe as dd
from config import setup_cpu_cluster, OUTPUT_PATH
from pathlib import Path
import gc
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
        days = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        first_4_hours = [0, 1, 2, 3 ]
        second_4_hours = [4, 5, 6, 7 ]
        third_4_hours = [8, 9, 10, 11 ]
        fourth_4_hours = [12, 13, 14, 15]
        fifth_4_hours = [16, 17, 18, 19]
        sixth_4_hours = [20, 21, 22, 23]
        hours = [first_4_hours, second_4_hours, third_4_hours, fourth_4_hours, fifth_4_hours, sixth_4_hours]

        resume_from_day = 0
        resume_from_hour = first_4_hours

        for day in days:
            ### this is to resume from a specific day and hour
            if day < resume_from_day:
                continue
            for h in hours:
                if day == resume_from_day and h != resume_from_hour:
                    continue
            ### this is to resume from a specific day and hour

                start_time = time.time()
                df = dd.read_parquet(
                datapath,
                split_row_groups=True,
                columns=["um", "dm"],
                #filesystem="arrow",
                filters=[("day", "==", day), ("hour", "in", h)],
                #dtype_backend="pyarrow"
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

                hour_suffix = "first_4_hours" if h == first_4_hours else "second_4_hours" if h == second_4_hours else "third_4_hours" if h == third_4_hours else "fourth_4_hours" if h == fourth_4_hours else "fifth_4_hours" if h == fifth_4_hours else "sixth_4_hours"

                output_path = Path(OUTPUT_PATH) / "callgraph_ranked_ms" / f"callgraph_ranked_ms_day_{day}_{hour_suffix}.csv"
                result.to_csv(output_path, index=False)

                # free memory
                #del df, total_calls, unique_callers, ranked, result
                #gc.collect()


if __name__ == "__main__":
    main()

