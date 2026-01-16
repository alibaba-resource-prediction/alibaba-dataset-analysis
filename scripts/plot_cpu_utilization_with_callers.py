"""
Plot CPU utilization with callers using the MS in plots/top_cpu_utilization/top_cpu_utilization_day_<day>.txt

For each day:
- Reads MS list from plots/top_cpu_utilization/top_cpu_utilization_day_<day>.txt
- Finds top callers for each MS using graph_edges_full.csv
- Loads MSMetrics data for the MS and their callers
- Plots CPU utilization time series
- Saves to plots/top_cpu_utilization_with_callers/<day>/<MSNAME_<N_CALLERS>_callers.png

this helps us see visually how the CPU utilization of a MS is affected by its callers
Next it would be interesting to either overlay when the calls happened in the timeline or do a correlation analysis
"""

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUTPUT_PATH, setup_cpu_cluster, DAYS

def main():
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        edges_path = OUTPUT_PATH + "graph_edges_full.csv"
        
        edges = pd.read_csv(edges_path)
        # Filter out invalid MS names
        bad = {"UNKNOWN", "UNAVAILABLE", "USER"}
        edges = edges[
            (~edges["dm"].isin(bad)) &
            (~edges["um"].isin(bad))
        ]
        
        for day in DAYS:
            print(f"\nProcessing day {day}...")
            
            ms_file = Path(f"plots/top_cpu_utilization/top_cpu_utilization_day_{day}.txt")
            
            with open(ms_file, 'r') as f:
                ms_list = [line.strip() for line in f if line.strip()]
            
            print(f"  Found {len(ms_list)} MS to process")
            
            ms_callers_map = {}  # ms -> list of top caller MS names
            ms_caller_counts_map = {}  # ms -> dict of caller -> count
            all_ms_to_load = set()  # All unique MS we need data for
            
            for ms in ms_list:
                ms_callers = edges[edges["dm"] == ms].sort_values("count", ascending=False)
                
                if ms_callers.empty:
                    continue
                
                # Get top callers (limit to reasonable number for plotting)
                top_callers = ms_callers.head(10)["um"].tolist()
                caller_counts = ms_callers.head(10)[["um", "count"]].set_index("um")["count"].to_dict()
                
                ms_callers_map[ms] = top_callers
                ms_caller_counts_map[ms] = caller_counts
                all_ms_to_load.add(ms)
                all_ms_to_load.update(top_callers)
            
            if not all_ms_to_load:
                print(f"  No MS with callers found, skipping day {day}")
                continue
            
            all_ms_to_load = list(all_ms_to_load)
            print(f"  Loading data for {len(all_ms_to_load)} unique MS (targets + callers)...")
            
            try:
                ms_metrics = dd.read_parquet(
                    datapath,
                    split_row_groups=True,
                    columns=["day", "timestamp", "msname", "cpu_utilization"],
                    filters=[("day", "==", day), ("msname", "in", all_ms_to_load)],
                )
                
                # Compute and aggregate by msname and timestamp (mean across instances)
                metrics_df = (
                    ms_metrics.groupby(["msname", "timestamp"])["cpu_utilization"]
                    .mean()
                    .reset_index()
                ).compute()
                
                
            except Exception as e:
                print(f"  Error loading metrics data for day {day}: {e}")
                continue
            
            # Create output directory for this day
            output_dir = Path(f"plots/top_cpu_utilization_with_callers/{day}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Second pass: plot each MS using pre-loaded data
            for ms in ms_list:
                print(f"    Processing {ms}...")
                
                if ms not in ms_callers_map:
                    print(f"      No callers found for {ms}, skipping")
                    continue
                
                top_callers = ms_callers_map[ms]
                caller_counts = ms_caller_counts_map[ms]
                
                # Check if target MS has data
                target_data = metrics_df[metrics_df["msname"] == ms]
                if target_data.empty:
                    print(f"      No metrics data found for target {ms} on day {day}, skipping")
                    continue
                
                # Create plot
                plt.figure(figsize=(14, 8))
                
                # Plot target MS
                target_sorted = target_data.sort_values("timestamp")
                plt.plot(
                    target_sorted["timestamp"],
                    target_sorted["cpu_utilization"],
                    label=f"{ms} (target)",
                    linewidth=2.5,
                    color="red",
                    marker="o",
                    markersize=3
                )
                
                # Plot callers
                found_callers = 0
                for caller in top_callers:
                    caller_data = metrics_df[metrics_df["msname"] == caller]
                    if not caller_data.empty:
                        caller_sorted = caller_data.sort_values("timestamp")
                        count = caller_counts.get(caller, 0)
                        plt.plot(
                            caller_sorted["timestamp"],
                            caller_sorted["cpu_utilization"],
                            label=f"{caller} (calls: {count})",
                            alpha=0.7,
                            linewidth=1.5,
                            marker="o",
                            markersize=2
                        )
                        found_callers += 1
                
                plt.title(f"CPU Utilization for {ms} and Top Callers (Day {day})")
                plt.xlabel("Timestamp")
                plt.ylabel("Mean CPU Utilization")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                output_path = output_dir / f"{ms}_{found_callers}_callers.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"      Saved plot: {output_path} (showing {found_callers} callers)")
        
        print("\nDone!")


if __name__ == "__main__":
    main()