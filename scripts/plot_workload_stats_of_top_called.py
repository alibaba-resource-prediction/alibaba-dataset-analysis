"""
Plot CPU and memory utilization by hour for top N MS from callgraph ranked data.
Reads raw MSMetrics data for the top MS from callgraph ranked CSVs.
Iterates through all days.

In the plots created there is very little data. I dont get it... how could the most called MS in the call graph have so little data for their cpu and memory utilization?
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
from config import OUTPUT_PATH, setup_cpu_cluster, DAYS
from pathlib import Path


def main():
    # Process each day
    for day in DAYS:
        # Read all callgraph ranked files for this day to get top MS
        ranked_dir = Path(OUTPUT_PATH) / "callgraph_ranked_ms"
        ranked_files = sorted(ranked_dir.glob(f"callgraph_ranked_ms_day_{day}_*.csv"))
        
        if not ranked_files:
            print(f"No ranked CSV files found for day {day}, skipping...")
            continue
        
        # Aggregate ranked data to get top MS for the day
        print(f"Reading ranked data for day {day}...")
        total_calls_dict = {}
        for ranked_file in ranked_files:
            df = pd.read_csv(ranked_file)
            for _, row in df.iterrows():
                dm = row["dm"]
                total_calls_dict[dm] = total_calls_dict.get(dm, 0) + row["total_calls"]
        
        # Get top 100 MS
        ranked_list = sorted(total_calls_dict.items(), key=lambda x: x[1], reverse=True)
        top_100_ms = [ms for ms, _ in ranked_list[:100]]
        
        print(f"Reading MSMetrics data for top 100 MS on day {day}...")
        
        # Read raw MSMetrics data for top MS
        with setup_cpu_cluster() as cluster:
            client = cluster.get_client()
            
            datapath = "/home/mpds/data/bronze/table=MSMetrics"
            df = dd.read_parquet(
                datapath,
                split_row_groups=True,
                columns=["msname", "day", "hour", "cpu_utilization", "memory_utilization"],
                filters=[
                    ("msname", "in", top_100_ms),
                    ("day", "==", day),
                ],
            )
            
            # Aggregate by msname and hour (mean across all instances)
            hourly_stats = (
                df.groupby(["msname", "hour"])
                .agg({
                    "cpu_utilization": "mean",
                    "memory_utilization": "mean",
                })
                .reset_index()
                .compute()
            )
        
        # Reset index and remove any duplicates
        hourly_stats = hourly_stats.reset_index(drop=True)
        hourly_stats = hourly_stats.drop_duplicates(subset=["msname", "hour"], keep="first")
        
        # Sort by hour
        hourly_stats = hourly_stats.sort_values(["msname", "hour"])
        
        # Create plots directory for this day
        plots_dir = Path("plots") / "workload_stats" / f"day_{day}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot for disjoint groups: 1-10, 11-20, 21-30, ..., 91-100
        for n in range(10, 101, 10):
            start_idx = n - 10  # 0, 10, 20, ..., 90
            end_idx = n  # 10, 20, 30, ..., 100
            group_ms = top_100_ms[start_idx:end_idx]
            
            plot_data = hourly_stats[hourly_stats["msname"].isin(group_ms)].copy()
            plot_data = plot_data.reset_index(drop=True)
            
            # Create figure with two subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: CPU utilization
            ax1 = axes[0]
            sns.lineplot(data=plot_data, x="hour", y="cpu_utilization", hue="msname", ax=ax1, marker="o", markersize=3)
            ax1.set_xlabel("Hour")
            ax1.set_ylabel("Mean CPU Utilization")
            ax1.set_title(f"Ranks {start_idx+1}-{end_idx} MS - CPU Utilization by Hour (Day {day})")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Memory utilization
            ax2 = axes[1]
            sns.lineplot(data=plot_data, x="hour", y="memory_utilization", hue="msname", ax=ax2, marker="o", markersize=3)
            ax2.set_xlabel("Hour")
            ax2.set_ylabel("Mean Memory Utilization")
            ax2.set_title(f"Ranks {start_idx+1}-{end_idx} MS - Memory Utilization by Hour (Day {day})")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = plots_dir / f"workload_stats_ranks_{start_idx+1}_{end_idx}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            print(f"Saved plot for ranks {start_idx+1}-{end_idx} MS to {output_path}")


if __name__ == "__main__":
    main()

