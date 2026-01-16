"""
Rank MS by their CPU utilization.
"""
import dask.dataframe as dd
from config import setup_cpu_cluster, OUTPUT_PATH, DAYS, HOUR_GROUPS, get_hour_suffix
from pathlib import Path

def main():
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        datapath = "/home/mpds/data/bronze/table=MSMetrics"

        for day in DAYS:
            for hours in HOUR_GROUPS:

                ms_metrics = dd.read_parquet(datapath,
                split_row_groups=True,
                columns=["timestamp", "msname", "cpu_utilization"],
                filters=[("day", "==", day), ("hour", "in", hours)],
                )
                g = (
                    ms_metrics.groupby(["msname", "timestamp"])
                    .agg({"cpu_utilization": "mean"})
                    .sort_values(by="cpu_utilization", ascending=False)
                    .compute()
                )
                g = g.reset_index()

                output_path = Path(OUTPUT_PATH) / "msmetrics_rank_ms" / f"msmetrics_rank_ms_day_{day}_hours_{get_hour_suffix(hours)}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                g.to_parquet(output_path)
                print(f"Wrote MSMetrics rank data for day {day} hours {hours} to {output_path}")

if __name__ == "__main__":
    main()