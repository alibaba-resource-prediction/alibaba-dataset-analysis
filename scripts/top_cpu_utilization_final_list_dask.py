"""
Build a final list of top microservices by CPU utilization using Dask.

Reads MSMetrics data directly via Dask CPU cluster.
For each day: top 1000 MS by average CPU utilization.
Unions those lists and takes distinct MS names.
Writes the result to plots/top_cpu_utilization/TOP_MS_FINAL_DASK.txt.
"""

from pathlib import Path

import dask.dataframe as dd

from config import setup_cpu_cluster, DAYS

MSMETRICS_PATH = "/home/mpds/data/bronze/table=MSMetrics"


def main() -> None:
    output_dir = Path("plots") / "top_cpu_utilization"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_top_ms: set[str] = set()

    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        for day in DAYS:
            print(f"Processing day {day}...")
            df = dd.read_parquet(
                MSMETRICS_PATH,
                split_row_groups=True,
                columns=["msname", "timestamp", "cpu_utilization"],
                filters=[("day", "==", day)],
            )
            # Match original pipeline: first aggregate by (msname, timestamp), then by msname
            by_ms_ts = (
                df.groupby(["msname", "timestamp"])["cpu_utilization"]
                .mean()
                .reset_index()
            )
            grouped = by_ms_ts.groupby("msname")["cpu_utilization"].mean()
            top_series = grouped.nlargest(1000)
            result = top_series.compute()
            top_ms = result.index.tolist()
            all_top_ms.update(top_ms)
            print(f"  Day {day}: {len(top_ms)} top MS; union size so far: {len(all_top_ms)}")

        final_list = sorted(all_top_ms)
        output_path = output_dir / "TOP_MS_FINAL_DASK.txt"

        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(final_list))

        print(f"\nWrote {len(final_list)} distinct top CPU MS to {output_path}")


if __name__ == "__main__":
    main()
