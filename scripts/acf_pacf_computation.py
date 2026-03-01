from dask.distributed import Client
import dask.dataframe as dd
from config import setup_gpu_cluster
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
import argparse
import sys


def main(ms_file, output_csv, nlags=40, target_days=[0, 1, 2], compute_acf=False, compute_pacf=False):
    if not compute_acf and not compute_pacf:
        print("Error: You must specify at least one of --acf or --pacf")
        sys.exit(1)

    with open(ms_file) as f:
        target_ms = [line.strip() for line in f if line.strip()]

    with setup_gpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        datapath = "/home/mpds/data/bronze/table=MSMetrics"

        ms_metrics = dd.read_parquet(
            datapath,
            split_row_groups=True,
            columns=["day", "timestamp", "msname", "cpu_utilization"],
            filters=[
                ("day", "in", target_days),
                ("msname", "in", target_ms),
            ],
        )

        ms_metrics = (
            ms_metrics
            .groupby(["msname", "timestamp"])["cpu_utilization"]
            .mean()
            .reset_index()
        )

        pdf = ms_metrics.compute()

    rows = []

    for ms, group in pdf.groupby("msname"):
        group = group.sort_values("timestamp")
        series = group["cpu_utilization"].diff().dropna().values

        if len(series) < 2:
            print(f"{ms}: not enough data")
            continue

        n = len(series)
        max_lag_acf = min(nlags, n - 1)
        max_lag_pacf = min(nlags, (n // 2) - 1)

        if compute_acf:
            acf_vals = acf(series, nlags=max_lag_acf, fft=True)
            for lag, val in enumerate(acf_vals):
                rows.append({
                    "MSNAME": ms,
                    "LAG": lag,
                    "TYPE": "ACF",
                    "VALUE": val
                })
        
        if compute_pacf:
            if max_lag_pacf < 1:
                print(f"{ms}: not enough data for PACF")
            else:
                pacf_vals = pacf(series, nlags=max_lag_pacf, method="ywm")
                for lag, val in enumerate(pacf_vals):
                    rows.append({
                        "MSNAME": ms,
                        "LAG": lag,
                        "TYPE": "PACF",
                        "VALUE": val
                    })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ACF and/or PACF per MS and save as CSV")
    parser.add_argument("--ms_file", required=True, help="Text file with MS names (one per line)")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    parser.add_argument("--nlags", type=int, default=40, help="Number of lags")
    parser.add_argument("--acf", action="store_true", help="Compute ACF")
    parser.add_argument("--pacf", action="store_true", help="Compute PACF")

    args = parser.parse_args()

    main(
        args.ms_file,
        args.output_csv,
        nlags=args.nlags,
        compute_acf=args.acf,
        compute_pacf=args.pacf,
    )
