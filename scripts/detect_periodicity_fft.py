#!/usr/bin/env python
from dask.distributed import Client
import dask.dataframe as dd
from config import setup_gpu_cluster
import pandas as pd
import numpy as np
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq
from math import comb
import argparse

def fishers_g_test(series, sampling_interval, min_period, max_period, alpha):
    """
    series: 1D numpy array of values
    sampling_interval: in minutes
    min_period, max_period: ignore periods outside this range
    alpha: significance level
    Returns: g_stat, p_value, is_periodic, dominant_period
    """

    if len(series) < 10:
        return None

    # Remove trend
    series = detrend(series)
    n = len(series)

    fft_vals = rfft(series)
    power = np.abs(fft_vals) ** 2
    freqs = rfftfreq(n, d=sampling_interval)

    power = power[1:]
    freqs = freqs[1:]
    periods = 1 / freqs

    # Restrict period range
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return None

    power = power[mask]
    periods = periods[mask]

    # Dominant peak
    peak_idx = np.argmax(power)
    dominant_period = periods[peak_idx]
    g = power[peak_idx] / np.sum(power)

    #m = len(power)
    #p_value = min(m * (1 - g)**(m - 1), 1.0)

    is_periodic = g < alpha

    return g, is_periodic, dominant_period

def main(ms_file,
         output_csv,
         target_days,
         sampling_interval,
         min_period,
         max_period,
         alpha,
         use_diff):

    with open(ms_file) as f:
        target_ms = [line.strip() for line in f if line.strip()]

    with setup_gpu_cluster() as cluster:
        client = cluster.get_client()
        print(f"Dask dashboard: {client.dashboard_link}")

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

        pdf = (
            ms_metrics.groupby(["msname", "timestamp"])["cpu_utilization"]
            .mean()
            .reset_index()
            .compute()
        )

    rows = []

    for ms, group in pdf.groupby("msname"):
        group = group.sort_values("timestamp")

        if use_diff:
            series = group["cpu_utilization"].diff().dropna().values
        else:
            series = group["cpu_utilization"].values

        if len(series) < 10:
            print(f"{ms}: not enough data")
            continue

        result = fishers_g_test(series,
                                sampling_interval=sampling_interval,
                                min_period=min_period,
                                max_period=max_period,
                                alpha=alpha)
        if result is None:
            continue

        g_stat, is_periodic, dominant_period = result

        rows.append({
            "MSNAME": ms,
            "DOMINANT_PERIOD_MIN": dominant_period,
            "G_STAT": g_stat,
            "IS_PERIODIC": is_periodic
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("G_STAT")

    result_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect periodicity per MS using FFT + Fisher's g-test"
    )
    parser.add_argument("--ms-file", required=True, help="Text file with MS names (one per line)")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--target-days", type=int, nargs="+", default=[0,1,2],
                        help="List of day numbers to include (default: 0 1 2)")
    parser.add_argument("--sampling-interval", type=float, default=1.0,
                        help="Sampling interval in minutes (default: 1.0)")
    parser.add_argument("--min-period", type=float, default=5.0,
                        help="Ignore periods below this value in minutes (default: 5)")
    parser.add_argument("--max-period", type=float, default=2000.0,
                        help="Ignore periods above this value in minutes (default: 2000)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Significance level for Fisher's g-test (default: 0.01)")
    parser.add_argument("--use-diff", action="store_true",
                        help="Take first difference of series before analysis")

    args = parser.parse_args()

    main(ms_file=args.ms_file,
         output_csv=args.output_csv,
         target_days=args.target_days,
         sampling_interval=args.sampling_interval,
         min_period=args.min_period,
         max_period=args.max_period,
         alpha=args.alpha,
         use_diff=args.use_diff)
