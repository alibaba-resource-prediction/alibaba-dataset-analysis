"""
Run Granger causality tests for caller CPU trends vs target CPU trends.

For each day:
- Reads MS list from plots/top_cpu_utilization/top_cpu_utilization_day_<day>.txt
- Gets top callers from graph_edges_full.csv
- Loads MSMetrics CPU utilization for targets and callers
- Transforms series (diff or pct-change)
- Runs Granger tests (caller -> target) across lags
- Saves CSV per day and a combined CSV
"""

import argparse
import warnings
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config import OUTPUT_PATH, setup_cpu_cluster, DAYS


BAD_MS = {"UNKNOWN", "UNAVAILABLE", "USER"}

warnings.filterwarnings(
    "ignore",
    message="verbose is deprecated",
    category=FutureWarning,
    module="statsmodels.tsa.stattools",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Granger causality tests for caller CPU trends."
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="*",
        default=DAYS,
        help="Days to process (default: all DAYS in config).",
    )
    parser.add_argument(
        "--top-callers",
        type=int,
        default=10,
        help="Number of top callers per target.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=6,
        help="Max lag (in timesteps) for Granger tests.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["diff", "pct", "both"],
        default="diff",
        help="Series transform to use: diff, pct, or both.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used for summary stats.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum aligned samples to run Granger tests.",
    )
    parser.add_argument(
        "--limit-ms",
        type=int,
        default=None,
        help="Process only the first N MS in the list.",
    )
    parser.add_argument(
        "--ms-list-template",
        type=str,
        default="plots/top_cpu_utilization/top_cpu_utilization_day_{day}.txt",
        help="Template path for MS list files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_PATH,
        help="Output directory for CSVs.",
    )
    return parser.parse_args()


def load_ms_list(path, limit_ms):
    with open(path, "r") as f:
        ms_list = [line.strip() for line in f if line.strip()]
    if limit_ms:
        ms_list = ms_list[:limit_ms]
    return ms_list


def transform_series(series, mode):
    if mode == "diff":
        out = series.diff()
    elif mode == "pct":
        out = series.pct_change()
    else:
        raise ValueError(f"Unknown transform mode: {mode}")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_transformed_series(df, mode):
    df = df.sort_values("timestamp")
    transformed = transform_series(df["cpu_utilization"], mode)
    return pd.DataFrame({"timestamp": df["timestamp"], "value": transformed})


def best_pvalue(lag_pvalues):
    best_lag_val = None
    best_p = np.nan
    for lag, pval in lag_pvalues.items():
        if pd.isna(pval):
            continue
        if pd.isna(best_p) or pval < best_p:
            best_p = pval
            best_lag_val = lag
    return best_lag_val, best_p


def fstat_to_partial_r2(fstat, df_num, df_denom):
    if pd.isna(fstat) or pd.isna(df_num) or pd.isna(df_denom):
        return np.nan
    if df_num <= 0 or df_denom <= 0:
        return np.nan
    return (fstat * df_num) / (fstat * df_num + df_denom)


def best_r2(lag_r2):
    best_lag_val = None
    best_val = np.nan
    best_abs = -1.0
    for lag, r2 in lag_r2.items():
        if pd.isna(r2):
            continue
        if r2 > best_abs:
            best_abs = r2
            best_lag_val = lag
            best_val = r2
    return best_lag_val, best_val


def run_granger_tests(merged, max_lag):
    data = merged[["target", "caller"]].to_numpy()
    tests = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    lag_pvalues = {}
    lag_fstats = {}
    lag_r2 = {}
    for lag, result in tests.items():
        fstat, pval, df_denom, df_num = result[0]["ssr_ftest"]
        lag_pvalues[lag] = pval
        lag_fstats[lag] = fstat
        lag_r2[lag] = fstat_to_partial_r2(fstat, df_num, df_denom)
    best_lag_val, best_p = best_pvalue(lag_pvalues)
    best_r2_lag, best_r2_val = best_r2(lag_r2)
    return lag_pvalues, lag_fstats, lag_r2, best_lag_val, best_p, best_r2_lag, best_r2_val


def build_weighted_caller_series(metrics_by_ms, callers, caller_counts):
    frames = []
    for caller in callers:
        df = metrics_by_ms.get(caller)
        if df is None:
            continue
        frames.append(
            df[["timestamp", "cpu_utilization"]].rename(
                columns={"cpu_utilization": caller}
            )
        )

    if not frames:
        return None

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")

    weights = pd.Series({c: caller_counts.get(c, 1.0) for c in merged.columns if c != "timestamp"})
    if weights.empty:
        return None
    weights = weights / weights.sum()

    weighted_sum = merged[weights.index].mul(weights, axis=1).sum(axis=1)
    weight_presence = merged[weights.index].notna().mul(weights, axis=1).sum(axis=1)
    weighted_avg = weighted_sum / weight_presence

    return pd.DataFrame(
        {"timestamp": merged["timestamp"], "cpu_utilization": weighted_avg}
    )


def transforms_from_arg(arg):
    if arg == "both":
        return ["diff", "pct"]
    return [arg]


def main():
    args = parse_args()

    datapath = "/home/mpds/data/bronze/table=MSMetrics"
    edges_path = OUTPUT_PATH + "graph_edges_full.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(edges_path)
    edges = edges[(~edges["dm"].isin(BAD_MS)) & (~edges["um"].isin(BAD_MS))]

    transforms = transforms_from_arg(args.transform)

    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        for transform in transforms:
            all_results = []
            for day in args.days:
                ms_list_path = args.ms_list_template.format(day=day)
                ms_list = load_ms_list(ms_list_path, args.limit_ms)
                if not ms_list:
                    print(f"Day {day}: no MS list found or empty, skipping.")
                    continue

                ms_callers_map = {}
                ms_caller_counts_map = {}
                all_ms_to_load = set()

                for ms in ms_list:
                    ms_callers = edges[edges["dm"] == ms].sort_values("count", ascending=False)
                    if ms_callers.empty:
                        continue
                    top_callers = ms_callers.head(args.top_callers)["um"].tolist()
                    caller_counts = (
                        ms_callers.head(args.top_callers)[["um", "count"]]
                        .set_index("um")["count"]
                        .to_dict()
                    )
                    ms_callers_map[ms] = top_callers
                    ms_caller_counts_map[ms] = caller_counts
                    all_ms_to_load.add(ms)
                    all_ms_to_load.update(top_callers)

                if not all_ms_to_load:
                    print(f"Day {day}: no MS with callers found, skipping.")
                    continue

                ms_metrics = dd.read_parquet(
                    datapath,
                    split_row_groups=True,
                    columns=["day", "timestamp", "msname", "cpu_utilization"],
                    filters=[("day", "==", day), ("msname", "in", list(all_ms_to_load))],
                )

                metrics_df = (
                    ms_metrics.groupby(["msname", "timestamp"])["cpu_utilization"]
                    .mean()
                    .reset_index()
                    .compute()
                )

                if metrics_df.empty:
                    print(f"Day {day}: no metrics data found, skipping.")
                    continue

                metrics_by_ms = {
                    ms: df.sort_values("timestamp")
                    for ms, df in metrics_df.groupby("msname")
                }

                day_results = []
                for ms in ms_list:
                    if ms not in metrics_by_ms:
                        continue
                    callers = ms_callers_map.get(ms, [])
                    if not callers:
                        continue

                    available_callers = [c for c in callers if c in metrics_by_ms]
                    if not available_callers:
                        continue

                    total_callers = len(callers)
                    callers_with_data = len(available_callers)

                    target_series = build_transformed_series(metrics_by_ms[ms], transform)

                    for rank, caller in enumerate(available_callers, start=1):
                        caller_series = build_transformed_series(metrics_by_ms[caller], transform)
                        merged = pd.merge(
                            target_series,
                            caller_series,
                            on="timestamp",
                            suffixes=("_target", "_caller"),
                        ).dropna()
                        if len(merged) < args.min_samples:
                            continue

                        merged = merged.rename(
                            columns={"value_target": "target", "value_caller": "caller"}
                        )

                        try:
                            (
                                lag_pvalues,
                                lag_fstats,
                                lag_r2,
                                best_lag_val,
                                best_p,
                                best_r2_lag,
                                best_r2_val,
                            ) = run_granger_tests(merged, args.max_lag)
                        except Exception:
                            continue

                        n_significant = sum(p < args.alpha for p in lag_pvalues.values() if not pd.isna(p))

                        day_results.append(
                            {
                                "day": day,
                                "transform": transform,
                                "target_ms": ms,
                                "series_type": "caller",
                                "caller_ms": caller,
                                "caller_rank": rank,
                                "call_count": ms_caller_counts_map[ms].get(caller, 0),
                                "n_callers_total": total_callers,
                                "n_callers_with_data": callers_with_data,
                                "n_samples": len(merged),
                                "max_lag": args.max_lag,
                                "alpha": args.alpha,
                                "best_lag": best_lag_val,
                                "best_pvalue": best_p,
                                "best_r2_lag": best_r2_lag,
                                "best_r2": best_r2_val,
                                "n_significant_lags": n_significant,
                                **{f"lag_{lag}_pvalue": p for lag, p in lag_pvalues.items()},
                                **{f"lag_{lag}_fstat": f for lag, f in lag_fstats.items()},
                                **{f"lag_{lag}_r2": r2 for lag, r2 in lag_r2.items()},
                            }
                        )

                    weighted_series = build_weighted_caller_series(
                        metrics_by_ms, available_callers, ms_caller_counts_map[ms]
                    )
                    if weighted_series is not None:
                        caller_series = build_transformed_series(weighted_series, transform)
                        merged = pd.merge(
                            target_series,
                            caller_series,
                            on="timestamp",
                            suffixes=("_target", "_caller"),
                        ).dropna()
                        if len(merged) >= args.min_samples:
                            merged = merged.rename(
                                columns={"value_target": "target", "value_caller": "caller"}
                            )
                            try:
                                (
                                    lag_pvalues,
                                    lag_fstats,
                                    lag_r2,
                                    best_lag_val,
                                    best_p,
                                    best_r2_lag,
                                    best_r2_val,
                                ) = run_granger_tests(merged, args.max_lag)
                            except Exception:
                                continue
                            n_significant = sum(
                                p < args.alpha for p in lag_pvalues.values() if not pd.isna(p)
                            )
                            day_results.append(
                                {
                                    "day": day,
                                    "transform": transform,
                                    "target_ms": ms,
                                    "series_type": "weighted_callers",
                                    "caller_ms": "<weighted_callers>",
                                    "caller_rank": 0,
                                    "call_count": sum(ms_caller_counts_map[ms].values()),
                                    "n_callers_total": total_callers,
                                    "n_callers_with_data": callers_with_data,
                                    "n_samples": len(merged),
                                    "max_lag": args.max_lag,
                                    "alpha": args.alpha,
                                    "best_lag": best_lag_val,
                                    "best_pvalue": best_p,
                                    "best_r2_lag": best_r2_lag,
                                    "best_r2": best_r2_val,
                                    "n_significant_lags": n_significant,
                                    **{f"lag_{lag}_pvalue": p for lag, p in lag_pvalues.items()},
                                    **{f"lag_{lag}_fstat": f for lag, f in lag_fstats.items()},
                                    **{f"lag_{lag}_r2": r2 for lag, r2 in lag_r2.items()},
                                }
                            )

                if day_results:
                    day_df = pd.DataFrame(day_results)
                    day_output = output_dir / f"caller_cpu_granger_{transform}_day_{day}.csv"
                    day_df.to_csv(day_output, index=False)
                    print(f"Day {day} ({transform}): wrote {len(day_df)} rows to {day_output}")
                    all_results.extend(day_results)
                else:
                    print(f"Day {day} ({transform}): no Granger results (insufficient data).")

            if all_results:
                all_df = pd.DataFrame(all_results)
                all_output = output_dir / f"caller_cpu_granger_{transform}_all_days.csv"
                all_df.to_csv(all_output, index=False)
                print(f"Wrote combined results to {all_output}")


if __name__ == "__main__":
    main()
