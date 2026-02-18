"""
Plot Granger causality summaries from caller_cpu_granger*.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import OUTPUT_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Granger causality summaries."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=OUTPUT_PATH,
        help="CSV path or directory containing caller_cpu_granger_*_all_days.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/caller_cpu_granger",
        help="Directory to write plots.",
    )
    return parser.parse_args()


def load_inputs(path):
    input_path = Path(path)
    if input_path.is_dir():
        files = list(input_path.glob("caller_cpu_granger_*_all_days.csv"))
        if not files:
            raise FileNotFoundError(f"No Granger CSVs in {input_path}")
        frames = [pd.read_csv(f) for f in files]
        return pd.concat(frames, ignore_index=True)
    if input_path.exists():
        return pd.read_csv(input_path)
    raise FileNotFoundError(f"Missing input: {input_path}")


def save_plot(fig, output_dir, name):
    output_path = output_dir / name
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(args.input)
    if df.empty:
        print("Input CSV is empty, nothing to plot.")
        return

    df = df[df["series_type"].isin(["caller", "weighted_callers"])].copy()
    if df.empty:
        print("No caller or weighted series rows found.")
        return

    if "transform" not in df.columns:
        df["transform"] = "diff"

    df["best_pvalue"] = df["best_pvalue"].clip(lower=1e-300)
    df["neglog10_best_p"] = -np.log10(df["best_pvalue"])

    lag_cols = [c for c in df.columns if c.startswith("lag_") and c.endswith("_pvalue")]
    lag_cols = sorted(lag_cols, key=lambda c: int(c.split("_")[1]))

    sns.set_theme(style="whitegrid")

    for transform in sorted(df["transform"].unique()):
        subset = df[df["transform"] == transform].copy()
        if subset.empty:
            continue

        transform_dir = output_dir / transform
        transform_dir.mkdir(parents=True, exist_ok=True)

        # 1) Histogram of best p-values (as -log10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            data=subset,
            x="neglog10_best_p",
            hue="series_type",
            bins=40,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"Best Granger p-value (-log10), transform={transform}")
        ax.set_xlabel("-log10(p-value)")
        ax.set_ylabel("Density")
        save_plot(fig, transform_dir, "best_pvalue_hist.png")

        # 2) Best lag distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(
            data=subset,
            x="best_lag",
            hue="series_type",
            ax=ax,
        )
        ax.set_title(f"Best Lag Distribution, transform={transform}")
        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Count")
        save_plot(fig, transform_dir, "best_lag_counts.png")

        # 3) Share of significant lags by lag
        if lag_cols:
            lag_long = subset.melt(
                id_vars=["series_type"],
                value_vars=lag_cols,
                var_name="lag",
                value_name="pvalue",
            )
            lag_long["lag"] = lag_long["lag"].str.extract(r"lag_(\d+)_pvalue")[0]
            lag_long = lag_long.dropna(subset=["lag"])
            lag_long["lag"] = lag_long["lag"].astype(int)
            alpha = subset["alpha"].dropna().iloc[0] if "alpha" in subset.columns else 0.05
            lag_long["is_significant"] = lag_long["pvalue"] < alpha
            lag_summary = (
                lag_long.groupby(["series_type", "lag"])["is_significant"]
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(
                data=lag_summary,
                x="lag",
                y="is_significant",
                hue="series_type",
                marker="o",
                ax=ax,
            )
            ax.set_title(f"Share of Significant Lags (p<{alpha}), transform={transform}")
            ax.set_xlabel("Lag (timesteps)")
            ax.set_ylabel("Fraction significant")
            save_plot(fig, transform_dir, "significant_share_by_lag.png")

        # 4) Sample size vs best p-value
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=subset,
            x="n_samples",
            y="neglog10_best_p",
            hue="series_type",
            alpha=0.4,
            ax=ax,
        )
        ax.set_title(f"Samples vs Best p-value (-log10), transform={transform}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("-log10(best p-value)")
        save_plot(fig, transform_dir, "samples_vs_best_pvalue.png")

        # 5) Best R2 histogram
        if "best_r2" in subset.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(
                data=subset,
                x="best_r2",
                hue="series_type",
                bins=40,
                element="step",
                stat="density",
                common_norm=False,
                ax=ax,
            )
            ax.set_title(f"Best Partial R2, transform={transform}")
            ax.set_xlabel("Best partial R2")
            ax.set_ylabel("Density")
            save_plot(fig, transform_dir, "best_r2_hist.png")

            # 6) Best R2 boxplot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=subset,
                x="series_type",
                y="best_r2",
                ax=ax,
            )
            ax.set_title(f"Best Partial R2 by Series Type, transform={transform}")
            ax.set_xlabel("Series type")
            ax.set_ylabel("Best partial R2")
            save_plot(fig, transform_dir, "best_r2_box.png")

            # 7) Sample size vs best R2
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=subset,
                x="n_samples",
                y="best_r2",
                hue="series_type",
                alpha=0.4,
                ax=ax,
            )
            ax.set_title(f"Samples vs Best Partial R2, transform={transform}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Best partial R2")
            save_plot(fig, transform_dir, "samples_vs_best_r2.png")

        # 8) Mean partial R2 by lag
        lag_r2_cols = [c for c in subset.columns if c.startswith("lag_") and c.endswith("_r2")]
        if lag_r2_cols:
            lag_r2_cols = sorted(lag_r2_cols, key=lambda c: int(c.split("_")[1]))
            lag_r2_long = subset.melt(
                id_vars=["series_type"],
                value_vars=lag_r2_cols,
                var_name="lag",
                value_name="r2",
            )
            lag_r2_long["lag"] = lag_r2_long["lag"].str.extract(r"lag_(\d+)_r2")[0]
            lag_r2_long = lag_r2_long.dropna(subset=["lag"])
            lag_r2_long["lag"] = lag_r2_long["lag"].astype(int)
            lag_r2_summary = (
                lag_r2_long.groupby(["series_type", "lag"])["r2"]
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(
                data=lag_r2_summary,
                x="lag",
                y="r2",
                hue="series_type",
                marker="o",
                ax=ax,
            )
            ax.set_title(f"Mean Partial R2 by Lag, transform={transform}")
            ax.set_xlabel("Lag (timesteps)")
            ax.set_ylabel("Mean partial R2")
            save_plot(fig, transform_dir, "mean_r2_by_lag.png")


if __name__ == "__main__":
    main()
