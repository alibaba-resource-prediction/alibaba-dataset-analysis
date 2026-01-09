"""
Plot CPU utilization for the top 100 microservices per day.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_PATH, DAYS, HOUR_GROUPS, get_hour_suffix


def daily_rank_path(day: int, hours: list[int]) -> Path:
    suffix = get_hour_suffix(hours)
    filename = f"msmetrics_rank_ms_day_{day}_hours_{suffix}.parquet"
    return Path(OUTPUT_PATH) / "msmetrics_rank_ms" / filename


def load_day_frames(day: int) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for hours in HOUR_GROUPS:
        path = daily_rank_path(day, hours)

        df = pd.read_parquet(path)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif df.index.name:
            df = df.reset_index()

        frames.append(df[["msname", "timestamp", "cpu_utilization"]])

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def summarize_top_cpu(df: pd.DataFrame, top_n: int = 100) -> list[str]:
    grouped = (
        df.groupby("msname", as_index=False)["cpu_utilization"]
        .mean()
        .nlargest(top_n, columns="cpu_utilization")
    )
    return grouped["msname"].tolist()


def plot_day(day: int, df: pd.DataFrame, top_ms: list[str], output_dir: Path) -> None:
    subset = df[df["msname"].isin(top_ms)].copy()
    base_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
    subset["timestamp"] = base_date + pd.to_timedelta(subset["timestamp"], unit="ms")
    subset.sort_values(["msname", "timestamp"], inplace=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    for _, group in subset.groupby("msname"):
        ax.plot(
            group["timestamp"],
            group["cpu_utilization"],
            linewidth=0.9,
            alpha=0.6,
        )

    ax.set_title(f"Top 100 CPU Utilization by MS over Time (Day {day})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("CPU utilization")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    output_path = output_dir / f"top_cpu_utilization_day_{day}.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved day {day} plot to {output_path}")
    write_top_list(day, top_ms, output_dir)


def write_top_list(day: int, top_ms: list[str], output_dir: Path) -> None:
    txt_path = output_dir / f"top_cpu_utilization_day_{day}.txt"
    with txt_path.open("w", encoding="utf-8") as txt:
        txt.write("\n".join(top_ms))
    print(f"Wrote top-MS list for day {day} to {txt_path}")


def main() -> None:
    output_dir = Path("plots") / "top_cpu_utilization"
    output_dir.mkdir(parents=True, exist_ok=True)

    for day in DAYS:
        df = load_day_frames(day)
        top_ms = summarize_top_cpu(df)
        plot_day(day, df, top_ms, output_dir)


if __name__ == "__main__":
    main()

