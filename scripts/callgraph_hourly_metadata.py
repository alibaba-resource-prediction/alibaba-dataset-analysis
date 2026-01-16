"""Dump per-day/hour metadata (rows and bytes) for the CallGraph table."""

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parse_day_hour(path: str) -> Tuple[int, int]:
    """Extract day/hour from a hive-partitioned path."""
    day = hour = None
    for section in path.replace("\\", "/").split("/"):
        if section.startswith("day="):
            day = int(section.split("=", 1)[-1])
        elif section.startswith("hour="):
            hour = int(section.split("=", 1)[-1])
    return day if day is not None else -1, hour if hour is not None else -1


def iter_file_stats(dataset_path: str) -> Iterable[Tuple[int, int, int, int]]:
    """Yield day/hour, row count, and total bytes for each parquet file."""
    dataset = ds.dataset(dataset_path, partitioning="hive")
    for fragment in dataset.get_fragments():
        file_path = Path(fragment.path)
        if not file_path.is_absolute():
            file_path = Path(dataset_path).joinpath(file_path)
        if not file_path.exists():
            continue

        day, hour = parse_day_hour(str(file_path))
        parquet_file = pq.ParquetFile(str(file_path))
        total_rows = parquet_file.metadata.num_rows
        total_bytes = 0
        for rg_index in range(parquet_file.metadata.num_row_groups):
            rg = parquet_file.metadata.row_group(rg_index)
            if rg.total_byte_size is not None:
                total_bytes += rg.total_byte_size

        yield day, hour, total_rows, total_bytes


def summarize(dataset_path: str) -> pd.DataFrame:
    """Build a day/hour summary DataFrame."""
    stats = defaultdict(lambda: {"rows": 0, "bytes": 0})
    for day, hour, rows, byte_size in iter_file_stats(dataset_path):
        key = (day, hour)
        stats[key]["rows"] += rows
        stats[key]["bytes"] += byte_size

    records = []
    for (day, hour), values in stats.items():
        records.append(
            {
                "day": day,
                "hour": hour,
                "rows": values["rows"],
                "bytes": values["bytes"],
                "gb": values["bytes"] / (1024 ** 3),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(["day", "hour"]).reset_index(drop=True)


def main() -> None:
    path = "/home/mpds/data/bronze/table=CallGraph" 

    df = summarize(path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

