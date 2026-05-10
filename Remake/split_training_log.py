"""
Split warehouse training logs into smaller files.

Examples:
  python split_training_log.py "%USERPROFILE%\\AppData\\LocalLow\\DefaultCompany\\Warehouse\\warehouse_stats.csv" --gap-minutes 30
  python split_training_log.py warehouse_stats.csv --rows-per-file 50000
  python split_training_log.py training_log.xlsx --split-after-rows 10000,25000 --output-format xlsx
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV/XLSX training log by row counts, exact row breakpoints, "
            "or large gaps between timestamp rows."
        )
    )
    parser.add_argument("input_file", help="CSV, XLSX, or XLS file to split.")
    parser.add_argument(
        "--output-dir",
        help="Directory for split files. Defaults to '<input name>_split'.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        help="Write this many data rows per output file. The header is repeated in every file.",
    )
    parser.add_argument(
        "--split-after-rows",
        help=(
            "Comma-separated 1-based data row numbers after which to start a new file. "
            "Example: 10000,25000"
        ),
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Timestamp column used with --gap-minutes. Default: timestamp.",
    )
    parser.add_argument(
        "--gap-minutes",
        type=float,
        help=(
            "Start a new output file when the timestamp gap is larger than this. "
            "Useful when multiple training runs were appended into one CSV."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "xlsx"),
        default="csv",
        help="Output file format. Default: csv.",
    )
    parser.add_argument(
        "--prefix",
        default="split",
        help="Prefix for generated files. Default: split.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding. Default: utf-8-sig.",
    )
    return parser.parse_args()


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip().strip('"')
    if not text:
        return None

    for timestamp_format in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, timestamp_format)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def ensure_column(fieldnames: Iterable[str] | None, column: str) -> None:
    if not fieldnames or column not in fieldnames:
        available = ", ".join(fieldnames or [])
        raise SystemExit(f"Column '{column}' was not found. Available columns: {available}")


def parse_row_breaks(value: str | None) -> set[int]:
    if not value:
        return set()

    breaks: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            row_number = int(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid row number in --split-after-rows: {item}") from exc

        if row_number < 1:
            raise SystemExit("--split-after-rows values must be 1 or greater.")

        breaks.add(row_number)

    return breaks


def split_csv_by_row_count(
    input_path: Path,
    output_dir: Path,
    rows_per_file: int,
    output_format: str,
    prefix: str,
    encoding: str,
) -> None:
    if output_format == "xlsx":
        split_excel_with_pandas(input_path, output_dir, rows_per_file, set(), None, output_format, prefix)
        return

    row_counts: dict[str, int] = {}
    current_handle: Any = None
    current_writer: csv.DictWriter[str] | None = None
    current_part = 0
    rows_in_part = 0

    try:
        with input_path.open("r", newline="", encoding=encoding) as source:
            reader = csv.DictReader(source)

            for row in reader:
                if current_writer is None or rows_in_part >= rows_per_file:
                    if current_handle is not None:
                        current_handle.close()

                    current_part += 1
                    rows_in_part = 0
                    output_name = f"{prefix}_rows_part_{current_part:03d}.csv"
                    current_handle = (output_dir / output_name).open("w", newline="", encoding=encoding)
                    current_writer = csv.DictWriter(current_handle, fieldnames=reader.fieldnames)
                    current_writer.writeheader()
                    row_counts[output_name] = 0

                current_writer.writerow(row)
                row_counts[Path(current_handle.name).name] += 1
                rows_in_part += 1
    finally:
        if current_handle is not None:
            current_handle.close()

    print_summary(output_dir, row_counts)


def split_csv_after_rows(
    input_path: Path,
    output_dir: Path,
    row_breaks: set[int],
    output_format: str,
    prefix: str,
    encoding: str,
) -> None:
    if output_format == "xlsx":
        split_excel_with_pandas(input_path, output_dir, None, row_breaks, None, output_format, prefix)
        return

    row_counts: dict[str, int] = {}
    current_handle: Any = None
    current_writer: csv.DictWriter[str] | None = None
    current_part = 0

    try:
        with input_path.open("r", newline="", encoding=encoding) as source:
            reader = csv.DictReader(source)

            for data_row_number, row in enumerate(reader, start=1):
                if current_writer is None:
                    current_part += 1
                    output_name = f"{prefix}_rows_part_{current_part:03d}.csv"
                    current_handle = (output_dir / output_name).open("w", newline="", encoding=encoding)
                    current_writer = csv.DictWriter(current_handle, fieldnames=reader.fieldnames)
                    current_writer.writeheader()
                    row_counts[output_name] = 0

                current_writer.writerow(row)
                row_counts[Path(current_handle.name).name] += 1

                if data_row_number in row_breaks:
                    current_handle.close()
                    current_handle = None
                    current_writer = None
    finally:
        if current_handle is not None:
            current_handle.close()

    print_summary(output_dir, row_counts)


def split_csv_by_time_gap(
    input_path: Path,
    output_dir: Path,
    timestamp_column: str,
    gap_minutes: float,
    output_format: str,
    prefix: str,
    encoding: str,
) -> None:
    if output_format == "xlsx":
        split_excel_with_pandas(input_path, output_dir, None, set(), gap_minutes, output_format, prefix, timestamp_column)
        return

    gap_seconds = gap_minutes * 60
    previous_timestamp: datetime | None = None
    current_handle: Any = None
    current_writer: csv.DictWriter[str] | None = None
    current_run = 0
    row_counts: dict[str, int] = {}

    try:
        with input_path.open("r", newline="", encoding=encoding) as source:
            reader = csv.DictReader(source)
            ensure_column(reader.fieldnames, timestamp_column)

            for row in reader:
                row_timestamp = parse_timestamp(row.get(timestamp_column))
                start_new_run = current_writer is None

                if row_timestamp is not None and previous_timestamp is not None:
                    delta_seconds = (row_timestamp - previous_timestamp).total_seconds()
                    if delta_seconds > gap_seconds or delta_seconds < -60:
                        start_new_run = True

                if start_new_run:
                    if current_handle is not None:
                        current_handle.close()

                    current_run += 1
                    timestamp_label = (
                        row_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
                        if row_timestamp is not None
                        else f"unknown_time_{current_run:03d}"
                    )
                    output_name = f"{prefix}_run_{current_run:03d}_{timestamp_label}.csv"
                    current_handle = (output_dir / output_name).open("w", newline="", encoding=encoding)
                    current_writer = csv.DictWriter(current_handle, fieldnames=reader.fieldnames)
                    current_writer.writeheader()
                    row_counts[output_name] = 0

                current_writer.writerow(row)
                row_counts[Path(current_handle.name).name] += 1

                if row_timestamp is not None:
                    previous_timestamp = row_timestamp
    finally:
        if current_handle is not None:
            current_handle.close()

    print_summary(output_dir, row_counts)


def split_excel_with_pandas(
    input_path: Path,
    output_dir: Path,
    rows_per_file: int | None,
    row_breaks: set[int],
    gap_minutes: float | None,
    output_format: str,
    prefix: str,
    timestamp_column: str = "timestamp",
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Excel input/output needs pandas and openpyxl. Install them with: pip install pandas openpyxl"
        ) from exc

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    row_counts: dict[str, int] = {}

    if rows_per_file is not None:
        current_part = 0
        for start in range(0, len(df), rows_per_file):
            current_part += 1
            group = df.iloc[start : start + rows_per_file]
            output_name = f"{prefix}_rows_part_{current_part:03d}.{output_format}"
            write_dataframe(group, output_dir / output_name, output_format)
            row_counts[output_name] = len(group)
    elif row_breaks:
        current_part = 0
        start = 0
        breakpoints = sorted(row_breaks)

        for breakpoint in breakpoints:
            end = min(breakpoint, len(df))
            if end <= start:
                continue

            current_part += 1
            group = df.iloc[start:end]
            output_name = f"{prefix}_rows_part_{current_part:03d}.{output_format}"
            write_dataframe(group, output_dir / output_name, output_format)
            row_counts[output_name] = len(group)
            start = end

        if start < len(df):
            current_part += 1
            group = df.iloc[start:]
            output_name = f"{prefix}_rows_part_{current_part:03d}.{output_format}"
            write_dataframe(group, output_dir / output_name, output_format)
            row_counts[output_name] = len(group)
    else:
        if timestamp_column not in df.columns:
            raise SystemExit(
                f"Column '{timestamp_column}' was not found. Available columns: {', '.join(df.columns)}"
            )

        gap_seconds = (gap_minutes or 0) * 60
        previous_timestamp: datetime | None = None
        current_rows: list[int] = []
        current_run = 0

        def flush_run() -> None:
            nonlocal current_rows, current_run
            if not current_rows:
                return

            current_run += 1
            first_timestamp = parse_timestamp(df.iloc[current_rows[0]][timestamp_column])
            timestamp_label = (
                first_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
                if first_timestamp is not None
                else f"unknown_time_{current_run:03d}"
            )
            output_name = f"{prefix}_run_{current_run:03d}_{timestamp_label}.{output_format}"
            write_dataframe(df.iloc[current_rows], output_dir / output_name, output_format)
            row_counts[output_name] = len(current_rows)
            current_rows = []

        for position, (_, row) in enumerate(df.iterrows()):
            row_timestamp = parse_timestamp(row[timestamp_column])
            start_new_run = False

            if row_timestamp is not None and previous_timestamp is not None:
                delta_seconds = (row_timestamp - previous_timestamp).total_seconds()
                start_new_run = delta_seconds > gap_seconds or delta_seconds < -60

            if start_new_run:
                flush_run()

            current_rows.append(position)
            if row_timestamp is not None:
                previous_timestamp = row_timestamp

        flush_run()

    print_summary(output_dir, row_counts)


def write_dataframe(df: Any, output_path: Path, output_format: str) -> None:
    if output_format == "xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def print_summary(output_dir: Path, row_counts: dict[str, int]) -> None:
    print(f"Created {len(row_counts)} file(s) in: {output_dir}")
    for name, count in row_counts.items():
        print(f"  {name}: {count} rows")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    row_breaks = parse_row_breaks(args.split_after_rows)
    split_method_count = sum(
        [
            args.rows_per_file is not None,
            bool(row_breaks),
            args.gap_minutes is not None,
        ]
    )

    if split_method_count == 0:
        raise SystemExit("Choose a split method: --rows-per-file, --split-after-rows, or --gap-minutes.")

    if split_method_count > 1:
        raise SystemExit("Use only one split method at a time.")

    if args.rows_per_file is not None and args.rows_per_file < 1:
        raise SystemExit("--rows-per-file must be 1 or greater.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.with_name(f"{input_path.stem}_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        if args.rows_per_file is not None:
            split_csv_by_row_count(
                input_path,
                output_dir,
                args.rows_per_file,
                args.output_format,
                args.prefix,
                args.encoding,
            )
        elif row_breaks:
            split_csv_after_rows(input_path, output_dir, row_breaks, args.output_format, args.prefix, args.encoding)
        else:
            split_csv_by_time_gap(
                input_path,
                output_dir,
                args.timestamp_column,
                args.gap_minutes,
                args.output_format,
                args.prefix,
                args.encoding,
            )
    elif suffix in (".xlsx", ".xls"):
        split_excel_with_pandas(
            input_path,
            output_dir,
            args.rows_per_file,
            row_breaks,
            args.gap_minutes,
            args.output_format,
            args.prefix,
            args.timestamp_column,
        )
    else:
        raise SystemExit("Unsupported input type. Use .csv, .xlsx, or .xls.")


if __name__ == "__main__":
    main()
