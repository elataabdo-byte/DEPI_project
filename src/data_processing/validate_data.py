"""Non-destructive data-quality and window-index validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def validate_dataframe(df: pd.DataFrame, sensor_columns: list[str] | None = None) -> dict:
    sensors = sensor_columns or [c for c in df.columns if c not in {
        "timestamp", "segment", "ground_truth", "window_id", "window_label", "split"
    } and pd.api.types.is_numeric_dtype(df[c])]
    missing_sensors = [c for c in sensors if c not in df.columns]
    if missing_sensors: raise ValueError(f"Configured sensor columns missing: {missing_sensors}")
    report = {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sensors": sensors,
        "sensor_missing": df[sensors].isna().sum().astype(int).to_dict(),
        "sensor_infinite": {c: int(np.isinf(pd.to_numeric(df[c], errors="coerce")).sum()) for c in sensors},
    }
    if "timestamp" in df:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        report["invalid_timestamps"] = int(ts.isna().sum())
        report["duplicate_timestamps"] = int(ts.duplicated().sum())
        report["timestamps_monotonic"] = bool(ts.dropna().is_monotonic_increasing)
    if "ground_truth" in df:
        report["label_counts"] = {str(k): int(v) for k,v in df["ground_truth"].value_counts(dropna=False).items()}
        report["labels_valid_binary"] = bool(df["ground_truth"].dropna().isin([0,1]).all())
    return report

def validate_windows(data: pd.DataFrame, index: pd.DataFrame, gap_threshold_seconds: int = 30) -> dict:
    required_data = {"timestamp", "segment", "window_id", "ground_truth"}
    if required_data - set(data.columns): raise ValueError(f"Windowed data missing: {sorted(required_data - set(data.columns))}")
    if "window_id" not in index: raise ValueError("Window index missing window_id")
    d = data.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="raise")
    counts = d.groupby("window_id").agg(
        actual_rows=("timestamp", "size"), actual_start=("timestamp", "min"),
        actual_end=("timestamp", "max"), segment_count=("segment", "nunique"),
        actual_label=("ground_truth", lambda s: int(s.max() > 0)))
    idx = index.set_index("window_id")
    shared = counts.index.intersection(idx.index)
    return {
        "data_windows": int(counts.shape[0]), "index_windows": int(idx.shape[0]),
        "same_window_ids": bool(counts.index.sort_values().equals(idx.index.sort_values())),
        "windows_crossing_segments": int((counts["segment_count"] > 1).sum()),
        "windows_with_gap_over_threshold": int(sum(
            (g.sort_values("timestamp")["timestamp"].diff().dt.total_seconds() > gap_threshold_seconds).any()
            for _, g in d.groupby("window_id")
        )),
        "row_counts_match": bool((counts.loc[shared, "actual_rows"].to_numpy() == idx.loc[shared, "n_rows"].to_numpy()).all()),
        "labels_match": bool((counts.loc[shared, "actual_label"].to_numpy() == idx.loc[shared, "window_label"].to_numpy()).all()),
    }

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate a Sentinel CSV and optionally its window index.")
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--index"); parser.add_argument("--gap-seconds", type=int, default=30)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    report = validate_dataframe(df)
    if args.index:
        report["window_validation"] = validate_windows(df, pd.read_csv(args.index), args.gap_seconds)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
