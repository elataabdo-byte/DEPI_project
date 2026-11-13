import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULTS = {
    "name": "dataset",
    "work_dir": ".",
    "timestamp_col": "timestamp",
    "segment_col": "segment",
    "gt_mode": "column",          # "column" | "intervals" | "none"
    "gt_col": "ground_truth",
    "window_seconds": 300,
    "stride_seconds": 300,        # NOTE: windows are non-overlapping; stride is only recorded
    "gap_threshold_seconds": 30,
    "digital_values": [0, 1],
    "force_analog": [],
    "force_digital": [],
    "exclude_cols": [],           # extra non-signal columns (e.g. "id")
    "minute_features": {
        "enabled": False,
        "required_digital": None,
        "min_rows_for_frozen": 3,
    },
}

# Internal names that are never treated as signals
INTERNAL_EXCLUDE = {
    "timestamp", "segment", "ground_truth", "window_id", "minute", "load_start",
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    cfg = {**DEFAULTS, **user}
    work = Path(cfg["work_dir"])
    cfg.setdefault("input_path", str(work / "3_labeled" / "data_labeled.csv"))
    cfg.setdefault("output_dir", str(work / "4_windows"))
    cfg["minute_features"] = {**DEFAULTS["minute_features"], **user.get("minute_features", {})}
    cfg["has_gt"] = cfg["gt_mode"] != "none"
    return cfg


def detect_signal_types(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str]]:
    """Return (analog_signals, digital_signals) detected from the data."""
    exclude = INTERNAL_EXCLUDE | set(cfg["exclude_cols"])
    binary_values = set(cfg["digital_values"])
    force_analog = set(cfg["force_analog"])
    force_digital = set(cfg["force_digital"])
    analog, digital = [], []

    for col in df.columns:
        if col in exclude:
            continue
        if col in force_digital:
            digital.append(col)
            continue
        if col in force_analog:
            analog.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  [skip] non-numeric column: {col}")
            continue

        values = df[col].dropna().unique()
        if values.size == 0:
            print(f"  [skip] all-NaN column: {col}")
            continue

        if set(values).issubset(binary_values):
            digital.append(col)
        else:
            analog.append(col)

    return analog, digital



def load_data(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["input_path"])
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loaded: {path}  ({size_mb:.2f} MB)")

    df = pd.read_csv(path)

    # Standardize the column names used internally
    renames = {}
    if cfg["timestamp_col"] != "timestamp":
        renames[cfg["timestamp_col"]] = "timestamp"
    if cfg["has_gt"] and cfg["gt_col"] != "ground_truth":
        renames[cfg["gt_col"]] = "ground_truth"
    if cfg["segment_col"] != "segment" and cfg["segment_col"] in df.columns:
        renames[cfg["segment_col"]] = "segment"
    df = df.rename(columns=renames)

    if "timestamp" not in df.columns:
        raise KeyError(f"Timestamp column '{cfg['timestamp_col']}' not found in {path}")
    if cfg["has_gt"] and "ground_truth" not in df.columns:
        raise KeyError(
            f"Ground-truth column '{cfg['gt_col']}' not found, but gt_mode is "
            f"'{cfg['gt_mode']}'. Use gt_mode 'none' if the data has no labels."
        )
    if "segment" not in df.columns:
        print("  [info] no segment column found: treating all rows as segment 0")
        df["segment"] = 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    print(f"Shape: {df.shape}")
    print(f"Segments: {df['segment'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def run_quality_checks(df, analog_signals, digital_signals, gap_threshold) -> None:
    signals = analog_signals + digital_signals

    print("\n=== Data quality checks ===")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate timestamps per segment: "
          f"{df.duplicated(subset=['segment', 'timestamp']).sum()}")
    print(f"Missing values total: {int(df.isnull().sum().sum())}")

    non_numeric = df[signals].apply(pd.to_numeric, errors="coerce").isna().sum()
    print(f"Non-numeric values per column:\n{non_numeric[non_numeric > 0]}")

    inf_counts = np.isinf(df[signals].select_dtypes(include="number")).sum()
    print(f"Infinite values per column:\n{inf_counts[inf_counts > 0]}")

    gaps = (
        df.sort_values(["segment", "timestamp"])
          .groupby("segment")["timestamp"]
          .diff()
          .dt.total_seconds()
    )
    print(f"Gaps > {gap_threshold}s: {int((gaps > gap_threshold).sum())}")
    print(f"Non-positive gaps: {int((gaps <= 0).sum())}")



def build_windows(source: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split readings into fixed-size windows, restarting after large gaps."""
    window_seconds = cfg["window_seconds"]
    gap_threshold = cfg["gap_threshold_seconds"]
    has_gt = cfg["has_gt"]

    df = source.copy()
    df["_original_order"] = np.arange(len(df))
    df = df.sort_values(["segment", "timestamp"]).reset_index(drop=True)

    df["_segment_start"] = df.groupby("segment")["timestamp"].transform("min")
    df["_elapsed"] = (df["timestamp"] - df["_segment_start"]).dt.total_seconds()
    df["_time_bin"] = (df["_elapsed"] // window_seconds).astype(int)
    df["_gap"] = df.groupby("segment")["timestamp"].diff().dt.total_seconds()

    df["_new_window"] = (
        df["_gap"].isna()
        | df["_time_bin"].ne(df.groupby("segment")["_time_bin"].shift())
        | df["_gap"].gt(gap_threshold)
    )
    df["_window_number"] = df.groupby("segment")["_new_window"].cumsum().astype(int)

    df["window_id"] = (
        "segment_" + df["segment"].astype(str)
        + "_window_" + df["_window_number"].astype(str)
    )

    agg = dict(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        n_rows=("timestamp", "size"),
    )
    if has_gt:
        agg["n_abnormal"] = ("ground_truth", "sum")
    windows_index = df.groupby(["segment", "window_id"], as_index=False).agg(**agg)

    if has_gt:
        windows_index["window_label"] = (windows_index["n_abnormal"] > 0).astype(int)
    else:
        windows_index["n_abnormal"] = 0
        windows_index["window_label"] = -1  # unknown: no ground truth

    windows_index = windows_index[
        ["window_id", "segment", "start_time", "end_time",
         "n_rows", "window_label", "n_abnormal"]
    ]

    df = (
        df.sort_values("_original_order")
          .drop(columns=[
              "_original_order", "_segment_start", "_elapsed",
              "_time_bin", "_gap", "_new_window", "_window_number",
          ])
          .reset_index(drop=True)
    )
    return df, windows_index


def validate_windows(df: pd.DataFrame, windows_index: pd.DataFrame, cfg: dict) -> None:
    """Ensure windows_index matches the assigned rows in data_windowed."""
    gap_threshold = cfg["gap_threshold_seconds"]
    print("\n=== Windowing validation ===")

    gaps = (
        df.sort_values(["segment", "timestamp"])
          .groupby("window_id")["timestamp"]
          .diff()
          .dt.total_seconds()
    )
    max_gap = gaps.groupby(df["window_id"]).max()
    print(f"Windows with gap > threshold: {int((max_gap > gap_threshold).sum())}")

    agg = dict(
        rows=("timestamp", "size"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    )
    if cfg["has_gt"]:
        agg["label"] = ("ground_truth", lambda s: int(s.max() > 0))
    per_window = df.groupby("window_id").agg(**agg).sort_index()

    idx = windows_index.set_index("window_id").sort_index()

    assert idx.index.equals(per_window.index), "window_id mismatch"
    assert (idx["n_rows"] == per_window["rows"]).all(), "n_rows mismatch"
    assert (
        pd.to_datetime(idx["start_time"]).reset_index(drop=True)
        == per_window["start"].reset_index(drop=True)
    ).all(), "start_time mismatch"
    assert (
        pd.to_datetime(idx["end_time"]).reset_index(drop=True)
        == per_window["end"].reset_index(drop=True)
    ).all(), "end_time mismatch"
    if cfg["has_gt"]:
        assert (
            idx["window_label"].reset_index(drop=True)
            == per_window["label"].reset_index(drop=True)
        ).all(), "label mismatch"

    print("windows_index matches data_windowed.")



def build_minute_table(frame, analog_signals, digital_signals, cfg) -> pd.DataFrame:
    mf = cfg["minute_features"]
    required = mf["required_digital"]
    min_rows = mf["min_rows_for_frozen"]
    has_gt = cfg["has_gt"]

    rows = frame.sort_values(["segment", "timestamp"]).copy()
    rows["minute"] = rows["timestamp"].dt.floor("min")
    if required:
        rows["load_start"] = (
            rows.groupby("segment")[required].diff().eq(-1).astype(int)
        )

    grouped = rows.groupby(["segment", "minute"])
    agg = dict(n_rows=("timestamp", "size"))
    if has_gt:
        agg["ground_truth"] = ("ground_truth", "max")
    if required:
        agg["load_starts"] = ("load_start", "sum")
    counts = grouped.agg(**agg)

    analog = grouped[analog_signals].agg(["mean", "std", "min", "max"])
    analog.columns = [f"{sig}_{stat}" for sig, stat in analog.columns]

    parts = [counts, analog]
    if digital_signals:
        parts.append(grouped[digital_signals].mean().add_suffix("_mean"))
    table = pd.concat(parts, axis=1).reset_index()

    if required:
        table["loaded_fraction"] = 1 - table[f"{required}_mean"]

    spread = (
        table[[f"{s}_max" for s in analog_signals]].to_numpy()
        - table[[f"{s}_min" for s in analog_signals]].to_numpy()
    )
    table["analog_range_max"] = spread.max(axis=1)
    table["is_frozen"] = (
        (table["n_rows"] >= min_rows) & (table["analog_range_max"] == 0)
    )

    std_cols = [f"{s}_std" for s in analog_signals]
    table[std_cols] = table[std_cols].fillna(0.0)
    return table


def build_frozen_periods(table: pd.DataFrame, has_gt: bool) -> pd.DataFrame:
    columns = ["period_id", "start", "end", "minutes", "anomalous_minutes"]
    keep = ["minute", "ground_truth"] if has_gt else ["minute"]
    frozen = table.loc[table["is_frozen"], keep].sort_values("minute").copy()
    if frozen.empty:
        return pd.DataFrame(columns=columns)

    frozen["period_id"] = frozen["minute"].diff().ne(pd.Timedelta(minutes=1)).cumsum()
    agg = dict(
        start=("minute", "min"),
        end=("minute", "max"),
        minutes=("minute", "size"),
    )
    if has_gt:
        agg["anomalous_minutes"] = ("ground_truth", "sum")
    periods = frozen.groupby("period_id").agg(**agg)
    if not has_gt:
        periods["anomalous_minutes"] = 0
    return periods.reset_index()



def main() -> None:
    parser = argparse.ArgumentParser(description="Generic windowing step")
    parser.add_argument("--config", required=True, help="Path to the JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    has_gt = cfg["has_gt"]
    mf = cfg["minute_features"]
    required = mf["required_digital"]

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "windowed":        out_dir / "data_windowed.csv",
        "windows_index":   out_dir / "windows_index.csv",
        "config":          out_dir / "windowing_config.json",
    }
    if mf["enabled"]:
        output_files["minute_features"] = out_dir / "data_minute_features.csv"
        output_files["frozen_periods"] = out_dir / "frozen_periods.csv"

    print(f"Dataset: {cfg['name']}  |  gt_mode: {cfg['gt_mode']}  |  "
          f"minute_features: {mf['enabled']}")

    df = load_data(cfg)

   
    analog_signals, digital_signals = detect_signal_types(df, cfg)
    print("\n=== Detected signal types ===")
    print(f"Analog  ({len(analog_signals)}): {analog_signals}")
    print(f"Digital ({len(digital_signals)}): {digital_signals}")

    if not analog_signals:
        raise RuntimeError("No analog signals detected. Check the input file / config.")
    if mf["enabled"]:
        if required and required not in digital_signals:
            raise RuntimeError(
                f"'{required}' was expected to be a digital signal but wasn't "
                f"detected. Add it to force_digital or fix the config."
            )

    run_quality_checks(df, analog_signals, digital_signals, cfg["gap_threshold_seconds"])

    data_windowed, windows_index = build_windows(df, cfg)
    validate_windows(data_windowed, windows_index, cfg)

    if mf["enabled"]:
        minute_table = build_minute_table(
            data_windowed, analog_signals, digital_signals, cfg,
        )
        frozen_periods = build_frozen_periods(minute_table, has_gt)
        minute_features = (
            minute_table.loc[~minute_table["is_frozen"]]
                        .drop(columns=["analog_range_max", "is_frozen"], errors="ignore")
                        .copy()
        )

        print("\n=== Minute features validation ===")
        frozen_keys = set(zip(
            minute_table.loc[minute_table["is_frozen"], "segment"],
            minute_table.loc[minute_table["is_frozen"], "minute"],
        ))
        kept_keys = set(zip(minute_features["segment"], minute_features["minute"]))
        print(f"Frozen minutes left in output: {len(frozen_keys & kept_keys)}")
        print(f"Duplicate (segment, minute) rows: "
              f"{int(minute_features.duplicated(['segment', 'minute']).sum())}")
        print(f"Missing values: {int(minute_features.isna().sum().sum())}")
        if has_gt:
            print(f"Anomalous minutes kept: {int(minute_features['ground_truth'].sum())}")

        minute_features.to_csv(output_files["minute_features"], index=False)
        frozen_periods.to_csv(output_files["frozen_periods"], index=False)

    run_config = {
        "dataset": cfg["name"],
        "window_size_seconds": cfg["window_seconds"],
        "stride_seconds": cfg["stride_seconds"],
        "window_alignment": "segment_start",
        "overlap": False,
        "gap_threshold_seconds": cfg["gap_threshold_seconds"],
        "gap_policy": (
            f"Start a new window when consecutive readings are more than "
            f"{cfg['gap_threshold_seconds']} seconds apart."
        ),
        "allow_partial_windows": True,
        "gt_mode": cfg["gt_mode"],
        "labeling_rule": (
            "window_label = 1 if any reading in the window has ground_truth = 1; "
            "otherwise 0."
            if has_gt else
            "No ground truth available: window_label = -1 (unknown)."
        ),
        "analog_signals": analog_signals,
        "digital_signals": digital_signals,
        "signal_detection": "auto (binary columns -> digital, rest -> analog)",
        "signal_overrides": {
            "force_analog": sorted(cfg["force_analog"]),
            "force_digital": sorted(cfg["force_digital"]),
        },
        "minute_features": mf,
    }

    data_windowed.to_csv(output_files["windowed"], index=False)
    windows_index.to_csv(output_files["windows_index"], index=False)
    with open(output_files["config"], "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)

    print("\n=== Summary ===")
    print(f"Original rows:  {len(df)}")
    print(f"Windowed rows:  {len(data_windowed)}")
    print(f"Windows:        {len(windows_index)}")
    if has_gt:
        print(f"Original anomalies: {int(df['ground_truth'].sum())}")
        print(f"Final anomalies:    {int(data_windowed['ground_truth'].sum())}")
        print(f"Anomalous windows:  {int(windows_index['window_label'].sum())}")
    else:
        print("Ground truth: none (window_label = -1)")
    print("\nFiles written:")
    for name, path in output_files.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
