""" 
  1. Load labeled readings and auto-detect analog/digital signals.
  2. Run data quality checks.
  3. Split readings into 5-minute windows per segment,
     restarting a window after any gap > 30 seconds.
  4. Build a window index (start/end/rows/label).
  5. Build minute-level features, flag frozen minutes,
     and export clean minute features + frozen periods.

Signal classification rules (in priority order):
  1. FORCE_DIGITAL  -> always digital
  2. FORCE_ANALOG   -> always analog
  3. SIGNAL_EXCLUDE -> never treated as a signal
  4. non-numeric    -> skipped
  5. all values in DIGITAL_VALUES -> digital
  6. otherwise      -> analog
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_PATH = Path("moddata_labeled.csv")
OUTPUT_DIR = Path(".")

WINDOW_SECONDS = 300
STRIDE_SECONDS = 300
GAP_THRESHOLD_SECONDS = 30
MIN_ROWS_FOR_FROZEN = 3

# Columns that are metadata / labels — never treated as signals
SIGNAL_EXCLUDE = {
    "timestamp", "segment", "ground_truth",
    "window_id", "minute", "load_start",
}

# Values that indicate a binary (digital) column
DIGITAL_VALUES = {0, 1}

 
FORCE_ANALOG: set[str] = set()
FORCE_DIGITAL: set[str] = set()

# The pipeline assumes this column exists as a digital signal.
REQUIRED_DIGITAL = "COMP"

OUTPUT_FILES = {
    "windowed":        OUTPUT_DIR / "data_windowed.csv",
    "windows_index":   OUTPUT_DIR / "windows_index.csv",
    "config":          OUTPUT_DIR / "windowing_config.json",
    "minute_features": OUTPUT_DIR / "data_minute_features.csv",
    "frozen_periods":  OUTPUT_DIR / "frozen_periods.csv",
}


# ------------------------------------------------------------------
# Signal type detection
# ------------------------------------------------------------------
def detect_signal_types(
    df: pd.DataFrame,
    exclude: set[str] = SIGNAL_EXCLUDE,
    binary_values: set = DIGITAL_VALUES,
    force_analog: set[str] = FORCE_ANALOG,
    force_digital: set[str] = FORCE_DIGITAL,
) -> tuple[list[str], list[str]]:
    """Return (analog_signals, digital_signals) detected from the data."""
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


# ------------------------------------------------------------------
# Loading & quality checks
# ------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loaded: {path}  ({size_mb:.2f} MB)")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    print(f"Shape: {df.shape}")
    print(f"Segments: {df['segment'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def run_quality_checks(
    df: pd.DataFrame,
    analog_signals: list[str],
    digital_signals: list[str],
) -> None:
    signals = analog_signals + digital_signals

    print("\n=== Data quality checks ===")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate timestamps per segment: "
          f"{df.duplicated(subset=['segment', 'timestamp']).sum()}")
    print(f"Missing values total: {int(df.isnull().sum().sum())}")

    non_numeric = (
        df[signals].apply(pd.to_numeric, errors="coerce").isna().sum()
    )
    print(f"Non-numeric values per column:\n{non_numeric[non_numeric > 0]}")

    inf_counts = np.isinf(
        df[signals].select_dtypes(include="number")
    ).sum()
    print(f"Infinite values per column:\n{inf_counts[inf_counts > 0]}")

    gaps = (
        df.sort_values(["segment", "timestamp"])
          .groupby("segment")["timestamp"]
          .diff()
          .dt.total_seconds()
    )
    print(f"Gaps > {GAP_THRESHOLD_SECONDS}s: "
          f"{int((gaps > GAP_THRESHOLD_SECONDS).sum())}")
    print(f"Non-positive gaps: {int((gaps <= 0).sum())}")


# ------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------
def build_windows(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split readings into fixed-size windows, restarting after large gaps."""
    df = source.copy()
    df["_original_order"] = np.arange(len(df))
    df = df.sort_values(["segment", "timestamp"]).reset_index(drop=True)

    df["_segment_start"] = (
        df.groupby("segment")["timestamp"].transform("min")
    )
    df["_elapsed"] = (
        df["timestamp"] - df["_segment_start"]
    ).dt.total_seconds()
    df["_time_bin"] = (df["_elapsed"] // WINDOW_SECONDS).astype(int)
    df["_gap"] = (
        df.groupby("segment")["timestamp"].diff().dt.total_seconds()
    )

    df["_new_window"] = (
        df["_gap"].isna()
        | df["_time_bin"].ne(df.groupby("segment")["_time_bin"].shift())
        | df["_gap"].gt(GAP_THRESHOLD_SECONDS)
    )
    df["_window_number"] = (
        df.groupby("segment")["_new_window"].cumsum().astype(int)
    )

    df["window_id"] = (
        "segment_" + df["segment"].astype(str)
        + "_window_" + df["_window_number"].astype(str)
    )

    windows_index = (
        df.groupby(["segment", "window_id"], as_index=False)
          .agg(
              start_time=("timestamp", "min"),
              end_time=("timestamp", "max"),
              n_rows=("timestamp", "size"),
              n_abnormal=("ground_truth", "sum"),
          )
    )
    windows_index["window_label"] = (
        windows_index["n_abnormal"] > 0
    ).astype(int)
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


def validate_windows(df: pd.DataFrame, windows_index: pd.DataFrame) -> None:
    """Ensure windows_index matches the assigned rows in data_windowed."""
    print("\n=== Windowing validation ===")

    gaps = (
        df.sort_values(["segment", "timestamp"])
          .groupby("window_id")["timestamp"]
          .diff()
          .dt.total_seconds()
    )
    max_gap = gaps.groupby(df["window_id"]).max()
    print(f"Windows with gap > threshold: "
          f"{int((max_gap > GAP_THRESHOLD_SECONDS).sum())}")

    per_window = df.groupby("window_id").agg(
        rows=("timestamp", "size"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        label=("ground_truth", lambda s: int(s.max() > 0)),
    ).sort_index()

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
    assert (
        idx["window_label"].reset_index(drop=True)
        == per_window["label"].reset_index(drop=True)
    ).all(), "label mismatch"

    print("windows_index matches data_windowed.")


# ------------------------------------------------------------------
# Minute-level features
# ------------------------------------------------------------------
def build_minute_table(
    frame: pd.DataFrame,
    analog_signals: list[str],
    digital_signals: list[str],
) -> pd.DataFrame:
    rows = frame.sort_values(["segment", "timestamp"]).copy()
    rows["minute"] = rows["timestamp"].dt.floor("min")
    rows["load_start"] = (
        rows.groupby("segment")[REQUIRED_DIGITAL].diff().eq(-1).astype(int)
    )

    grouped = rows.groupby(["segment", "minute"])
    counts = grouped.agg(
        n_rows=("timestamp", "size"),
        ground_truth=("ground_truth", "max"),
        load_starts=("load_start", "sum"),
    )
    analog = grouped[analog_signals].agg(["mean", "std", "min", "max"])
    analog.columns = [f"{sig}_{stat}" for sig, stat in analog.columns]
    digital = grouped[digital_signals].mean().add_suffix("_mean")

    table = pd.concat([counts, analog, digital], axis=1).reset_index()
    table["loaded_fraction"] = 1 - table[f"{REQUIRED_DIGITAL}_mean"]

    spread = (
        table[[f"{s}_max" for s in analog_signals]].to_numpy()
        - table[[f"{s}_min" for s in analog_signals]].to_numpy()
    )
    table["analog_range_max"] = spread.max(axis=1)
    table["is_frozen"] = (
        (table["n_rows"] >= MIN_ROWS_FOR_FROZEN)
        & (table["analog_range_max"] == 0)
    )

    std_cols = [f"{s}_std" for s in analog_signals]
    table[std_cols] = table[std_cols].fillna(0.0)
    return table


def build_frozen_periods(table: pd.DataFrame) -> pd.DataFrame:
    columns = ["period_id", "start", "end", "minutes", "anomalous_minutes"]
    frozen = (
        table.loc[table["is_frozen"], ["minute", "ground_truth"]]
             .sort_values("minute")
             .copy()
    )
    if frozen.empty:
        return pd.DataFrame(columns=columns)

    frozen["period_id"] = (
        frozen["minute"].diff().ne(pd.Timedelta(minutes=1)).cumsum()
    )
    periods = frozen.groupby("period_id").agg(
        start=("minute", "min"),
        end=("minute", "max"),
        minutes=("minute", "size"),
        anomalous_minutes=("ground_truth", "sum"),
    )
    return periods.reset_index()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    df = load_data(INPUT_PATH)

    # ---- Auto-detect signal types (with manual overrides) ----
    analog_signals, digital_signals = detect_signal_types(df)
    print("\n=== Detected signal types ===")
    print(f"Analog  ({len(analog_signals)}): {analog_signals}")
    print(f"Digital ({len(digital_signals)}): {digital_signals}")
    if FORCE_ANALOG:
        print(f"Forced analog : {sorted(FORCE_ANALOG)}")
    if FORCE_DIGITAL:
        print(f"Forced digital: {sorted(FORCE_DIGITAL)}")

    if not analog_signals:
        raise RuntimeError("No analog signals detected. Check the input file.")
    if not digital_signals:
        raise RuntimeError("No digital signals detected. Check the input file.")
    if REQUIRED_DIGITAL not in digital_signals:
        raise RuntimeError(
            f"'{REQUIRED_DIGITAL}' was expected to be a digital signal but "
            f"wasn't detected. Add it to FORCE_DIGITAL or fix the dataset."
        )

    run_quality_checks(df, analog_signals, digital_signals)

    data_windowed, windows_index = build_windows(df)
    validate_windows(data_windowed, windows_index)

    minute_table = build_minute_table(
        data_windowed, analog_signals, digital_signals,
    )
    frozen_periods = build_frozen_periods(minute_table)
    minute_features = (
        minute_table.loc[~minute_table["is_frozen"]]
                     .drop(columns=["analog_range_max", "is_frozen"],
                           errors="ignore")
                     .copy()
    )

    print("\n=== Minute features validation ===")
    frozen_keys = set(zip(
        minute_table.loc[minute_table["is_frozen"], "segment"],
        minute_table.loc[minute_table["is_frozen"], "minute"],
    ))
    kept_keys = set(zip(
        minute_features["segment"], minute_features["minute"],
    ))
    print(f"Frozen minutes left in output: {len(frozen_keys & kept_keys)}")
    print(f"Duplicate (segment, minute) rows: "
          f"{int(minute_features.duplicated(['segment', 'minute']).sum())}")
    print(f"Missing values: {int(minute_features.isna().sum().sum())}")
    print(f"Anomalous minutes kept: "
          f"{int(minute_features['ground_truth'].sum())}")

    config = {
        "window_size_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "window_alignment": "segment_start",
        "overlap": False,
        "gap_threshold_seconds": GAP_THRESHOLD_SECONDS,
        "gap_policy": (
            f"Start a new window when consecutive readings are more than "
            f"{GAP_THRESHOLD_SECONDS} seconds apart."
        ),
        "allow_partial_windows": True,
        "labeling_rule": (
            "window_label = 1 if any reading in the window has "
            "ground_truth = 1; otherwise 0."
        ),
        "analog_signals": analog_signals,
        "digital_signals": digital_signals,
        "signal_detection": "auto (binary columns -> digital, rest -> analog)",
        "signal_overrides": {
            "force_analog": sorted(FORCE_ANALOG),
            "force_digital": sorted(FORCE_DIGITAL),
        },
        "required_digital_signal": REQUIRED_DIGITAL,
        "min_rows_for_frozen": MIN_ROWS_FOR_FROZEN,
    }

    data_windowed.to_csv(OUTPUT_FILES["windowed"], index=False)
    windows_index.to_csv(OUTPUT_FILES["windows_index"], index=False)
    minute_features.to_csv(OUTPUT_FILES["minute_features"], index=False)
    frozen_periods.to_csv(OUTPUT_FILES["frozen_periods"], index=False)
    with open(OUTPUT_FILES["config"], "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Original rows:  {len(df)}")
    print(f"Windowed rows:  {len(data_windowed)}")
    print(f"Windows:        {len(windows_index)}")
    print(f"Original anomalies: {int(df['ground_truth'].sum())}")
    print(f"Final anomalies:    {int(data_windowed['ground_truth'].sum())}")
    print(f"Anomalous windows:  {int(windows_index['window_label'].sum())}")
    print("\nFiles written:")
    for name, path in OUTPUT_FILES.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()