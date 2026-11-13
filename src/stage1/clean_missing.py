"""
Generic cleaning pipeline:
  - Duplicate handling
  - Frozen-logger detection (per segment, before missing-value filling)
  - Missing-value handling
  - ON/OFF -> 1/0 conversion

Usage:
    python scripts/clean_missing_generic.py --config configs/metro.json
    python scripts/clean_missing_generic.py --config configs/pump.json
    python scripts/clean_missing_generic.py --config configs/iot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    # paths
    "work_dir": ".",
    # columns
    "timestamp_col": "timestamp",
    "segment_col": "segment",
    # duplicates
    "dup_policy": "none",                 # "none" | "drop"
    # missing values
    "missing_policy": "none",             # "none" | "drop_rows" | "ffill" | "interpolate"
    "max_missing_col_fraction": 0.9,
    "protect_cols": [],
    "leftover_policy": "global_stat",     # "none" | "global_stat" | "drop_rows"
    # frozen logger
    "frozen_min_duration_minutes": 5,
    "frozen_sensor_cols": [],             # [] => auto-detect numeric continuous sensors
    "frozen_action": "flag",              # "flag" | "drop"
}

FROZEN_FLAG_COL = "frozen_logger_flag"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = {**DEFAULTS, **json.load(f)}

    work = Path(cfg["work_dir"])
    cfg.setdefault("input_path", str(work / "1_timestamp" / "data_ts_fixed.csv"))
    cfg.setdefault("clean_output_path", str(work / "2_clean" / "data_clean.csv"))
    return cfg


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame, ts: str, seg: str | None) -> pd.DataFrame:
    before = len(df)

    df = df.drop_duplicates()
    after_exact = len(df)

    keys = [seg, ts] if seg else [ts]
    df = df.drop_duplicates(subset=keys, keep="first")

    print(
        f"Duplicates removed: {before - after_exact} exact rows, "
        f"{after_exact - len(df)} repeated timestamps"
    )
    return df


# ---------------------------------------------------------------------------
# Frozen logger detection
# ---------------------------------------------------------------------------

def _select_frozen_sensor_cols(
    df: pd.DataFrame,
    cfg: dict,
    ts: str,
    seg: str | None,
) -> list[str]:
    """Return the sensor columns used for frozen-logger detection."""

    configured = list(cfg.get("frozen_sensor_cols") or [])
    if configured:
        missing = [c for c in configured if c not in df.columns]
        if missing:
            raise KeyError(f"Frozen sensor columns not found: {missing}")
        return configured

    # Auto-detect: numeric / boolean, excluding protected columns,
    # constant columns, and columns with <= 1 distinct value.
    excluded = {ts, FROZEN_FLAG_COL}
    if seg:
        excluded.add(seg)
    excluded |= set(cfg.get("protect_cols") or [])

    sensor_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not (pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])):
            continue
        if df[col].nunique(dropna=True) > 1:
            sensor_cols.append(col)

    return sensor_cols


def _detect_frozen_in_group(
    g: pd.DataFrame,
    sensor_cols: list[str],
    ts: str,
    min_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Detect frozen periods inside a single segment (sorted by timestamp)."""

    if g.empty or not sensor_cols:
        return pd.DataFrame(
            columns=["segment", "start_time", "end_time", "duration",
                     "duration_seconds", "repeated_steps"]
        )

    sensors = g[sensor_cols]

    # All sensors unchanged compared with the previous row
    same_as_prev = sensors.eq(sensors.shift(1)).all(axis=1)
    if len(same_as_prev):
        same_as_prev.iloc[0] = False

    # Run ids for consecutive True blocks
    run_id = same_as_prev.ne(same_as_prev.shift(fill_value=False)).cumsum()

    rows: list[dict] = []
    for _, run in g.loc[same_as_prev].groupby(run_id[same_as_prev]):
        if run.empty:
            continue

        first_pos = g.index.get_loc(run.index[0])
        last_pos = g.index.get_loc(run.index[-1])

        # Include the row before the run so the total frozen duration
        # reflects the moment when values stopped changing.
        start_pos = max(0, first_pos - 1)

        start_time = g.iloc[start_pos][ts]
        end_time = g.iloc[last_pos][ts]
        duration = end_time - start_time

        if duration >= min_duration:
            rows.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "duration_seconds": duration.total_seconds(),
                "repeated_steps": int(len(run)),
            })

    return pd.DataFrame(rows)


def detect_frozen_logger(
    df: pd.DataFrame,
    cfg: dict,
    ts: str,
    seg: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect continuous sensor-value repetitions per segment.

    Detection runs BEFORE missing-value filling so that imputed values
    cannot create artificial frozen periods.
    """

    min_duration = pd.Timedelta(minutes=cfg["frozen_min_duration_minutes"])
    sensor_cols = _select_frozen_sensor_cols(df, cfg, ts, seg)

    print("\n========== Frozen Logger Detection ==========")
    print(f"Minimum duration : {min_duration}")
    print(f"Sensor columns   : {sensor_cols if sensor_cols else '[none]'}")

    df = df.copy()
    df[FROZEN_FLAG_COL] = False

    if not sensor_cols:
        print("[info] no usable sensor columns: skipping frozen detection")
        print("=============================================\n")
        return df, pd.DataFrame(
            columns=["segment", "start_time", "end_time", "duration",
                     "duration_seconds", "repeated_steps"]
        )

    all_periods: list[pd.DataFrame] = []

    if seg:
        groups = df.groupby(seg, sort=False)
    else:
        groups = [("__all__", df)]

    for seg_value, g in groups:
        g = g.sort_values(ts, kind="stable")
        periods = _detect_frozen_in_group(g, sensor_cols, ts, min_duration)

        if periods.empty:
            continue

        periods.insert(0, "segment", seg_value)
        all_periods.append(periods)

        # Flag every row whose timestamp falls inside a frozen period
        for _, p in periods.iterrows():
            mask = (g[ts] >= p["start_time"]) & (g[ts] <= p["end_time"])
            df.loc[g.index[mask], FROZEN_FLAG_COL] = True

    if all_periods:
        frozen_periods = (
            pd.concat(all_periods, ignore_index=True)
            .sort_values("start_time", kind="stable")
            .reset_index(drop=True)
        )
    else:
        frozen_periods = pd.DataFrame(
            columns=["segment", "start_time", "end_time", "duration",
                     "duration_seconds", "repeated_steps"]
        )

    flagged = int(df[FROZEN_FLAG_COL].sum())
    print(f"Candidate frozen-logger periods: {len(frozen_periods)}")
    print(f"Flagged rows: {flagged}")
    print("=============================================\n")

    return df, frozen_periods


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def drop_bad_columns(df: pd.DataFrame, cfg: dict, keep: set[str]) -> pd.DataFrame:
    frac = df.isna().mean()
    bad = [
        c for c in df.columns
        if c not in keep and frac[c] > cfg["max_missing_col_fraction"]
    ]
    if bad:
        print(
            f"Dropping columns missing > "
            f"{cfg['max_missing_col_fraction']:.0%}: {bad}"
        )
        df = df.drop(columns=bad)
    return df


def fill_missing(
    df: pd.DataFrame,
    cfg: dict,
    ts: str,
    seg: str | None,
    keep: set[str],
) -> pd.DataFrame:
    policy = cfg["missing_policy"]
    cols = [c for c in df.columns if c not in keep]
    total_before = int(df[cols].isna().sum().sum())

    if policy == "drop_rows":
        df = df.dropna(subset=cols)

    elif policy in ("ffill", "interpolate"):
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            if policy == "interpolate" and num_cols:
                g[num_cols] = g[num_cols].interpolate(
                    method="linear", limit_direction="both"
                )
            g[cols] = g[cols].ffill().bfill()
            return g

        if seg:
            df = pd.concat(
                [_fill(g) for _, g in df.groupby(seg, sort=False)]
            ).sort_index()
        else:
            df = _fill(df)

        left = int(df[cols].isna().sum().sum())
        lp = cfg["leftover_policy"]

        if left and lp == "global_stat":
            for c in cols:
                if not df[c].isna().any():
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    mode = df[c].mode(dropna=True)
                    df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "")
            print(
                f"Leftover (whole segment empty): {left} "
                f"filled with column median/mode"
            )
        elif left and lp == "drop_rows":
            df = df.dropna(subset=cols)
            print(f"Leftover (whole segment empty): {left} rows dropped")
        elif left and lp != "none":
            raise ValueError(f"Unknown leftover_policy: {lp}")

    elif policy != "none":
        raise ValueError(f"Unknown missing_policy: {policy}")

    total_after = int(df[cols].isna().sum().sum())
    print(f"Missing values ({policy}): {total_before} -> {total_after}")
    return df


# ---------------------------------------------------------------------------
# ON/OFF -> 1/0
# ---------------------------------------------------------------------------

def replace_on_off(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ON/OFF (case-insensitive) to 1/0 in object/string columns."""

    converted_cols: list[str] = []

    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue

        non_na = df[col].dropna().astype(str).str.strip().str.lower()
        unique_vals = set(non_na.unique())

        if unique_vals and unique_vals.issubset({"on", "off"}):
            df[col] = (
                df[col].astype("string").str.strip().str.lower()
                .map({"on": 1, "off": 0})
            )
            converted_cols.append(col)

    if converted_cols:
        print(f"Converted ON/OFF -> 1/0 in columns: {converted_cols}")
    else:
        print("[info] no ON/OFF columns detected")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic duplicates + frozen logger + missing values"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = Path(cfg["input_path"])

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path}  shape={df.shape}")

    ts = cfg["timestamp_col"]
    if ts not in df.columns:
        raise KeyError(f"Timestamp column '{ts}' not found")

    # Parse & sort timestamps once, up-front.
    df[ts] = pd.to_datetime(df[ts], errors="coerce")
    invalid_ts = int(df[ts].isna().sum())
    if invalid_ts:
        print(f"[warning] Invalid timestamps: {invalid_ts}")

    df = df.dropna(subset=[ts]).sort_values(ts, kind="stable").reset_index(drop=True)

    seg = cfg["segment_col"] if cfg["segment_col"] in df.columns else None
    if seg is None:
        print("[info] no segment column: treating data as one segment")

    keep = (
        set(cfg["protect_cols"])
        | {ts, FROZEN_FLAG_COL}
        | ({seg} if seg else set())
    )

    # Step 1: duplicates
    if cfg["dup_policy"] == "drop":
        df = drop_duplicates(df, ts, seg)
        df = df.sort_values(ts, kind="stable").reset_index(drop=True)
    elif cfg["dup_policy"] != "none":
        raise ValueError(f"Unknown dup_policy: {cfg['dup_policy']}")

    # Step 2: frozen-logger detection (before missing-value filling)
    df, frozen_periods = detect_frozen_logger(df=df, cfg=cfg, ts=ts, seg=seg)

    out = Path(cfg["clean_output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    frozen_output = out.parent / "frozen_logger_periods.csv"
    frozen_periods.to_csv(frozen_output, index=False)

    # Step 3: optional removal of frozen candidate rows
    frozen_action = cfg["frozen_action"]
    if frozen_action == "drop":
        before = len(df)
        df = df.loc[~df[FROZEN_FLAG_COL]].copy()
        print(f"Frozen candidate rows removed: {before - len(df)}")
    elif frozen_action != "flag":
        raise ValueError(
            f"Unknown frozen_action: {frozen_action}. Use 'flag' or 'drop'."
        )

    # Step 4: missing values
    if cfg["missing_policy"] != "none":
        df = drop_bad_columns(df, cfg, keep)
    df = fill_missing(df, cfg, ts, seg, keep)

    # Step 5: ON/OFF -> 1/0
    df = replace_on_off(df)

    # Step 6: save cleaned data
    df.to_csv(out, index=False)

    print("\n========== Cleaning Completed ==========")
    print(f"Saved cleaned data : {out}")
    print(f"Saved frozen periods: {frozen_output}")
    print(f"Final shape        : {df.shape}")
    print(f"Frozen action      : {frozen_action}")
    print("========================================\n")


if __name__ == "__main__":
    main()