"""Timestamp cleaning, stable sorting, gap audit, and segmentation.

Preserves the notebook's policies: invalid timestamps are excluded, duplicate
timestamps are retained, and a new segment begins when a gap exceeds
gap_multiplier * the empirical median interval.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class TimestampConfig:
    input_csv: Path
    output_dir: Path
    timestamp_column: str = "timestamp"
    segment_column: str = "segment"
    drop_columns: tuple[str, ...] = ("Unnamed: 0",)
    chunksize: int = 200_000
    gap_multiplier: float = 3.0
    encoding: str = "utf-8-sig"

    def __post_init__(self):
        self.input_csv = Path(self.input_csv)
        self.output_dir = Path(self.output_dir)
        if self.chunksize <= 0 or self.gap_multiplier <= 0:
            raise ValueError("chunksize and gap_multiplier must be positive")

def inspect_and_load(cfg: TimestampConfig) -> pd.DataFrame:
    if not cfg.input_csv.is_file():
        raise FileNotFoundError(f"Input file not found: {cfg.input_csv.resolve()}")
    chunks = []
    for chunk in pd.read_csv(cfg.input_csv, dtype=str, keep_default_na=False,
                             chunksize=cfg.chunksize, encoding=cfg.encoding):
        chunk = chunk.drop(columns=[c for c in cfg.drop_columns if c in chunk.columns])
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"Input CSV contains no rows: {cfg.input_csv}")
    return pd.concat(chunks, ignore_index=True)

def process_timestamps(df: pd.DataFrame, cfg: TimestampConfig) -> tuple[pd.DataFrame, dict]:
    if cfg.timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column {cfg.timestamp_column!r}")
    parsed = pd.to_datetime(df[cfg.timestamp_column], errors="coerce")
    invalid_count = int(parsed.isna().sum())
    result = df.copy()
    result[cfg.timestamp_column] = parsed
    result = result.dropna(subset=[cfg.timestamp_column])
    result = result.sort_values(cfg.timestamp_column, kind="stable").reset_index(drop=True)
    dup_mask = result[cfg.timestamp_column].duplicated(keep=False)
    dup_rows = int(dup_mask.sum())
    dup_groups = int(result.loc[dup_mask].groupby(cfg.timestamp_column).ngroups) if dup_rows else 0
    exact_dups = int(result.loc[dup_mask].duplicated(keep="first").sum()) if dup_rows else 0
    return result, {"invalid_count": invalid_count, "dup_rows": dup_rows,
                    "dup_groups": dup_groups, "exact_dups": exact_dups}

def analyze_sampling_frequency(df: pd.DataFrame, cfg: TimestampConfig) -> tuple[pd.Series, dict]:
    dt = df[cfg.timestamp_column].diff()
    return dt, {"n_intervals": int(dt.count()), "median_interval": dt.median(),
                "min_interval": dt.min(), "max_interval": dt.max()}

def apply_segmentation(df: pd.DataFrame, cfg: TimestampConfig) -> tuple[pd.DataFrame, pd.Timedelta, pd.Timedelta]:
    if df.empty:
        raise ValueError("No valid timestamps remain after timestamp parsing")
    result = df.copy()
    dt = result[cfg.timestamp_column].diff()
    typical_interval = dt.median()
    if pd.isna(typical_interval) or typical_interval <= pd.Timedelta(0):
        raise ValueError("Cannot determine a positive median sampling interval")
    gap_threshold = cfg.gap_multiplier * typical_interval
    result[cfg.segment_column] = (dt > gap_threshold).cumsum().astype(int)
    return result, typical_interval, gap_threshold

def build_gap_table(df: pd.DataFrame, cfg: TimestampConfig, typical_interval: pd.Timedelta,
                    gap_threshold: pd.Timedelta) -> pd.DataFrame:
    seg = df[cfg.segment_column].to_numpy()
    ts = df[cfg.timestamp_column].to_numpy()
    gap_idx = np.flatnonzero(np.r_[False, seg[1:] > seg[:-1]])
    cols = ["gap_id", "previous_timestamp", "next_timestamp", "observed_interval_s",
            "threshold_s", "reference_interval_s", "estimated_missing_samples"]
    if not len(gap_idx):
        return pd.DataFrame(columns=cols)
    observed = (ts[gap_idx] - ts[gap_idx - 1]) / np.timedelta64(1, "s")
    ref_s = float(typical_interval.total_seconds())
    return pd.DataFrame({
        "gap_id": np.arange(len(gap_idx)),
        "previous_timestamp": ts[gap_idx - 1], "next_timestamp": ts[gap_idx],
        "observed_interval_s": observed, "threshold_s": float(gap_threshold.total_seconds()),
        "reference_interval_s": ref_s,
        "estimated_missing_samples": np.maximum(0, np.round(observed / ref_s) - 1).astype(int)
    })[cols]

def build_segments_summary(df: pd.DataFrame, cfg: TimestampConfig) -> pd.DataFrame:
    grouped = df.groupby(cfg.segment_column, sort=True)[cfg.timestamp_column]
    out = pd.DataFrame({"start_timestamp": grouped.min(), "end_timestamp": grouped.max(),
                        "row_count": grouped.size()}).reset_index()
    out["duration_s"] = (out["end_timestamp"] - out["start_timestamp"]).dt.total_seconds()
    return out

def write_log(cfg: TimestampConfig, df: pd.DataFrame, gaps: pd.DataFrame,
              segments: pd.DataFrame, audit: dict, typical: pd.Timedelta,
              threshold: pd.Timedelta) -> Path:
    ts = df[cfg.timestamp_column]
    timezone_note = (f"timezone-aware (offset: {ts.dt.tz})" if ts.dt.tz is not None else
                     "timezone-naive; source timezone is unspecified")
    lines = [
        f"# Timestamp processing log — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
        f"- Input: `{cfg.input_csv.resolve()}`", f"- Output dir: `{cfg.output_dir.resolve()}`",
        f"- Rows after cleaning: **{len(df):,}**",
        f"- Invalid timestamps excluded: {audit['invalid_count']:,}",
        f"- Duplicate timestamp rows retained: {audit['dup_rows']:,} ({audit['dup_groups']:,} groups; {audit['exact_dups']:,} exact duplicates)",
        f"- Typical interval (median): {typical}", f"- Gap threshold: {threshold}",
        f"- Segments created: **{df[cfg.segment_column].nunique():,}**",
        f"- Gaps detected: **{len(gaps):,}**", f"- Timezone: {timezone_note}",
        f"- Timestamp range: {ts.min()} → {ts.max()}", "",
        "## Caveats", "- A time gap is not evidence of a machine failure.",
        "- Missing-sample estimates assume a regular interval.",
        "- Duplicate timestamps are retained; sampling interval is empirical.", ""
    ]
    if len(gaps):
        lines.insert(-2, f"- Total observed gap time: {gaps['observed_interval_s'].sum()/86400:.2f} days")
        lines.insert(-2, f"- Longest gap: {gaps['observed_interval_s'].max():.1f} seconds")
        lines.insert(-2, f"- Estimated missing samples: {int(gaps['estimated_missing_samples'].sum()):,}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "timestamp_log.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def run(cfg: TimestampConfig) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    raw = inspect_and_load(cfg)
    df, audit = process_timestamps(raw, cfg)
    df, typical, threshold = apply_segmentation(df, cfg)
    gaps = build_gap_table(df, cfg, typical, threshold)
    segments = build_segments_summary(df, cfg)
    if not df[cfg.timestamp_column].is_monotonic_increasing:
        raise AssertionError("Timestamps are not chronological after sorting")
    if int(segments["row_count"].sum()) != len(df):
        raise AssertionError("Segment summary row count mismatch")
    paths = {
        "data": cfg.output_dir / "data_ts_fixed.csv",
        "gaps": cfg.output_dir / "timestamp_gaps.csv",
        "segments": cfg.output_dir / "timestamp_segments_summary.csv",
    }
    df.to_csv(paths["data"], index=False)
    gaps.to_csv(paths["gaps"], index=False)
    segments.to_csv(paths["segments"], index=False)
    paths["log"] = write_log(cfg, df, gaps, segments, audit, typical, threshold)
    return paths

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Parse timestamps, audit gaps, and assign segments.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gap-multiplier", type=float, default=3.0)
    args = parser.parse_args()
    outputs = run(TimestampConfig(Path(args.input), Path(args.outdir), gap_multiplier=args.gap_multiplier))
    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}")

if __name__ == "__main__":
    main()
