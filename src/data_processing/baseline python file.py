from __future__ import annotations

# ───── 1. Imports ─────

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ───── 2. Constants ─────

log = logging.getLogger("baseline")

MAD_SCALE = 1.4826

IQR_TO_STD = 1.349

LEVELS = ("warning", "critical")

METHODS = ("iqr", "mad")

DEFAULT_OFFICIAL_FAILURES = (
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
)

HEALTHY_START = pd.Timestamp("2020-02-01 00:00:00")
HEALTHY_END = pd.Timestamp("2020-04-10 23:59:59")
HEALTHY_EXCLUDED_EPISODES = (
    ("2020-03-12 00:00:00", "2020-03-12 14:00:00"),
    ("2020-03-28 23:00:00", "2020-03-29 21:00:00"),
)
ANALOG_SIGNALS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
MIN_ROWS_FOR_FROZEN = 3

# ───── 3. Configuration ─────


@dataclass(frozen=True)
class BaselineConfig:
    """Settings for the baseline pipeline: paths, column names, states and multipliers."""

    data_path: Path = field(
        default_factory=lambda: Path(os.environ.get("BASELINE_DATA_PATH", "data_labeled.csv"))
    )
    output_dir: Path = Path("outputs/metrics")
    timestamp_col: str | None = "timestamp"
    label_col: str = "ground_truth"
    id_col: str | None = None
    sensors: tuple[str, ...] | None = None
    exclude_cols: tuple[str, ...] = ("segment",)
    state_col: str = "state"
    state_source_col: str = "Motor_current"
    state_labels: tuple[str, ...] = ("off", "unloaded", "loaded")
    state_edges: tuple[float, ...] = (1.0, 4.5)
    digital_max_unique: int = 2
    min_normal_rows: int = 30
    train_fraction: float = 1.0
    iqr_multipliers: tuple[float, float] = (1.5, 3.0)
    mad_multipliers: tuple[float, float] = (3.5, 6.0)
    official_failures: tuple[tuple[str, str], ...] = DEFAULT_OFFICIAL_FAILURES
    strict: bool = False

    def __post_init__(self) -> None:
        if len(self.state_labels) != len(self.state_edges) + 1:
            raise ValueError("state_labels must have exactly one more item than state_edges.")
        if list(self.state_edges) != sorted(self.state_edges):
            raise ValueError("state_edges must be in ascending order.")
        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1].")

    @property
    def quality_path(self) -> Path:
        return self.output_dir / "baseline_quality_report.csv"

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / "sensor_baseline.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "baseline_integrity_summary.json"

    @property
    def state_baseline_path(self) -> Path:
        return self.output_dir / "sensor_baseline_by_state.csv"

    @property
    def thresholds_path(self) -> Path:
        return self.output_dir / "sensor_thresholds_by_state.csv"

    @property
    def outliers_path(self) -> Path:
        return self.output_dir / "outlier_summary_by_state.csv"

    @property
    def failure_check_path(self) -> Path:
        return self.output_dir / "official_failure_check.csv"


# ───── 4. Loading & validation ─────


class DataValidationError(Exception):
    """Raised when the dataset fails a validation check."""


def load_data(cfg: BaselineConfig) -> pd.DataFrame:
    """Read the CSV file and parse the timestamp column."""
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"File is empty: {path}") from exc
    if df.empty:
        raise DataValidationError(f"File has no rows: {path}")
    if cfg.timestamp_col and cfg.timestamp_col in df.columns:
        df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    log.info("Loaded %s -> %d rows x %d columns", path.name, *df.shape)
    return df


def validate_schema(df: pd.DataFrame, cfg: BaselineConfig) -> None:
    """Check required columns exist and the label column is a complete 0/1 series."""
    required = [cfg.label_col, cfg.state_source_col]
    required += [c for c in (cfg.timestamp_col, cfg.id_col) if c]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Required columns missing: {missing}. Available: {list(df.columns)}")

    label = df[cfg.label_col]
    if label.isna().any():
        raise DataValidationError(f"'{cfg.label_col}' contains {int(label.isna().sum())} missing values.")
    invalid = set(pd.unique(label)) - {0, 1}
    if invalid:
        raise DataValidationError(f"'{cfg.label_col}' must be 0/1, found other values: {sorted(invalid)}")
    log.info("Schema OK. Label distribution: %s", label.value_counts().sort_index().to_dict())


def infer_sensors(df: pd.DataFrame, cfg: BaselineConfig) -> list[str]:
    """Return the sensor columns: explicit from config, otherwise auto-detected."""
    non_sensor = {cfg.timestamp_col, cfg.label_col, cfg.id_col, cfg.state_col, *cfg.exclude_cols}
    non_sensor.discard(None)

    if cfg.sensors:
        absent = [c for c in cfg.sensors if c not in df.columns]
        if absent:
            raise DataValidationError(f"Configured sensors missing from data: {absent}")
        sensors = list(cfg.sensors)
    else:
        sensors = [
            c for c in df.columns
            if c not in non_sensor and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not sensors:
            raise DataValidationError("No numeric sensor columns found. Set BaselineConfig.sensors.")
        log.info("Auto-detected %d sensors: %s", len(sensors), sensors)

    non_numeric = [c for c in sensors if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Sensor columns must be numeric, got non-numeric: {non_numeric}")
    return sensors


def _report(message: str, strict: bool) -> None:
    """Raise in strict mode, otherwise log the message as a warning."""
    if strict:
        raise DataValidationError(message)
    log.warning(message)


def _check_timestamps(df: pd.DataFrame, cfg: BaselineConfig) -> dict:
    """Check timestamps for parse failures, duplicates and ordering."""
    ts_col = cfg.timestamp_col
    timestamps = df[ts_col]
    keys = [c for c in (cfg.id_col, ts_col) if c]

    info = {
        "unparseable": int(timestamps.isna().sum()),
        "duplicate_timestamps": int(df.duplicated(subset=keys).sum()),
    }
    groups = [g for _, g in df.groupby(cfg.id_col)[ts_col]] if cfg.id_col else [timestamps]
    info["unordered_groups"] = int(sum(not g.dropna().is_monotonic_increasing for g in groups))

    if info["unparseable"]:
        _report(f"{info['unparseable']} timestamps could not be parsed.", cfg.strict)
    if info["duplicate_timestamps"]:
        _report(f"{info['duplicate_timestamps']} duplicate timestamps.", cfg.strict)
    if info["unordered_groups"]:
        _report("Timestamps are not sorted ascending.", cfg.strict)
    if timestamps.notna().any():
        info["start"], info["end"] = str(timestamps.min()), str(timestamps.max())
    return info


def check_integrity(df: pd.DataFrame, sensors: list[str], cfg: BaselineConfig) -> dict:
    """Run integrity checks and return a JSON-serialisable summary."""
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "sensors": sensors,
        "dtypes": {c: str(df[c].dtype) for c in sensors},
    }

    missing = df[sensors].isna().sum()
    infinite = np.isinf(df[sensors]).sum()
    summary["missing_total"] = int(missing.sum())
    summary["inf_total"] = int(infinite.sum())
    if missing.sum():
        _report(f"Missing values in sensors: {missing[missing > 0].to_dict()}", cfg.strict)
    if infinite.sum():
        _report(f"Infinite values in sensors: {infinite[infinite > 0].to_dict()}", cfg.strict)

    summary["duplicate_rows"] = int(df.duplicated().sum())
    if summary["duplicate_rows"]:
        _report(f"{summary['duplicate_rows']} fully duplicated rows.", cfg.strict)

    if cfg.timestamp_col is None:
        log.warning("No timestamp column configured - skipping timestamp checks.")
        summary["timestamp"] = "skipped"
    else:
        summary["timestamp"] = _check_timestamps(df, cfg)

    summary["normal_rows"] = int((df[cfg.label_col] == 0).sum())
    summary["anomaly_rows"] = int((df[cfg.label_col] == 1).sum())
    log.info("Integrity checks finished.")
    return summary


def integrity_overview(summary: dict) -> pd.DataFrame:
    """Flatten the integrity summary into a one-column table."""
    flat = {k: v for k, v in summary.items() if k not in ("dtypes", "sensors")}
    return pd.json_normalize(flat, sep="_").T.rename(columns={0: "value"})


# ───── 5. Profiling & baselines ─────


def _sensor_kind(n_unique: int, digital_max_unique: int) -> str:
    return "digital" if n_unique <= digital_max_unique else "analogue"


def build_quality_report(df: pd.DataFrame, sensors: list[str], digital_max_unique: int = 2) -> pd.DataFrame:
    """Build the per-sensor data quality table over the full dataset."""
    rows = []
    for sensor in sensors:
        column = df[sensor]
        finite = column.replace([np.inf, -np.inf], np.nan).dropna()
        n_unique = int(finite.nunique())
        rows.append({
            "sensor": sensor,
            "dtype": str(column.dtype),
            "n_rows": len(df),
            "n_missing": int(column.isna().sum()),
            "missing_pct": round(100 * column.isna().mean(), 4),
            "n_inf": int(np.isinf(column).sum()),
            "n_unique": n_unique,
            "zero_fraction": float((finite == 0).mean()) if len(finite) else np.nan,
            "is_constant": n_unique <= 1,
            "sensor_type": _sensor_kind(n_unique, digital_max_unique),
        })
    return pd.DataFrame(rows)


def _sensor_statistics(values: pd.Series, name: str, cfg: BaselineConfig, declared_type: str | None) -> dict:
    """Compute descriptive statistics for one sensor."""
    x = values.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        log.warning("Sensor %s has no valid normal readings - filled with NaN.", name)
        return {"sensor": name, "n": 0, "sensor_type": "unknown", "zscore_usable": False}

    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    median = x.median()
    mad = float((x - median).abs().median())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    n_unique = int(x.nunique())
    kind = declared_type or _sensor_kind(n_unique, cfg.digital_max_unique)
    is_digital = kind == "digital"
    is_constant = n_unique <= 1 or std == 0.0
    if is_constant:
        log.warning("Sensor %s is constant in normal data (std=0): z-scores undefined.", name)

    return {
        "sensor": name,
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": std,
        "median": float(median),
        "Q1": float(q1),
        "Q3": float(q3),
        "IQR": float(q3 - q1),
        "MAD": mad,
        "MAD_scaled": MAD_SCALE * mad,
        "min": float(x.min()),
        "max": float(x.max()),
        "p01": float(x.quantile(0.01)),
        "p99": float(x.quantile(0.99)),
        "n_unique": n_unique,
        "sensor_type": kind,
        "on_fraction": float((x == x.max()).mean()) if is_digital and n_unique == 2 else np.nan,
        "zscore_usable": bool(not is_digital and not is_constant),
    }


def compute_baseline(
    df: pd.DataFrame,
    sensors: list[str],
    cfg: BaselineConfig,
    sensor_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute the strict baseline from normal readings only (label == 0)."""
    normal = df[df[cfg.label_col] == 0]
    n_total, n_normal = len(df), len(normal)
    if n_normal < cfg.min_normal_rows:
        raise DataValidationError(
            f"Only {n_normal} normal rows (< min_normal_rows={cfg.min_normal_rows}); baseline unreliable."
        )
    log.info("Using %d / %d rows (%.1f%%) as normal baseline data.", n_normal, n_total, 100 * n_normal / n_total)

    declared = sensor_types or {}
    rows = [_sensor_statistics(normal[s], s, cfg, declared.get(s)) for s in sensors]
    return pd.DataFrame(rows).set_index("sensor")


def assign_states(df: pd.DataFrame, cfg: BaselineConfig) -> pd.Series:
    """Label each row with an operating state by cutting the state source column."""
    bins = [-np.inf, *cfg.state_edges, np.inf]
    states = pd.cut(df[cfg.state_source_col], bins=bins, right=False, labels=list(cfg.state_labels))
    return states.astype(str)


def summarize_states(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    """Count normal and fault rows per operating state."""
    counts = pd.crosstab(df[cfg.state_col], df[cfg.label_col]).reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["normal_rows", "fault_rows"]
    counts = counts.reindex([s for s in cfg.state_labels if s in counts.index])
    counts.insert(0, "rows", counts.sum(axis=1))
    counts["share_of_normal_pct"] = 100 * counts["normal_rows"] / counts["normal_rows"].sum()
    counts["share_of_fault_pct"] = 100 * counts["fault_rows"] / max(counts["fault_rows"].sum(), 1)
    return counts.round(3)


def split_train_eval(df: pd.DataFrame, cfg: BaselineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically into training and evaluation rows."""
    if cfg.train_fraction >= 1.0:
        return df, df
    ordered = df.sort_values(cfg.timestamp_col, kind="stable") if cfg.timestamp_col else df
    cut = int(len(ordered) * cfg.train_fraction)
    log.info("Chronological split: %d training rows, %d evaluation rows.", cut, len(ordered) - cut)
    return ordered.iloc[:cut], ordered.iloc[cut:]


def compute_baseline_by_state(
    train: pd.DataFrame,
    sensors: list[str],
    cfg: BaselineConfig,
    sensor_types: dict[str, str],
) -> pd.DataFrame:
    """Compute the normal-only baseline separately for each operating state."""
    frames = []
    for state in cfg.state_labels:
        subset = train[train[cfg.state_col] == state]
        try:
            part = compute_baseline(subset, sensors, cfg, sensor_types)
        except DataValidationError as exc:
            log.warning("State '%s' skipped: %s", state, exc)
            continue
        part.insert(0, "state", state)
        frames.append(part)
    if not frames:
        raise DataValidationError("No operating state has enough normal rows for a baseline.")
    return pd.concat(frames)


# ───── 6. Thresholds & outliers ─────


def build_thresholds(baseline_by_state: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    """Derive IQR and MAD warning / critical limits per state and sensor."""
    stats = baseline_by_state.reset_index()
    iqr_scale = stats["IQR"].where(stats["IQR"] > 0, IQR_TO_STD * stats["std"])
    mad_scale = stats["MAD_scaled"].where(stats["MAD_scaled"] > 0, stats["std"])

    thresholds = stats[["state", "sensor"]].copy()
    thresholds["scale_fallback"] = (stats["IQR"] == 0) | (stats["MAD"] == 0)
    for level, k_iqr, k_mad in zip(LEVELS, cfg.iqr_multipliers, cfg.mad_multipliers):
        thresholds[f"iqr_{level}_low"] = stats["Q1"] - k_iqr * iqr_scale
        thresholds[f"iqr_{level}_high"] = stats["Q3"] + k_iqr * iqr_scale
        thresholds[f"mad_{level}_low"] = stats["median"] - k_mad * mad_scale
        thresholds[f"mad_{level}_high"] = stats["median"] + k_mad * mad_scale
    return thresholds


def evaluate_outliers(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    sensors: list[str],
    cfg: BaselineConfig,
) -> pd.DataFrame:
    """Compute outlier rates per state, sensor, method and level (vectorised)."""
    labels = list(cfg.state_labels)
    codes = pd.Categorical(df[cfg.state_col], categories=labels).codes.astype(np.int64)
    truth = df[cfg.label_col].to_numpy().astype(np.int64)
    group = codes * 2 + truth
    group_size = np.bincount(group, minlength=2 * len(labels))

    rows = []
    for sensor in sensors:
        per_state = thresholds[thresholds["sensor"] == sensor].set_index("state").reindex(labels)
        values = df[sensor].to_numpy()
        for method in METHODS:
            for level in LEVELS:
                low = per_state[f"{method}_{level}_low"].to_numpy()[codes]
                high = per_state[f"{method}_{level}_high"].to_numpy()[codes]
                outside = ((values < low) | (values > high)).astype(float)
                hits = np.bincount(group, weights=outside, minlength=2 * len(labels))
                for code, state in enumerate(labels):
                    n_normal, n_fault = group_size[2 * code], group_size[2 * code + 1]
                    if n_normal + n_fault == 0:
                        continue
                    rows.append({
                        "state": state,
                        "sensor": sensor,
                        "method": method,
                        "level": level,
                        "normal_out_pct": 100 * hits[2 * code] / n_normal if n_normal else np.nan,
                        "fault_out_pct": 100 * hits[2 * code + 1] / n_fault if n_fault else np.nan,
                    })
    return pd.DataFrame(rows)


def pivot_outliers(outliers: pd.DataFrame) -> pd.DataFrame:
    """Pivot the outlier table into a state x sensor overview."""
    return outliers.pivot_table(
        index=["state", "sensor"],
        columns=["method", "level"],
        values=["normal_out_pct", "fault_out_pct"],
    ).round(2)


def check_official_failures(df: pd.DataFrame, cfg: BaselineConfig) -> tuple[pd.DataFrame, int]:
    """Compare labels with the official failure windows; return the table and stray-label count."""
    columns = ["window_start", "window_end", "rows", "labelled_anomaly", "unlabelled", "status"]
    if not cfg.timestamp_col or not cfg.official_failures:
        return pd.DataFrame(columns=columns), 0

    timestamps = df[cfg.timestamp_col]
    is_anomaly = df[cfg.label_col] == 1
    covered = pd.Series(False, index=df.index)
    rows = []
    for start, end in cfg.official_failures:
        inside = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
        covered |= inside
        n_rows = int(inside.sum())
        n_labelled = int(is_anomaly[inside].sum())
        rows.append({
            "window_start": start,
            "window_end": end,
            "rows": n_rows,
            "labelled_anomaly": n_labelled,
            "unlabelled": n_rows - n_labelled,
            "status": "OK" if n_rows > 0 and n_rows == n_labelled else "MISMATCH",
        })

    table = pd.DataFrame(rows, columns=columns)
    outside = int((is_anomaly & ~covered).sum())
    if (table["status"] == "MISMATCH").any():
        _report("Some official failure windows are not fully labelled as anomalies.", cfg.strict)
    if outside:
        _report(f"{outside} anomaly labels fall outside every official failure window.", cfg.strict)
    return table, outside


# ───── 7. Healthy-period selection ─────


def flag_frozen_rows(frame: pd.DataFrame, cfg: BaselineConfig) -> pd.Series:
    """Flag rows in minutes where every analogue signal is constant (frozen logger)."""
    minute = frame[cfg.timestamp_col].dt.floor("min")
    keys = [frame["segment"], minute] if "segment" in frame.columns else [minute]
    grouped = frame.groupby(keys)
    spread = (
        grouped[ANALOG_SIGNALS].transform("max") - grouped[ANALOG_SIGNALS].transform("min")
    ).max(axis=1)
    rows_in_minute = grouped[cfg.timestamp_col].transform("size")
    return (rows_in_minute >= MIN_ROWS_FOR_FROZEN) & (spread == 0)


def select_healthy_rows(frame: pd.DataFrame, cfg: BaselineConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Return the healthy-row mask and a step-by-step selection summary."""
    timestamps = frame[cfg.timestamp_col]
    in_window = timestamps.between(HEALTHY_START, HEALTHY_END)
    is_normal = frame[cfg.label_col].eq(0)
    in_episode = pd.Series(False, index=frame.index)
    for start, end in HEALTHY_EXCLUDED_EPISODES:
        in_episode |= timestamps.between(pd.Timestamp(start), pd.Timestamp(end))
    is_frozen = flag_frozen_rows(frame, cfg)

    mask = in_window & is_normal & ~in_episode & ~is_frozen
    counts = {
        "all rows": len(frame),
        "inside healthy window": int(in_window.sum()),
        "... and ground_truth == 0": int((in_window & is_normal).sum()),
        "... and not in excluded episodes": int((in_window & is_normal & ~in_episode).sum()),
        "... and not frozen (final)": int(mask.sum()),
    }
    steps = pd.DataFrame({"rows": counts})
    steps["removed_by_step"] = (-steps["rows"].diff()).astype("Int64")
    return mask, steps


# ───── 8. Orchestration & CLI ─────


@dataclass
class PipelineResult:
    """Container for every table produced by run_pipeline."""

    df: pd.DataFrame
    sensors: list[str]
    analogue_sensors: list[str]
    integrity: dict
    quality: pd.DataFrame
    baseline: pd.DataFrame
    state_summary: pd.DataFrame
    baseline_by_state: pd.DataFrame
    thresholds: pd.DataFrame
    outliers: pd.DataFrame
    outliers_pivot: pd.DataFrame
    failure_check: pd.DataFrame
    anomalies_outside_windows: int


def save_outputs(result: PipelineResult, cfg: BaselineConfig) -> None:
    """Write all pipeline tables and the integrity summary to the output directory."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result.quality.to_csv(cfg.quality_path, index=False)
    result.baseline.to_csv(cfg.baseline_path)
    result.baseline_by_state.to_csv(cfg.state_baseline_path)
    result.thresholds.to_csv(cfg.thresholds_path, index=False)
    result.outliers.to_csv(cfg.outliers_path, index=False)
    result.failure_check.to_csv(cfg.failure_check_path, index=False)
    with open(cfg.summary_path, "w", encoding="utf-8") as handle:
        json.dump(result.integrity, handle, indent=2, default=str)

    for path in list_output_files(cfg)["path"]:
        log.info("saved: %s", path)


def list_output_files(cfg: BaselineConfig) -> pd.DataFrame:
    """List the files currently present in the output directory."""
    files = sorted(cfg.output_dir.glob("*")) if cfg.output_dir.exists() else []
    return pd.DataFrame({
        "file": [f.name for f in files],
        "size_kb": [round(f.stat().st_size / 1024, 1) for f in files],
        "path": [str(f.resolve()) for f in files],
    })


def run_pipeline(cfg: BaselineConfig, save: bool = True) -> PipelineResult:
    """Run validation, profiling, baselines, thresholds and label checks end to end."""
    df = load_data(cfg)
    validate_schema(df, cfg)
    sensors = infer_sensors(df, cfg)

    integrity = check_integrity(df, sensors, cfg)
    quality = build_quality_report(df, sensors, cfg.digital_max_unique)
    sensor_types = dict(zip(quality["sensor"], quality["sensor_type"]))
    analogue = [s for s in sensors if sensor_types[s] == "analogue"]

    df[cfg.state_col] = assign_states(df, cfg)
    train, evaluation = split_train_eval(df, cfg)

    baseline = compute_baseline(train, sensors, cfg, sensor_types)
    state_summary = summarize_states(df, cfg)
    baseline_by_state = compute_baseline_by_state(train, analogue, cfg, sensor_types)
    thresholds = build_thresholds(baseline_by_state, cfg)
    outliers = evaluate_outliers(evaluation, thresholds, analogue, cfg)
    failure_check, outside = check_official_failures(df, cfg)

    result = PipelineResult(
        df=df,
        sensors=sensors,
        analogue_sensors=analogue,
        integrity=integrity,
        quality=quality,
        baseline=baseline,
        state_summary=state_summary,
        baseline_by_state=baseline_by_state,
        thresholds=thresholds,
        outliers=outliers,
        outliers_pivot=pivot_outliers(outliers),
        failure_check=failure_check,
        anomalies_outside_windows=outside,
    )
    if save:
        save_outputs(result, cfg)
    return result


def main() -> None:
    """Command-line entry point: full pipeline followed by the healthy-period baseline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    parser = argparse.ArgumentParser(description="Data quality and normal-only sensor baseline.")
    parser.add_argument("--data", type=Path, required=False, help="Path to the labelled CSV file.")
    parser.add_argument("--out", type=Path, default=Path("outputs/metrics"), help="Output directory.")
    parser.add_argument("--strict", action="store_true", help="Raise on integrity problems instead of warning.")
    args = parser.parse_args()

    config = BaselineConfig(
        data_path=Path(r"C:\Users\compu import\Desktop\TEST\phase1_imputed\labeled_data_imputed.csv"),
        output_dir=args.out,
        strict=args.strict,
    )
    result = run_pipeline(config)

    healthy_mask, healthy_selection = select_healthy_rows(result.df, config)
    healthy_df = result.df.loc[healthy_mask]
    log.info("Healthy-period rows selected: %d", len(healthy_df))

    sensor_types = dict(zip(result.quality["sensor"], result.quality["sensor_type"]))
    healthy_baseline = compute_baseline(healthy_df, result.sensors, config, sensor_types)
    healthy_baseline_by_state = compute_baseline_by_state(
        healthy_df, result.analogue_sensors, config, sensor_types
    )
    healthy_thresholds = build_thresholds(healthy_baseline_by_state, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    healthy_baseline.to_csv(config.output_dir / "sensor_baseline_healthy.csv")
    healthy_baseline_by_state.to_csv(config.output_dir / "sensor_baseline_by_state_healthy.csv")
    healthy_thresholds.to_csv(config.output_dir / "sensor_thresholds_by_state_healthy.csv", index=False)
    healthy_selection.to_csv(config.output_dir / "healthy_selection_summary.csv")
    for path in list_output_files(config)["path"]:
        log.info("saved: %s", path)


if __name__ == "__main__":
    main()