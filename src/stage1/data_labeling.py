import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULTS = {
    "work_dir": ".",
    "timestamp_col": "timestamp",
    "gt_mode": "intervals",
    "gt_col": "ground_truth",
    "events_path": None,
    "events_start_col": "start_time",
    "events_end_col": "end_time",
    "label_col_raw": None,
    "label_map": {},
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = {**DEFAULTS, **json.load(f)}
    work = Path(cfg["work_dir"])

    # Prefer an explicitly configured labeling path, then the dataset's
    # general input path. Keep the old pipeline location as a backward-compatible fallback.
    cfg["labeling_input_path"] = (
        cfg.get("labeling_input_path")
        or cfg.get("input")
        or str(work / "2_clean" / "data_clean.csv")
    )
    cfg["labeling_output_path"] = (
        cfg.get("labeling_output_path")
        or str(work / "3_labeled" / "data_labeled.csv")
    )
    return cfg


def label_failures(sensors_df: pd.DataFrame, failures_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply inclusive failure-event intervals to timestamped rows."""
    ts, gt = cfg["timestamp_col"], cfg["gt_col"]
    s_col, e_col = cfg["events_start_col"], cfg["events_end_col"]
    if ts not in sensors_df.columns:
        raise ValueError(f"Sensor data missing column: {ts}")
    missing = {s_col, e_col} - set(failures_df.columns)
    if missing:
        raise ValueError(f"Failure events missing columns: {sorted(missing)}")
    data = sensors_df.copy()
    events = failures_df.copy()
    data[ts] = pd.to_datetime(data[ts], errors="raise")
    events[s_col] = pd.to_datetime(events[s_col], errors="raise")
    events[e_col] = pd.to_datetime(events[e_col], errors="raise")
    if (events[e_col] < events[s_col]).any():
        raise ValueError("Every failure event must end at or after its start")
    data = data.sort_values(ts).reset_index(drop=True)
    data[gt] = 0
    for _, event in events.iterrows():
        mask = data[ts].between(event[s_col], event[e_col], inclusive="both")
        data.loc[mask, gt] = 1
    if not data.empty and int(data[gt].sum()) == 0:
        raise ValueError("No rows were labeled; check timestamps and failure-event intervals")
    return data


def label_by_column(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Map a raw status column (text) to 0/1 using label_map."""
    raw, gt, mapping = cfg["label_col_raw"], cfg["gt_col"], cfg["label_map"]
    if not raw or raw not in df.columns:
        raise ValueError(f"label_col_raw {raw!r} not found in the data")
    if not mapping:
        raise ValueError("label_map is empty in the config")
    unknown = set(df[raw].dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Values in '{raw}' missing from label_map: {sorted(unknown)}")
    out = df.copy()
    out[gt] = out[raw].map(mapping).astype(int)
    return out


def run(cfg: dict) -> Path:
    inp, outp = Path(cfg["labeling_input_path"]), Path(cfg["labeling_output_path"])
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")
    df = pd.read_csv(inp)
    mode = cfg["gt_mode"]

    if mode == "intervals":
        if not cfg["events_path"]:
            raise ValueError("gt_mode 'intervals' needs events_path in the config")
        ev = Path(cfg["events_path"])
        if not ev.is_file():
            raise FileNotFoundError(f"Failure-events file not found: {ev.resolve()}")
        df = label_failures(df, pd.read_csv(ev), cfg)
    elif mode == "column":
        df = label_by_column(df, cfg)
    elif mode == "none":
        print("gt_mode 'none': no labels available, data passed through unchanged")
    else:
        raise ValueError(f"Unknown gt_mode: {mode}")

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    if mode != "none":
        gt = cfg["gt_col"]
        print(f"Rows: {len(df)}  |  abnormal rows: {int(df[gt].sum())}")
    return outp


def main() -> None:
    parser = argparse.ArgumentParser(description="Add the ground_truth column")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run(load_config(args.config))
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
