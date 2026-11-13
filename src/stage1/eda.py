import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
cfg = json.load(open(parser.parse_args().config, encoding="utf-8"))

name = cfg.get("name", "dataset")
ts_col = cfg.get("timestamp_col", "timestamp")
mult = cfg.get("gap_multiplier", 3.0)
out_dir = Path(cfg.get("work_dir", ".")) / "0_eda"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(cfg["raw_input_path"], encoding="utf-8-sig")
df = df.drop(columns=[c for c in cfg.get("drop_columns", []) if c in df.columns])

ts = pd.to_datetime(df[ts_col], errors="coerce")
diffs = ts.dropna().sort_values().diff().dropna()
median = diffs.median()
n_gaps = int((diffs > median * mult).sum())

missing = df.isna().sum()
missing = missing[missing > 0]
numeric = df.select_dtypes("number")

lines = [f"EDA report: {name}", ""]
lines += ["Size",
          f"  rows: {len(df):,}",
          f"  columns: {df.shape[1]}", ""]
lines += ["Time",
          f"  range: {ts.min()} to {ts.max()}",
          f"  invalid timestamps: {int(ts.isna().sum())}",
          f"  sorted: {ts.dropna().is_monotonic_increasing}",
          f"  median interval: {median}",
          f"  longest gap: {diffs.max()}",
          f"  gaps over {mult} x median: {n_gaps}", ""]
lines += ["Duplicates and missing",
          f"  duplicate rows: {int(df.duplicated().sum())}",
          f"  duplicate timestamps: {int(ts.duplicated().sum())}",
          f"  missing values: {int(missing.sum())}"]
lines += [f"    {c}: {n:,} ({n / len(df):.1%})" for c, n in missing.items()]
lines += ["", "Columns", "  name | type | min | max | mean"]
for c in df.columns:
    if c in numeric.columns:
        lines.append(f"  {c} | numeric | {numeric[c].min():.4g} | "
                     f"{numeric[c].max():.4g} | {numeric[c].mean():.4g}")
    else:
        lines.append(f"  {c} | text")

lines += ["", "Ground truth"]
mode = cfg.get("gt_mode", "none")
if mode == "column":
    lines.append(f"  status column: {cfg['label_col_raw']}")
    lines += [f"    {k}: {v:,}"
              for k, v in df[cfg["label_col_raw"]].value_counts().items()]
elif mode == "intervals":
    events = pd.read_csv(cfg["events_path"])
    s, e = cfg.get("events_start_col", "start_time"), cfg.get("events_end_col", "end_time")
    lines.append(f"  failure intervals: {len(events)}")
    lines += [f"    {r[s]} to {r[e]}" for _, r in events.iterrows()]
else:
    lines.append("  none")

report = out_dir / "eda_report.txt"
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:20]))
print("\nSaved", report)
