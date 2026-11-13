"""
export_scenarios.py

Scenario per window export. For each dataset, apply the calibrated
scenario rules to every window in a given split and write one row
per window: window_id, timestamp, segment, scenario, sensors.

Does not use model outputs. Reads only:
  {split_dir}/{prefix}/{prefix}_{part}.csv
  {split_dir}/{prefix}/{prefix}_features.json
  {out_dir}/{prefix}/{prefix}_scenario_params.json

Writes:
  {out_dir}/{prefix}/{prefix}_scenarios_{part}.csv
"""

import argparse
import json
import os

import pandas as pd

import scenario_rules as sr


def load_features(split_dir, prefix):
    path = os.path.join(split_dir, prefix, f"{prefix}_features.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(out_dir, prefix):
    path = os.path.join(out_dir, prefix, f"{prefix}_scenario_params.json")
    return sr.load_model(path)


def load_split(split_dir, prefix, part, feats):
    path = os.path.join(split_dir, prefix, f"{prefix}_{part}.csv")
    head = pd.read_csv(path, nrows=0)
    cols = ["timestamp", "window_id", "segment"] + feats
    if "ground_truth" in head.columns:
        cols = cols + ["ground_truth"]
    return pd.read_csv(path, usecols=cols, parse_dates=["timestamp"])


def window_table(df, analog, gt_col=None):
    g = df.groupby("window_id", sort=True)
    meta = pd.DataFrame({
        "ts": g["timestamp"].min(),
        "segment": g["segment"].first(),
    })
    if gt_col and gt_col in df.columns:
        meta["gt"] = g[gt_col].max()
    means = g[analog].mean()
    stds = g[analog].std(ddof=0)
    return means, stds, meta


def export(split_dir, out_dir, prefix, part):
    print(f"\n=== {prefix} / {part} ===")

    feats = load_features(split_dir, prefix)
    model = load_model(out_dir, prefix)
    analog = list(model["sensors"].keys())

    df = load_split(split_dir, prefix, part, feats)
    gt_col = "ground_truth" if "ground_truth" in df.columns else None
    means, stds, meta = window_table(df, analog, gt_col)
    print(f"  windows: {len(means)}")
    print(f"  analog sensors: {len(analog)}")
    print(f"  ref_mode: {model['selection']['ref_mode']}")

    codes = sr.detect_series(
        model,
        means[analog],
        stds[analog],
        means.index.values,
        meta["segment"].values,
    )

    rows = []
    for i, wid in enumerate(codes.index):
        
        scenario, sensors = sr.summarize_window(codes.iloc[i], min_sensors=3) 
        row = {
            "window_id": int(wid),
            "timestamp": meta.loc[wid, "ts"],
            "segment": meta.loc[wid, "segment"],
            "scenario": scenario,
            "sensors": ",".join(sensors) if sensors else "",
        }
        if gt_col:
            row["ground_truth"] = int(meta.loc[wid, "gt"])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, prefix, f"{prefix}_scenarios_{part}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {out_path}")
    print(f"  scenario distribution:")
    for name, cnt in out_df["scenario"].value_counts().items():
        pct = 100.0 * cnt / len(out_df)
        print(f"    {name:>16}: {cnt:>6}  ({pct:>5.2f}%)")

    if gt_col:
        normal = out_df[out_df["ground_truth"] == 0]
        anomaly = out_df[out_df["ground_truth"] == 1]
        if len(normal) > 0:
            print(f"  on normal windows (n={len(normal)}):")
            for name, cnt in normal["scenario"].value_counts().items():
                if name == "normal":
                    continue
                pct = 100.0 * cnt / len(normal)
                print(f"    {name:>16}: {cnt:>6}  ({pct:>5.2f}%)")
        if len(anomaly) > 0:
            print(f"  on anomaly windows (n={len(anomaly)}):")
            for name, cnt in anomaly["scenario"].value_counts().items():
                if name == "normal":
                    continue
                pct = 100.0 * cnt / len(anomaly)
                print(f"    {name:>16}: {cnt:>6}  ({pct:>5.2f}%)")

    return out_df


def main():
    parser = argparse.ArgumentParser(description="Scenario per window export.")
    parser.add_argument("--split-dir", default="D:/processed_outputs")
    parser.add_argument("--out-dir", default="scenario_out")
    parser.add_argument("--prefixes", nargs="+", default=["metro", "pump", "iot"])
    parser.add_argument("--parts", nargs="+", default=["test"])
    args = parser.parse_args()

    for prefix in args.prefixes:
        for part in args.parts:
            try:
                export(args.split_dir, args.out_dir, prefix, part)
            except Exception as e:
                print(f"\n=== {prefix} / {part} ===")
                print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()