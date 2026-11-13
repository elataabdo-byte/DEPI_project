import argparse
import json
import os

import numpy as np
import pandas as pd

DEFAULT_SPLIT = {
    "mode": "events",
    "test_events": 2,
    
    "lead_hours": 24,
    "val_ratio": 0.15,
    "gap_windows": 2,
    "test_ratio": 0.15,
    "train_ratio": 0.70,
}


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_windows(df, ts_col):
    """One row per window_id: first/last row index, first timestamp, has_gt."""
    g = df.groupby("window_id", sort=False)
    w = pd.DataFrame({
        "start": g.apply(lambda x: x.index.min(), include_groups=False),
        "end": g.apply(lambda x: x.index.max(), include_groups=False),
        "ts": g[ts_col].min(),
        "has_gt": g["_gt"].max().astype(int),
    }).reset_index()
    w = w.sort_values("start").reset_index(drop=True)
    if not (w["start"].iloc[1:].values > w["end"].iloc[:-1].values).all():
        raise ValueError("window_id values are not contiguous blocks in time order")
    return w


def window_id_is_valid(df, ts_col, window_seconds):
    wid = df["window_id"].values
    if (np.diff(wid) < 0).any():
        return False
    g = df.groupby("window_id", sort=False)[ts_col]
    span = (g.max() - g.min()).dt.total_seconds()
    return bool((span <= window_seconds * 1.5).all())


def rebuild_window_id(df, ts_col, seg_col, window_seconds):
    t0 = df.groupby(seg_col)[ts_col].transform("min")
    bucket = ((df[ts_col] - t0).dt.total_seconds() // window_seconds).astype(int)
    key = df[seg_col].astype(str) + "_" + bucket.astype(str)
    return pd.factorize(key)[0] + 1


def failure_starts(df, ts_col):
    gt = df["_gt"].values
    rising = np.where((gt[1:] == 1) & (gt[:-1] == 0))[0] + 1
    if gt[0] == 1:
        rising = np.r_[0, rising]
    return df[ts_col].iloc[rising].tolist()


def apply_rules(df, rules):
    """Return int Series: 1 if any rule violated, else 0."""
    gt = pd.Series(0, index=df.index, dtype=int)
    for col, rng in rules.items():
        if col not in df.columns:
            continue
        if "min" in rng:
            gt = gt | (df[col] < rng["min"]).astype(int)
        if "max" in rng:
            gt = gt | (df[col] > rng["max"]).astype(int)
    return gt


def split_dataset(data_path, config_path, out_dir, prefix):
    print(f"\n=== {prefix} ===  data={data_path}  config={config_path}")
    cfg = load_config(config_path)
    sp = {**DEFAULT_SPLIT, **cfg.get("split", {})}
    ts_col = cfg.get("timestamp_col", cfg.get("timestamp_column", "timestamp"))
    seg_col = cfg.get("segment_col", "segment")
    gt_mode = cfg.get("gt_mode", "none")
    gt_col = cfg.get("gt_col", "ground_truth")
    label_raw = cfg.get("label_col_raw")

    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in cfg.get("drop_columns", []) if c in df.columns])
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        raise ValueError("unparseable timestamps found")
    df = df.sort_values(ts_col, kind="stable").reset_index(drop=True)

    win_s = cfg.get("window_seconds", 180)
    rebuilt = False
    if "window_id" not in df.columns or not window_id_is_valid(df, ts_col, win_s):
        print(f"WARNING: window_id missing/invalid -> rebuilt from {seg_col} + {win_s}s buckets")
        df["window_id"] = rebuild_window_id(df, ts_col, seg_col, win_s)
        rebuilt = True

    # ---- ground truth per mode
    if gt_mode == "rules":
        df["_gt"] = apply_rules(df, cfg.get("rules", {}))
        has_gt = True
    elif gt_mode == "column" and gt_col in df.columns:
        df["_gt"] = df[gt_col].astype(int)
        has_gt = True
    elif gt_mode == "intervals" and gt_col in df.columns:
        df["_gt"] = df[gt_col].astype(int)
        has_gt = True
    elif gt_mode == "intervals":
        df["_gt"] = apply_rules(df, cfg.get("rules", {}))
        has_gt = True
    else:
        df["_gt"] = 0
        has_gt = False

    print(f"rows={len(df)}  period={df[ts_col].min()} -> {df[ts_col].max()}  "
          f"has_ground_truth={has_gt}  gt_rows={int(df['_gt'].sum())}")

    o = os.path.join(out_dir, prefix)
    os.makedirs(o, exist_ok=True)

    clean_export_df = df.drop(columns=["_gt"], errors="ignore")
    clean_export_path = os.path.join(o, f"{prefix}_clean_data.csv")
    clean_export_df.to_csv(clean_export_path, index=False)
    print(f"  -> Clean dataset exported to: {clean_export_path}")

    # ---- non-features / features
    meta_cols = [c for c in [ts_col, seg_col, "window_id", gt_col, label_raw] if c and c in df.columns]
    non_feat = set(meta_cols) | set(cfg.get("exclude_cols", [])) | set(cfg.get("protect_cols", [])) | {"_gt"}
    cand = [c for c in df.select_dtypes(include="number").columns if c not in non_feat]

    # ---- windows and split
    w = build_windows(df, ts_col)
    seg_map = df.drop_duplicates("window_id").set_index("window_id")[seg_col]
    w["segment"] = w["window_id"].map(seg_map)
    n_w = len(w)
    gap = int(sp["gap_windows"])
    events = failure_starts(df, ts_col) if (has_gt and gt_mode == "intervals") else []
    split_mode = sp.get("mode", "events")

    if split_mode == "segments":
        seg_first = df.groupby(seg_col)[ts_col].min().sort_values()
        segments = seg_first.index.tolist()
        n_seg = len(segments)
        tr_cut = max(1, int(round(n_seg * sp.get("train_ratio", 0.70))))
        va_cut = max(tr_cut + 1, int(round(n_seg * (sp.get("train_ratio", 0.70) + sp.get("val_ratio", 0.15)))))
        train_segs = set(segments[:tr_cut])
        val_segs   = set(segments[tr_cut:va_cut])
        test_segs  = set(segments[va_cut:])

        def _span(segs):
            ww = w[w["segment"].isin(segs)]
            if len(ww) == 0:
                return (0, 0)
            return (int(ww.index.min()), int(ww.index.max() + 1))

        tr_a, tr_b = _span(train_segs)
        va_a, va_b = _span(val_segs)
        te_a, te_b = _span(test_segs)
        idx = {"train": (tr_a, tr_b), "val": (va_a, va_b), "test": (te_a, te_b)}
        mode = f"segments_ratio train={len(train_segs)} val={len(val_segs)} test={len(test_segs)}"
    else:
        if has_gt and len(events) >= 1:
            k = min(int(sp["test_events"]), len(events))
            t0 = events[-k] - pd.Timedelta(hours=sp["lead_hours"])
            test_w = int(np.searchsorted(w["ts"].values, np.datetime64(t0), side="right") - 1)
            test_w = max(test_w, 0)
            mode = f"test starts {sp['lead_hours']}h before the last {k} failure(s)"
        else:
            test_w = int(n_w * (1 - sp["test_ratio"]))
            mode = "no ground truth -> plain tail split"
            if has_gt:
                mode = "no failures found -> plain tail split"
        val_end = test_w - gap
        val_len = int(round(sp["val_ratio"] * n_w))
        val_start = val_end - val_len
        train_end = val_start - gap
        if train_end <= 0:
            raise ValueError("Not enough windows for Train; reduce test_events/val_ratio")
        idx = {"train": (0, train_end), "val": (val_start, val_end), "test": (test_w, n_w)}

    # ---- build parts, filtering has_gt for train/val
    parts = {}
    dropped = {}
    for name, (a, b) in idx.items():
        ww = w.iloc[a:b]
        if name != "test":
            keep = ww[ww["has_gt"] == 0]
            dropped[name] = int(len(ww) - len(keep))
            ww = keep
        else:
            dropped[name] = 0
        ids = set(ww["window_id"])
        parts[name] = df[df["window_id"].isin(ids)].copy()

    # ---- features: drop constant in Train
    train = parts["train"]
    const = [c for c in cand if train[c].nunique(dropna=True) <= 1]
    features = [c for c in cand if c not in const]
    mu = train[features].mean()
    sd = train[features].std(ddof=0).replace(0, 1.0)

    # ---- sanity checks
    report = {
        "dataset": prefix,
        "cut_mode": mode,
        "gt_mode": gt_mode,
        "window_id_rebuilt": rebuilt,
        "window_seconds": win_s,
        "n_features": len(features),
        "features": features,
        "constant_features_removed": const,
        "failure_events_total": len(events),
        "windows_dropped_for_gt": dropped,
        "gap_windows": gap,
        "parts": {},
    }
    problems = []

    ids = {n: set(p["window_id"]) for n, p in parts.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if ids[a] & ids[b]:
            problems.append(f"window overlap between {a} and {b}")

    if (len(parts["train"]) and len(parts["val"]) and len(parts["test"])
            and not (parts["train"][ts_col].max() < parts["val"][ts_col].min()
                     <= parts["val"][ts_col].max() < parts["test"][ts_col].min())):
        problems.append("parts are not strictly ordered in time")

    for n in ["train", "val"]:
        if len(parts[n]) and parts[n]["_gt"].sum() > 0:
            problems.append(f"{n} contains failure rows")

    if has_gt and len(parts["test"]) and parts["test"]["_gt"].sum() == 0:
        problems.append("TEST HAS NO FAILURES (or none detected)")

    if set(features) & non_feat:
        problems.append("non-feature column leaked into features")

    for n, p in parts.items():
        if len(p) and p[features].isna().any().any():
            problems.append(f"NaN in {n} features")

    report["problems"] = problems

    # ---- save
    for name, p in parts.items():
        p_out = p.drop(columns=["_gt"])
        m = [c for c in meta_cols if c in p_out.columns]
        p_out[[ts_col] + features + [c for c in m if c != ts_col]].to_csv(
            os.path.join(o, f"{prefix}_{name}.csv"), index=False)
        ((p_out[features] - mu) / sd).to_csv(
            os.path.join(o, f"{prefix}_X_{name}.csv"), index=False)
        p_out[m].to_csv(
            os.path.join(o, f"{prefix}_meta_{name}.csv"), index=False)

        report["parts"][name] = {
            "rows": int(len(p)),
            "windows": int(p["window_id"].nunique()) if len(p) else 0,
            "from": str(p[ts_col].min()) if len(p) else None,
            "to": str(p[ts_col].max()) if len(p) else None,
            "gt_rows": int(p["_gt"].sum()),
            "gt_windows": int(p.groupby("window_id")["_gt"].max().sum()) if len(p) else 0,
        }

    with open(os.path.join(o, f"{prefix}_split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(o, f"{prefix}_scaler.json"), "w") as f:
        json.dump({"fit_on": "train", "mean": mu.to_dict(), "std": sd.to_dict()}, f, indent=2)
    with open(os.path.join(o, f"{prefix}_features.json"), "w") as f:
        json.dump(features, f, indent=2)

    # ---- print
    print(f"cut: {mode}")
    for n, r in report["parts"].items():
        print(f"  {n:5s} rows={r['rows']:>8}  windows={r['windows']:>6}  "
              f"{r['from']} -> {r['to']}  gt_windows={r['gt_windows']}")
    print(f"  features={len(features)}  removed_constant={const}  "
          f"gt windows dropped={dropped}")
    print("  CHECKS:", "OK" if not problems else problems)

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="processed_outputs")
    ap.add_argument("--task", nargs=3, action="append",
                    metavar=("PREFIX", "DATA_CSV", "CONFIG_JSON"),
                    help="repeatable, e.g. --task pump datasets/data__pump_clean.csv configs/pump.json")
    a = ap.parse_args()
    tasks = a.task or [
        ("iot",   "datasets/data_clean_IOT.csv", "configs/iot.json"),
        ("metro", "datasets/source.csv",         "configs/metro.json"),
        ("pump",  "datasets/data__pump_clean.csv", "configs/pump.json"),
    ]
    for prefix, data, cfgp in tasks:
        try:
            split_dataset(data, cfgp, a.out, prefix)
        except Exception as e:
            print(f"FAILED {prefix}: {e}")