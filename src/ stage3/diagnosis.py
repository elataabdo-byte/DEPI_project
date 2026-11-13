 

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Model definition (must match trainLSTM.py)
# ============================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size, latent_dim, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        B, T, _ = x.size()
        _, (h, _) = self.encoder(x)
        z = self.enc_fc(h[-1])
        h_dec = self.dec_fc(z)
        dec_in = h_dec.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in)
        return self.out_fc(dec_out)


def load_model(model_path, report, n_features, device):
    cfg = report["config"]
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def per_feature_errors(model, X, batch_size, device):
    """Return [N, n_features] mean squared error per feature per sequence."""
    n = len(X)
    n_features = X.shape[2]
    out = np.zeros((n, n_features), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon = model(xb)
            err = torch.mean((recon - xb) ** 2, dim=1)
            out[i:i + batch_size] = err.cpu().numpy()
    return out


# ============================================================
# Helpers
# ============================================================

def wid_to_first_index(meta):
    """Map window_id -> index of its first row in the meta dataframe."""
    wid_col = meta["window_id"].values
    mapping = {}
    for idx, wid in enumerate(wid_col):
        if wid not in mapping:
            mapping[int(wid)] = idx
    return mapping


def confidence_level(score):
    if score is None:
        return None
    if score > 1.5:
        return "High"
    if score > 0.5:
        return "Medium"
    return "Low"


def status_from(pred, gt):
    if gt is None:
        return None
    if pred == 1 and gt == 1:
        return "TP"
    if pred == 1 and gt == 0:
        return "FP"
    if pred == 0 and gt == 1:
        return "FN"
    return "TN"


# ============================================================
# Core per-dataset routine
# ============================================================

def diagnose_dataset(prefix, data_dir, out_dir, device, batch_size=64):
    print(f"\n{'=' * 72}")
    print(f"  {prefix.upper()}")
    print(f"{'=' * 72}")
    t0 = time.time()

    # --- Load metadata
    with open(os.path.join(data_dir, f"{prefix}_report.json"), encoding="utf-8") as f:
        report = json.load(f)
    with open(os.path.join(data_dir, f"{prefix}_threshold.json"), encoding="utf-8") as f:
        threshold = float(json.load(f)["value"])
    with open(os.path.join(data_dir, f"{prefix}_features.json"), encoding="utf-8") as f:
        features = json.load(f)

    n_features = report["data"]["n_features"]
    print(f"  n_features     : {n_features}")
    print(f"  threshold      : {threshold:.6f}")

    # --- Load arrays
    errors = np.load(os.path.join(data_dir, f"{prefix}_test_errors.npy"))
    preds = np.load(os.path.join(data_dir, f"{prefix}_test_preds.npy"))
    X = np.load(os.path.join(data_dir, f"{prefix}_seq_test.npy"))
    gt_path = os.path.join(data_dir, f"{prefix}_seq_test_gt.npy")
    gt_raw = np.load(gt_path) if os.path.exists(gt_path) else None
    wids = np.load(os.path.join(data_dir, f"{prefix}_seq_test_wid.npy"))
    meta = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_meta_test.csv"),
        usecols=["timestamp", "window_id", "segment"],
        parse_dates=["timestamp"],
    )

    has_gt = gt_raw is not None and int(gt_raw.sum()) > 0
    n_seq = len(errors)
    seq_len = X.shape[1]
    n_anom = int(preds.sum())

    print(f"  sequences      : {n_seq}")
    print(f"  seq_len        : {seq_len}")
    print(f"  anomalous      : {n_anom}")
    print(f"  has ground truth: {has_gt}")

    # --- Timestamps per sequence
    wid_to_idx = wid_to_first_index(meta)
    last_row = len(meta) - 1
    ts_start = np.empty(n_seq, dtype=object)
    ts_end = np.empty(n_seq, dtype=object)
    segments = np.empty(n_seq, dtype=object)

    for i in range(n_seq):
        start_idx = wid_to_idx.get(int(wids[i]))
        if start_idx is None:
            ts_start[i] = ts_end[i] = segments[i] = None
            continue
        end_idx = min(start_idx + seq_len - 1, last_row)
        ts_start[i] = meta.iloc[start_idx]["timestamp"]
        ts_end[i] = meta.iloc[end_idx]["timestamp"]
        segments[i] = meta.iloc[start_idx]["segment"]

    # --- Confidence score (normalized excess over threshold)
    conf_score = (errors - threshold) / threshold
    conf_level = np.array(
        [confidence_level(float(s)) if p == 1 else None
         for s, p in zip(conf_score, preds)],
        dtype=object,
    )

    # --- Culprits + z-scores (anomalous only)
    culprit_data = {}
    if n_anom > 0:
        print(f"  computing per-feature errors for {n_anom} anomalous sequences...")
        model = load_model(
            os.path.join(data_dir, f"{prefix}_model.pt"),
            report, n_features, device,
        )
        anom_idx = np.where(preds == 1)[0]
        X_anom = X[anom_idx]
        pfe = per_feature_errors(model, X_anom, batch_size, device)

        for local_i, global_i in enumerate(anom_idx):
            feat_err = pfe[local_i]
            z = X[global_i]
            z_max = np.max(np.abs(z), axis=0)
            z_mean = np.mean(z, axis=0)
            top = np.argsort(feat_err)[-3:][::-1]
            culprit_data[int(global_i)] = [
                {
                    "sensor": features[fi] if fi < len(features) else f"f{fi}",
                    "error": float(feat_err[fi]),
                    "zscore_max": float(z_max[fi]),
                    "direction": "above" if z_mean[fi] >= 0 else "below",
                }
                for fi in top
            ]

    # --- Assemble rows
    rows = []
    for i in range(n_seq):
        p = int(preds[i])
        g = int(gt_raw[i]) if has_gt else None
        row = {
            "seq_id": i,
            "window_id_start": int(wids[i]),
            "timestamp_start": ts_start[i],
            "timestamp_end": ts_end[i],
            "segment": segments[i],
            "reconstruction_error": round(float(errors[i]), 6),
            "threshold": round(threshold, 6),
            "confidence_score": round(float(conf_score[i]), 4),
            "confidence_level": conf_level[i],
            "prediction": p,
            "gt": g,
            "status": status_from(p, g),
        }
        culprits = culprit_data.get(i, [])
        for slot in range(3):
            key = f"culprit_{slot + 1}"
            if slot < len(culprits):
                c = culprits[slot]
                row[f"{key}_sensor"] = c["sensor"]
                row[f"{key}_error"] = round(c["error"], 6)
                row[f"{key}_zscore_max"] = round(c["zscore_max"], 4)
                row[f"{key}_direction"] = c["direction"]
            else:
                row[f"{key}_sensor"] = None
                row[f"{key}_error"] = None
                row[f"{key}_zscore_max"] = None
                row[f"{key}_direction"] = None
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_diagnosis.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"  saved: {out_path}  ({len(df)} rows)")

    # --- Summary
    print(f"  elapsed        : {time.time() - t0:.1f}s")
    if has_gt:
        tp = int(((preds == 1) & (gt_raw == 1)).sum())
        fp = int(((preds == 1) & (gt_raw == 0)).sum())
        fn = int(((preds == 0) & (gt_raw == 1)).sum())
        tn = int(((preds == 0) & (gt_raw == 0)).sum())
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    lv = pd.Series([c for c in conf_level if c is not None])
    if len(lv) > 0:
        print(f"  confidence levels (anomalous only):")
        for name, cnt in lv.value_counts().items():
            print(f"    {name:>8}: {cnt:>6}")

    if culprit_data:
        counter = Counter()
        for lst in culprit_data.values():
            if lst:
                counter[lst[0]["sensor"]] += 1
        print(f"  top culprit sensors (anomalous only):")
        for name, cnt in counter.most_common(5):
            pct = 100.0 * cnt / n_anom
            print(f"    {name:>20}: {cnt:>5}  ({pct:>5.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Per-sequence diagnosis for anomaly detection.")
    parser.add_argument("--data-dir", default="D:/processed_outputs")
    parser.add_argument("--out-dir", default="diagnosis_out")
    parser.add_argument("--prefixes", nargs="+", default=["metro", "pump", "iot"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    for prefix in args.prefixes:
        data_dir = os.path.join(args.data_dir, prefix)
        if not os.path.isdir(data_dir):
            print(f"\nSKIP {prefix}: {data_dir} not found")
            continue
        try:
            diagnose_dataset(prefix, data_dir, args.out_dir, device, args.batch_size)
        except Exception as e:
            print(f"\n{prefix}: FAILED — {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
