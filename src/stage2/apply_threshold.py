"""
apply_threshold.py

Recompute test predictions using a new threshold percentile.
Does not retrain or modify the model.
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    for task in config["tasks"]:
        prefix = task["prefix"]
        data_dir = task["data_dir"]
        pct = float(task["percentile"])

        val_errors = np.load(os.path.join(data_dir, f"{prefix}_val_errors.npy"))
        test_errors = np.load(os.path.join(data_dir, f"{prefix}_test_errors.npy"))
        gt = np.load(os.path.join(data_dir, f"{prefix}_seq_test_gt.npy"))

        threshold = float(np.percentile(val_errors, pct))
        preds = (test_errors > threshold).astype(np.int32)

        np.save(os.path.join(data_dir, f"{prefix}_test_preds.npy"), preds)
        with open(os.path.join(data_dir, f"{prefix}_threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"value": threshold, "percentile": pct}, f, indent=2)

        tp = int(((gt == 1) & (preds == 1)).sum())
        fp = int(((gt == 0) & (preds == 1)).sum())
        fn = int(((gt == 1) & (preds == 0)).sum())
        tn = int(((gt == 0) & (preds == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n=== {prefix} ===")
        print(f"  percentile = {pct}")
        print(f"  threshold  = {threshold:.6f}")
        print(f"  tp={tp}  fp={fp}  fn={fn}  tn={tn}")
        print(f"  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")


if __name__ == "__main__":
    main()