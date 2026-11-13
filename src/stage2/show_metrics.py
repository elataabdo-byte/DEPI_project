"""
show_metrics.py

Load saved predictions and ground truth for each dataset and print
the full confusion matrix, accuracy, precision, recall, and F1.
"""

import json
import os

import numpy as np


def compute_all(gt, preds):
    tp = int(((gt == 1) & (preds == 1)).sum())
    fp = int(((gt == 0) & (preds == 1)).sum())
    fn = int(((gt == 1) & (preds == 0)).sum())
    tn = int(((gt == 0) & (preds == 0)).sum())
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def show(prefix, data_dir):
    gt_path = os.path.join(data_dir, f"{prefix}_seq_test_gt.npy")
    preds_path = os.path.join(data_dir, f"{prefix}_test_preds.npy")
    errors_path = os.path.join(data_dir, f"{prefix}_test_errors.npy")
    thr_path = os.path.join(data_dir, f"{prefix}_threshold.json")

    if not (os.path.exists(gt_path) and os.path.exists(preds_path)):
        print(f"\n{prefix}: missing files, skip")
        return

    gt = np.load(gt_path)
    preds = np.load(preds_path)
    errors = np.load(errors_path)

    threshold = None
    if os.path.exists(thr_path):
        with open(thr_path, encoding="utf-8") as f:
            threshold = json.load(f).get("value")

    print(f"\n{'=' * 60}")
    print(f"  {prefix.upper()}")
    print(f"{'=' * 60}")

    if threshold is not None:
        print(f"  threshold     = {threshold:.6f}")
    print(f"  test size     = {len(gt)}")
    print(f"  gt positives  = {int(gt.sum())}  ({100 * gt.sum() / len(gt):.2f}%)")
    print(f"  predictions   = {int(preds.sum())}  ({100 * preds.sum() / len(preds):.2f}%)")
    print(f"  errors mean   = {errors.mean():.6f}")
    print(f"  errors median = {np.median(errors):.6f}")

    if gt.sum() == 0:
        print("\n  No ground truth positives. Cannot compute accuracy.")
        return

    m = compute_all(gt, preds)

    print("\n  --- Confusion matrix ---")
    print(f"                  Pred Normal   Pred Anomaly")
    print(f"  Actual Normal      TN={m['tn']:>6}      FP={m['fp']:>6}")
    print(f"  Actual Anomaly     FN={m['fn']:>6}      TP={m['tp']:>6}")

    print("\n  --- Metrics ---")
    print(f"  Accuracy   = (TP + TN) / Total")
    print(f"             = ({m['tp']} + {m['tn']}) / {m['total']}")
    print(f"             = {m['accuracy']:.4f}  ({100 * m['accuracy']:.2f}%)")
    print()
    print(f"  Precision  = TP / (TP + FP)  = {m['tp']} / {m['tp'] + m['fp']}  = {m['precision']:.4f}")
    print(f"  Recall     = TP / (TP + FN)  = {m['tp']} / {m['tp'] + m['fn']}  = {m['recall']:.4f}")
    print(f"  F1         = 2*P*R / (P + R) = {m['f1']:.4f}")

    print("\n  --- Interpretation ---")
    if m["accuracy"] >= 0.95:
        print(f"  Accuracy {100 * m['accuracy']:.1f}% looks high, but remember:")
        neg = m["tn"] + m["fp"]
        pos = m["tp"] + m["fn"]
        print(f"  Dataset is imbalanced: {neg} normal vs {pos} anomalous")
        print(f"  A model that says 'all normal' would get {100 * neg / m['total']:.1f}% accuracy and 0% recall.")
        print(f"  Use F1 ({m['f1']:.4f}) for balanced evaluation.")


def main():
    for prefix in ["iot", "pump", "metro"]:
        data_dir = f"D:/processed_outputs/{prefix}"
        if not os.path.exists(data_dir):
            print(f"{prefix}: dir not found, skip")
            continue
        show(prefix, data_dir)


if __name__ == "__main__":
    main()