"""
make_sequences.py

Builds fixed-length sequences from split X/meta CSV files for LSTM-Autoencoder training.

Inputs per task (from config):
  {data_dir}/{prefix}_features.json       -> list of feature column names
  {data_dir}/{prefix}_X_{split}.csv       -> normalized features per row
  {data_dir}/{prefix}_meta_{split}.csv    -> meta columns (segment, ground_truth, window_id)

Outputs per task (to out_dir):
  {prefix}_seq_{split}.npy                -> [N, seq_len, n_features] float32
  {prefix}_seq_{split}_gt.npy             -> [N] int32  (1 if any row in seq has gt=1)
  {prefix}_seq_{split}_wid.npy            -> [N] int64  (window_id at seq start, -1 if unavailable)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def load_json(path):
    """Read a JSON file and return its parsed contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sequences(
    x_df,
    meta_df,
    feature_cols,
    seq_len,
    stride,
    seg_col="segment",
    gt_col="ground_truth",
    ignore_segment=False,
):
    """
    Build sliding-window sequences that never cross segment boundaries.

    Parameters
    ----------
    x_df : DataFrame
        Normalized feature values, one row per timestamp.
    meta_df : DataFrame
        Metadata aligned row-by-row with x_df. May contain segment,
        ground_truth, and window_id columns.
    feature_cols : list of str
        Feature column names, in the order expected by the model.
    seq_len : int
        Number of consecutive rows per sequence.
    stride : int
        Step size between consecutive sequences.
    seg_col : str
        Column identifying a segment. Sequences do not cross segments
        when this column exists.
    gt_col : str
        Ground-truth column. A sequence is labeled 1 if any row inside
        it has a positive value.
    ignore_segment : bool
        If True, treat the whole file as a single segment.

    Returns
    -------
    X : ndarray, shape (N, seq_len, n_features), float32
    gt : ndarray, shape (N,), int32
    wid : ndarray, shape (N,), int64
    skipped_segments : int
        Number of segments shorter than seq_len that were skipped.
    """
    df = meta_df.copy()
    for col in feature_cols:
        df[col] = x_df[col].values

    if ignore_segment or seg_col not in df.columns:
        groups = [(None, df)]
    else:
        groups = df.groupby(seg_col, sort=False)

    x_list, gt_list, wid_list = [], [], []
    skipped_segments = 0

    for _, group in groups:
        group = group.reset_index(drop=True)
        features = group[feature_cols].values.astype(np.float32)
        n_rows = len(features)

        if n_rows < seq_len:
            skipped_segments += 1
            continue

        for start in range(0, n_rows - seq_len + 1, stride):
            end = start + seq_len
            x_list.append(features[start:end])

            if gt_col in group.columns:
                gt_value = int(group[gt_col].iloc[start:end].max())
            else:
                gt_value = 0
            gt_list.append(gt_value)

            if "window_id" in group.columns:
                wid_value = int(group["window_id"].iloc[start])
            else:
                wid_value = -1
            wid_list.append(wid_value)

    if not x_list:
        return None, None, None, skipped_segments

    return (
        np.stack(x_list).astype(np.float32),
        np.array(gt_list, dtype=np.int32),
        np.array(wid_list, dtype=np.int64),
        skipped_segments,
    )


def process_task(task):
    """Process one task entry from the config."""
    prefix = task["prefix"]
    data_dir = task["data_dir"]
    out_dir = task.get("out_dir", data_dir)
    seq_len = int(task.get("seq_len", 60))
    stride = int(task.get("stride", seq_len))
    ignore_segment = bool(task.get("ignore_segment", False))

    print(
        f"\n=== {prefix} ===  dir={data_dir}  "
        f"seq_len={seq_len}  stride={stride}  ignore_segment={ignore_segment}"
    )

    features_path = os.path.join(data_dir, f"{prefix}_features.json")
    if not os.path.exists(features_path):
        print(f"  ERROR: missing {features_path}")
        return

    feature_cols = load_json(features_path)
    print(f"  features: {len(feature_cols)}")

    for split in SPLITS:
        x_path = os.path.join(data_dir, f"{prefix}_X_{split}.csv")
        meta_path = os.path.join(data_dir, f"{prefix}_meta_{split}.csv")

        if not (os.path.exists(x_path) and os.path.exists(meta_path)):
            print(f"  SKIP {split}: missing input files")
            continue

        x_df = pd.read_csv(x_path)
        meta_df = pd.read_csv(meta_path)

        if len(x_df) != len(meta_df):
            print(f"  ERROR {split}: X={len(x_df)} meta={len(meta_df)} mismatch")
            continue

        sequences, gt, wid, skipped_segments = build_sequences(
            x_df,
            meta_df,
            feature_cols,
            seq_len,
            stride,
            seg_col="segment",
            gt_col="ground_truth",
            ignore_segment=ignore_segment,
        )

        if sequences is None:
            print(f"  WARNING {split}: no sequences (all segments < {seq_len})")
            continue

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{prefix}_seq_{split}.npy"), sequences)
        np.save(os.path.join(out_dir, f"{prefix}_seq_{split}_gt.npy"), gt)
        np.save(os.path.join(out_dir, f"{prefix}_seq_{split}_wid.npy"), wid)

        anomalies = int(gt.sum())
        total = len(gt)
        ratio = 100.0 * anomalies / max(total, 1)
        print(
            f"  {split:5s}: X={sequences.shape}  "
            f"anomalies={anomalies}/{total} ({ratio:.1f}%)  "
            f"skipped_segments={skipped_segments}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build sequences for LSTM-Autoencoder training from split CSVs."
    )
    parser.add_argument("--config", required=True, help="Path to the sequences config JSON.")
    args = parser.parse_args()

    config = load_json(args.config)
    for task in config["tasks"]:
        process_task(task)


if __name__ == "__main__":
    main()