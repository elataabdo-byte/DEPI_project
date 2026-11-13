"""
train_lstm.py

Train per-dataset LSTM Autoencoders for time-series anomaly detection.

For each task in the config:
  - Load pre-built sequences (train, val, test) from disk
  - Train an LSTM Autoencoder on the train split
  - Compute per-sequence reconstruction errors on the val split
  - Set the anomaly threshold to a percentile of val errors
  - Evaluate on the test split using the provided ground truth
  - Save model weights, threshold, report, and per-sequence arrays
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sequences(data_dir, prefix, split):
    x_path = os.path.join(data_dir, f"{prefix}_seq_{split}.npy")
    gt_path = os.path.join(data_dir, f"{prefix}_seq_{split}_gt.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"missing {x_path}")
    X = np.load(x_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.int32) if os.path.exists(gt_path) else None
    return X, gt


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        recon = model(xb)
        loss = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def eval_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            recon = model(xb)
            total += criterion(recon, xb).item()
            count += 1
    return total / max(count, 1)


def compute_errors(model, X, batch_size, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon = model(xb)
            per_seq = torch.mean((recon - xb) ** 2, dim=(1, 2))
            chunks.append(per_seq.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def compute_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def train_task(task, device):
    prefix = task["prefix"]
    data_dir = task["data_dir"]
    hidden_size = int(task.get("hidden_size", 64))
    latent_dim = int(task.get("latent_dim", 16))
    num_layers = int(task.get("num_layers", 1))
    epochs = int(task.get("epochs", 50))
    batch_size = int(task.get("batch_size", 32))
    lr = float(task.get("learning_rate", 1e-3))
    patience = int(task.get("patience", 10))
    pct = float(task.get("threshold_percentile", 99.0))

    print(f"\n=== {prefix} ===")
    print(f"  dir={data_dir}")
    print(f"  config: hidden={hidden_size} latent={latent_dim} layers={num_layers} "
          f"epochs={epochs} batch={batch_size} lr={lr} patience={patience} pct={pct}")

    X_train, _ = load_sequences(data_dir, prefix, "train")
    X_val, _ = load_sequences(data_dir, prefix, "val")
    X_test, gt_test = load_sequences(data_dir, prefix, "test")
    n_features = X_train.shape[2]
    print(f"  shapes: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(X_val)),
        batch_size=batch_size, shuffle=False,
    )

    model = LSTMAutoencoder(n_features, hidden_size, latent_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_loss(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(f"  epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  [{elapsed:.1f}s]")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping (best_epoch={best_epoch})")
                break

    model.load_state_dict(best_state)

    val_errors = compute_errors(model, X_val, batch_size, device)
    threshold = float(np.percentile(val_errors, pct))
    val_fpr = float((val_errors > threshold).mean())
    print(f"  val errors: mean={val_errors.mean():.6f} std={val_errors.std():.6f}")
    print(f"  threshold p{pct}={threshold:.6f}  val_fpr={val_fpr:.4f}")

    test_errors = compute_errors(model, X_test, batch_size, device)
    test_preds = (test_errors > threshold).astype(int)

    report = {
        "prefix": prefix,
        "config": {
            "hidden_size": hidden_size,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "patience": patience,
            "threshold_percentile": pct,
        },
        "data": {
            "n_features": int(n_features),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
        },
        "threshold": {
            "value": threshold,
            "percentile": pct,
            "val_fpr": val_fpr,
        },
        "val_errors": {
            "mean": float(val_errors.mean()),
            "std": float(val_errors.std()),
            "min": float(val_errors.min()),
            "max": float(val_errors.max()),
        },
        "test_errors": {
            "mean": float(test_errors.mean()),
            "std": float(test_errors.std()),
            "min": float(test_errors.min()),
            "max": float(test_errors.max()),
        },
    }

    if gt_test is not None and gt_test.sum() > 0:
        metrics = compute_metrics(gt_test, test_preds)
        report["metrics"] = metrics
        print(f"  test: tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} tn={metrics['tn']}")
        print(f"  precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")
    else:
        report["metrics"] = None
        print(f"  test: no ground truth (suspected {int(test_preds.sum())}/{len(test_preds)})")

    model_path = os.path.join(data_dir, f"{prefix}_model.pt")
    torch.save(model.state_dict(), model_path)

    threshold_path = os.path.join(data_dir, f"{prefix}_threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(report["threshold"], f, indent=2)

    np.save(os.path.join(data_dir, f"{prefix}_val_errors.npy"), val_errors)
    np.save(os.path.join(data_dir, f"{prefix}_test_errors.npy"), test_errors)
    np.save(os.path.join(data_dir, f"{prefix}_test_preds.npy"), test_preds)

    report_path = os.path.join(data_dir, f"{prefix}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  saved: {model_path}")
    print(f"  saved: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    print(f"device={device}  seed={seed}")

    for task in config["tasks"]:
        train_task(task, device)


if __name__ == "__main__":
    main()