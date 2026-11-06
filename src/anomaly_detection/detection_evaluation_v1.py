from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


CSV =r"C:\Users\compu import\Desktop\TEST\source  without ground for mira.csv"
FAILURE_LOG =r"C:\Users\compu import\Desktop\TEST\failure_events_clean (2).csv"
MODEL_FILE = "model.joblib"
HEALTHY_ID = 3784
FAILURE_ID = 27177          # first window of the official failure of 29-30 May 2020 (segment 181)
HOUR = 72                 # 72 windows x 5 min = 6 hours before and 6 hours after (12 hours in total)
WIN = 30                  # rows per window (5 min)
OFFSETS = [0, 10, 20]     # window grid shifted by this many rows
RATIO_MIN = 0.5           # window is anomalous if >= 50% of its rows are above the row threshold
Z_LIMIT = 3.0             # classic 3-sigma limit for the z-score baseline
SEED = 42
FORBIDDEN = {"timestamp", "ground_truth", "segment", "window_id"}

df = pd.read_csv(CSV, parse_dates=["timestamp"])
assert "ground_truth" not in df.columns, "This script is meant for the file WITHOUT ground_truth"

#Official failure log -> row flag
log = pd.read_csv(FAILURE_LOG, parse_dates=["start_time", "end_time"])
log = log.rename(columns={"start_time": "window_start", "end_time": "window_end"})
in_failure = np.zeros(len(df), dtype=bool)
for r in log.itertuples():
    in_failure |= ((df["timestamp"] >= r.window_start) & (df["timestamp"] <= r.window_end)).to_numpy()
df["in_failure"] = in_failure

#Model in all sensors
train = df[df["window_id"].between(HEALTHY_ID - HOUR, HEALTHY_ID + HOUR)]
assert not train["in_failure"].any(), "Training period overlaps an official failure!"

if Path(MODEL_FILE).exists():
    saved = joblib.load(MODEL_FILE)
    model, SENSORS, row_threshold = saved["model"], saved["sensors"], saved["row_threshold"]
    print(f"Loaded {MODEL_FILE}")
else:
    SENSORS = [c for c in df.columns if c not in FORBIDDEN | {"in_failure"}
               and pd.api.types.is_numeric_dtype(df[c])]
    model = IsolationForest(n_estimators=200, random_state=SEED).fit(train[SENSORS])
    all_scores = -model.score_samples(df[SENSORS])
    q1, q3 = np.percentile(all_scores, [25, 75])
    row_threshold = q3 + 1.5 * (q3 - q1)
    print(f"{MODEL_FILE} not found -> trained the model again (same settings as 2.2)")
assert FORBIDDEN.isdisjoint(model.feature_names_in_), "Forbidden column in model input!"
joblib.dump({"model": model, "sensors": SENSORS, "row_threshold": row_threshold}, MODEL_FILE)   
print(f"{len(SENSORS)} sensors | Isolation Forest row threshold = {row_threshold:.4f}")

# z-score baseline: healthy mean/std from the training rows; row score = max |z| over the sensors
mu, sd = train[SENSORS].mean(), train[SENSORS].std()
used = [s for s in SENSORS if sd[s] > 0]                # a sensor that never moves in training has no z-score
dropped = sorted(set(SENSORS) - set(used))
print(f"z-score baseline: {len(used)} sensors used" + (f" (no variation in training, skipped: {dropped})" if dropped else ""))

#The two prototype periods
periods = []
for name, wid in [("healthy", HEALTHY_ID), ("failure", FAILURE_ID)]:
    p = df[df["window_id"].between(wid - HOUR, wid + HOUR)].copy()
    p["period"] = name
    periods.append(p)
proto = pd.concat(periods)
proto["if_score"] = -model.score_samples(proto[SENSORS])
proto["z_score"] = ((proto[used] - mu[used]) / sd[used]).abs().max(axis=1)
METHODS = {"isolation_forest": ("if_score", row_threshold), "zscore_3sigma": ("z_score", Z_LIMIT)}

#Evaluation at every offset
def metrics(y, p):
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=round(pr, 3), recall=round(rc, 3),
                f1=round(f1, 3), false_alarms=fp)

def windows(period_rows, off):
    """Cut one period into full 30-row windows, grid shifted by `off` rows."""
    key = (np.arange(len(period_rows)) - off) // WIN
    t = period_rows.assign(key=key)
    t = t[t.key >= 0]
    size = t.groupby("key").size()
    return t[t.key.isin(size[size == WIN].index)]

rows_eval, rows_lead, status0 = [], [], []
for off in OFFSETS:
    w_all = []
    for pname, prows in proto.groupby("period", sort=False):
        w = windows(prows, off)
        g = w.groupby("key")
        tbl = pd.DataFrame({"period": pname, "start": g.timestamp.min(),
                            "label": g.in_failure.any().astype(int)})
        for m, (col, thr) in METHODS.items():
            tbl[m] = (g[col].apply(lambda x: (x > thr).mean()) >= RATIO_MIN).astype(int)
            tbl[m + "_ratio"] = g[col].apply(lambda x: (x > thr).mean()).round(3)
        w_all.append(tbl)
    tbl = pd.concat(w_all).reset_index(drop=True)
    y = tbl["label"]

    for m in METHODS:
        pred = tbl[m]
        rows_eval.append({"method": m, "offset_rows": off, "windows": len(tbl),
                          "failure_windows": int(y.sum()), **metrics(y, pred)})
        # lead time: windows of the failure period, before the first failure window
        fp_ = tbl[tbl.period == "failure"].reset_index(drop=True)
        first = int(fp_.index[fp_.label == 1][0])
        before = fp_[:first][m].to_numpy()
        lead = 0
        while lead < len(before) and before[len(before) - 1 - lead] == 1:
            lead += 1
        rows_lead.append({"method": m, "offset_rows": off,
                          "failure_windows": int(fp_.label.sum()),
                          "failure_windows_detected": int(fp_.loc[fp_.label == 1, m].sum()),
                          "windows_before_failure": len(before),
                          "lead_windows_in_a_row": lead,
                          "flagged_in_last_hour": int(before[-12:].sum())})
    if off == 0:
        status0 = tbl

ev, lead = pd.DataFrame(rows_eval), pd.DataFrame(rows_lead)
print("\n=== Detection evaluation (prototype windows) ===")
print(ev.to_string(index=False))
print("\n=== Lead time before the failure ===")
print(lead.to_string(index=False))

ev.to_csv("eval_detection.csv", index=False)
lead.to_csv("eval_lead_time.csv", index=False)
status0.to_csv("window_status_prototype.csv", index=False)
print("\nSaved: eval_detection.csv, eval_lead_time.csv, window_status_prototype.csv")