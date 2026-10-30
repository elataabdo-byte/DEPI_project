import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


CSV = "source.csv"
HEALTHY_ID = 3784
FAILURE_ID = 18558       # confirmed failure (official log: 18 Apr 2020)
HOUR = 12                # 12 windows x 5 min = 1 hour before and 1 hour after
RATIO_MIN = 0.5          # window is anomalous if >= 50% of its 30 rows are above the threshold
SEED = 42
FORBIDDEN = {"timestamp", "ground_truth", "segment", "window_id"}

df = pd.read_csv(CSV, parse_dates=["timestamp"])
SENSORS = [c for c in df.columns
           if c not in FORBIDDEN and pd.api.types.is_numeric_dtype(df[c])]
print("Sensors:", SENSORS, f"({len(SENSORS)})")

def around(wid):
    """Window +- 1 hour (12 windows before and after)."""
    return df[df["window_id"].between(wid - HOUR, wid + HOUR)].copy()

healthy_p, failure_p = around(HEALTHY_ID), around(FAILURE_ID)

#Check the two windows
assert df.loc[df.window_id == HEALTHY_ID, "ground_truth"].sum() == 0, "Healthy window has failure rows!"
assert df.loc[df.window_id == FAILURE_ID, "ground_truth"].sum() > 0, "Failure window has NO failure rows!"

# -1) Train on the healthy period only
train = healthy_p[healthy_p["ground_truth"] == 0]
model = IsolationForest(n_estimators=200, random_state=SEED).fit(train[SENSORS])
assert FORBIDDEN.isdisjoint(model.feature_names_in_), "Forbidden column in model input!"
print(f"Training rows: {len(train)} | abnormal rows in training: {int(train.ground_truth.sum())}")

# -2) Threshold from the scores of ALL data (IQR)
def score(d):
    return -model.score_samples(d[SENSORS])      # higher = more anomalous

all_scores = score(df)
q1, q3 = np.percentile(all_scores, [25, 75])
threshold = q3 + 1.5 * (q3 - q1)
print(f"Q1={q1:.4f}  Q3={q3:.4f}  threshold = Q3 + 1.5*IQR = {threshold:.4f}  (on {len(all_scores):,} rows)")

# -3) Window status for both periods
healthy_p["score"], failure_p["score"] = score(healthy_p), score(failure_p)

def window_table(d):
    g = d.groupby("window_id")
    t = pd.DataFrame({"truth_abnormal_rows": g.ground_truth.sum(),
                      "anomaly_ratio": g.score.apply(lambda x: (x > threshold).mean()),
                      "mean_score": g.score.mean()})
    t["status"] = np.where(t.anomaly_ratio >= RATIO_MIN, "anomalous", "normal")
    t["truth"] = np.where(t.truth_abnormal_rows > 0, "failure", "healthy")
    return t.round(3)

for name, wid, d in [("HEALTHY", HEALTHY_ID, healthy_p), ("FAILURE", FAILURE_ID, failure_p)]:
    t = window_table(d)
    c = t.loc[wid]
    print(f"\n=== {name} period (window {wid}) | {d.timestamp.min()} -> {d.timestamp.max()} ===")
    print(f"Target window {wid}: truth={c.truth} | anomaly_ratio={c.anomaly_ratio} | status={c.status}")
    print(f"Windows flagged anomalous: {(t.status == 'anomalous').sum()} of {len(t)}")
    print(t.to_string())


joblib.dump({"model": model, "sensors": SENSORS, "row_threshold": threshold}, "model.joblib")
pd.concat([window_table(healthy_p).assign(period="healthy"),
           window_table(failure_p).assign(period="failure")]).to_csv("window_status_v1.csv")
print("Saved: model.joblib, window_status_v1.csv")