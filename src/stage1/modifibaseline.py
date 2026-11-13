import os
import json
import pickle
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet

# =========================================================
# CONFIG
 
CSV_PATH = "data_clean IOT.csv"
OUT_DIR  = "baseline_artifacts"
FIG_DIR  = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CONTINUOUS_SENSORS = ["pH", "TDS", "DHT_temp", "DHT_humidity", "water_temp"]
ALL_SENSORS        = CONTINUOUS_SENSORS + ["water_level"]

PHYSICAL = {
    "pH":           (0, 14),
    "TDS":          (0, 5000),
    "DHT_temp":     (-20, 60),
    "DHT_humidity": (0, 100),
    "water_temp":   (-5, 50),
    "water_level":  (0, 3),
}

MAHAL_THRESHOLD  = 0.995
IF_CONTAMINATION = 0.01

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

# =========================================================
# STEP 1 : LOAD + PHYSICAL FILTER
 
print("=" * 70)
print("STEP 1: LOAD + PHYSICAL FILTER")
 

df = pd.read_csv(CSV_PATH, low_memory=False, on_bad_lines="warn")
df.columns = [c.strip() for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

with open(CSV_PATH, "rb") as f:
    source_hash = hashlib.sha256(f.read()).hexdigest()

in_physical = pd.Series(True, index=df.index)
for col, (lo, hi) in PHYSICAL.items():
    in_physical &= df[col].between(lo, hi)

reference = df.loc[in_physical].copy().reset_index(drop=True)
excluded  = df.loc[~in_physical].copy().reset_index(drop=True)

print(f"Total rows            : {len(df):,}")
print(f"Reference rows        : {len(reference):,}")
print(f"Excluded (impossible) : {len(excluded):,}")

cols_keep = ["id", "timestamp", "segment", "window_id"] + ALL_SENSORS + \
            ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]
excluded[cols_keep].to_csv(f"{OUT_DIR}/excluded_physical.csv", index=False)
reference[["id", "timestamp"]].to_csv(f"{OUT_DIR}/reference_ids.csv", index=False)

# =========================================================
# STEP 2 : MAHALANOBIS
 
print("STEP 2: MAHALANOBIS (multivariate, robust covariance)")
 
X       = reference[CONTINUOUS_SENSORS].astype(float).values
mcd     = MinCovDet(random_state=42).fit(X)
mu      = mcd.location_
cov     = mcd.covariance_
cov_inv = np.linalg.inv(cov)
d2_ref  = np.einsum("ij,jk,ik->i", X - mu, cov_inv, X - mu)

threshold_mahal = float(np.quantile(d2_ref, MAHAL_THRESHOLD))
mahal_anomaly   = d2_ref > threshold_mahal

print(f"Condition number : {np.linalg.cond(cov):.2e}")
print(f"Threshold (p{MAHAL_THRESHOLD*100:.1f}) : {threshold_mahal:.2f}")
print(f"Flag rate        : {mahal_anomaly.mean()*100:.3f}%")

with open(f"{OUT_DIR}/multivariate.json", "w") as f:
    json.dump({
        "sensors":         CONTINUOUS_SENSORS,
        "mean_vector":     mu.tolist(),
        "covariance":      cov.tolist(),
        "threshold_p995":  threshold_mahal,
        "condition_number":float(np.linalg.cond(cov)),
    }, f, indent=2)

# =========================================================
# STEP 3 :  ISOLATION FOREST
 
 
print("STEP 3: ISOLATION FOREST (levels + first differences)")
 

X_if = reference[CONTINUOUS_SENSORS].astype(float).copy()
for col in CONTINUOUS_SENSORS:
    X_if[f"{col}_diff"] = reference[col].astype(float).diff()
X_if = X_if.replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler  = RobustScaler(quantile_range=(5, 95)).fit(X_if.values)
X_if_s  = scaler.transform(X_if.values)

iso = IsolationForest(
    n_estimators   = 300,
    contamination  = IF_CONTAMINATION,
    random_state   = 42,
    n_jobs         = -1,
).fit(X_if_s)

if_score       = -iso.score_samples(X_if_s)
threshold_if   = float(np.quantile(if_score, 1 - IF_CONTAMINATION))
if_anomaly     = if_score > threshold_if

print(f"Threshold (p{(1-IF_CONTAMINATION)*100:.0f}) : {threshold_if:.4f}")
print(f"Flag rate                                : {if_anomaly.mean()*100:.3f}%")

with open(f"{OUT_DIR}/isolation_forest.pkl", "wb") as f:
    pickle.dump({
        "model":         iso,
        "scaler":        scaler,
        "features":      list(X_if.columns),
        "sensors":       CONTINUOUS_SENSORS,
        "contamination": IF_CONTAMINATION,
        "threshold":     threshold_if,
    }, f)

# =========================================================
# STEP 4 :  AGREEMENT
print("STEP 4: CROSS-METHOD AGREEMENT")
 

reference["mahal_d2"]                = d2_ref
reference["mahal_anomaly"]           = mahal_anomaly.astype(int)
reference["if_score"]                = if_score
reference["if_anomaly"]              = if_anomaly.astype(int)
reference["n_methods_flagging"]      = reference["mahal_anomaly"] + reference["if_anomaly"]
reference["any_anomaly"]             = (reference["n_methods_flagging"] > 0).astype(int)
reference["high_confidence_anomaly"] = (reference["n_methods_flagging"] == 2).astype(int)

print("Agreement distribution (rows per #methods flagging):")
print(reference["n_methods_flagging"].value_counts().sort_index().to_string())
print(f"\nFlagged by at least one method : {reference['any_anomaly'].sum():,} "
      f"({reference['any_anomaly'].mean()*100:.2f}%)")
print(f"Flagged by both methods        : {reference['high_confidence_anomaly'].sum():,} "
      f"({reference['high_confidence_anomaly'].mean()*100:.2f}%)")

reference.to_csv(f"{OUT_DIR}/reference_with_flags.csv", index=False)

# =========================================================
# STEP 5 :  FIGURES
 
 
print("STEP 5: GENERATING FIGURES")
 

# ---- 01. Distributions ----------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, ALL_SENSORS):
    sns.histplot(reference[col], bins=60, kde=True, ax=ax, color="steelblue")
    ax.set_title(f"{col}   (n = {len(reference):,})")
    ax.set_xlabel(col); ax.set_ylabel("count")
axes.flat[-1].axis("off")
plt.suptitle("Reference period — sensor distributions", fontsize=14, y=1.00)
plt.savefig(f"{FIG_DIR}/01_distributions.png")
plt.close()

# ---- 02. Correlation heatmap ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
corr = reference[CONTINUOUS_SENSORS].corr(method="spearman")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Sensor correlation (Spearman) — reference period")
plt.savefig(f"{FIG_DIR}/02_correlation.png")
plt.close()

# ---- 03. Pairplot with anomaly coloring -----------------
idx_sample = np.random.RandomState(42).choice(
    len(reference), size=min(3000, len(reference)), replace=False
)
sample = reference.iloc[idx_sample].copy()
sample["label"] = sample["any_anomaly"].map({0: "normal", 1: "anomaly"})
g = sns.pairplot(
    sample[CONTINUOUS_SENSORS + ["label"]],
    hue="label",
    palette={"normal": "lightgray", "anomaly": "crimson"},
    plot_kws={"s": 8, "alpha": 0.6},
    diag_kind="hist",
    height=1.8,
)
g.fig.suptitle("Pairwise sensor relationships — flagged vs normal", y=1.01)
g.savefig(f"{FIG_DIR}/03_pairplot.png")
plt.close()

# ---- 04. Mahalanobis distribution -----------------------
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(d2_ref, bins=100, ax=ax, color="steelblue")
ax.axvline(threshold_mahal, color="crimson", linestyle="--",
           label=f"threshold (p{MAHAL_THRESHOLD*100:.1f}) = {threshold_mahal:.1f}")
ax.set_yscale("log")
ax.set_xlabel("Mahalanobis d²"); ax.set_ylabel("count (log scale)")
ax.set_title(f"Mahalanobis distance — {mahal_anomaly.sum():,} flagged above threshold")
ax.legend()
plt.savefig(f"{FIG_DIR}/04_mahalanobis_distribution.png")
plt.close()

# ---- 05. Isolation Forest score distribution ------------
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(if_score, bins=100, ax=ax, color="darkgreen")
ax.axvline(threshold_if, color="crimson", linestyle="--",
           label=f"threshold (p{(1-IF_CONTAMINATION)*100:.0f}) = {threshold_if:.3f}")
ax.set_yscale("log")
ax.set_xlabel("Isolation Forest anomaly score"); ax.set_ylabel("count (log scale)")
ax.set_title(f"Isolation Forest scores — {if_anomaly.sum():,} flagged above threshold")
ax.legend()
plt.savefig(f"{FIG_DIR}/05_iforest_distribution.png")
plt.close()

# ---- 06. Scatter: pairs of sensors, colored by method ---
pairs = [("pH", "TDS"), ("DHT_temp", "DHT_humidity"), ("pH", "water_temp")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (xcol, ycol) in zip(axes, pairs):
    ax.scatter(reference[xcol], reference[ycol],
               c="lightgray", s=4, alpha=0.4, label="normal")
    m_m = reference["mahal_anomaly"] == 1
    ax.scatter(reference.loc[m_m, xcol], reference.loc[m_m, ycol],
               c="crimson", s=20, alpha=0.8, edgecolors="none",
               label="mahalanobis")
    m_i = reference["if_anomaly"] == 1
    ax.scatter(reference.loc[m_i, xcol], reference.loc[m_i, ycol],
               c="orange", s=20, alpha=0.8, marker="x",
               label="iforest")
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title(f"{xcol} vs {ycol}")
    ax.legend(loc="best", fontsize=8)
plt.suptitle("Anomaly locations in sensor space", fontsize=14, y=1.02)
plt.savefig(f"{FIG_DIR}/06_scatter_anomalies.png")
plt.close()

# ---- 07. Time series with high-confidence anomalies -----
fig, axes = plt.subplots(len(CONTINUOUS_SENSORS), 1, figsize=(15, 12), sharex=True)
m_high = reference["high_confidence_anomaly"] == 1
for ax, col in zip(axes, CONTINUOUS_SENSORS):
    ax.plot(reference["timestamp"], reference[col],
            color="steelblue", lw=0.4, alpha=0.7)
    ax.scatter(reference.loc[m_high, "timestamp"], reference.loc[m_high, col],
               c="crimson", s=18, alpha=0.9, zorder=5,
               label="high-confidence anomaly (both methods)")
    ax.set_ylabel(col)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("timestamp")
plt.suptitle("Time series — high-confidence anomalies", fontsize=14)
plt.savefig(f"{FIG_DIR}/07_timeseries_anomalies.png")
plt.close()

# ---- 08. Agreement bar chart ----------------------------
fig, ax = plt.subplots(figsize=(7, 5))
counts = reference["n_methods_flagging"].value_counts().sort_index()
colors = ["lightgray", "orange", "crimson"][:len(counts)]
ax.bar(counts.index.astype(str), counts.values, color=colors)
ax.set_xlabel("Number of methods flagging the row")
ax.set_ylabel("Row count")
ax.set_title("Cross-method agreement")
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
plt.savefig(f"{FIG_DIR}/08_agreement.png")
plt.close()

# ---- 09. Excluded rows visualization --------------------
excl_cols = ["TDS", "DHT_temp", "DHT_humidity", "water_temp"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, excl_cols):
    if len(excluded) > 0 and col in excluded.columns:
        ax.scatter(range(len(excluded)), excluded[col],
                   c="crimson", s=30, alpha=0.8, edgecolors="black")
        lo, hi = PHYSICAL[col]
        ax.axhline(lo, color="gray", linestyle=":", label=f"range [{lo}, {hi}]")
        ax.axhline(hi, color="gray", linestyle=":")
        ax.set_title(f"Excluded {col}   ({len(excluded)} rows)")
        ax.set_ylabel(col); ax.set_xlabel("excluded row index")
        ax.legend(fontsize=8)
plt.suptitle("Physically-impossible values excluded from reference", fontsize=14, y=1.02)
plt.savefig(f"{FIG_DIR}/09_excluded_rows.png")
plt.close()

print(f"\nFigures saved in {FIG_DIR}/")
for fname in sorted(os.listdir(FIG_DIR)):
    print(f"  - {fname}")

# =========================================================
# STEP 6 : METADATA
 
metadata = {
    "version":            datetime.now().strftime("v1_%Y%m%d_%H%M%S"),
    "source_file":        CSV_PATH,
    "source_hash":        source_hash,
    "generated":          datetime.now().isoformat(),
    "total_rows":         int(len(df)),
    "reference_rows":     int(len(reference)),
    "excluded_rows":      int(len(excluded)),
    "physical_ranges":    PHYSICAL,
    "continuous_sensors": CONTINUOUS_SENSORS,
    "mahalanobis": {
        "threshold_percentile": MAHAL_THRESHOLD,
        "threshold_value":      threshold_mahal,
        "flagged_rows":         int(mahal_anomaly.sum()),
        "flag_rate":            float(mahal_anomaly.mean()),
        "condition_number":     float(np.linalg.cond(cov)),
    },
    "isolation_forest": {
        "contamination":    IF_CONTAMINATION,
        "n_estimators":     300,
        "threshold_value":  threshold_if,
        "flagged_rows":     int(if_anomaly.sum()),
        "flag_rate":        float(if_anomaly.mean()),
    },
    "agreement": {
        "flagged_by_any":  int(reference["any_anomaly"].sum()),
        "flagged_by_both": int(reference["high_confidence_anomaly"].sum()),
    },
}
with open(f"{OUT_DIR}/reference_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

 
print(f"All artifacts saved in: {OUT_DIR}/")