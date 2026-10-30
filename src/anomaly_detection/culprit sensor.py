import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------- CONFIG ----------------------
CSV       = r"C:\Users\compu import\Desktop\TEST\source.csv"
BASE_CSV  = r"C:\Users\compu import\Desktop\TEST\outputs\metrics\sensor_baseline_by_state.csv"
THR_CSV   = r"C:\Users\compu import\Desktop\TEST\outputs\metrics\sensor_thresholds_by_state.csv"
ORS_CSV   = r"C:\Users\compu import\Desktop\TEST\outputs\metrics\outlier_summary_by_state.csv"

HEALTHY_ID   = 3784
FAILURE_ID   = 18558
HOUR         = 12
RATIO_MIN    = 0.5
SEED         = 42
FORBIDDEN    = {"timestamp", "ground_truth", "segment", "window_id"}

N_MIN         = 15      # minimum rows per state for valid stats (30 rows/window)
BASELINE_CAP  = 25.0    # baseline outlier% above which we mark unstable
DIGITAL_DELTA = 20.0    # pp change threshold for digital signals

# ---------------------- LOAD ----------------------
df   = pd.read_csv(CSV, parse_dates=["timestamp"])
base = pd.read_csv(BASE_CSV)
thr  = pd.read_csv(THR_CSV)
ors  = pd.read_csv(ORS_CSV)

# ---------------------- STATE ----------------------
if "state" not in df.columns:
    # A "trip" = COMP=0 but the machine is physically still in loaded conditions
    # (e.g. compressor shut off by a safety action during a failure).
    # Detect it from analogue signals instead of trusting COMP alone.
    loaded_sig  = (df["TP2"] > 5) | (df["Motor_current"] > 4.75)
    trip_loaded = (df["COMP"] == 0) & loaded_sig

    df["state"] = np.select(
        [
            trip_loaded,                                     # COMP=0 but loaded → treat as loaded
            (df["COMP"] == 1) & (df["DV_eletric"] == 0),    # normal loaded
            df["COMP"] == 0,                                 # real off (idle, values near zero)
        ],
        ["loaded", "loaded", "off"],
        default="unloaded",
    )
    print("→ state derived (with trip detection). Counts:\n",
          df["state"].value_counts())
# ---------------------- SLICES ----------------------
def around(wid):
    return df[df["window_id"].between(wid - HOUR, wid + HOUR)].copy()

healthy_p   = around(HEALTHY_ID)
failure_p   = around(FAILURE_ID)
failure_win = df[df["window_id"] == FAILURE_ID].copy()   # <== target only

print("=" * 80)
print(f"Healthy ctx: {healthy_p.timestamp.min()} -> {healthy_p.timestamp.max()} | rows={len(healthy_p)}")
print(f"Failure ctx: {failure_p.timestamp.min()} -> {failure_p.timestamp.max()} | rows={len(failure_p)}")
print(f"Target win {FAILURE_ID}: rows={len(failure_win)} "
      f"| states={failure_win['state'].value_counts().to_dict()}")
print("=" * 80)

# ============== PART 1: IsolationForest (keep your detection) ==============
SENSORS = [c for c in df.columns
           if c not in FORBIDDEN and pd.api.types.is_numeric_dtype(df[c])]

train = healthy_p[healthy_p["ground_truth"] == 0]
model = IsolationForest(n_estimators=200, random_state=SEED).fit(train[SENSORS])

def score(d):
    return -model.score_samples(d[SENSORS])

all_scores = score(df)
q1, q3 = np.percentile(all_scores, [25, 75])
threshold = q3 + 1.5 * (q3 - q1)

def window_table(d):
    d = d.copy()
    d["score"] = score(d)
    g = d.groupby("window_id")
    t = pd.DataFrame({
        "truth_abnormal_rows": g.ground_truth.sum(),
        "anomaly_ratio":       g.score.apply(lambda x: (x > threshold).mean()),
        "mean_score":          g.score.mean(),
    })
    t["status"] = np.where(t.anomaly_ratio >= RATIO_MIN, "anomalous", "normal")
    return t.round(3)

print("\n=== IsolationForest: window status in failure context ===")
ft = window_table(failure_p)
print(ft.to_string())

# ============== PART 2: Culprit analysis on target window 18558 ==============
analogue = (base[base["sensor_type"] == "analogue"]
              .loc[lambda d: d["sensor"] != "window_id", "sensor"]
              .unique().tolist())
digital  = (base[base["sensor_type"] == "digital"]
              .loc[lambda d: d["sensor"] != "window_id", "sensor"]
              .unique().tolist())

# -------- analogues (IQR + MAD, warning + critical) --------
rows = []
for state in failure_win["state"].unique():
    sdf = failure_win[failure_win["state"] == state]
    n = len(sdf)
    if n < N_MIN:
        print(f"\n[skip] state={state} n={n} < {N_MIN}")
        continue

    st  = thr[thr["state"] == state]
    sor = ors[ors["state"] == state]

    for sensor in analogue:
        st_s = st[st["sensor"] == sensor]
        if st_s.empty:
            continue
        for method in ("mad", "iqr"):
            for level in ("warning", "critical"):
                lo_c, hi_c = f"{method}_{level}_low", f"{method}_{level}_high"
                if lo_c not in st_s.columns:
                    continue
                lo = st_s[lo_c].values[0]
                hi = st_s[hi_c].values[0]
                vals = sdf[sensor].values
                outlier = int(((vals < lo) | (vals > hi)).sum())
                fault_pc = round(outlier / n * 100, 2)

                b = sor[(sor["sensor"] == sensor)
                        & (sor["method"] == method)
                        & (sor["level"] == level)]
                base_pc = float(b["normal_out_pct"].values[0]) if not b.empty else np.nan

                rows.append(dict(state=state, sensor=sensor, n_rows=n,
                                 method=method, level=level,
                                 fault_pct=fault_pc, baseline_pct=base_pc))

ano = pd.DataFrame(rows)
if not ano.empty:
    ano["abs_lift"] = (ano["fault_pct"] - ano["baseline_pct"]).round(2)
    ano["rel_lift"] = (ano["fault_pct"] / ano["baseline_pct"].clip(lower=0.1)).round(1)
    ano["specific"] = (ano["baseline_pct"].notna()
                        & (ano["baseline_pct"] <= BASELINE_CAP)
                        & (ano["fault_pct"] > ano["baseline_pct"]))

    ano_top = (ano[ano["specific"]]
                 .sort_values(["state", "level", "rel_lift"],
                              ascending=[True, True, False])
                 .groupby(["state", "level"]).head(5)
                 .reset_index(drop=True))

    print("\n" + "=" * 80)
    print(f"ANALOGUE CULPRITS in window {FAILURE_ID}")
    print("=" * 80)
    for (state, level), g in ano_top.groupby(["state", "level"]):
        print(f"\n--- state={state} | level={level} | n={g['n_rows'].iloc[0]} ---")
        print(g[["sensor", "method", "fault_pct", "baseline_pct",
                 "abs_lift", "rel_lift"]].to_string(index=False))

# -------- digital (on_fraction delta vs same-state baseline) --------
healthy_only = df[df["ground_truth"] == 0]
dig_rows = []
for state in failure_win["state"].unique():
    sdf = failure_win[failure_win["state"] == state]
    hdf = healthy_only[healthy_only["state"] == state]
    n = len(sdf)
    if n < N_MIN or len(hdf) == 0:
        continue
    for sensor in digital:
        f_on = float(sdf[sensor].mean())
        h_on = float(hdf[sensor].mean())
        delta_pp = round((f_on - h_on) * 100, 2)
        dig_rows.append(dict(state=state, sensor=sensor, n_rows=n,
                             fault_on_pct=round(f_on * 100, 2),
                             baseline_on_pct=round(h_on * 100, 2),
                             delta_pp=delta_pp,
                             flag="CHANGED" if abs(delta_pp) > DIGITAL_DELTA else "-"))

dig = pd.DataFrame(dig_rows)
if not dig.empty:
    dig = dig.sort_values("delta_pp", key=abs, ascending=False)
    print("\n" + "=" * 80)
    print(f"DIGITAL SIGNAL CHANGES in window {FAILURE_ID}")
    print("=" * 80)
    print(dig.to_string(index=False))

# ---------------------- SAVE ----------------------
if not ano.empty:
    ano_top.to_csv(f"culprit_analogue_win_{FAILURE_ID}.csv", index=False)
if not dig.empty:
    dig.to_csv(f"culprit_digital_win_{FAILURE_ID}.csv", index=False)
print(f"\nSaved: culprit_analogue_win_{FAILURE_ID}.csv  (and digital, if any)")

print("\n=== Raw values in window 18558 vs baselines ===")
for s in ["TP2", "DV_pressure", "Motor_current", "H1"]:
    win_vals = failure_win[s]
    off_row  = base[(base.state=="off")    & (base.sensor==s)]
    ld_row   = base[(base.state=="loaded") & (base.sensor==s)]
    print(f"\n{s}:")
    print(f"  window 18558:  min={win_vals.min():.4f}  median={win_vals.median():.4f}  max={win_vals.max():.4f}")
    if not off_row.empty:
        r = off_row.iloc[0]
        print(f"  off baseline:  median={r['median']:.4f}  Q1={r['Q1']:.4f}  Q3={r['Q3']:.4f}")
    if not ld_row.empty:
        r = ld_row.iloc[0]
        print(f"  loaded baseline: median={r['median']:.4f}  Q1={r['Q1']:.4f}  Q3={r['Q3']:.4f}")