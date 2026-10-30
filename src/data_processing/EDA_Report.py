"""
MetroPT-3 Exploratory Data Analysis (EDA)
Converted from EDA_Report.ipynb and enhanced with publication-ready visualizations.

Usage:
  python EDA_Report.py
Optional environment variables:
  DATA_PATH  path to the input CSV
  OUT_DIR    directory for CSV reports and visualization PNGs
"""
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
except ImportError:
    sns = None
    plt.style.use("seaborn-v0_8-whitegrid")

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)

# Default matches the Sentinel project folder structure; override via DATA_PATH.
_default_data = Path(r"C:\Users\compu import\Desktop\TEST\MetroPT3(AirCompressor).csv")
_candidates = [
    _default_data,
    Path(r"C:\Users\compu import\Desktop\TEST\MetroPT3(AirCompressor).csv"),
    Path(r" C:\Users\compu import\Desktop\TEST\MetroPT3(AirCompressor).csv"),
    Path("data_raw") / "data_raw.csv",
]
_env_data = os.environ.get("DATA_PATH")
DATA_PATH = Path(_env_data) if _env_data else next((p for p in _candidates if p.exists()), _default_data)
OUT_DIR = Path(os.environ.get("OUT_DIR", str(DATA_PATH.parent if DATA_PATH.parent != Path(".") else Path("eda_output"))))
PLOT_DIR = OUT_DIR / "eda_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent chart style and high-resolution output.
FIG_DPI = 180
FIGSIZE_WIDE = (13, 5.5)
COLOR_MAIN = "#2878B5"
COLOR_ACCENT = "#E07A2D"
COLOR_BAD = "#C74440"
COLOR_GOOD = "#2A9D8F"
COLOR_NEUTRAL = "#7A8793"

def save_plot(filename):
    """Save the current figure and display it."""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()

def show_table(obj, max_rows=40):
    """Print a readable table in a normal Python terminal."""
    if isinstance(obj, pd.Series):
        print(obj.to_string())
    elif isinstance(obj, pd.DataFrame):
        print(obj.head(max_rows).to_string())
        if len(obj) > max_rows:
            print(f"... ({len(obj) - max_rows} additional rows; full results are saved where applicable)")
    else:
        print(obj)

def display(obj):
    # Keeps notebook-originated display(...) calls usable in a standalone .py script.
    show_table(obj)



# ========================================================================
# MetroPT-3 — Exploratory Data Analysis (EDA)
# ========================================================================


# ========================================================================
# 1) Load the data
# ========================================================================


# --- Notebook code cell 3 ---
df = pd.read_csv(DATA_PATH, low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
print("Invalid timestamps:", df["timestamp"].isna().sum())

# "Unnamed: 0" is an extra index column (0, 10, 20, ...). Dropped in memory only; data_raw.csv is untouched.
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.sort_values("timestamp").reset_index(drop=True)
print(df.shape)
df.head()


# ========================================================================
# Sensor definitions (from the dataset PDF)
# ========================================================================


# --- Notebook code cell 5 ---
ANALOG = {
    "TP2": "bar",              # pressure on the compressor
    "TP3": "bar",              # pressure at the pneumatic panel
    "H1": "bar",               # pressure drop when the cyclonic separator filter discharges
    "DV_pressure": "bar",      # pressure drop when towers discharge; ~0 = compressor under load
    "Reservoirs": "bar",       # downstream reservoir pressure (close to TP3)
    "Oil_temperature": "°C",
    "Motor_current": "A",      # ~0 off, ~4 offloaded, ~7 under load, ~9 starting
}
DIGITAL = {
    "COMP": "0/1",             # air intake valve (active = off or offloaded)
    "DV_eletric": "0/1",       # outlet valve control (active = under load)
    "Towers": "0/1",           # which drying tower is in use
    "MPG": "0/1",              # start under load when pressure < 8.2 bar
    "LPS": "0/1",              # active when pressure < 7 bar
    "Pressure_switch": "0/1",  # discharge in the drying towers
    "Oil_level": "0/1",        # active when oil is BELOW expected level
    "Caudal_impulses": "0/1",  # pulses of air flow from APU to reservoirs
}
A, D = list(ANALOG), list(DIGITAL)

absent = [c for c in A + D if c not in df.columns]
print("Missing expected columns:", absent if absent else "none")


# ========================================================================
# 2) Overview
# ========================================================================


# --- Notebook code cell 7 ---
print("Rows x columns:", df.shape)
print("Period:", df["timestamp"].min(), "->", df["timestamp"].max())
print("Duration:", df["timestamp"].max() - df["timestamp"].min())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
df.dtypes


# ========================================================================
# 3) Missing values
# ========================================================================


# --- Notebook code cell 9 ---
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(4),
})
missing


# ========================================================================
# 4) Time gaps
# ========================================================================


# --- Notebook code cell 11 ---
dt = df["timestamp"].diff()
typical = dt.median()
print("Typical sampling interval:", typical)
print("\nMost common steps:")
print(dt.value_counts().head(5))

gap_mask = dt > 3 * typical
gaps = pd.DataFrame({
    "gap_start": df["timestamp"].shift(1)[gap_mask],
    "gap_end": df["timestamp"][gap_mask],
    "gap_length": dt[gap_mask],
}).sort_values("gap_length", ascending=False)

n_missing_readings = int(((gaps["gap_length"] / typical) - 1).sum()) if len(gaps) else 0
print("\nNumber of gaps:", len(gaps))
print("Total time inside gaps:", gaps["gap_length"].sum())
print("Approx. readings missing because of gaps:", n_missing_readings)
gaps.head(15)


# ========================================================================
# 4.1) Frozen-logger periods (screening)
# ========================================================================


# --- Notebook code cell 13 ---
MIN_FROZEN_DURATION = pd.Timedelta(minutes=5)

same_as_previous = df[A].eq(df[A].shift(1)).all(axis=1)
same_as_previous.iloc[0] = False
run_id = same_as_previous.ne(same_as_previous.shift(fill_value=False)).cumsum()
candidate_rows = []
for _, run in df.loc[same_as_previous].groupby(run_id[same_as_previous]):
    first_idx = run.index[0]
    last_idx = run.index[-1]
    start_idx = max(df.index.min(), first_idx - 1)
    start_time = df.loc[start_idx, 'timestamp']
    end_time = df.loc[last_idx, 'timestamp']
    duration = end_time - start_time
    if duration >= MIN_FROZEN_DURATION:
        candidate_rows.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'duration_seconds': duration.total_seconds(),
            'repeated_steps': int(len(run)),
        })

frozen_periods = pd.DataFrame(candidate_rows, columns=[
    'start_time', 'end_time', 'duration', 'duration_seconds', 'repeated_steps'
])

df['frozen_logger_flag'] = False
for _, period in frozen_periods.iterrows():
    df.loc[df['timestamp'].between(period['start_time'], period['end_time']), 'frozen_logger_flag'] = True

print('Minimum duration:', MIN_FROZEN_DURATION)
print('Candidate frozen-logger periods:', len(frozen_periods))
print('Flagged rows:', int(df['frozen_logger_flag'].sum()))
display(frozen_periods.head(20))
frozen_periods.to_csv(OUT_DIR / 'frozen_logger_periods.csv', index=False)



# --- Notebook code cell 14 ---
# Daily data coverage
per_day = df.set_index("timestamp").resample("D").size()
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
ax.plot(per_day.index, per_day.values, color=COLOR_MAIN, linewidth=1.7)
ax.fill_between(per_day.index, per_day.values, alpha=0.15, color=COLOR_MAIN)
ax.set(title="Daily Data Coverage", xlabel="Date", ylabel="Readings per day")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
save_plot("01_daily_data_coverage.png")



# ========================================================================
# 5) Analog sensors
# ========================================================================


# --- Notebook code cell 16 ---
desc = df[A].describe(percentiles=[.01, .25, .5, .75, .99]).T
desc["skew"] = df[A].skew()
desc["unit"] = pd.Series(ANALOG)
desc


# ========================================================================
# 5.2 Distributions (plots use a 200k-row random sample for speed; all statistics use the full data)
# ========================================================================


# --- Notebook code cell 18 ---
# Distribution plots use a random sample for speed; statistics use the full dataset.
sample = df[A].sample(min(200_000, len(df)), random_state=0)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, c in zip(axes.ravel(), A):
    vals = sample[c].dropna()
    if sns is not None:
        sns.histplot(vals, bins=55, kde=True, ax=ax, color=COLOR_MAIN, edgecolor=None)
    else:
        ax.hist(vals, bins=55, color=COLOR_MAIN, alpha=0.85)
    ax.set_title(f"{c} ({ANALOG[c]})", fontweight="bold")
    ax.set_xlabel(ANALOG[c])
    ax.set_ylabel("Frequency")
for ax in axes.ravel()[len(A):]:
    ax.axis("off")
fig.suptitle("Analog Sensor Distributions", fontsize=16, fontweight="bold", y=1.02)
save_plot("02_analog_sensor_distributions.png")



# --- Notebook code cell 19 ---
# Boxplots help compare spread and extreme values; points are hidden for readability.
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, c in zip(axes.ravel(), A):
    vals = sample[c].dropna()
    ax.boxplot(vals, vert=True, patch_artist=True, showfliers=False,
               boxprops={"facecolor": "#B8D8EA", "edgecolor": COLOR_MAIN},
               medianprops={"color": COLOR_BAD, "linewidth": 2},
               whiskerprops={"color": COLOR_MAIN}, capprops={"color": COLOR_MAIN})
    ax.set_title(f"{c} ({ANALOG[c]})", fontweight="bold")
    ax.set_ylabel(ANALOG[c])
for ax in axes.ravel()[len(A):]:
    ax.axis("off")
fig.suptitle("Analog Sensor Spread (outlier points hidden for clarity)", fontsize=15, fontweight="bold", y=1.02)
save_plot("03_analog_sensor_boxplots.png")



# ========================================================================
# 5.3 Outliers
# ========================================================================


# --- Notebook code cell 21 ---
q1, q3 = df[A].quantile(.25), df[A].quantile(.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out_count = ((df[A] < lower) | (df[A] > upper)).sum()

PHYS = {  # assumed plausible bounds (min, max)
    "TP2": (-0.1, 12), "TP3": (-0.1, 12), "H1": (-0.1, 12),
    "DV_pressure": (-0.1, 12), "Reservoirs": (-0.1, 12),
    "Oil_temperature": (0, 120), "Motor_current": (0, 15),
}
phys_count = pd.Series({
    c: int(((df[c] < PHYS[c][0]) | (df[c] > PHYS[c][1])).sum()) for c in A
})

outliers = pd.DataFrame({
    "iqr_lower": lower, "iqr_upper": upper,
    "outliers_iqr": out_count,
    "outliers_iqr_pct": (out_count / len(df) * 100).round(3),
    "outside_physical_range": phys_count,
    "outside_physical_pct": (phys_count / len(df) * 100).round(3),
})
outliers


# ========================================================================
# 5.4 Sensor ranges against physical expectations
# ========================================================================


# --- Notebook code cell 23 ---
REVIEW_PCT = 1.0  # more than this % outside the operating band -> "review"

EXPECTED = {  # sensor: (operating_min, operating_max, what we expect physically)
    "TP2":             (-0.05, 11.0, "~0 bar when off/offloaded; rises to line pressure under load"),
    "TP3":             (7.0, 10.5,   "panel pressure; LPS activates below 7 bar"),
    "H1":              (-0.05, 11.0, "near line pressure; drops toward 0 at separator discharge"),
    "DV_pressure":     (-0.05, 10.5, "~0 under load; spikes when the towers discharge"),
    "Reservoirs":      (7.0, 10.5,   "follows TP3"),
    "Oil_temperature": (40.0, 85.0,  "warmed-up compressor oil"),
    "Motor_current":   (0.0, 10.0,   "~0 off, ~4 offloaded, ~7 loaded, ~9 starting"),
}

rows = []
for c in A:
    s = df[c]
    hard_min, hard_max = PHYS[c]
    op_min, op_max, note = EXPECTED[c]
    below_op, above_op = int((s < op_min).sum()), int((s > op_max).sum())
    pct_op = round((below_op + above_op) / len(s) * 100, 3)
    hard = int(((s < hard_min) | (s > hard_max)).sum())
    if hard:
        verdict = "violates physical limits"
    elif pct_op > REVIEW_PCT:
        verdict = "review: often outside operating band"
    elif pct_op > 0:
        verdict = "minor excursions"
    else:
        verdict = "within expectations"
    rows.append({
        "sensor": c, "unit": ANALOG[c],
        "obs_min": s.min(), "obs_p1": desc.loc[c, "1%"],
        "obs_p99": desc.loc[c, "99%"], "obs_max": s.max(),
        "expected_min": op_min, "expected_max": op_max,
        "below_band": below_op, "above_band": above_op,
        "pct_outside_band": pct_op, "hard_violations": hard,
        "verdict": verdict, "expectation": note,
    })
ranges = pd.DataFrame(rows).set_index("sensor")

# When do the excursions happen? (top days per sensor)
for c in ranges.index[ranges["pct_outside_band"] > 0]:
    lo, hi = EXPECTED[c][:2]
    m = (df[c] < lo) | (df[c] > hi)
    print(f"{c}: {int(m.sum())} rows outside [{lo}, {hi}]; most affected days:")
    print(df.loc[m, "timestamp"].dt.date.value_counts().head(5).to_string(), "\n")

# Cross-checks between related sensors
gap = (df["TP3"] - df["Reservoirs"]).abs()
print(f"|TP3 - Reservoirs|: median {gap.median():.3f} bar, 99th pct {gap.quantile(.99):.3f} bar, max {gap.max():.3f} bar")
print(f"TP3 < 7 bar: {(df['TP3'] < 7).mean() * 100:.3f}% of rows | LPS active: {df['LPS'].mean() * 100:.3f}%")
ranges


# ========================================================================
# 6) Digital sensors
# ========================================================================


# --- Notebook code cell 25 ---
rows = []
for c in D:
    s = df[c]
    nun = s.nunique(dropna=True)
    rows.append({
        "sensor": c,
        "n_unique": nun,
        "values": sorted(s.dropna().unique().tolist()) if nun <= 5 else "many",
        "pct_active (value=1)": round(s.mean() * 100, 3),
        "state_changes": int((s.diff().fillna(0) != 0).sum()),
    })
digital_summary = pd.DataFrame(rows).set_index("sensor")
digital_summary


# ========================================================================
# 6.1 Signal semantics: which COMP state means the compressor is producing air
# ========================================================================


# --- Notebook code cell 27 ---
PRODUCING_AIR_VALUE = 0   # COMP is active (1) when off/offloaded, so 0 = intake open = producing air
MOTOR_OFF_A = 0.5         # below this the motor is considered stopped (~0 A when off)

df["_motor_stopped"] = df["Motor_current"] < MOTOR_OFF_A
by_comp = df.groupby("COMP").agg(
    rows=("COMP", "size"),
    motor_current_mean=("Motor_current", "mean"),
    tp2_mean=("TP2", "mean"),
    dv_eletric_active=("DV_eletric", "mean"),
    motor_stopped_pct=("_motor_stopped", "mean"),
)
by_comp["share_pct"] = (by_comp["rows"] / len(df) * 100).round(2)
by_comp["motor_stopped_pct"] = (by_comp["motor_stopped_pct"] * 100).round(2)
df = df.drop(columns="_motor_stopped")
display(by_comp.round(3))

# The state with the higher compressor pressure (TP2) is the one where air is produced.
air_state = by_comp["tp2_mean"].idxmax()
if air_state == PRODUCING_AIR_VALUE:
    print(f"Confirmed: COMP == {PRODUCING_AIR_VALUE} is the producing-air state (higher TP2 and motor current than COMP == 1).")
else:
    print(f"WARNING: data suggests COMP == {air_state} is the producing-air state. Check PRODUCING_AIR_VALUE.")

print(f"COMP and DV_eletric differ in {(df['COMP'] != df['DV_eletric']).mean() * 100:.2f}% of rows (mirror images)")
print(f"COMP and MPG are equal in {(df['COMP'] == df['MPG']).mean() * 100:.2f}% of rows")

df["producing_air"] = df["COMP"].eq(PRODUCING_AIR_VALUE)
print(f"Share of time producing air: {df['producing_air'].mean() * 100:.2f}%")


# ========================================================================
# 7) Failures vs sensor readings
# ========================================================================


# --- Notebook code cell 29 ---
failures = pd.DataFrame({
    "failure_id": [1, 2, 3, 4],
    "start_time": pd.to_datetime(["2020-04-18 00:00", "2020-05-29 23:30", "2020-06-05 10:00", "2020-07-15 14:30"]),
    "end_time":   pd.to_datetime(["2020-04-18 23:59", "2020-05-30 06:00", "2020-06-07 14:30", "2020-07-15 19:00"]),
    "failure_type": "Air leak",
})

df["in_failure"] = False
for _, r in failures.iterrows():
    df.loc[df["timestamp"].between(r["start_time"], r["end_time"]), "in_failure"] = True

print("Rows inside failure windows:", int(df["in_failure"].sum()),
      f"({df['in_failure'].mean() * 100:.2f}%)")

cmp_ = df.groupby("in_failure")[A + D].mean().T
cmp_.columns = ["normal_mean", "failure_mean"]
cmp_["change_pct"] = ((cmp_["failure_mean"] - cmp_["normal_mean"]) / cmp_["normal_mean"].replace(0, np.nan) * 100).round(2)
cmp_


# --- Notebook code cell 30 ---
# Hourly sensor trends with failure windows shaded.
hourly = df.set_index("timestamp")[A].resample("1h").mean()
fig, axes = plt.subplots(len(A), 1, figsize=(15, 2.25 * len(A)), sharex=True)
if len(A) == 1:
    axes = [axes]
for ax, c in zip(axes, A):
    ax.plot(hourly.index, hourly[c], lw=0.9, color=COLOR_MAIN)
    for _, r in failures.iterrows():
        ax.axvspan(r["start_time"], r["end_time"], color=COLOR_BAD, alpha=0.16, linewidth=0)
    ax.set_ylabel(f"{c}\\n({ANALOG[c]})")
    ax.grid(True, alpha=0.25)
axes[0].set_title("Hourly Analog Sensor Trends — Failure Windows Shaded", fontsize=14, fontweight="bold")
axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
save_plot("06_failure_window_sensor_trends.png")



# ========================================================================
# 8) Sensor summary and approved sensors
# ========================================================================


# --- Notebook code cell 32 ---
MAX_MISSING_PCT = 5.0
MAX_PHYSICAL_PCT = 5.0

rows = []
for c in A + D:
    s = df[c]
    is_analog = c in ANALOG
    row = {
        "sensor": c,
        "type": "analog" if is_analog else "digital",
        "unit": ANALOG.get(c, DIGITAL.get(c)),
        "count": int(s.count()),
        "missing_pct": round(s.isna().mean() * 100, 4),
        "min": s.min(), "max": s.max(),
        "mean": s.mean(), "std": s.std(), "median": s.median(),
        "n_unique": int(s.nunique()),
    }
    if is_analog:
        row.update({
            "q1": q1[c], "q3": q3[c],
            "outliers_iqr": int(out_count[c]),
            "outliers_iqr_pct": float(outliers.loc[c, "outliers_iqr_pct"]),
            "outside_physical_pct": float(outliers.loc[c, "outside_physical_pct"]),
            "pct_active": np.nan,
        })
    else:
        row.update({
            "q1": np.nan, "q3": np.nan,
            "outliers_iqr": np.nan, "outliers_iqr_pct": np.nan,
            "outside_physical_pct": np.nan,
            "pct_active": round(s.mean() * 100, 3),
        })
    rows.append(row)

summary = pd.DataFrame(rows)


def decide(r):
    reasons = []
    if r["missing_pct"] > MAX_MISSING_PCT:
        reasons.append(f"missing > {MAX_MISSING_PCT}%")
    if r["n_unique"] <= 1:
        reasons.append("constant value")
    if pd.notna(r["outside_physical_pct"]) and r["outside_physical_pct"] > MAX_PHYSICAL_PCT:
        reasons.append(f"outside physical range > {MAX_PHYSICAL_PCT}%")
    return pd.Series({
        "status": "approved" if not reasons else "rejected",
        "reason": "; ".join(reasons),
    })


summary = pd.concat([summary, summary.apply(decide, axis=1)], axis=1)
summary.to_csv(OUT_DIR / "sensor_summary.csv", index=False)
summary


# --- Notebook code cell 33 ---
approved = summary.loc[summary["status"] == "approved", ["sensor", "type", "unit"]]
approved.to_csv(OUT_DIR / "approved_sensors.csv", index=False)

print(f"Approved: {len(approved)} of {len(summary)} sensors")
print(approved["sensor"].tolist())
rejected = summary.loc[summary["status"] == "rejected", ["sensor", "reason"]]
print("\nRejected:")
print(rejected if len(rejected) else "none")


# ========================================================================
# Notes and findings (fill in after running)
# ========================================================================


# ---------------------------------------------------------------------------
# Additional diagnostic visualizations and machine-readable reports
# ---------------------------------------------------------------------------
# Missing values: count per column (only columns with missing values are shown).
missing_nonzero = missing.loc[missing["missing_count"] > 0].sort_values("missing_pct", ascending=True)
if len(missing_nonzero):
    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(missing_nonzero))))
    ax.barh(missing_nonzero.index, missing_nonzero["missing_pct"], color=COLOR_ACCENT)
    ax.set(title="Missing Values by Column", xlabel="Missing values (%)", ylabel="Column")
    save_plot("04_missing_values.png")
else:
    print("Missing-value plot skipped: no missing values detected.")

# Correlation matrix for analog sensors.
corr = df[A].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 7))
if sns is not None:
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
else:
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(A)), A, rotation=45, ha="right")
    ax.set_yticks(range(len(A)), A)
    fig.colorbar(im, ax=ax)
ax.set_title("Analog Sensor Correlation Matrix", fontsize=14, fontweight="bold")
save_plot("05_analog_sensor_correlations.png")

# Digital sensor activity rates.
digital_active = (df[D].mean() * 100).sort_values()
fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.barh(digital_active.index, digital_active.values, color=COLOR_MAIN)
ax.set_xlim(0, 100)
ax.set(title="Digital Sensor Active Rate (value = 1)", xlabel="Rows with value 1 (%)", ylabel="Digital sensor")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
save_plot("07_digital_sensor_active_rates.png")

# Sensor operating-band review, where the calculated table is available.
if "ranges" in globals() and not ranges.empty:
    range_plot = ranges["pct_outside_band"].sort_values()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(range_plot.index, range_plot.values, color=[
        COLOR_BAD if v > REVIEW_PCT else COLOR_ACCENT if v > 0 else COLOR_GOOD
        for v in range_plot.values
    ])
    ax.axvline(REVIEW_PCT, color=COLOR_BAD, linestyle="--", linewidth=1.5,
               label=f"Review threshold ({REVIEW_PCT}%)")
    ax.set(title="Analog Readings Outside Expected Operating Bands",
           xlabel="Outside operating band (%)", ylabel="Sensor")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.legend()
    save_plot("08_operating_band_review.png")
    ranges.to_csv(OUT_DIR / "sensor_operating_ranges.csv")

# Save other tables and event data to make the script useful outside a notebook.
missing.to_csv(OUT_DIR / "missing_values_report.csv")
gaps.to_csv(OUT_DIR / "timestamp_gaps.csv", index=False)
digital_summary.to_csv(OUT_DIR / "digital_sensor_summary.csv")
cmp_.to_csv(OUT_DIR / "failure_vs_normal_sensor_comparison.csv")
summary.to_csv(OUT_DIR / "sensor_summary.csv", index=False)
approved.to_csv(OUT_DIR / "approved_sensors.csv", index=False)

print("\\nEDA completed successfully.")
print(f"Input file: {DATA_PATH}")
print(f"Reports folder: {OUT_DIR.resolve()}")
print(f"Visualization folder: {PLOT_DIR.resolve()}")
print("Charts generated: daily coverage, analog distributions, analog boxplots, missing values (if any),")
print("                  analog correlations, failure-window trends, digital activity, operating-band review.")
