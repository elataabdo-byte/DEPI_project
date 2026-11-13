"""
scenario_rules.py  -  Phase 2 / E2: scenario rules + parameters (general, one code for all datasets)

Scenarios (shared names):
    normal | single_noise | repeated_noise | drift | shift | sudden_fault

Input is WINDOW-LEVEL series per sensor: x[t] = window mean, v[t] = window std (within-window).
Every rule needs a HISTORY of windows (never a single window).

Rules (see docs for the maths):
  sudden_fault  : flatline  (window std < flat_eps for >= F windows)     OR
                  out-of-bounds (window mean outside Normal range +- margin for >= F windows)
  shift         : mean of last m windows differs from the median of the L windows before them by > theta*sigma_e,
                  all m windows on the same side, stable (low spread), no slope  -> new constant level
  drift         : linear-regression slope over last N windows > alpha*sigma_e per window, with R^2 >= r2
  repeated_noise: >= rep_count "events" inside the last rep_horizon windows, or variance inflated in >= 3 of last 5
                  windows  (an event = |x - local_median| > k1*sigma_e  or  window std > noise_thr*typical std)
  single_noise  : exactly one event in window t-1 and the series is back to normal in window t (needs 1 window after)
  normal        : none of the above

Priority when several fire:  sudden_fault > shift > drift > repeated_noise > single_noise > normal

All thresholds are calibrated per dataset / per sensor from NORMAL TRAIN windows only:
    threshold = max(guide default, q_{calib_q} of the statistic on Normal windows)
so the defaults are lower bounds, never looser than the data allows.
"""
import json
import warnings
warnings.filterwarnings("ignore")
from collections import deque

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

SCENARIOS = ["normal", "single_noise", "repeated_noise", "drift", "shift", "sudden_fault"]
_CODE = {n: i for i, n in enumerate(SCENARIOS)}

DEFAULT_PARAMS = {
    "L": 12,                 # history windows used as the local reference (median)
    "k1": 3.5,               # spike / event size in sigma_e units         (guide: 3.5)
    "k2": 1.0,               # "back to normal" band in sigma_e units      (guide: 1.0)
    "shift_theta": 2.5,      # level difference in sigma_e units           (guide: 2.5)
    "shift_m": 4,            # consecutive windows to confirm a shift      (guide: >= 4)
    "shift_stable": 1.5,     # max std of the recent m windows (sigma_e units)
    "drift_N": 6,            # windows used for the slope                  (guide: >= 6)
    "drift_alpha": 0.3,      # min |slope| in sigma_e per window (guide 0.05 abs. units -> made relative)
    "drift_r2": 0.7,         # linearity of the trajectory
    "flat_eps": 1e-3,        # window std below this = flat                (guide: 0.001)
    "fault_min_windows": 5,  # sustained windows                           (guide: >= 5)
    "oob_margin": 0.5,       # out-of-bounds margin as a fraction of the Normal range
    "noise_k": 3.0,          # window std inflation factor                 (guide: 3.0)
    "rep_count": 3,          # events needed for repeated noise            (guide: 3 cycles)
    "rep_horizon": 10,
    "calib_q": 0.999,
    "max_fault_run": 200,    # if a sensor is flat normally for longer than this, flatline rule is disabled
}


# --------------------------------------------------------------------------------------------
def _runs(window_id, segment=None):
    """Run ids: a new run starts when window_id is not consecutive or the segment changes."""
    wid = np.asarray(window_id)
    brk = np.r_[True, np.diff(wid) != 1]
    if segment is not None:
        seg = np.asarray(segment)
        brk |= np.r_[True, seg[1:] != seg[:-1]]
    return np.cumsum(brk) - 1


def _run_length(flag):
    """Length of the current consecutive-True run at every position."""
    f = np.asarray(flag, dtype=bool)
    out = np.zeros(len(f), dtype=int)
    c = 0
    for i, b in enumerate(f):
        c = c + 1 if b else 0
        out[i] = c
    return out


def _stats_run(x, v, b, P):
    """Vectorised statistics for ONE contiguous run of windows (one sensor)."""
    n = len(x)
    L, m, N = P["L"], P["shift_m"], P["drift_N"]
    mode = P.get("ref_mode", "local")
    nan = np.full(n, np.nan)
    xs = pd.Series(x)
    med = xs.rolling(L, min_periods=L).median()
    if mode == "baseline":                                      # guide semantics: distance from the Normal baseline
        s_e = b["scale_base"]
        ref1 = np.full(n, b["mu"])
        refm_full = np.full(n, b["mu"])
    else:                                                       # local: distance from the previous L windows
        s_e = b["scale_loc"]
        ref1 = med.shift(1).values                              # median of the L windows before t
        refm_full = med.shift(m).values
    e = (x - ref1) / s_e                                        # single-window deviation (sigma_e units)
    nr = v / b["v_ref"]                                         # variance inflation ratio

    def prev(a, k=1):
        return np.r_[np.full(k, np.nan), a[:-k]] if n > k else np.full(n, np.nan)

    st = {"e": e, "nr": nr}
    # --- single noise: candidate is window t-1, window t must be back to normal
    st["peak_prev"] = np.abs(prev(e))
    st["nr_prev"] = prev(nr)
    st["back_dev"] = np.abs(x - prev(ref1)) / s_e
    st["back_nr"] = nr
    # --- shift
    refm = refm_full
    W = np.vstack([np.full((m - 1, m), np.nan), swv(x, m)]) if n >= m else np.full((n, m), np.nan)
    D = (W - refm[:, None]) / s_e
    md = np.nanmean(D, axis=1) if n else nan
    st["shift_stat"] = np.abs(md)
    st["shift_side_min"] = np.nanmin(D * np.sign(md)[:, None], axis=1)
    st["shift_spread"] = np.nanstd(W, axis=1) / s_e
    tm = np.arange(m) - (m - 1) / 2
    wc = W - np.nanmean(W, axis=1)[:, None]
    st["shift_slope_total"] = np.abs((wc * tm).sum(axis=1) / (tm ** 2).sum()) * (m - 1) / s_e
    # --- drift
    Wd = np.vstack([np.full((N - 1, N), np.nan), swv(x, N)]) if n >= N else np.full((n, N), np.nan)
    td = np.arange(N) - (N - 1) / 2
    wdc = Wd - Wd.mean(axis=1)[:, None]
    slope = (wdc * td).sum(axis=1) / (td ** 2).sum()
    ssr = slope ** 2 * (td ** 2).sum()
    sst = (wdc ** 2).sum(axis=1)
    st["drift_stat"] = np.abs(slope) / s_e
    with np.errstate(divide="ignore", invalid="ignore"):
        st["drift_r2"] = np.where(sst > 0, ssr / sst, 0.0)
    # --- fault
    st["flat_run"] = _run_length(v < P["flat_eps"])
    st["oob_run"] = _run_length((x < b["lo"]) | (x > b["hi"]))
    return pd.DataFrame(st)


def sensor_stats(x, v, run_id, b, P):
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    run_id = np.asarray(run_id)
    parts = []
    for r in np.unique(run_id):
        idx = np.where(run_id == r)[0]
        parts.append(_stats_run(x[idx], v[idx], b, P).set_index(idx))
    return pd.concat(parts).sort_index()


def _events(st, b, P):
    return ((np.abs(st["e"]) > b["k1"]) | (st["nr"] > b["noise_thr"])).fillna(False).values


def classify(st, b, P, run_id):
    """Return scenario code per window (int array) for one sensor."""
    n = len(st)
    code = np.zeros(n, dtype=int)
    rid = np.asarray(run_id)
    ev = _events(st, b, P)                                      # window exceeds k1*sigma or inflated variance
    prev_ev = np.r_[False, ev[:-1]]
    back = ((st["back_dev"] <= P["k2"]).fillna(False).values & (st["back_nr"] <= b["noise_thr"]).fillna(False).values)
    rev = prev_ev & back & ~ev                                  # a transient excursion, confirmed one window later
    rev_cnt = pd.Series(rev.astype(int)).groupby(rid).transform(lambda s: s.rolling(P["rep_horizon"], min_periods=1).sum()).values
    hi_nr = (st["nr"] > b["noise_thr"]).astype(int)
    nr_cnt = pd.Series(hi_nr.values).groupby(rid).transform(lambda s: s.rolling(5, min_periods=1).sum()).values
    repeated = (rev_cnt >= P["rep_count"]) | (nr_cnt >= 3)
    single = rev & ~repeated
    drift = ((st["drift_stat"] > b["drift_alpha"]) & (st["drift_r2"] >= P["drift_r2"])).fillna(False).values
    shift = ((st["shift_stat"] > b["shift_theta"]) & (st["shift_side_min"] > 0.5 * b["shift_theta"])
             & (st["shift_spread"] < P["shift_stable"]) & (st["shift_slope_total"] < 1.0)).fillna(False).values
    fault = (st["flat_run"].values >= b["fault_flat"]) | (st["oob_run"].values >= P["fault_min_windows"])
    # priority: lowest first, later assignments overwrite
    code[single] = _CODE["single_noise"]
    code[repeated] = _CODE["repeated_noise"]
    code[drift] = _CODE["drift"]
    code[shift] = _CODE["shift"]
    code[fault] = _CODE["sudden_fault"]
    return code


# --------------------------------------------------------------------------------------------
def calibrate(train_means, train_stds, train_wid, train_seg=None, val=None, params=None, analog=None):
    """
    Per-sensor baseline from NORMAL TRAIN windows; thresholds from NORMAL VALIDATION windows
    (val = (means, stds, window_id, segment)); if val is None the thresholds come from train.
    """
    P = {**DEFAULT_PARAMS, **(params or {})}
    mode = P.get("ref_mode", "local")
    rid_tr = _runs(train_wid, train_seg)
    if val is not None:
        vm, vs, vw, vseg = val
        rid_v = _runs(vw, vseg)
    cols = analog if analog is not None else list(train_means.columns)
    base = {}
    for c in cols:
        x = train_means[c].values.astype(float)
        v = train_stds[c].values.astype(float)
        ref1 = pd.Series(x).groupby(rid_tr).transform(lambda s: s.rolling(P["L"], min_periods=P["L"]).median().shift(1)).values
        e0 = x - ref1
        mad = np.nanmedian(np.abs(e0 - np.nanmedian(e0))) * 1.4826
        sig_loc = max(mad if mad > 0 else np.nanstd(e0), 1e-9)
        sig_base = max(float(np.nanstd(x)), 1e-9)
        xa = x if val is None else np.r_[x, vm[c].values.astype(float)]
        va = v if val is None else np.r_[v, vs[c].values.astype(float)]
        pos = v[v > P["flat_eps"]]
        v_ref = float(np.median(pos)) if len(pos) else P["flat_eps"]
        lo_q, hi_q = np.nanquantile(xa, 1 - P["calib_q"]), np.nanquantile(xa, P["calib_q"])
        rng = max(hi_q - lo_q, 1e-9)
        b = {"mu": float(np.nanmean(x)), "sigma_x": sig_base, "scale_base": sig_base, "scale_loc": float(sig_loc),
             "v_ref": v_ref, "lo": float(lo_q - P["oob_margin"] * rng), "hi": float(hi_q + P["oob_margin"] * rng),
             "k1": P["k1"], "noise_thr": P["noise_k"], "shift_theta": P["shift_theta"],
             "drift_alpha": P["drift_alpha"], "fault_flat": P["fault_min_windows"]}
        # thresholds from validation (Normal) statistics
        if val is None:
            tx, tv, trid = x, v, rid_tr
        else:
            tx, tv, trid = vm[c].values.astype(float), vs[c].values.astype(float), rid_v
        st = pd.concat([_stats_run(tx[trid == r], tv[trid == r], b, {**P, "ref_mode": mode}) for r in np.unique(trid)]).reset_index(drop=True)
        q = P["calib_q"]
        b["k1"] = float(max(P["k1"], np.nanquantile(np.abs(st["e"]), q)))
        b["noise_thr"] = float(max(P["noise_k"], np.nanquantile(st["nr"], q)))
        ok = (st["shift_spread"] < P["shift_stable"]) & (st["shift_slope_total"] < 1.0) & (st["shift_side_min"] > 0)
        if ok.sum() > 30:
            b["shift_theta"] = float(max(P["shift_theta"], np.nanquantile(st.loc[ok, "shift_stat"], q)))
        okd = st["drift_r2"] >= P["drift_r2"]
        if okd.sum() > 30:
            b["drift_alpha"] = float(max(P["drift_alpha"], np.nanquantile(st.loc[okd, "drift_stat"], q)))
        # flatline: run length that is "normal" for this sensor (train + val)
        fr_tr = _run_length(v < P["flat_eps"])
        fr = fr_tr if val is None else np.r_[fr_tr, _run_length(vs[c].values < P["flat_eps"])]
        f_s = int(max(P["fault_min_windows"], np.quantile(fr, q) + 1))
        b["flat_rule_enabled"] = f_s <= P["max_fault_run"]
        b["fault_flat"] = f_s if b["flat_rule_enabled"] else 10 ** 9
        base[c] = b
    return {"params": P, "sensors": base}


def fit(train, val, params=None, analog=None):
    """
    Choose the reference mode per dataset (data-driven, no test labels):
      baseline mode  = the guide's semantics (distance from Normal baseline) -> used when the guide's numbers
                       (k1=3.5, theta=2.5) still hold on Normal validation (median calibrated factor <= 1.5x guide)
      local mode     = distance from the previous L windows -> used when Normal data is non-stationary.
    train / val = (means, stds, window_id, segment)
    """
    P0 = {**DEFAULT_PARAMS, **(params or {})}
    mb = calibrate(*train, val=val, params={**P0, "ref_mode": "baseline"}, analog=analog)
    k1 = np.median([b["k1"] for b in mb["sensors"].values()])
    th = np.median([b["shift_theta"] for b in mb["sensors"].values()])
    stationary = (k1 <= 1.5 * P0["k1"]) and (th <= 1.5 * P0["shift_theta"])
    model = mb if stationary else calibrate(*train, val=val, params={**P0, "ref_mode": "local"}, analog=analog)
    model["selection"] = {"ref_mode": model["params"]["ref_mode"], "median_k1_in_baseline_mode": float(k1),
                          "median_shift_theta_in_baseline_mode": float(th), "guide_k1": P0["k1"],
                          "guide_theta": P0["shift_theta"]}
    return model


def detect_series(model, means, stds, window_id, segment=None):
    """Offline: scenario code per window x sensor (DataFrame of ints)."""
    P = model["params"]
    rid = _runs(window_id, segment)
    out = {}
    for c, b in model["sensors"].items():
        st = sensor_stats(means[c].values, stds[c].values, rid, b, P)
        out[c] = classify(st, b, P, rid)
    return pd.DataFrame(out, index=means.index)


def summarize_window(codes_row, min_sensors=1):
    """
    Window-level label = highest-priority scenario that at least
    `min_sensors` sensors agree on.

    When min_sensors=1 (legacy), behaves like before: any single
    sensor's highest-priority code wins. When min_sensors > 1, a
    scenario is only reported if enough sensors agree; otherwise the
    window falls back to the next lower-priority scenario, and finally
    to 'normal'.
    """
    if len(codes_row) == 0:
        return SCENARIOS[0], []

    for code in sorted(codes_row.unique(), reverse=True):
        if code == 0:
            return SCENARIOS[0], []
        sensors = [c for c, k in codes_row.items() if k == code]
        if len(sensors) >= min_sensors:
            return SCENARIOS[code], sensors

    return SCENARIOS[0], []

class ScenarioDetector:
    """Streaming wrapper: feed one window at a time, get per-sensor scenario of the latest window."""

    def __init__(self, model, buffer_windows=40):
        self.m = model
        self.buf_x = deque(maxlen=buffer_windows)
        self.buf_v = deque(maxlen=buffer_windows)

    def update(self, window_means: dict, window_stds: dict):
        self.buf_x.append(window_means)
        self.buf_v.append(window_stds)
        X = pd.DataFrame(list(self.buf_x))
        V = pd.DataFrame(list(self.buf_v))
        P = self.m["params"]
        rid = np.zeros(len(X), dtype=int)
        res = {}
        for c, b in self.m["sensors"].items():
            st = sensor_stats(X[c].values, V[c].values, rid, b, P)
            res[c] = SCENARIOS[int(classify(st, b, P, rid)[-1])]
        top = max(res.values(), key=lambda s: _CODE[s])
        if top == "normal":
            return {"detected_scenario": "Not Observed", "confidence_score": 0.0, "sensors": [],
                    "evidence": "All sensor readings remain within normal operational parameters and do not match "
                                "the thresholds of the six defined scenarios."}
        return {"detected_scenario": top, "sensors": [c for c, s in res.items() if s == top],
                "per_sensor": {c: s for c, s in res.items() if s != "normal"}}


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
