"""
scenario_pipeline.py - E2 end-to-end, per dataset:
  1) window stats (mean/std per window per sensor) from the Phase-1 split files
  2) fit scenario parameters (baseline from Normal Train, thresholds from Normal Validation)
  3) evaluate the rules on Val/Test (false-alarm rate on Normal windows, hit rate on failure windows, episodes)
  4) run synthetic Test Cases injected into REAL Normal windows of the same dataset

Usage:
  python scenario_pipeline.py --split-dir processed_outputs --prefix pump --out scenario_out
Needs  <split-dir>/<prefix>/<prefix>_{train,val,test}.csv and <prefix>_features.json  (output of split_data.py)
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import scenario_rules as sr

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- window stats
def window_stats(split_dir, prefix):
    base = os.path.join(split_dir, prefix)
    feats = json.load(open(os.path.join(base, f"{prefix}_features.json")))
    out = {}
    for part in ["train", "val", "test"]:
        head = pd.read_csv(os.path.join(base, f"{prefix}_{part}.csv"), nrows=1)
        gt_col = "ground_truth" if "ground_truth" in head.columns else None
        df = pd.read_csv(os.path.join(base, f"{prefix}_{part}.csv"),
                         usecols=["timestamp", "window_id", "segment"] + feats + ([gt_col] if gt_col else []),
                         parse_dates=["timestamp"])
        g = df.groupby("window_id", sort=True)
        meta = pd.DataFrame({"ts": g["timestamp"].min(), "segment": g["segment"].first()})
        meta["gt"] = g[gt_col].max() if gt_col else 0
        out[part] = (g[feats].mean(), g[feats].std(ddof=0), meta)
    train_raw = pd.read_csv(os.path.join(base, f"{prefix}_train.csv"), usecols=feats)
    nun = train_raw.nunique()
    digital = [c for c in feats if nun[c] <= 3]          # on/off signals: scenario rules are for analog sensors
    return out, feats, digital


# ---------------------------------------------------------------- evaluation
def episodes(flag):
    f = np.asarray(flag, bool)
    return int(((f[1:] & ~f[:-1]).sum()) + (1 if f[0] else 0))


def evaluate(model, data, analog, has_gt):
    res = {}
    for part in ["val", "test"]:
        m, s, meta = data[part]
        codes = sr.detect_series(model, m[analog], s[analog], m.index.values, meta["segment"].values)
        normal = (meta["gt"] == 0).values
        r = {"windows_normal": int(normal.sum()), "windows_failure": int((~normal).sum()), "scenarios": {}}
        for i, sc in enumerate(sr.SCENARIOS):
            if i == 0:
                continue
            hit = (codes == i).any(axis=1).values
            sensors = sorted(codes.columns[(codes == i).any()].tolist())
            d = {"windows_any_sensor": int(hit.sum()), "episodes": episodes(hit), "sensors": sensors,
                 "rate_on_normal": round(float(hit[normal].mean()), 4) if normal.any() else None}
            if (~normal).any():
                d["rate_on_failure"] = round(float(hit[~normal].mean()), 4)
            r["scenarios"][sc] = d
        res[part] = r
    return res


# ---------------------------------------------------------------- test cases (injection into real windows)
def _longest_run(wid):
    wid = np.asarray(wid)
    brk = np.r_[True, np.diff(wid) != 1]
    rid = np.cumsum(brk)
    best = max(np.unique(rid), key=lambda r: (rid == r).sum())
    return np.where(rid == best)[0]


def test_cases(model, data, analog):
    """Each case is built from a real Normal stretch of the validation data; amplitude = 2x the calibrated threshold."""
    P = model["params"]
    m, s, meta = data["val"]
    idx = _longest_run(m.index.values)
    idx = idx[(meta["gt"].values[idx] == 0)]
    T = 40
    rows = []
    mode = P["ref_mode"]
    for c in analog:
        b = model["sensors"][c]
        x0 = m[c].values[idx].astype(float)
        v0 = s[c].values[idx].astype(float)
        scale = b["scale_base"] if mode == "baseline" else b["scale_loc"]
        ref = b["mu"] if mode == "baseline" else np.nanmedian(x0)
        n = min(len(x0), 400)
        # candidate start: stretch of T windows where the real signal stays near the reference (needed for "return to normal")
        starts = [i for i in range(14, n - T, 2)
                  if np.all(np.abs(x0[i - 2:i + 10] - ref) <= 0.9 * scale)]   # calm only where we inject
        rid = np.zeros(T, dtype=int)

        def run(x, v):
            st = sr.sensor_stats(x, v, rid, b, P)
            return sr.classify(st, b, P, rid)

        def add(case, expect, window, ok_fn):
            rows.append({"sensor": c, **case, "expected": expect, "window": window, **ok_fn})

        if not starts:
            rows.append({"sensor": c, "case": "all", "result": "SKIPPED", "detail": "no calm real stretch to inject into"})
            continue
        t0s = starts[0]
        xb = x0[t0s - 14:t0s - 14 + T].copy()
        vb = v0[t0s - 14:t0s - 14 + T].copy()
        t0 = 14
        A = 2 * b["k1"] * scale
        # 0 normal
        code = run(xb, vb)
        rows.append({"sensor": c, "case": "normal", "expected": "normal", "result": "PASS" if (code[12:] == 0).all() else "FAIL",
                     "detail": f"alarms={[sr.SCENARIOS[k] for k in code[12:] if k]}"})
        # 1 single noise (spike that returns)
        x, v = xb.copy(), vb.copy()
        x[t0] += A
        code = run(x, v)
        rows.append({"sensor": c, "case": "single_noise", "expected": "single_noise",
                     "result": "PASS" if code[t0 + 1] == 1 else "FAIL", "detail": sr.SCENARIOS[code[t0 + 1]]})
        # 2a repeated noise: three returning spikes
        x, v = xb.copy(), vb.copy()
        for k, sgn in zip([t0, t0 + 2, t0 + 4], [1, -1, 1]):
            x[k] += sgn * A
        code = run(x, v)
        ok = (code[t0 + 5:t0 + 8] == 2).any()
        rows.append({"sensor": c, "case": "repeated_noise(spikes)", "expected": "repeated_noise", "result": "PASS" if ok else "FAIL",
                     "detail": [sr.SCENARIOS[k] for k in code[t0 + 5:t0 + 8]]})
        # 2b repeated noise: sustained variance inflation
        x, v = xb.copy(), vb.copy()
        v[t0:t0 + 4] = np.maximum(v[t0:t0 + 4], b["v_ref"]) * 2 * b["noise_thr"]
        code = run(x, v)
        ok = (code[t0 + 2:t0 + 7] == 2).any()
        rows.append({"sensor": c, "case": "repeated_noise(variance)", "expected": "repeated_noise", "result": "PASS" if ok else "FAIL",
                     "detail": [sr.SCENARIOS[k] for k in code[t0 + 2:t0 + 7]]})
        # 3 drift: linear ramp over 10 windows
        x, v = xb.copy(), vb.copy()
        slope = 1.3 * b["drift_alpha"] * scale                    # 1.3x the calibrated minimum slope, 8 windows
        if ref + 8 * slope > b["hi"] - 0.1 * scale:
            rows.append({"sensor": c, "case": "drift", "expected": "drift", "result": "N/A",
                         "detail": "a drift this steep leaves the sensor's normal range within 8 windows"})
        else:
            x[t0:t0 + 8] += np.arange(1, 9) * slope
            x[t0 + 8:] += 8 * slope
            code = run(x, v)
            ok = (code[t0 + 5:t0 + 12] == 3).any()
            rows.append({"sensor": c, "case": "drift", "expected": "drift", "result": "PASS" if ok else "FAIL",
                         "detail": [sr.SCENARIOS[k] for k in code[t0 + 5:t0 + 12]]})
        # 4 shift: step to a new constant level for 12 windows
        x, v = xb.copy(), vb.copy()
        x[t0:t0 + 12] += 2 * b["shift_theta"] * scale
        code = run(x, v)
        ok = (code[t0 + 3:t0 + 12] == 4).any()
        rows.append({"sensor": c, "case": "shift", "expected": "shift", "result": "PASS" if ok else "FAIL",
                     "detail": [sr.SCENARIOS[k] for k in code[t0 + 3:t0 + 12]]})
        # 5a sudden fault: flatline
        if b["flat_rule_enabled"]:
            x, v = xb.copy(), vb.copy()
            F = int(b["fault_flat"])
            x[t0:t0 + F + 2] = x[t0]
            v[t0:t0 + F + 2] = 0.0
            code = run(x, v) if t0 + F + 2 <= T else None
            if code is None:
                rows.append({"sensor": c, "case": "sudden_fault(flatline)", "expected": "sudden_fault", "result": "SKIPPED",
                             "detail": f"needs {F} flat windows (> stretch)"})
            else:
                ok = (code[t0 + F - 1:t0 + F + 2] == 5).any()
                rows.append({"sensor": c, "case": "sudden_fault(flatline)", "expected": "sudden_fault", "result": "PASS" if ok else "FAIL",
                             "detail": [sr.SCENARIOS[k] for k in code[t0 + F - 1:t0 + F + 2]]})
        else:
            rows.append({"sensor": c, "case": "sudden_fault(flatline)", "expected": "sudden_fault", "result": "N/A",
                         "detail": "sensor is flat normally -> flatline rule disabled for it"})
        # 5b sudden fault: saturation (out of bounds)
        x, v = xb.copy(), vb.copy()
        x[t0:t0 + 8] = b["hi"] + scale
        code = run(x, v)
        ok = (code[t0 + P["fault_min_windows"] - 1:t0 + 8] == 5).any()
        rows.append({"sensor": c, "case": "sudden_fault(out_of_bounds)", "expected": "sudden_fault", "result": "PASS" if ok else "FAIL",
                     "detail": [sr.SCENARIOS[k] for k in code[t0 + 3:t0 + 8]]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main(split_dir, prefix, out_dir):
    data, feats, digital = window_stats(split_dir, prefix)
    analog = [c for c in feats if c not in digital]
    mtr, str_, mt = data["train"]
    mv, sv, mvm = data["val"]
    model = sr.fit((mtr[analog], str_[analog], mtr.index.values, mt["segment"].values),
                   (mv[analog], sv[analog], mv.index.values, mvm["segment"].values), analog=analog)
    model["digital_sensors_excluded"] = digital
    has_gt = bool((data["test"][2]["gt"] > 0).any())
    ev = evaluate(model, data, analog, has_gt)
    tc = test_cases(model, data, analog)
    o = os.path.join(out_dir, prefix)
    os.makedirs(o, exist_ok=True)
    sr.save_model(model, os.path.join(o, f"{prefix}_scenario_params.json"))
    json.dump(ev, open(os.path.join(o, f"{prefix}_scenario_eval.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    tc.to_csv(os.path.join(o, f"{prefix}_scenario_testcases.csv"), index=False, encoding="utf-8-sig")
    summ = tc[tc["result"].isin(["PASS", "FAIL"])].groupby("case")["result"].apply(lambda r: f"{(r == 'PASS').sum()}/{len(r)}")
    print(f"\n=== {prefix}: ref_mode={model['selection']['ref_mode']}  analog={len(analog)}  digital_excluded={digital}")
    print("Test cases (pass/total sensors):", summ.to_dict())
    for part in ["val", "test"]:
        print(f" {part}:", {k: (v['rate_on_normal'], v.get('rate_on_failure'), v['episodes']) for k, v in ev[part]["scenarios"].items()})
    return model, ev, tc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="processed_outputs")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default="scenario_out")
    a = ap.parse_args()
    main(a.split_dir, a.prefix, a.out)
