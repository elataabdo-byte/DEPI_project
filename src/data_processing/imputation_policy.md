# Imputation Policy & Temporal Handling - MetroPT Dataset

## 1. Executive Summary & Verification
The dataset was thoroughly audited for explicit missing values (`NaN`/`null`), invalid strings, and negative sentinel values (e.g., `-999`). 
- **Explicit Missing Records:** 0 across all 15 operational sensor channels.
- **Sampling Interval:** Nominal sensor acquisition operates at a 10-second interval (0.1 Hz).

## 2. Imputation Strategy Justification
Industrial pneumatic compressor data represents continuous thermodynamic and electrical physical states. Imputation policies must respect temporal causality:
- **Continuous Physical Sensors (e.g., Pressures, Temperatures, Motor Current):**
  - *Method:* Segment-bounded Linear Interpolation (`interpolate(method='time')`).
  - *Rationale:* Physical states (like air pressure and oil temperature) do not experience instantaneous step changes; smooth linear transitions reflect physical continuity.
- **Discrete & Control Signals (e.g., COMP, DV_eletric, Towers, Pressure_switch):**
  - *Method:* Forward Fill (`ffill()`) within active segments.
  - *Rationale:* Discrete relay/valve states remain latch-locked until the next actuation signal occurs.
- **Segment-Based Execution:**
  - Imputation is strictly executed per `segment` (`groupby('segment')`). 
  - Crossing segment boundaries is strictly prohibited to prevent data leakage between distinct operational runs.

## 3. Policy on Long Gaps & Data Deletion
- **Short Gaps (<= 30 seconds):**
  - Represent normal sensor polling jitter or network dropouts. Maximum observed within-segment gap is 30.0s (<= 2 missed cycles). These are safe for segment-bounded interpolation.
- **Long Gaps (> 30 seconds / Hours to Days):**
  - Represent system maintenance, depot shutdowns, or non-operational compressor states.
  - **Handling Rule:** Long gaps are **not** synthesized or interpolated. Segment boundaries created during the timestamp phase segment the time series, isolating operational periods without corrupting true machine idle periods. No raw timestamps are deleted to preserve physical chronology.
