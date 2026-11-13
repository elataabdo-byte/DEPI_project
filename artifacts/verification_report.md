
# Verification Report

**Source file:** `data_clean IOT.csv`
**Generated:** 2026-11-10 17:27:13
**Rows:** 50,570    **Columns:** 15

---

## 1. Basic Verification

| Check | Result |
|---|---|
| Shape | 50,570 x 15 |
| Total NaN | 0 |
| Duplicate rows | 0 |
| Duplicate timestamps | 0 |
| Duplicate id | 0 |
| Timestamp monotonic increasing | True |
| Time range | 2023-11-26 10:57:52 to 2023-12-26 21:36:40 |

Column dtypes:

```
id                          int64
timestamp          datetime64[us]
pH                        float64
TDS                       float64
water_level                 int64
DHT_temp                  float64
DHT_humidity              float64
water_temp                float64
pH_reducer                  int64
add_water                   int64
nutrients_adder             int64
humidifier                  int64
ex_fan                      int64
segment                     int64
window_id                   int64
```

## 2. Physical Range Violations (count only, no deletion)

| Sensor | Outside range | Valid range |
|---|---|---|
| pH | 0 | [0, 14] |
| TDS | 25 | [0, 5000] |
| DHT_temp | 8 | [5, 45] |
| DHT_humidity | 3 | [0, 100] |
| water_temp | 72 | [5, 40] |
| water_level | 0 | [0, 3] |

## 3. Actuators (expected 0/1)

- `pH_reducer`: [0, 1]
- `add_water`: [0, 1]
- `nutrients_adder`: [0, 1]
- `humidifier`: [0, 1]
- `ex_fan`: [0, 1]

## 4. Zeros and Negatives per Sensor

| Sensor | Zeros | Negatives |
|---|---|---|
| pH | 0 | 0 |
| TDS | 0 | 25 |
| DHT_temp | 0 | 0 |
| DHT_humidity | 0 | 0 |
| water_temp | 72 | 0 |
| water_level | 904 | 0 |

## 5. Distributions

```
                count         mean         std     min       0.1%       1%      5%      50%       95%      99%      99.9%      max
pH            50570.0     6.001514    0.553016    0.27    3.27000    5.370    5.50     5.79     6.890     6.98     7.3900    11.57
TDS           50570.0  1154.134377  247.507196 -283.91  464.22949  550.629  689.48  1234.65  1460.654  1602.54  1662.8400  2278.35
DHT_temp      50570.0    24.316781    1.209319   12.30   20.00000   20.000   23.00    24.40    25.900    26.20    27.2000    70.00
DHT_humidity  50570.0    71.714188   25.956493   25.00   46.70000   52.800   58.80    71.70    81.700    82.80    83.1431  3312.60
water_temp    50570.0    21.494904    2.174898    0.00    0.00000   18.060   18.35    21.52    24.660    24.93    24.9900    25.00
water_level   50570.0     1.258829    0.479218    0.00    0.00000    0.000    1.00     1.00     2.000     2.00     3.0000     3.00
```

## 6. Anomaly Locations

### TDS < 0

- Count: **25**
- Segments: [34, 36]
- Time range: 2023-12-12 22:35:21 to 2023-12-13 01:19:42

### DHT_temp outside [5, 45]

- Count: **8**
- Segments: [64, 65]
- Time range: 2023-12-21 08:30:45 to 2023-12-21 08:40:18

### DHT_humidity outside [0, 100]

- Count: **3**
- Segments: [57, 58]
- Time range: 2023-12-19 15:44:28 to 2023-12-19 23:46:48

### water_temp <= 0

- Count: **72**
- Segments: [53]
- Time range: 2023-12-19 12:33:40 to 2023-12-19 12:40:25

Actuator state during `water_temp <= 0`:

```
   pH_reducer  add_water  nutrients_adder  humidifier  ex_fan
0           0          0               72          72      72
1          72         72                0           0       0
```

All rows with `water_temp <= 0` coincide with `add_water = 1` and `pH_reducer = 1`. This is an operating state (pump active), not a sensor fault.

## 7. pH Extremes

| Threshold | Count |
|---|---|
| pH < 1 | 15 |
| pH < 4 | 232 |
| pH < 5 | 270 |
| pH > 7.5 | 42 |
| pH > 9 | 12 |
| pH > 10 | 3 |

## 8. water_level Distribution

| Level | Count |
|---|---|
| 0 | 904 |
| 1 | 35,725 |
| 2 | 13,889 |
| 3 | 52 |

## 9. Data Structure

- Total segments: **100**
- Windows per segment (mean): 22.7
- Rows per segment: min=1, median=61, max=10,089

## 10. Classification of Findings

### Normal operation (kept in reference period)

- pH variation between ~3 and ~9, driven by `pH_reducer` and `nutrients_adder`
- `water_temp <= 0` during `add_water = 1` and `pH_reducer = 1`
- `water_level = 0` (level-low state)

### Sensor faults (excluded from reference period)

| Type | Count | Segments |
|---|---|---|
| TDS < 0 | 25 | 34, 36 |
| DHT_temp outside [5, 45] | 8 | 64, 65 |
| DHT_humidity outside [0, 100] | 3 | 57, 58 |

Total rows flagged as sensor fault: **36** (0.071% of dataset).

These rows will be excluded only when computing the baseline statistics. They remain in the dataset for downstream analysis.
 
