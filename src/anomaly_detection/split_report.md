# Phase 2 - Task 1: Feature Separation & Event-Based Split Report

## 1. Overview & Data Summary
- **Total Samples (After Cleaning):** 1,462,200
- **Time Range:** 2020-02-01 00:00:00 to 2020-09-01 03:56:32
- **Sensor Features Count:** 15
- **Metadata Columns:** ['timestamp', 'segment', 'ground_truth', 'window_id']

---

## 2. Cutoff Timestamps & Split Logic
Splits were constructed based on historical failure events:
- **Cutoff 1 (Train / Validation):** `2020-04-12 00:00:00`
- **Cutoff 2 (Validation / Test):** `2020-07-01 00:00:00`

---

## 3. Failure Events Distribution
- **Train Set:** Event 1 (2020-04-18) and Event 2 (2020-05-29 to 2020-05-30)
- **Validation Set:** Event 3 (2020-06-05 to 2020-06-07)
- **Test Set:** Event 4 (2020-07-15)

---

## 4. Split Statistics
| Split | Row Count | Percentage |
| :--- | :--- | :--- |
| **Train** | 521,898 | 35.69% |
| **Validation** | 502,345 | 34.36% |
| **Test** | 437,957 | 29.95% |
| **Total** | **1,462,200** | **100.00%** |

---

## 5. Verification & Acceptance Criteria
- **Temporal Order:** Verified ($\max(Train) < \min(Val) < \min(Test)$).
- **Data Integrity:** No missing values in sensor features, row counts match cleaned dataset.
