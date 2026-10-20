# Phase 2 - Task 1: Feature Separation & Event-Based Split Report

## 1. Overview & Data Summary
- **Total Samples (After Cleaning):** 1,498,426
- **Time Range:** 2020-02-01 00:00:00 to 2020-09-01 03:59:50
- **Sensor Features Count:** 15
- **Metadata Columns:** ['timestamp', 'segment', 'ground_truth', 'window_id']
- **Removed Frozen Period:** `2020-06-22 15:05:00` to `2020-06-25 05:08:35` (18,522 rows)

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
| **Train** | 522,848 | 34.89% |
| **Validation** | 531,976 | 35.50% |
| **Test** | 443,602 | 29.60% |
| **Total** | **1,498,426** | **100.00%** |

---

## 5. Verification & Acceptance Criteria
- **Temporal Order:** Verified ($\max(Train) < \min(Val) < \min(Test)$).
- **Data Integrity:** No missing values in sensor features, row counts match cleaned dataset.
