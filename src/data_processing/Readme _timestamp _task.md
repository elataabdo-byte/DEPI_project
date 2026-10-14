# MetroPT-3 Timestamp Processing, Gap Analysis and Segmentation

**Phase 1 - Task 2 (Timestamp).** Notebook: `time_stamp.ipynb`

## 1. Purpose

The MetroPT-3 air-compressor dataset is a multivariate sensor time series. Before missing-value handling, ground-truth
labeling and windowing can be done, the time axis must be trustworthy. This task:

1. Parses and validates the `timestamp` column.
2. Sorts the records chronologically (stable sort).
3. Investigates duplicate timestamps without deleting anything.
4. Measures the real sampling interval from the data.
5. Detects time gaps and splits the series into continuous segments.
6. Exports the cleaned dataset together with auditable gap, segment and log files.

## 2. Input and Output

| Item | Description |
|---|---|
| Input | `MetroPT3(AirCompressor).csv` (1,516,948 rows; 16 columns after dropping the pandas index column `Unnamed: 0`) |
| Configuration | Environment variables `TS_INPUT_CSV` and `TS_OUTPUT_DIR`, or the `Config` dataclass in Step 1 |

### Deliverables

| File | Content |
|---|---|
| `data_ts_fixed.csv` | Cleaned dataset: parsed and chronologically sorted timestamps, all sensor columns unchanged, plus a `segment` column |
| `timestamp_gaps.csv` | One row per detected gap: `gap_id`, `previous_timestamp`, `next_timestamp`, `observed_interval_s`, `threshold_s`, `reference_interval_s`, `estimated_missing_samples` |
| `timestamp_segments_summary.csv` | One row per segment: `segment`, `start_timestamp`, `end_timestamp`, `row_count`, `duration_s` |
| `timestamp_log.md` | Processing log generated from the computed results (row counts, duplicates, interval, gaps, segments, timezone, caveats) |

## 3. Method

| Step | What the notebook does |
|---|---|
| 1. Configuration | A single `Config` dataclass holds paths, chunk size, timestamp and segment column names, columns to drop, and the gap multiplier. No hard-coded paths in the logic. |
| 2. Ingestion | The CSV is read in chunks of 200,000 rows with every field kept as text (`dtype=str`, `keep_default_na=False`), so sensor values are carried through exactly as stored. The index artefact `Unnamed: 0` is dropped. |
| 3. Timestamp processing | Timestamps are parsed with `pd.to_datetime(errors="coerce")`. Invalid values are counted and reported. Rows are sorted with a stable sort so that the original row order breaks ties. Duplicate timestamps are investigated (rows, groups, exact duplicates) and kept under the `keep_all` policy. |
| 4. Sampling interval | Consecutive time differences are computed; the median is used as the typical interval because it is robust to long shutdown gaps. Minimum and maximum intervals are reported. |
| 5. Gap detection and segmentation | A gap is any interval strictly greater than `3 x median interval`. A cumulative sum over the gap flags assigns an incrementing `segment` id starting at 0, so downstream windowing can be restricted to continuous blocks. |
| 6. Validation and export | Assertions check chronological order, that segment ids start at 0, that the segment summary matches the segment count and total row count, and that every gap exceeds the threshold. Only then are the four output files written. |

## 4. Results (from the executed run)

| Metric | Value |
|---|---|
| Rows after cleaning | 1,516,948 |
| Columns in output | 16 (including `segment`, excluding `Unnamed: 0`) |
| Time range | 2020-02-01 00:00:00 to 2020-09-01 03:59:50 |
| Median sampling interval | 10 seconds |
| Minimum / maximum interval | 8 seconds / 2 days 00:01:58 |
| Gap threshold | 30 seconds (3 x median) |
| Gaps detected | 331 |
| Segments created | 332 |

The notebook prints a message only when invalid timestamps or duplicate timestamps are found; no such message
appeared in the run, so none were reported for this file. The authoritative figures (including total gap time,
longest gap and estimated missing samples) are in the generated `timestamp_log.md`.

## 5. Design Decisions

- **Duplicates are never deleted.** Removing a row only to make a timestamp unique can destroy valid
  measurements. The policy is `keep_all`, and any duplicates would remain visible and be counted in the log.
- **The sampling interval is empirical.** It is computed from this run rather than assumed from documentation.
- **The gap rule is a heuristic.** `3 x median` is configurable through `gap_multiplier` and is not a universal rule.
- **Sensor values are preserved as text.** The data is read as strings and written back unchanged; only the
  timestamp column is converted to a datetime type.
- **Missing-sample estimates** use `round(observed interval / median interval) - 1` (floored at 0), for example
  a 40-second interval at a 10-second median gives 3 missing samples. This assumes regular sampling inside the gap.

## 6. Important Caveats

- A gap is an **observed temporal gap**. It is not evidence of a machine failure or shutdown; linking gaps to
  failures requires independent evidence such as the failure log.
- Estimated missing samples are **suspected**, not confirmed.
- The sampling is not perfectly regular: intervals vary by a few seconds around 10 seconds (jitter), so
  "one record every 10 seconds" is only approximately true.
- **Time zone:** the timestamps carry no offset and the dataset documentation does not specify a time zone. The
  values are kept as recorded; no conversion was applied.
- No values were interpolated, imputed or inserted.

## 7. How to Run

1. Place `MetroPT3(AirCompressor).csv` in the working directory, or set the paths:

   ```powershell
   $env:TS_INPUT_CSV  = "C:\path\to\MetroPT3(AirCompressor).csv"
   $env:TS_OUTPUT_DIR = "C:\path\to\output"
   ```

2. Open `time_stamp.ipynb` and run all cells in order.
3. Check the console summary and `timestamp_log.md`.

Requirements: Python 3.9 or newer, `pandas`, `numpy`, and Jupyter.

## 8. Impact on Downstream Tasks

| Downstream task | Use of this output |
|---|---|
| Missing values | Gaps are known and flagged, so imputation is not applied across shutdown periods. |
| Ground-truth labeling | Timestamps are standardized and sorted, so failure intervals can be aligned with the timeline. |
| Windowing | Sliding windows must be built inside each `segment` so that no window spans a gap. |
