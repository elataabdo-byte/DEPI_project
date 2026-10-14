# Timestamp processing log — 2026-10-14 12:31 UTC

- Input: `C:\Users\compu import\Desktop\TEST\MetroPT3(AirCompressor).csv`
- Output dir: `C:\Users\compu import\Desktop\TEST`
- Rows after cleaning: **1,516,948**
- Invalid timestamps excluded: 0
- Duplicate rows (kept): 0 (0 groups, 0 exact)
- Typical interval (median): 0 days 00:00:10
- Gap threshold (3× median): 0 days 00:00:30
- Segments created: **332**
- Gaps detected: **331**
- Timezone: timezone-naive (no offset in source). Assumption: recorded in local wall-clock time; the MetroPT-3 documentation does NOT specify a timezone.
- Timestamp range: 2020-02-01 00:00:00 → 2020-09-01 03:59:50
- Total observed gap time: 37.90 days
- Longest gap: 172918.0 s
- Estimated missing samples (if regular): 327,097

## Caveats
- A gap = observed temporal gap; **not** evidence of a machine failure.
- Missing-sample estimates assume regular sampling at the median interval.
- Duplicates were **kept** (`keep_all`); they remain visible in the output.
- Sampling rate is **empirical** (from this run), not assumed from documentation.
