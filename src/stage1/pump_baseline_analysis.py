import json
from datetime import datetime
import numpy as np
import pandas as pd
input_file = "sensor_baseline.csv"
df = pd.read_csv(input_file)

original_row_count = len(df)
time_range = {
    "start": str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
    "end": str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A"
}

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5, limit_direction="both")
df[numeric_cols] = df[numeric_cols].fillna(value=np.nan)

if "status" in df.columns:
    df_baseline = df[df["status"] == "NORMAL"].copy()
else:
    df_baseline = df.copy()

for col in numeric_cols:
    if col in df_baseline.columns:
        median = df_baseline[col].median()
        mad = (df_baseline[col] - median).abs().median()
        if mad > 0:
            modified_z_score = 0.6745 * (df_baseline[col] - median) / mad
            df_baseline = df_baseline[modified_z_score.abs() < 3.5]

baseline_output_file = "sensor_baseline.csv"
df_baseline[numeric_cols].to_csv(baseline_output_file, index=False)

metadata = {
    "computation_timestamp": datetime.now().isoformat(),
    "source_file_name": input_file,
    "original_row_count": original_row_count,
    "baseline_row_count": len(df_baseline),
    "time_range": time_range,
    "filter_applied": "NORMAL status + MAD-based outlier removal (threshold 3.5)",
    "pipeline_version": "v2.1_production"
}

with open("sensor_baseline_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
