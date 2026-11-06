import os
import pandas as pd
import numpy as np
import json


def find_file_path(filename):
    possible_dirs = [
        r"C:\Users\compu import\Desktop\TEST\outputs\metrics",
        os.path.join(os.getcwd(), "outputs", "metrics"),
        os.path.join(os.getcwd(), "..", "outputs", "metrics"),
        r"C:\DEPI_project-main\DEPI_project-main\src\anomaly_detection",
        r"C:\DEPI_project-main\DEPI_project-main\src\data_processing",
        os.getcwd(),
        r"C:\Users\compu import\Desktop\TEST",
    ]

    for directory in possible_dirs:
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return os.path.normpath(full_path)

    return filename


def compute_and_save_critical_limits(
    failure_events_clean_name="failure_events_clean (2).csv",
    source_name="source.csv",
    baseline_name="sensor_baseline.csv",
    config_output_path="config_limits.py",
    safety_factor=4.0,
    pre_failure_minutes=30,
):
    print("=" * 70)
    print("Loading and locating input files...")
    print("=" * 70)

    failure_events_clean_path = find_file_path(failure_events_clean_name)
    source_path = find_file_path(source_name)
    baseline_path = find_file_path(baseline_name)

    print("Detected file paths:")
    print(f"  - Failure Events : {failure_events_clean_path}")
    print(f"  - Source Data    : {source_path}")
    print(f"  - Sensor Baseline: {baseline_path}")

    try:
        df_failures = pd.read_csv(failure_events_clean_path)
        df_data = pd.read_csv(source_path)
        df_baseline = pd.read_csv(baseline_path)

    except FileNotFoundError as e:
        print(f"\nError: Unable to locate the required file -> {e}")
        print("\nPlease ensure the files exist in one of the following directories:")

        for directory in [
            r"C:\Users\compu import\Desktop\TEST\outputs\metrics",
            r"C:\Users\compu import\Desktop\TEST",
            os.getcwd(),
        ]:
            print(f"  - {directory}")

        return

    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])
    df_failures["start_time"] = pd.to_datetime(df_failures["start_time"])

    if "end_time" in df_failures.columns:
        df_failures["end_time"] = pd.to_datetime(df_failures["end_time"])
    else:
        df_failures["end_time"] = df_failures["start_time"]

    baseline_dict = df_baseline.set_index("sensor")[["mean", "std"]].to_dict("index")

    EXCLUDE = {"window_id", "timestamp", "ground_truth", "segment"}
    sensors = [sensor for sensor in baseline_dict.keys() if sensor not in EXCLUDE]

    failure_intervals = [
        (row["start_time"], row["end_time"])
        for _, row in df_failures.iterrows()
    ]

    print("\n" + "=" * 70)
    print("2. Analyzing Sensor Readings Before Failure Events...")
    print("=" * 70)

    sensor_max_values = {sensor: [] for sensor in sensors}
    sensor_min_values = {sensor: [] for sensor in sensors}
    skipped = 0

    for _, row in df_failures.iterrows():
        start_time = row["start_time"]
        pre_start = start_time - pd.Timedelta(minutes=pre_failure_minutes)

        overlaps = any(
            (pre_start < failure_end)
            and (start_time > failure_start)
            and (failure_start != start_time)
            for failure_start, failure_end in failure_intervals
        )

        if overlaps:
            skipped += 1
            continue

        pre_failure_data = df_data[
            (df_data["timestamp"] < start_time)
            & (df_data["timestamp"] >= pre_start)
        ]

        if pre_failure_data.empty:
            continue

        for sensor in sensors:
            if sensor in pre_failure_data.columns:
                sensor_max_values[sensor].append(pre_failure_data[sensor].max())
                sensor_min_values[sensor].append(pre_failure_data[sensor].min())

    print(f"Total failure events       : {len(df_failures)}")
    print(f"Skipped overlapping windows: {skipped}")

    print("\n" + "=" * 70)
    print("3. Calculating Critical Limits and Z-Scores...")
    print("=" * 70)

    critical_limits = {}
    reasoning_docs = {}

    for sensor in sensors:
        mean = float(baseline_dict[sensor]["mean"])
        std = float(baseline_dict[sensor]["std"])

        if std <= 0:
            critical_limits[sensor] = {
                "upper_z": float(safety_factor),
                "lower_z": -float(safety_factor),
                "upper_raw": None,
                "lower_raw": None,
                "method": "fallback_constant",
            }

            reasoning_docs[sensor] = (
                f"{sensor}: Standard deviation is zero. "
                f"Default limits of ±{safety_factor} sigma were applied."
            )

            continue

        upper_z = float(safety_factor)
        lower_z = float(safety_factor)

        upper_raw = mean + upper_z * std
        lower_raw = mean - lower_z * std

        method = "baseline_K_sigma"

        critical_limits[sensor] = {
            "upper_z": round(upper_z, 2),
            "lower_z": round(-lower_z, 2),
            "upper_raw": round(upper_raw, 6),
            "lower_raw": round(lower_raw, 6),
            "method": method,
        }

        reasoning_docs[sensor] = (
            f"{sensor}: Upper limit={upper_z:.2f}σ "
            f"(raw={upper_raw:.4f}), "
            f"Lower limit={-lower_z:.2f}σ "
            f"(raw={lower_raw:.4f}). "
            f"Baseline: mean={mean:.4f}, std={std:.4f}. "
            f"Safety factor K={safety_factor}. Method={method}."
        )

    print("\nFinal Critical Limit Results:")

    for sensor, reasoning in reasoning_docs.items():
        print(f"- {reasoning}")

    config_content = f"""CRITICAL_LIMITS = {json.dumps(critical_limits, indent=4, ensure_ascii=False)}

REASONING = {json.dumps(reasoning_docs, indent=4, ensure_ascii=False)}
"""

    with open(config_output_path, "w", encoding="utf-8") as file:
        file.write(config_content)

    print("\n" + "=" * 70)
    print(f"Critical limits successfully saved to: {config_output_path}")
    print("=" * 70)


if __name__ == "__main__":
    compute_and_save_critical_limits(
        failure_events_clean_name="failure_events_clean (2).csv",
        source_name="source.csv",
        baseline_name="sensor_baseline.csv",
        config_output_path="config_limits.py",
        safety_factor=4.0,
        pre_failure_minutes=30,
    )
