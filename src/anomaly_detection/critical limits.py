import os
import pandas as pd
import numpy as np
import json


# ============================================================
# 1) البحث عن الملفات في عدة مسارات محتملة
# ============================================================
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


# ============================================================
# 2) الدالة الرئيسية
# ============================================================
def compute_and_save_critical_limits(
    failure_events_clean_name="failure_events_clean (2).csv",
    source_name="source.csv",
    baseline_name="sensor_baseline.csv",
    config_output_path="config_limits.py",
    safety_factor=4.0,          # K في "mean ± K*std"
    pre_failure_minutes=30,     # نافذة ما قبل العطل (دقائق)
):
    print("=" * 70)
    print("جاري البحث عن الملفات وتحميلها...")
    print("=" * 70)

    failure_events_clean_path = find_file_path(failure_events_clean_name)
    source_path = find_file_path(source_name)
    baseline_path = find_file_path(baseline_name)

    print(f"📁 المسارات التي تم رصدها:")
    print(f"  - Failure Events : {failure_events_clean_path}")
    print(f"  - Source Data    : {source_path}")
    print(f"  - Baseline       : {baseline_path}")

    # --------------------------------------------------------
    # تحميل الملفات
    # --------------------------------------------------------
    try:
        df_failures = pd.read_csv(failure_events_clean_path)
        df_data = pd.read_csv(source_path)
        df_baseline = pd.read_csv(baseline_path)
    except FileNotFoundError as e:
        print(f"\n❌ خطأ: لم يتم العثور على أحد الملفات -> {e}")
        print("\n💡 اتأكد إن الملفات دي موجودة في أحد المسارات:")
        for d in [
            r"C:\Users\compu import\Desktop\TEST\outputs\metrics",
            r"C:\Users\compu import\Desktop\TEST",
            os.getcwd(),
        ]:
            print(f"  - {d}")
        return

    # --------------------------------------------------------
    # تحويل التواريخ مرة واحدة (برّه اللوب = أسرع)
    # --------------------------------------------------------
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])
    df_failures["start_time"] = pd.to_datetime(df_failures["start_time"])
    if "end_time" in df_failures.columns:
        df_failures["end_time"] = pd.to_datetime(df_failures["end_time"])
    else:
        df_failures["end_time"] = df_failures["start_time"]

    baseline_dict = df_baseline.set_index("sensor")[["mean", "std"]].to_dict("index")
    EXCLUDE = {"window_id", "timestamp", "ground_truth", "segment"}
    sensors = [s for s in baseline_dict.keys() if s not in EXCLUDE]
    # قائمة فواصل الأعطال لتفادي التداخل
    failure_intervals = [
        (r["start_time"], r["end_time"])
        for _, r in df_failures.iterrows()
    ]

    # --------------------------------------------------------
    # 2) فحص القيم قبل الأعطال مباشرة
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. فحص قيم الحساسات قبل الأعطال مباشرة...")
    print("=" * 70)

    sensor_max_values = {s: [] for s in sensors}
    sensor_min_values = {s: [] for s in sensors}
    skipped = 0

    for _, row in df_failures.iterrows():
        start_time = row["start_time"]
        pre_start = start_time - pd.Timedelta(minutes=pre_failure_minutes)

        # تخطي النوافذ اللي تتداخل مع عطل تاني
        overlaps = any(
            (pre_start < f_end) and (start_time > f_start) and (f_start != start_time)
            for f_start, f_end in failure_intervals
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

    print(f"   إجمالي الأعطال       : {len(df_failures)}")
    print(f"   تم تخطي (تداخل نوافذ) : {skipped}")

    # --------------------------------------------------------
    # 3) حساب الحدود الحرجة (upper + lower) وتحويلها إلى Z-scores
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. حساب الحدود الحرجة (upper + lower) وتحويلها إلى Z-scores...")
    print("=" * 70)

    critical_limits = {}
    reasoning_docs = {}

    for sensor in sensors:
        mean = float(baseline_dict[sensor]["mean"])
        std = float(baseline_dict[sensor]["std"])

        # لو std = 0: نستخدم safety_factor مباشرة
        if std <= 0:
            critical_limits[sensor] = {
                "upper_z": float(safety_factor),
                "lower_z": -float(safety_factor),
                "upper_raw": None,
                "lower_raw": None,
                "method": "fallback_constant",
            }
            reasoning_docs[sensor] = (
                f"{sensor}: std=0 (ثابت في الـ baseline)، "
                f"استخدمنا ±{safety_factor} sigma كقيمة افتراضية."
            )
            continue

        # الحد الأساسي = mean ± K*std
        upper_z = float(safety_factor)
        lower_z = float(safety_factor)
        upper_raw = mean + upper_z * std
        lower_raw = mean - lower_z * std
        method = "baseline_K_sigma"

                 # الحد الأساسي = mean ± K*std
        upper_z = float(safety_factor)
        lower_z = float(safety_factor)
        upper_raw = mean + upper_z * std
        lower_raw = mean - lower_z * std
        method = "baseline_K_sigma"

        # (removed: pre-failure adjustment — contaminated by failure buildup)

        critical_limits[sensor] = {
            "upper_z": round(float(upper_z), 2),
            "lower_z": round(float(-lower_z), 2),
            "upper_raw": round(float(upper_raw), 6),
            "lower_raw": round(float(lower_raw), 6),
            "method": method,
        }
        critical_limits[sensor] = {
            "upper_z": round(float(upper_z), 2),
            "lower_z": round(float(-lower_z), 2),
            "upper_raw": round(float(upper_raw), 6),
            "lower_raw": round(float(lower_raw), 6),
            "method": method,
        }

        reasoning_docs[sensor] = (
            f"{sensor}: upper={upper_z:.2f}σ (raw={upper_raw:.4f}), "
            f"lower={lower_z:.2f}σ (raw={lower_raw:.4f}). "
            f"Baseline: mean={mean:.4f}, std={std:.4f}. "
            f"Safety factor K={safety_factor}. Method={method}."
        )

    # --------------------------------------------------------
    # طباعة النتائج
    # --------------------------------------------------------
    print("\n--- النتائج النهائية الموثقة ---")
    for sensor, reasoning in reasoning_docs.items():
        print(f"- {reasoning}")

    # --------------------------------------------------------
    # 4) حفظ الإعدادات في config_limits.py
    # --------------------------------------------------------
    config_content = f"""# تم التوليد تلقائياً لمهمة Critical limits per sensor (المهمة 2.5)
# Safety factor K = {safety_factor} | Pre-failure window = {pre_failure_minutes} min
CRITICAL_LIMITS = {json.dumps(critical_limits, indent=4, ensure_ascii=False)}

# التوثيق والأسباب الفنية (Rationale):
REASONING = {json.dumps(reasoning_docs, indent=4, ensure_ascii=False)}
"""

    with open(config_output_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    print("\n" + "=" * 70)
    print(f"4. ✅ تم حفظ الإعدادات والحدود بنجاح في: {config_output_path}")
    print("=" * 70)


# ============================================================
# نقطة الدخول
# ============================================================
if __name__ == "__main__":
    compute_and_save_critical_limits(
        failure_events_clean_name="failure_events_clean (2).csv",
        source_name="source.csv",
        baseline_name="sensor_baseline.csv",
        config_output_path="config_limits.py",
        safety_factor=4.0,
        pre_failure_minutes=30,
    )