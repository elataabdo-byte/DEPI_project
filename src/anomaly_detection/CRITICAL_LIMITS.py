import os
import pandas as pd
import numpy as np
import json

def find_file_path(filename):
    possible_dirs = [
        r"C:\DEPI_project-main\DEPI_project-main\src\anomaly_detection",
        r"C:\DEPI_project-main\DEPI_project-main\src\data_processing",
        os.getcwd()
    ]
    
    for directory in possible_dirs:
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return full_path
            
    return filename 
def compute_and_save_critical_limits(
    failure_events_clean_name="failure_events_clean.csv",
    source_name="source.csv",
    baseline_name="sensor_baseline.csv",
    config_output_path="config_limits.py"
):
    print("جاري البحث عن الملفات وتحميلها...")
    
    failure_events_clean_path = find_file_path(failure_events_clean_name)
    source_path = find_file_path(source_name)
    baseline_path = find_file_path(baseline_name)
    
    print(f"📁 المسارات التي تم رصدها:")
    print(f"  - Failure Events: {failure_events_clean_path}")
    print(f"  - Source Data: {source_path}")
    print(f"  - Baseline: {baseline_path}")

    try:
        df_failures = pd.read_csv(failure_events_clean_path)
        df_data = pd.read_csv(source_path)
        df_baseline = pd.read_csv(baseline_path)
    except FileNotFoundError as e:
        print(f"خطأ: لم يتم العثور على أحد الملفات المطلوبة -> {e}")
        return

    baseline_dict = df_baseline.set_index('sensor')[['mean', 'std']].to_dict('index')
    sensors = list(baseline_dict.keys())
    
    print("\n2. فحص قيم الحساسات قبل الأعطال مباشرة...")
    sensor_max_values = {sensor: [] for sensor in sensors}

    for _, row in df_failures.iterrows():
        start_time = pd.to_datetime(row['start_time'])
        
        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
        pre_failure_data = df_data[
            (df_data['timestamp'] < start_time) & 
            (df_data['timestamp'] >= start_time - pd.Timedelta(minutes=30))
        ]
        
        if pre_failure_data.empty:
            continue
            
        for sensor in sensors:
            if sensor in pre_failure_data.columns:
                max_val = pre_failure_data[sensor].max()
                sensor_max_values[sensor].append(max_val)

    print("3. اختيار الحدود الحرجة وتحويلها إلى Z-scores وتوثيق الأسباب...")
    critical_limits = {}
    reasoning_docs = {}

    for sensor in sensors:
        if sensor_max_values[sensor]:
            raw_critical_limit = np.percentile(sensor_max_values[sensor], 90)
        else:
            raw_critical_limit = baseline_dict[sensor]['mean'] + (4 * baseline_dict[sensor]['std'])

        mean = baseline_dict[sensor]['mean']
        std = baseline_dict[sensor]['std']
        
        if std > 0:
            z_score_limit = (raw_critical_limit - mean) / std
        else:
            z_score_limit = 4.0

        critical_limits[sensor] = round(float(z_score_limit), 2)
        
        reasoning_docs[sensor] = (
            f"The critical limit for {sensor} is {critical_limits[sensor]} (Z-score), "
            f"calculated based on values observed before failures (Mean={mean:.2f}, Std={std:.2f})."
        )

    print("\n--- النتائج النهائية الموثقة ---")
    for sensor, reasoning in reasoning_docs.items():
        print(f"- {sensor}: {reasoning}")

    config_content = f"""# تم التوليد تلقائياً لمهمة Critical limits per sensor (المهمة 2.5)
CRITICAL_LIMITS = {json.dumps(critical_limits, indent=4)}

# التوثيق والأسباب الفنية (Rationale):
REASONING = {json.dumps(reasoning_docs, indent=4)}
"""
    
    with open(config_output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
        
    print(f"\n4. تم حفظ الإعدادات والحدود بنجاح في الملف: {config_output_path}")

if __name__ == "__main__":
    compute_and_save_critical_limits()