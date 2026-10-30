"""Apply inclusive failure-event intervals to timestamped sensor rows."""
from pathlib import Path
import pandas as pd

def label_failures(sensors_df: pd.DataFrame, failures_df: pd.DataFrame) -> pd.DataFrame:
    required_data = {"timestamp"}
    required_events = {"start_time", "end_time"}
    if not required_data.issubset(sensors_df.columns):
        raise ValueError(f"Sensor data missing columns: {sorted(required_data - set(sensors_df.columns))}")
    if not required_events.issubset(failures_df.columns):
        raise ValueError(f"Failure events missing columns: {sorted(required_events - set(failures_df.columns))}")
    data = sensors_df.copy()
    events = failures_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    events["start_time"] = pd.to_datetime(events["start_time"], errors="raise")
    events["end_time"] = pd.to_datetime(events["end_time"], errors="raise")
    if (events["end_time"] < events["start_time"]).any():
        raise ValueError("Every failure event must end at or after its start")
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["ground_truth"] = 0
    for _, event in events.iterrows():
        mask = data["timestamp"].between(event["start_time"], event["end_time"], inclusive="both")
        data.loc[mask, "ground_truth"] = 1
    if not data.empty and int(data["ground_truth"].sum()) == 0:
        raise ValueError("No rows were labeled; check timestamps and failure-event intervals")
    return data

def run(input_csv: str | Path, events_csv: str | Path, output_csv: str | Path) -> Path:
    input_csv, events_csv, output_csv = map(Path, (input_csv, events_csv, output_csv))
    if not input_csv.is_file(): raise FileNotFoundError(f"Imputed input not found: {input_csv.resolve()}")
    if not events_csv.is_file(): raise FileNotFoundError(f"Failure-events file not found: {events_csv.resolve()}")
    data = label_failures(pd.read_csv(input_csv), pd.read_csv(events_csv))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)
    return output_csv

def main() -> None:
    input_csv  = Path(r"C:\Users\compu import\Desktop\TEST\phase1_imputed\data_imputed.csv")
    events_csv = Path(r"C:\Users\compu import\Desktop\TEST\failure_events_clean (2).csv")

    path = run(input_csv, events_csv, input_csv.with_name(f"labeled_{input_csv.name}"))
    print(f"Saved {path.resolve()}")

if __name__ == "__main__":
    main()
