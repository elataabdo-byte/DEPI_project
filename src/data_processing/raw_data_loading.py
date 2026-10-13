
from pathlib import Path
import pandas as pd

# Project root: Sentinel/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = Path(
    r"C:\DEPI\project\archive (2)\MetroPT3(AirCompressor).csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data_raw"
OUTPUT_PATH = OUTPUT_DIR / "data_raw.csv"


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}"
        )

    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)

    print("\nRows and Columns:", df.shape)
    print("\nColumn Names:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nFirst 5 Rows:\n", df.head())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        print("\nTimestamp range:", ts.min(), "->", ts.max())
        print("Invalid timestamps:", ts.isna().sum())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving raw data...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved successfully:", OUTPUT_PATH)
    print("Saved file exists:", OUTPUT_PATH.exists())


if __name__ == "__main__":
    main()
