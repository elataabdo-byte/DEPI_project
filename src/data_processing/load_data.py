
from pathlib import Path
import pandas as pd

# Locate the project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original dataset location (local computer only)
RAW_SOURCE_PATH = Path(
    r"C:\DEPI\project\archive (2)\MetroPT3(AirCompressor).csv"
)

# Local raw-data directory inside the project
OUTPUT_DIR = PROJECT_ROOT / "data_raw"
OUTPUT_PATH = OUTPUT_DIR / "data_raw.csv"

if not RAW_SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Original dataset not found: {RAW_SOURCE_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading raw data...")
df = pd.read_csv(RAW_SOURCE_PATH, low_memory=False)

print("\nRows and Columns:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# Save an unchanged copy of the raw dataset
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nRaw data saved to: {OUTPUT_PATH}")
print(f"Saved file exists: {OUTPUT_PATH.exists()}")
