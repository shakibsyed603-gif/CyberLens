import joblib
from config import PROCESSED_DATA_DIR
import os

processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")

data = joblib.load(processed_data_path)

print("\n=== FEATURE COLUMNS ===\n")
for col in data["feature_columns"]:
    print(col)
