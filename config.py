import os

# Data configuration
DATA_DIR = "data/raw"
MODEL_DIR = "models"
PROCESSED_DATA_DIR = "data/processed"

# Model configuration
MODEL_CONFIG = {
    'contamination': 0.1,  # Expected proportion of anomalies
    'n_estimators': 100,
    'random_state': 42,
    'max_samples': 'auto',
    'max_features': 1.0
}

# Feature columns to exclude from training
EXCLUDE_COLUMNS = [
    'Label', 'is_anomaly', 'difficulty'
]

# UI configuration
UI_CONFIG = {
    'page_title': "CyberLens",
    'page_icon': "🔐",
    'layout': "wide",
    'max_anomalies_display': 100
}

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
