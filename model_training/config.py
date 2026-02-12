import os


# Path to dataset
DATA_PATH = os.getenv("DATA_PATH", "C:\\Users\\Lenovo\\Downloads\\claim_history_dataset.csv")


# Model output paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")


# Train settings
TEST_SIZE = 0.5
RANDOM_STATE = 42

# Split strategy: 'stratified' (default), 'group_insured', 'group_hospital', 'time'
SPLIT_STRATEGY = os.getenv("SPLIT_STRATEGY", "stratified").strip().lower()

# Used when SPLIT_STRATEGY='time'
TIME_SPLIT_COL = os.getenv("TIME_SPLIT_COL", "checkout_date").strip()

# Threshold tuning for fraud classifier
TARGET_RECALL = float(os.getenv("TARGET_RECALL", "0.90"))
VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.20"))
OPTIMIZE_METRIC = os.getenv("OPTIMIZE_METRIC", "f1").strip().lower()

# Anomaly detection settings
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))