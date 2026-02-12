import logging
import os
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import feature_engineering
from train import train_model
from config import DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def run_pipeline():
    # Step 1 — Load data
    df = load_data(DATA_PATH)

    # Step 2 — Preprocess
    df = preprocess(df)

    # Step 3 — Feature engineering
    df = feature_engineering(df)

    # Step 4 — Train model
    df_combined = train_model(df, return_dataset=True)

    # Step 5 — Save combined preprocessed dataset (without is_fraud)
    os.makedirs("artifacts", exist_ok=True)
    preprocessed_path = "artifacts/preprocessed_dataset.csv"
    df_combined.to_csv(preprocessed_path, index=False)
    logging.info(f"Preprocessed dataset saved at: {preprocessed_path}")

if __name__ == "__main__":
    run_pipeline()
