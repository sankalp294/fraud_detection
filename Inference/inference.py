from __future__ import annotations

import json
import sys
import time
import logging
from pathlib import Path

import joblib
import pandas as pd

# -------------------------
# Project Root Setup
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # parent of Inference folder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Local imports
# -------------------------
from model_training.preprocessing import preprocess
from reasoning_engine import generate_reasoning

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Inference")

# -------------------------
# Paths
# -------------------------
MODELS_DIR = PROJECT_ROOT / "model_training" / "models"

MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
FEATURE_MAPS_PATH = MODELS_DIR / "feature_maps.pkl"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

# -------------------------
# Load models & metadata
# -------------------------
logger.info("Loading models and metadata...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
feature_bundle = joblib.load(FEATURE_MAPS_PATH) if FEATURE_MAPS_PATH.exists() else {}

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_columns = [str(c).strip() for c in metadata["fraud_classifier"]["feature_columns"]]
anomaly_feature_columns = [str(c).strip() for c in metadata["anomaly"]["feature_columns"]]
# fraud_threshold = float(metadata["fraud_classifier"]["optimal_threshold"]) # use metadata threshold for better model performance, but reasoning may be less consistent
fraud_threshold = float(0.6)  # Using fixed threshold for better reasoning consistency
anomaly_threshold = float(metadata["anomaly"]["threshold"])
logger.info("Models loaded successfully.")

# -------------------------
# Compute context features
# -------------------------
def compute_context_features(history_df: pd.DataFrame, new_claim_df: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()
    new_claim = new_claim_df.copy()
    logger.info("Computing context features for new claim...")

    # -------------------------
    # Policy-level features
    # -------------------------
    new_claim["cover_amount"] = pd.to_numeric(new_claim["cover_amount"], errors="coerce")
    new_claim["policy_total_claims"] = len(history_df)
    new_claim["policy_avg_claim"] = history_df["cover_amount"].astype(float).mean()

    history_df["checkin_date"] = pd.to_datetime(history_df["checkin_date"], errors="coerce")
    duration_months = max(((history_df["checkin_date"].max() - history_df["checkin_date"].min()).days) / 30, 1)
    new_claim["policy_claim_velocity"] = new_claim["policy_total_claims"] / duration_months

    # -------------------------
    # User-level features
    # -------------------------
    user_id = new_claim["insured_id"].iloc[0]
    user_history = history_df[history_df["insured_id"] == user_id]
    new_claim["user_total_claims"] = len(user_history)
    new_claim["user_max_claim"] = user_history["cover_amount"].astype(float).max() if len(user_history) > 0 else 0
    new_claim["insured_cancel_count"] = (user_history["checkout_status_name"] == "Cancelled").sum() if len(user_history) > 0 else 0

    # -------------------------
    # Hospital-level features
    # -------------------------
    hospital_id = str(new_claim["hospital_id"].iloc[0]).strip()
    history_df["hospital_id"] = history_df["hospital_id"].astype(str).str.strip()
    hospital_history = history_df[history_df["hospital_id"] == hospital_id]

    new_claim["hospital_total_claims"] = len(hospital_history)
    new_claim["hospital_cancel_count"] = (hospital_history["checkout_status_name"] == "Cancelled").sum() if len(hospital_history) > 0 else 0

    elapsed = time.time() - start_time
    logger.info("Context features computed | Time taken: %.3f sec", elapsed)
    logger.info(
        "Policy claims: %d | User claims: %d | Hospital claims: %d",
        new_claim["policy_total_claims"].iloc[0],
        new_claim["user_total_claims"].iloc[0],
        new_claim["hospital_total_claims"].iloc[0]
    )
    return new_claim

# -------------------------
# Predict fraud/anomaly
# -------------------------
def predict_new_claim(history_df: pd.DataFrame, new_claim_df: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()
    logger.info("Starting prediction for new claim...")
    
    # Attach context
    claim_with_context = compute_context_features(history_df, new_claim_df)

    # Preprocess new claim only
    claim_with_context = preprocess(claim_with_context, require_label=False)

    # Encode categoricals safely
    unknown_token = "__UNK__"
    cat_cols = claim_with_context.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col in label_encoders:
            le = label_encoders[col]
            vals = claim_with_context[col].astype(str)
            if unknown_token in le.classes_:
                vals = vals.where(vals.isin(le.classes_), other=unknown_token)
            else:
                fallback = "Unknown" if "Unknown" in le.classes_ else le.classes_[0]
                vals = vals.where(vals.isin(le.classes_), other=fallback)
            claim_with_context[col] = le.transform(vals)
        else:
            claim_with_context[col] = 0

    # Align features
    X = claim_with_context.reindex(columns=feature_columns, fill_value=0)
    logger.info("Feature alignment checked: %s", list(X.columns) == feature_columns)

    # Scale
    X_scaled = scaler.transform(X)

    # Fraud prediction
    fraud_prob = model.predict_proba(X_scaled)[:, 1]
    fraud_flag = (fraud_prob >= fraud_threshold).astype(int)

    # Anomaly prediction
    anomaly_input = X.reindex(columns=anomaly_feature_columns, fill_value=0)
    anomaly_scores = anomaly_model.decision_function(anomaly_input)
    is_sus = (anomaly_scores < anomaly_threshold).astype(int)

    # Attach results
    claim_with_context["fraud_prob"] = fraud_prob
    claim_with_context["fraud_flag"] = fraud_flag
    claim_with_context["fraud_threshold"] = fraud_threshold
    claim_with_context["anomaly_score"] = anomaly_scores
    claim_with_context["is_sus"] = is_sus

    # Generate reasoning
    claim_with_context["reason_codes"] = claim_with_context.apply(generate_reasoning, axis=1)

    elapsed = time.time() - start_time
    logger.info("Prediction completed | Time taken: %.3f sec", elapsed)
    return claim_with_context