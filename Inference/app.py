from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, request, jsonify

from inference import predict_new_claim
from api_client import fetch_claim_history

# -------------------------
# Project root setup
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FraudAPI")

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

REQUIRED_FIELDS = [
    "policy_id",
    "insured_id",
    "hospital_id",
    "checkin_date",
    "cover_amount",
    "check_option"
]

# -------------------------
# /predict endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
    data = request.get_json()
    logger.info("Incoming request: %s", data)

    # -------------------------
    # INPUT VALIDATION
    # -------------------------
    if not data:
        logger.warning("No input data provided")
        return jsonify({"status": "fail", "message": "No input data provided"}), 400

    if not isinstance(data, dict):
        logger.warning("Invalid format: expected single JSON object")
        return jsonify({"status": "fail", "message": "Send single JSON object"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in [None, "", " "]]
    if missing:
        logger.warning("Missing required fields: %s", missing)
        return jsonify({
            "status": "fail",
            "message": "Missing required fields",
            "missing_fields": missing
        }), 400

    # -------------------------
    # INSURED-POLICY VALIDATION
    # -------------------------
    policy_id = str(data["policy_id"]).strip()
    insured_id = str(data.get("insured_id", "")).strip()
    expected_prefix = f"{policy_id}-"
    if not insured_id.startswith(expected_prefix):
        logger.warning("Insured ID '%s' does not match Policy ID '%s'", insured_id, policy_id)
        return jsonify({
            "status": "fail",
            "message": "Insured ID does not belong to given Policy ID"
        }), 400

    # -------------------------
    # FETCH POLICY HISTORY
    # -------------------------
    try:
        history_df = fetch_claim_history(policy_id)
        logger.info("Policy history fetched: %d records", len(history_df))
    except ValueError as ve:
        logger.error("Policy not found: %s", ve)
        return jsonify({"status": "fail", "message": str(ve)}), 404
    except RuntimeError as re:
        logger.error("API/login failure: %s", re)
        return jsonify({"status": "error", "message": str(re)}), 502
    except Exception as e:
        logger.exception("Unexpected error fetching policy history")
        return jsonify({"status": "error", "message": f"Unexpected API error: {str(e)}"}), 500

    # -------------------------
    # PREDICTION
    # -------------------------
    try:
        new_claim_df = pd.DataFrame([data])
        result_df = predict_new_claim(history_df, new_claim_df)
        response = result_df[[
            "fraud_prob",
            "fraud_flag",
            "anomaly_score",
            "is_sus",
            "reason_codes"
        ]].to_dict(orient="records")

        elapsed = time.time() - start_time
        logger.info("Prediction completed in %.3f seconds", elapsed)

        return jsonify({"status": "success", "predictions": response})

    except Exception as e:
        logger.exception("Model inference failed")
        return jsonify({"status": "error", "message": f"Model inference failed: {str(e)}"}), 500


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)