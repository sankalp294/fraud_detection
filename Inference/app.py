from __future__ import annotations
import sys
from pathlib import Path

# -------------------------
# Project root
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify
import pandas as pd

from inference import load_policy_history, predict_new_claim

app = Flask(__name__)

# Minimal input for a new claim
REQUIRED_FIELDS = [
    "policy_id",
    "insured_id",
    "hospital_id",
    "checkin_date",
    "cover_amount",
    "check_option"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Accept single object only
        if not isinstance(data, dict):
            return jsonify({"error": "Send JSON object with new claim only"}), 400

        # Validate required fields
        missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in [None, "", " "]]
        if missing:
            return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

        policy_id = data["policy_id"]

        # Load full policy history
        history_df = load_policy_history(policy_id)

        # New claim as DataFrame
        new_claim_df = pd.DataFrame([data])

        # Predict
        result_df = predict_new_claim(history_df, new_claim_df)

        # Prepare response
        response = result_df[[
            "fraud_prob",
            "fraud_flag",
            "anomaly_score",
            "is_sus",
            "reason_codes"
        ]].to_dict(orient="records")

        return jsonify({"predictions": response})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)