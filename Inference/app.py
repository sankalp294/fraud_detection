from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from flask import Flask, request, jsonify
import pandas as pd

from inference import predict_new_claim
from api_client import fetch_claim_history


app = Flask(__name__)


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

    # -------------------------
    # INPUT VALIDATION
    # -------------------------
    data = request.get_json()

    if not data:
        return jsonify({"status": "fail", "message": "No input data provided"}), 400

    if not isinstance(data, dict):
        return jsonify({"status": "fail", "message": "Send single JSON object"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in [None, "", " "]]
    if missing:
        return jsonify({
            "status": "fail",
            "message": "Missing required fields",
            "missing_fields": missing
        }), 400

    policy_id = data["policy_id"]

    insured_id = str(data.get("insured_id", "")).strip()
    policy_id = str(policy_id).strip()

    # Validate insured belongs to policy
    expected_prefix = f"{policy_id}-"

    if not insured_id.startswith(expected_prefix):
        return jsonify({
            "status": "fail",
            "message": "Insured ID does not belong to given Policy ID"
        }), 400

    # -------------------------
    # FETCH POLICY HISTORY
    # -------------------------
    try:
        history_df = fetch_claim_history(policy_id)

    except ValueError as ve:
        # policy not found case
        return jsonify({
            "status": "fail",
            "message": str(ve)
        }), 404

    except RuntimeError as re:
        # login / API failure
        return jsonify({
            "status": "error",
            "message": str(re)
        }), 502

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Unexpected API error: {str(e)}"
        }), 500

    # -------------------------
    # PREDICT
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

        return jsonify({"status": "success", "predictions": response})

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model inference failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)