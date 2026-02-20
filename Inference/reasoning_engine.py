import numpy as np


def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0


def generate_reasoning(row):
    """
    Production-grade reasoning generator.
    Works with API-based data + model outputs.
    """

    fraud_flag = row.get("fraud_flag", 0)
    anomaly_flag = row.get("is_sus", 0)

    # ---------------------------------------------------
    # HARD NORMAL GUARD (NEW FIX)
    # ---------------------------------------------------
    if fraud_flag == 0 and anomaly_flag == 0:
        return ["Claim appears normal based on history and model signals"]

    reasons = []

    fraud_prob = safe_float(row.get("fraud_prob", 0))

    cover_amount = safe_float(row.get("cover_amount", 0))
    policy_avg = safe_float(row.get("policy_avg_claim", 0))
    policy_claims = safe_float(row.get("policy_total_claims", 0))
    user_claims = safe_float(row.get("user_total_claims", 0))
    user_max = safe_float(row.get("user_max_claim", 0))
    hospital_claims = safe_float(row.get("hospital_total_claims", 0))

    insured_cancel = safe_float(row.get("insured_cancel_count", 0))
    hospital_cancel = safe_float(row.get("hospital_cancel_count", 0))

    anomaly_score = safe_float(row.get("anomaly_score", 0))
    policy_velocity = safe_float(row.get("policy_claim_velocity", 0))


    # ---------------------------------------------------
    # MODEL-DRIVEN SIGNALS
    # ---------------------------------------------------

    if fraud_flag == 1:

        if fraud_prob >= 0.85:
            reasons.append("Very high fraud probability detected by model")

        elif fraud_prob >= 0.70:
            reasons.append("Strong fraud pattern detected from historical behaviour")

        elif fraud_prob >= 0.60:
            reasons.append("Moderate fraud risk based on model prediction")

    if anomaly_flag == 1:
        if anomaly_score < -0.08:
            reasons.append("Severe deviation from normal claim behaviour")

        elif anomaly_score < -0.04:
            reasons.append("Claim behaviour deviates from past patterns")


    # ---------------------------------------------------
    # CLAIM AMOUNT BEHAVIOUR
    # ---------------------------------------------------

    if policy_avg > 0:

        if cover_amount > 2 * policy_avg:
            reasons.append("Claim amount significantly higher than policy average")

        elif cover_amount > policy_avg:
            reasons.append("Claim amount higher than usual policy pattern")


    # ---------------------------------------------------
    # USER BEHAVIOUR
    # ---------------------------------------------------

    if user_claims >= 5:
        reasons.append("User has frequent claim history")

    if user_claims >= 2 and user_max > 600000:
        reasons.append("User previously made high-value claims")

    if insured_cancel >= 2:
        reasons.append("User has repeated claim cancellations")


    # ---------------------------------------------------
    # HOSPITAL BEHAVIOUR
    # ---------------------------------------------------

    if hospital_claims >= 15:
        reasons.append("Hospital frequently appears in claims data")

    if hospital_cancel >= 4:
        reasons.append("Hospital shows elevated cancellation pattern")


    # ---------------------------------------------------
    # POLICY BEHAVIOUR
    # ---------------------------------------------------

    if policy_claims >= 50:
        reasons.append("Policy has unusually high claim volume")

    if policy_velocity > 1.5:
        reasons.append("Claims occurring rapidly within policy duration")


    # ---------------------------------------------------
    # ANOMALY CONTEXT
    # ---------------------------------------------------

    if anomaly_flag == 1 and policy_velocity > 1:
        reasons.append("Anomalous claim timing pattern observed")


    # ---------------------------------------------------
    # FINAL FALLBACK CONTROL
    # ---------------------------------------------------

    if not reasons:
        reasons.append("Flagged by model due to hidden risk patterns")

    return reasons