import numpy as np

def generate_reasoning(row):
    """
    Generate human-readable reasoning for a single new claim
    based on history-derived context features.
    """

    reasons = []

    # -------------------------
    # FRAUD-BASED REASONS
    # -------------------------
    if row.get("fraud_flag") == 1:
        # Claim high compared to policy average
        policy_avg = row.get("policy_avg_claim", 0)
        if policy_avg > 0 and row.get("cover_amount", 0) > policy_avg:
            reasons.append("Claim amount is high compared to average policy claims")

        # High user claim frequency
        user_claims = row.get("user_total_claims", 0)
        policy_claims = row.get("policy_total_claims", 1)
        if user_claims > 0 and user_claims > max(10, 2 * policy_claims):
            reasons.append("User has unusually high number of claims")

        # User history of high claims
        user_claims = float(row.get("user_total_claims", 0) or 0)
        user_max = float(row.get("user_max_claim", 0) or 0)

        if user_claims >= 2 and user_max > 500000:
            reasons.append("User has history of high claim amounts")

        # Hospital frequent usage
        if row.get("hospital_total_claims", 0) > 20:
            reasons.append("Hospital frequently used in claims")

        # Cancellation patterns
        if row.get("insured_cancel_count", 0) > 2:
            reasons.append("User cancellation history is high")
        if row.get("hospital_cancel_count", 0) > 5:
            reasons.append("Hospital cancellation rate is high")

    # -------------------------
    # ANOMALY-BASED REASONS
    # -------------------------
    if row.get("is_sus") == 1:
        anomaly_score = row.get("anomaly_score", 0)
        if anomaly_score < -0.03:
            reasons.append("Claim pattern deviates from normal history")

        policy_velocity = row.get("policy_claim_velocity", 0)
        # Use relative threshold to policy avg claims
        if policy_velocity > 2 * max(1, row.get("policy_total_claims", 1)/12):
            reasons.append("Policy shows unusually high claim frequency")

    # -------------------------
    # FALLBACK
    # -------------------------
    if not reasons:
        # Only fallback if claim not flagged
        if row.get("fraud_flag") == 1 or row.get("is_sus") == 1:
            reasons.append("Flagged due to model threshold but no strong indicators detected")
        else:
            reasons.append("Claim appears normal based on historical context")

    return reasons