import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting feature engineering...")

    df = df.copy()

    # ---------------------------------------------------
    # USER FEATURES
    # ---------------------------------------------------
    if "insured_id" in df.columns:

        user_group = df.groupby("insured_id", dropna=False)

        df["user_total_claims"] = user_group["final_claim_payment"].transform("count")
        df["user_avg_claim"] = user_group["final_claim_payment"].transform("mean")
        df["user_max_claim"] = user_group["final_claim_payment"].transform("max")

        if "is_fraud" in df.columns:
            df["insured_cancel_count"] = user_group["is_fraud"].transform("sum")
        else:
            df["insured_cancel_count"] = 0

    else:
        df["user_total_claims"] = 0
        df["user_avg_claim"] = 0
        df["user_max_claim"] = 0
        df["insured_cancel_count"] = 0


    # ---------------------------------------------------
    # HOSPITAL FEATURES
    # ---------------------------------------------------
    if "hospital_id" in df.columns:

        hospital_group = df.groupby("hospital_id", dropna=False)

        df["hospital_total_claims"] = hospital_group["final_claim_payment"].transform("count")
        df["hospital_avg_claim"] = hospital_group["final_claim_payment"].transform("mean")
        df["hospital_max_claim"] = hospital_group["final_claim_payment"].transform("max")

        if "is_fraud" in df.columns:
            df["hospital_cancel_count"] = hospital_group["is_fraud"].transform("sum")
        else:
            df["hospital_cancel_count"] = 0

    else:
        df["hospital_total_claims"] = 0
        df["hospital_avg_claim"] = 0
        df["hospital_max_claim"] = 0
        df["hospital_cancel_count"] = 0


    # # ---------------------------------------------------
    # # POLICY FEATURES
    # # ---------------------------------------------------
    # if "policy_id" in df.columns:

    #     policy_group = df.groupby("policy_id", dropna=False)
    #     df["policy_claim_frequency"] = policy_group["final_claim_payment"].transform("count")

    # else:
    #     df["policy_claim_frequency"] = 0

    # ================= POLICY LEVEL FEATURES =================

    if "policy_id" in df.columns:

        df["policy_total_claims"] = df.groupby("policy_id")["policy_id"].transform("count")

        df["policy_avg_claim"] = df.groupby("policy_id")["cover_amount"].transform("mean")

        # velocity = claims per month
        df["checkin_date"] = pd.to_datetime(df["checkin_date"], errors="coerce")

        first_claim_date = df.groupby("policy_id")["checkin_date"].transform("min")
        last_claim_date  = df.groupby("policy_id")["checkin_date"].transform("max")

        policy_duration_months = (
            (last_claim_date - first_claim_date).dt.days / 30
        ).replace(0, 1)

        df["policy_claim_velocity"] = df["policy_total_claims"] / policy_duration_months



    # ---------------------------------------------------
    # RATIO FEATURES
    # ---------------------------------------------------
    if "cover_amount" in df.columns and "final_claim_payment" in df.columns:
        df["payment_to_cover_ratio"] = df["final_claim_payment"] / (df["cover_amount"] + 1)
    else:
        df["payment_to_cover_ratio"] = 0


    # ---------------------------------------------------
    # CLEAN NAN AFTER AGGREGATION
    # ---------------------------------------------------
    agg_cols = [
        "user_total_claims",
        "user_avg_claim",
        "user_max_claim",
        "insured_cancel_count",
        "hospital_total_claims",
        "hospital_avg_claim",
        "hospital_max_claim",
        "hospital_cancel_count",
        # "policy_claim_frequency",
        "payment_to_cover_ratio"
    ]

    for col in agg_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info("Feature engineering completed")
    return df
