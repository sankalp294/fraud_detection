import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _normalize_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "check_optio": "check_option",
        "cover_amou": "cover_amount",
        "currency_na": "currency_name",
        "disease_na": "disease_name",
        "hospital_na": "hospital_name",
        "final_clai": "final_claim_payment",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def preprocess(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:

    logger.info("Starting preprocessing...")

    df = _normalize_raw_schema(df)

    # ---------------------------------------------------
    # TARGET CREATION
    # ---------------------------------------------------
    if 'checkout_status_name' in df.columns:
        df['is_fraud'] = (
            df['checkout_status_name']
            .astype(str)
            .str.lower()
            .str.contains('cancel')
            .astype(int)
        )
    elif require_label:
        raise ValueError("'checkout_status_name' column missing")

    # ---------------------------------------------------
    # KEEP IDs AS RAW STRINGS (IMPORTANT)
    # ---------------------------------------------------
    id_cols = ["insured_id", "hospital_id", "policy_id"]

    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ---------------------------------------------------
    # NUMERIC CONVERSIONS (ONLY REAL NUMERIC FIELDS)
    # ---------------------------------------------------
    numeric_cols = [
        "cover_amount",
        "final_claim_payment",
        "currency_id"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---------------------------------------------------
    # MISSING HANDLING
    # ---------------------------------------------------
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna("UNKNOWN")

    # ---------------------------------------------------
    # DROP ONLY PURE TEXTUAL / IRRELEVANT FIELDS
    # (NOT IDs)
    # ---------------------------------------------------
    drop_cols = [
        'checkout_status_name',
        'company',
        'disease_name',
        'gender',
        'hospital_name',
        'full_name',
        'reference_no',
    ]

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    logger.info("Preprocessing completed")
    return df
