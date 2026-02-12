from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import joblib
import logging
import os
import pandas as pd
import json

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    SCALER_PATH,
    ANOMALY_MODEL_PATH,
    SPLIT_STRATEGY,
    TIME_SPLIT_COL,
    TARGET_RECALL,
    VALIDATION_SIZE,
    OPTIMIZE_METRIC,
    ANOMALY_CONTAMINATION,
)
from anomaly import generate_anomaly_scores

logger = logging.getLogger(__name__)

def train_model(df: pd.DataFrame, return_dataset: bool = True) -> pd.DataFrame:
    logger.info("Starting training pipeline...")

    df_original = df.copy()

    # -----------------------------
    # 1. Drop leakage / non-usable columns
    # -----------------------------
    drop_cols = [
        "reference_no",
        "policy_id",
        "full_name",
        "disease_name",
        "checkin_date",
        "checkout_date",
        "checkout_status_name",
        "checkout_status",  # proxy for label in many datasets; exclude from training + saved features
        "hospital_name",  # removed completely
        "final_claim_payment",  # not available at intake; avoid future-looking leakage
        "claim_ratio",  # derived from final_claim_payment; not available at intake
        "hospital_cancel_rate"  # legacy column if present
    ]
    df_model = df.drop(columns=drop_cols, errors="ignore").copy()

    # -----------------------------
    # 2. Target variable
    # -----------------------------
    if "is_fraud" not in df_model.columns:
        raise ValueError("Column 'is_fraud' not found in dataset.")
    y = df_model.pop("is_fraud")

    # Basic label distribution
    logger.info(f"Label distribution (overall): fraud_rate={float(y.mean()):.4f} ({int(y.sum())}/{len(y)})")

    # -----------------------------
    # 3. Train-test split FIRST (configurable)
    # -----------------------------
    X = df_model.copy()

    # Maps for inference-time feature construction (derived from training data)
    feature_maps = {}
    for group_col in ["hospital_id", "insured_id", "company"]:
        if group_col in X.columns:
            feature_maps[f"{group_col}_claim_count"] = X[group_col].value_counts(dropna=False).to_dict()

            # FIX: group using y separately (since y is popped from X)
            cancel_df = pd.DataFrame({
                group_col: X[group_col],
                "is_fraud": y
            })
            feature_maps[f"{group_col}_cancel_count"] = (
                cancel_df.groupby(group_col)["is_fraud"].sum().to_dict()
            )


    split_strategy = SPLIT_STRATEGY
    if split_strategy not in {"stratified", "group_insured", "group_hospital", "time"}:
        logger.warning(f"Unknown SPLIT_STRATEGY='{SPLIT_STRATEGY}', falling back to 'stratified'.")
        split_strategy = "stratified"

    if split_strategy == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    elif split_strategy in {"group_insured", "group_hospital"}:
        group_col = "insured_id" if split_strategy == "group_insured" else "hospital_id"
        if group_col not in X.columns:
            raise ValueError(f"Group split requested but column '{group_col}' not present in features.")

        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        groups = X[group_col]
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    else:  # time
        if TIME_SPLIT_COL not in df_original.columns:
            raise ValueError(
                f"Time split requested but '{TIME_SPLIT_COL}' not present in original dataset. "
                "Set TIME_SPLIT_COL or use SPLIT_STRATEGY='stratified'."
            )

        time_series = pd.to_datetime(df_original.loc[X.index, TIME_SPLIT_COL], errors="coerce")
        if time_series.isna().all():
            raise ValueError(f"Time split requested but '{TIME_SPLIT_COL}' could not be parsed as datetime.")

        order = time_series.sort_values().index
        cut = int(round((1.0 - TEST_SIZE) * len(order)))
        train_index = order[:cut]
        test_index = order[cut:]

        X_train, X_test = X.loc[train_index].copy(), X.loc[test_index].copy()
        y_train, y_test = y.loc[train_index].copy(), y.loc[test_index].copy()

    logger.info(
        f"Split strategy: {split_strategy} | "
        f"Train shape: {X_train.shape}, Test shape: {X_test.shape} | "
        f"fraud_rate_train={float(y_train.mean()):.4f}, fraud_rate_test={float(y_test.mean()):.4f}"
    )

    # -----------------------------
    # 3b. Validation split (for choosing an 'optimal' threshold)
    # -----------------------------
    val_size = float(VALIDATION_SIZE)
    if not (0.0 < val_size < 1.0):
        logger.warning(f"VALIDATION_SIZE={VALIDATION_SIZE} invalid; falling back to 0.2")
        val_size = 0.2

    if split_strategy in {"group_insured", "group_hospital"}:
        group_col = "insured_id" if split_strategy == "group_insured" else "hospital_id"
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=RANDOM_STATE)
        groups_val = X_train[group_col]
        fit_idx, val_idx = next(splitter_val.split(X_train, y_train, groups=groups_val))
        X_fit_raw, X_val_raw = X_train.iloc[fit_idx].copy(), X_train.iloc[val_idx].copy()
        y_fit, y_val = y_train.iloc[fit_idx].copy(), y_train.iloc[val_idx].copy()
    elif split_strategy == "time":
        time_series_train = pd.to_datetime(df_original.loc[X_train.index, TIME_SPLIT_COL], errors="coerce")
        order_train = time_series_train.sort_values().index
        cut_train = int(round((1.0 - val_size) * len(order_train)))
        fit_index = order_train[:cut_train]
        val_index = order_train[cut_train:]
        X_fit_raw, X_val_raw = X_train.loc[fit_index].copy(), X_train.loc[val_index].copy()
        y_fit, y_val = y_train.loc[fit_index].copy(), y_train.loc[val_index].copy()
    else:
        X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

    logger.info(
        f"Validation split: fit={X_fit_raw.shape}, val={X_val_raw.shape} | "
        f"fraud_rate_fit={float(y_fit.mean()):.4f}, fraud_rate_val={float(y_val.mean()):.4f}"
    )

    def _encode_fit_transform(X_fit: pd.DataFrame, X_other_list: list[pd.DataFrame]):
        cat_cols_local = X_fit.select_dtypes(include=["object"]).columns
        encoders_local: dict[str, LabelEncoder] = {}
        unknown_token_local = "__UNK__"
        for col in cat_cols_local:
            le = LabelEncoder()
            fit_vals = X_fit[col].astype(str)
            le.fit(pd.Index(fit_vals.unique()).append(pd.Index([unknown_token_local])))
            X_fit[col] = le.transform(fit_vals)
            for Xo in X_other_list:
                ov = Xo[col].astype(str)
                ov = ov.where(ov.isin(le.classes_), other=unknown_token_local)
                Xo[col] = le.transform(ov)
            encoders_local[col] = le
        return encoders_local

    def _add_cancel_count(
        X_fit: pd.DataFrame,
        y_fit_local: pd.Series,
        X_other_list: list[pd.DataFrame],
        group_col: str,
        out_col: str,
    ):
        tmp = pd.DataFrame({group_col: X_fit[group_col], "is_fraud": y_fit_local})
        grp_local = tmp.groupby(group_col)["is_fraud"].agg(["sum", "count"])

        # Leave-one-out cancelled count for training rows (exclude the current row)
        sum_local = X_fit[group_col].map(grp_local["sum"]).astype(float)
        loo_cancel_count = (sum_local - y_fit_local.astype(float)).clip(lower=0)
        X_fit[out_col] = loo_cancel_count.fillna(0).astype(int)

        # For other sets: cancelled count learned from fit set only
        cancel_count_map = grp_local["sum"].fillna(0)
        for Xo in X_other_list:
            Xo[out_col] = Xo[group_col].map(cancel_count_map).fillna(0).astype(int)

    # Prepare a tuning pipeline trained on FIT only, evaluated on VAL
    X_fit = X_fit_raw.copy()
    X_val = X_val_raw.copy()
    X_test_for_tuning = X_test.copy()

    _ = _encode_fit_transform(X_fit, [X_val, X_test_for_tuning])
    _add_cancel_count(X_fit, y_fit, [X_val, X_test_for_tuning], group_col="hospital_id", out_col="hospital_cancel_count")
    if "insured_id" in X_fit.columns:
        _add_cancel_count(X_fit, y_fit, [X_val, X_test_for_tuning], group_col="insured_id", out_col="insured_cancel_count")
    if "company" in X_fit.columns:
        _add_cancel_count(X_fit, y_fit, [X_val, X_test_for_tuning], group_col="company", out_col="company_cancel_count")

    scaler_tune = StandardScaler()
    X_fit_scaled = scaler_tune.fit_transform(X_fit)
    X_val_scaled = scaler_tune.transform(X_val)

    # Class imbalance handling
    pos_fit = float(y_fit.sum())
    neg_fit = float(len(y_fit) - y_fit.sum())
    scale_pos_weight_fit = (neg_fit / pos_fit) if pos_fit > 0 else 1.0

    model_tune = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight_fit,
        random_state=RANDOM_STATE
    )
    model_tune.fit(X_fit_scaled, y_fit)

    val_probs = model_tune.predict_proba(X_val_scaled)[:, 1]
    pr_p, pr_r, pr_t = precision_recall_curve(y_val, val_probs)

    chosen_threshold_opt = 0.5
    if len(pr_t) > 0:
        best_score = -1.0
        # thresholds align to precision/recall entries starting from index 1
        for i, thr in enumerate(pr_t, start=1):
            pred_i = (val_probs >= thr).astype(int)
            if OPTIMIZE_METRIC == "f1":
                score = f1_score(y_val, pred_i, zero_division=0)
            else:
                score = f1_score(y_val, pred_i, zero_division=0)
            if score > best_score:
                best_score = float(score)
                chosen_threshold_opt = float(thr)

    logger.info(f"Chosen optimal threshold (maximize {OPTIMIZE_METRIC} on VAL): {chosen_threshold_opt:.4f}")

    # -----------------------------
    # 4. Encode categorical columns (fit on TRAIN only)
    # -----------------------------
    label_encoders = _encode_fit_transform(X_train, [X_test])

    # -----------------------------
    # 5. Compute cancel counts (leave-one-out on TRAIN, train-derived mapping for TEST)
    # -----------------------------
    _add_cancel_count(X_train, y_train, [X_test], group_col="hospital_id", out_col="hospital_cancel_count")
    if "insured_id" in X_train.columns:
        _add_cancel_count(X_train, y_train, [X_test], group_col="insured_id", out_col="insured_cancel_count")
    if "company" in X_train.columns:
        _add_cancel_count(X_train, y_train, [X_test], group_col="company", out_col="company_cancel_count")

    # -----------------------------
    # 6. Train anomaly model on TRAIN only
    # -----------------------------
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ensure target or accidental leakage cols not included
    for col in ["is_fraud"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    logger.info(f"Preparing numeric features for anomaly detection... ({len(numeric_cols)} features)")
    anomaly_model, train_scores, threshold = generate_anomaly_scores(
        X_train[numeric_cols],
        contamination=ANOMALY_CONTAMINATION
    )
    test_scores = anomaly_model.decision_function(X_test[numeric_cols])

    X_train["anomaly_score"] = train_scores
    X_test["anomaly_score"] = test_scores
    X_train["is_sus"] = (X_train["anomaly_score"] < threshold).astype(int)
    X_test["is_sus"] = (X_test["anomaly_score"] < threshold).astype(int)

    logger.info(f"Suspicious cases in TRAIN: {X_train['is_sus'].sum()}")
    logger.info(f"Suspicious cases in TEST: {X_test['is_sus'].sum()}")

    # -----------------------------
    # 7. Fraud model features (exclude anomaly outputs)
    # -----------------------------
    X_train_fraud = X_train.drop(columns=["anomaly_score", "is_sus"], errors="ignore")
    X_test_fraud = X_test.drop(columns=["anomaly_score", "is_sus"], errors="ignore")

    feature_columns = [str(c).strip() for c in X_train_fraud.columns]
    anomaly_feature_columns = [str(c).strip() for c in numeric_cols]


    # -----------------------------
    # 8. Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fraud)
    X_test_scaled = scaler.transform(X_test_fraud)

    # -----------------------------
    # 9. Train XGBoost
    # -----------------------------
    pos_train = float(y_train.sum())
    neg_train = float(len(y_train) - y_train.sum())
    scale_pos_weight_train = (neg_train / pos_train) if pos_train > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight_train,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    # -----------------------------
    # 10. Evaluation
    # -----------------------------
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Default threshold (0.5)
    preds_05 = (probs >= 0.5).astype(int)

    # Optimal threshold chosen on validation (maximize F1)
    preds_opt = (probs >= chosen_threshold_opt).astype(int)

    # Recall-tuned threshold: maximize precision subject to recall >= TARGET_RECALL
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, probs)
    chosen_threshold = 0.5
    if len(pr_thresholds) > 0:
        candidate = []
        # thresholds align to precision/recall entries starting from index 1
        for i, thr in enumerate(pr_thresholds, start=1):
            if pr_recall[i] >= TARGET_RECALL:
                candidate.append((pr_precision[i], thr))
        if candidate:
            candidate.sort(reverse=True)  # highest precision
            chosen_threshold = float(candidate[0][1])

    preds_tuned = (probs >= chosen_threshold).astype(int)
    def _metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    roc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))

    m05 = _metrics(y_test, preds_05)
    mopt = _metrics(y_test, preds_opt)
    mt = _metrics(y_test, preds_tuned)

    logger.info("--------------------------------------------------")
    logger.info("MODEL TRAINING VALIDATION SUMMARY")
    logger.info("--------------------------------------------------")
    logger.info(f"ROC-AUC: {roc:.3f} | PR-AUC: {pr_auc:.3f}")
    logger.info(
        f"@thr=0.50 | Acc={m05['accuracy']*100:.2f}% "
        f"Prec={m05['precision']:.3f} Rec={m05['recall']:.3f} F1={m05['f1']:.3f} "
        f"TP={m05['tp']} FN={m05['fn']} FP={m05['fp']} TN={m05['tn']}"
    )
    logger.info(
        f"@thr={chosen_threshold_opt:.4f} (optimize {OPTIMIZE_METRIC} on VAL) | "
        f"Acc={mopt['accuracy']*100:.2f}% Prec={mopt['precision']:.3f} Rec={mopt['recall']:.3f} F1={mopt['f1']:.3f} "
        f"TP={mopt['tp']} FN={mopt['fn']} FP={mopt['fp']} TN={mopt['tn']}"
    )
    logger.info(
        f"@thr={chosen_threshold:.4f} (target_recall>={TARGET_RECALL:.2f}) | "
        f"Acc={mt['accuracy']*100:.2f}% Prec={mt['precision']:.3f} Rec={mt['recall']:.3f} F1={mt['f1']:.3f} "
        f"TP={mt['tp']} FN={mt['fn']} FP={mt['fp']} TN={mt['tn']}"
    )
    logger.info("--------------------------------------------------")

    # -----------------------------
    # 11. Save models
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    logger.info("Models saved successfully.")

    # Save feature maps and column order for inference
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "anomaly_feature_columns": anomaly_feature_columns,
            "feature_maps": feature_maps,
        },
        "models/feature_maps.pkl"
    )

    # Persist thresholds + evaluation metadata (useful for consistent inference later)
    metadata = {
        "split_strategy": split_strategy,
        "test_size": float(TEST_SIZE),
        "validation_size": float(val_size),
        "random_state": int(RANDOM_STATE),
        "fraud_rate_overall": float(y.mean()),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "anomaly": {
            "contamination": float(ANOMALY_CONTAMINATION),
            "threshold": float(threshold),
            "feature_columns": anomaly_feature_columns,
        },
        "fraud_classifier": {
            "default_threshold": 0.5,
            "optimal_threshold": float(chosen_threshold_opt),
            "optimize_metric": str(OPTIMIZE_METRIC),
            "tuned_threshold": float(chosen_threshold),
            "target_recall": float(TARGET_RECALL),
            "roc_auc": float(roc),
            "pr_auc": float(pr_auc),
            "metrics_at_0_5": m05,
            "metrics_at_optimal": mopt,
            "metrics_at_tuned": mt,
            "scale_pos_weight_train": float(scale_pos_weight_train),
            "feature_columns": feature_columns,
        },
    }
    with open("models/training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------
    # 12. Return combined dataset (optional)
    # -----------------------------
    if return_dataset:
        train_out = X_train.copy()
        test_out = X_test.copy()
        train_out["is_fraud"] = y_train
        test_out["is_fraud"] = y_test
        df_combined = pd.concat([train_out, test_out]).sort_index()
        return df_combined

    return pd.DataFrame()
