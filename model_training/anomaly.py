import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

def generate_anomaly_scores(df: pd.DataFrame, contamination: float = 0.05):
    """
    Train Isolation Forest on numeric features only and return scores and threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input training dataframe (numeric + encoded categorical features).
    contamination : float
        Expected proportion of anomalies in the dataset.

    Returns
    -------
    model : IsolationForest
        Trained anomaly detection model.
    scores : np.ndarray
        Decision function scores for the same dataset (higher = normal).
    threshold : float
        Threshold to define suspicious cases (scores below this = suspicious).
    """

    logger.info("Preparing numeric features for anomaly detection...")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns found for anomaly detection!")

    logger.info(f"Training Isolation Forest on {numeric_df.shape[1]} numeric features...")

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    model.fit(numeric_df)

    # Decision function (higher score = more normal)
    scores = model.decision_function(numeric_df)

    # Threshold to flag anomalies
    threshold = np.percentile(scores, 100 * contamination)
    logger.info(f"Anomaly threshold set at: {threshold:.4f}")

    return model, scores, threshold
