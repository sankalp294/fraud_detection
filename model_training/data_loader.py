import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {path}")


    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")


    logger.info(f"Dataset shape: {df.shape}")
    return df