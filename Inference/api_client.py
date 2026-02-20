from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# -------------------------
# Project root and dotenv
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("APIClient")

# -------------------------
# Environment variables
# -------------------------
LOGIN_URL = os.getenv("API_LOGIN_URL")
CLAIM_HISTORY_URL = os.getenv("API_CLAIM_HISTORY_URL")
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")

missing_env = [
    name for name, value in {
        "API_LOGIN_URL": LOGIN_URL,
        "API_CLAIM_HISTORY_URL": CLAIM_HISTORY_URL,
        "API_USERNAME": USERNAME,
        "API_PASSWORD": PASSWORD
    }.items() if not value
]

if missing_env:
    raise RuntimeError(f"Missing environment variables: {missing_env}")

# -------------------------
# Token cache
# -------------------------
TOKEN_CACHE = {"token": None, "created_at": None}
TOKEN_VALIDITY_SECONDS = 50 * 60  # 50 minutes

# -------------------------
# LOGIN
# -------------------------
def login_and_get_token() -> str:
    start_time = time.time()
    payload = {"username": USERNAME, "password": PASSWORD}
    logger.info("Logging in to fetch JWT token")

    try:
        response = requests.post(LOGIN_URL, data=payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Login failed: {response.text}")

        data = response.json()
        token = data.get("token")
        if not token:
            raise RuntimeError("Token missing in login response")

        TOKEN_CACHE["token"] = token
        TOKEN_CACHE["created_at"] = time.time()
        elapsed = time.time() - start_time
        logger.info("Login successful | Time taken: %.3f sec", elapsed)
        return token

    except Exception as e:
        logger.exception("Login API error")
        raise RuntimeError(f"Login API error: {str(e)}")

# -------------------------
# TOKEN MANAGER
# -------------------------
def get_valid_token() -> str:
    if TOKEN_CACHE["token"] is None:
        logger.info("No cached token, fetching new one")
        return login_and_get_token()

    age = time.time() - TOKEN_CACHE["created_at"]
    if age > TOKEN_VALIDITY_SECONDS:
        logger.info("Cached token expired, refreshing token")
        return login_and_get_token()

    return TOKEN_CACHE["token"]

# -------------------------
# FETCH CLAIM HISTORY
# -------------------------
def fetch_claim_history(policy_id: str) -> pd.DataFrame:
    start_time = time.time()
    token = get_valid_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"policy_id": policy_id}

    logger.info("Fetching claim history for policy_id: %s", policy_id)

    try:
        response = requests.post(CLAIM_HISTORY_URL, headers=headers, json=payload, timeout=15)

        # Token expired server-side
        if response.status_code == 401:
            logger.info("Token expired server-side, re-authenticating")
            token = login_and_get_token()
            headers["Authorization"] = f"Bearer {token}"
            response = requests.post(CLAIM_HISTORY_URL, headers=headers, json=payload, timeout=15)

        if response.status_code != 200:
            raise RuntimeError(f"Claim history API failed: {response.text}")

        history_data = response.json()
        if not history_data:
            logger.warning("No policy history found for policy_id: %s", policy_id)
            raise ValueError(f"No policy history found for policy_id: {policy_id}")

        elapsed = time.time() - start_time
        logger.info("Fetched %d claim records for policy_id %s in %.3f sec",
                    len(history_data), policy_id, elapsed)
        return pd.DataFrame(history_data)

    except ValueError:
        raise
    except Exception as e:
        logger.exception("Claim history API error")
        raise RuntimeError(f"Claim history API error: {str(e)}")