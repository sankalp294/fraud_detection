from __future__ import annotations

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import time


# -------------------------
# Load .env from project root
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# -------------------------
# Environment variables validation
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
TOKEN_CACHE = {
    "token": None,
    "created_at": None
}

TOKEN_VALIDITY_SECONDS = 50 * 60


# -------------------------
# LOGIN
# -------------------------
def login_and_get_token() -> str:
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }

    response = requests.post(LOGIN_URL, data=payload, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(f"Login failed: {response.text}")

    data = response.json()
    token = data.get("token")

    if not token:
        raise RuntimeError("Token missing in login response")

    TOKEN_CACHE["token"] = token
    TOKEN_CACHE["created_at"] = time.time()

    return token


# -------------------------
# TOKEN MANAGER
# -------------------------
def get_valid_token() -> str:

    if TOKEN_CACHE["token"] is None:
        return login_and_get_token()

    age = time.time() - TOKEN_CACHE["created_at"]

    if age > TOKEN_VALIDITY_SECONDS:
        return login_and_get_token()

    return TOKEN_CACHE["token"]


# -------------------------
# FETCH CLAIM HISTORY
# -------------------------
def fetch_claim_history(policy_id: str) -> pd.DataFrame:

    token = get_valid_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {"policy_id": policy_id}

    response = requests.post(
        CLAIM_HISTORY_URL,
        headers=headers,
        json=payload,
        timeout=15
    )

    # Token expired server-side
    if response.status_code == 401:
        token = login_and_get_token()
        headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            CLAIM_HISTORY_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

    if response.status_code != 200:
        raise RuntimeError(f"Claim history API failed: {response.text}")

    history_data = response.json()

    # 🚨 THIS is business failure — not technical failure
    if not history_data:
        raise ValueError(f"No policy history found for policy_id: {policy_id}")

    return pd.DataFrame(history_data)