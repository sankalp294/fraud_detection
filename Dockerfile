# Minimal Python container for fraud detection inference API
FROM python:3.10-slim

WORKDIR /app

# Disable Python buffering and pip cache
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies with minimal extras
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD python -c "import requests; requests.get('http://localhost:5000/predict', timeout=5)" || exit 1

EXPOSE 5000
CMD ["python", "Inference/app.py"]
