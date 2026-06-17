# Fraud Detection System

A comprehensive machine learning-based fraud detection system for insurance claims. This project combines data preprocessing, feature engineering, and advanced ML models to identify fraudulent claims with high accuracy.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Running the API](#running-the-api)
  - [Making Predictions](#making-predictions)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Model Architecture](#model-architecture)

## Features

- **Data Preprocessing**: Comprehensive data cleaning and normalization
- **Feature Engineering**: Automated feature extraction and selection
- **Multiple ML Models**: XGBoost classifier with anomaly detection
- **Scalable Architecture**: Modular pipeline design for easy updates
- **Flask API**: REST API for real-time fraud predictions
- **Configurable Workflows**: Support for multiple data split strategies (stratified, group-based, temporal)
- **Model Persistence**: Trained models and scalers saved for inference

## Project Structure

```
fraud_detection/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── model_training/          # Training pipeline
│   ├── __init__.py
│   ├── main.py             # Main entry point for training
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Load and validate dataset
│   ├── preprocessing.py    # Data cleaning and normalization
│   ├── feature_engineering.py  # Feature creation
│   ├── train.py            # Model training logic
│   ├── anomaly.py          # Anomaly detection models
│   ├── artifacts/          # Generated artifacts
│   │   └── preprocessed_dataset.csv
│   └── models/             # Trained model files
│       ├── fraud_model.pkl       # XGBoost classifier
│       ├── scaler.pkl            # Feature scaler
│       ├── anomaly_model.pkl     # Anomaly detector
│       └── training_metadata.json
│
└── Inference/              # API and inference
    ├── __init__.py
    ├── app.py             # Flask API application
    ├── inference.py       # Prediction logic
    ├── api_client.py      # External API client
    └── reasoning_engine.py # Decision logic
```

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for all dependencies:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - joblib
  - openpyxl
  - flask
  - requests
  - python-dotenv

## Installation

1. **Clone or download the repository**
   ```bash
   cd fraud_detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   - Place your claim history dataset at: `C:\Users\Lenovo\Downloads\claim_history_dataset.csv`
   - Or set the `DATA_PATH` environment variable:
     ```bash
     set DATA_PATH=path\to\your\dataset.csv  # Windows
     export DATA_PATH=path/to/your/dataset.csv  # macOS/Linux
     ```

## Usage

### Model Training

To train the fraud detection model:

```bash
cd model_training
python main.py
```

This will:
1. Load data from the configured path
2. Preprocess the data (cleaning, normalization)
3. Perform feature engineering
4. Train the XGBoost classifier and anomaly detection model
5. Save trained models to `models/` directory
6. Save preprocessed dataset to `artifacts/preprocessed_dataset.csv`

#### Training Configuration

Edit [model_training/config.py](model_training/config.py) to customize:
- `DATA_PATH`: Source dataset location
- `TEST_SIZE`: Train/test split ratio (default: 0.5)
- `RANDOM_STATE`: Random seed for reproducibility (default: 42)
- `SPLIT_STRATEGY`: Data splitting method:
  - `'stratified'` (default): Maintains class distribution
  - `'group_insured'`: Group by insured_id
  - `'group_hospital'`: Group by hospital_id
  - `'time'`: Time-based split

### Running the API

To start the inference API server:

```bash
cd Inference
python app.py
```

The Flask server will start on `http://localhost:5000`

### Making Predictions

**Request Format:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": "POL123",
    "insured_id": "INS456",
    "hospital_id": "HOSP789",
    "checkin_date": "2024-01-15",
    "cover_amount": 50000,
    "check_option": "standard"
  }'
```

**Response:**
```json
{
    "predictions": [
        {
            "anomaly_score": 0.017190661892136805,
            "fraud_flag": 0,
            "fraud_prob": 0.3311656713485718,
            "is_sus": 0,
            "reason_codes": [
                "Claim appears normal based on history and model signals"
            ]
        }
    ],
    "status": "success"
}
```

## Configuration

### Environment Variables

- `DATA_PATH`: Path to the claim history dataset (default: `C:\Users\Lenovo\Downloads\claim_history_dataset.csv`)
- `SPLIT_STRATEGY`: Data splitting strategy (default: `'stratified'`)

### Model Files

After training, the following files are created in `model_training/models/`:

| File | Description |
|------|-------------|
| `fraud_model.pkl` | Trained XGBoost classifier |
| `scaler.pkl` | Feature normalization scaler |
| `anomaly_model.pkl` | Anomaly detection model |
| `training_metadata.json` | Training statistics and metadata |

## API Endpoints

### POST `/predict`

Predicts whether a claim is fraudulent.

**Required Fields:**
- `policy_id` (string): Insurance policy identifier
- `insured_id` (string): Insured person identifier
- `hospital_id` (string): Hospital identifier
- `checkin_date` (string): Date of hospital checkin (YYYY-MM-DD format)
- `cover_amount` (float): Amount covered by policy
- `check_option` (string): Type of check option

**Response:**
- `predictions`: Array of prediction objects for each claim
  - `fraud_flag`: 0 (legitimate) or 1 (fraudulent)
  - `fraud_prob`: Probability of fraud (0-1)
  - `anomaly_score`: Anomaly detection score
  - `is_sus`: Suspicious flag (0 or 1)
  - `reason_codes`: Array of explanation strings for the decision
- `status`: "success" or error description

## Model Architecture

The fraud detection system uses a hybrid approach:

1. **XGBoost Classifier**: Primary prediction model
   - Trained on preprocessed and engineered features
   - Handles non-linear patterns in claim data
   - Provides probability scores

2. **Anomaly Detection**: Secondary validation
   - Identifies statistical outliers
   - Complements supervised learning predictions
   - Useful for detecting novel fraud patterns

3. **Feature Engineering Pipeline**:
   - Statistical aggregations per insured/hospital
   - Temporal features
   - Claim characteristics normalization
   - Derived risk indicators

## License

This project is confidential and for internal use only.

## Notes

- Ensure your dataset has the required columns before training
- The model expects consistent feature formats during inference
- Monitor model performance regularly and retrain with fresh data
- Adjust `TEST_SIZE` and `SPLIT_STRATEGY` based on your data characteristics
