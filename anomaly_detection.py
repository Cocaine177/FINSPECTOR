from pathlib import Path
import os
import joblib

# Get the directory where this script is located
script_dir = Path(__file__).parent
models_dir = script_dir / "Models"

# Model paths
model_path = os.path.join(os.path.dirname(__file__), "rf_stock_model.pkl") 
FEATURES_PATH = models_dir / "feature_names.pkl"

# Load model and feature names

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

def detect_anomalies(df, threshold=0.7):
    """Detect anomalies using the trained model"""
    df = df.copy()

    # Ensure all required features exist
    for i in range(10):
        if f'feature_{i}' not in df.columns:
            raise ValueError(f"Dataframe is missing required feature: feature_{i}")

    # Get only the features needed by the model
    X = df[[f'feature_{i}' for i in range(10)]]

    # Predict anomaly scores
    df["anomaly_score"] = model.predict_proba(X)[:, 1]
    df["suspicious"] = df["anomaly_score"].apply(
        lambda x: "🔴 Yes" if x > threshold else "🟢 No")

    return df
