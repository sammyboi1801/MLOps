from pathlib import Path
import joblib

# Path is built from this file's location, so it works no matter where uvicorn is started from
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "housing_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

# Loaded once when the server starts, then reused for every request
model = joblib.load(MODEL_PATH)

def predict_data(X):
    """
    Predict house prices for the input data.
    Args:
        X (list or numpy.ndarray): One row per house, with the 8 features in order
            (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude).
    Returns:
        y_pred (numpy.ndarray): Predicted house prices in dollars.
    """
    y_pred = model.predict(X) * 100_000
    return y_pred
