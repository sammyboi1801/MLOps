import json
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data import load_data, split_data

MODEL_PATH = "../model/housing_model.pkl"
METRICS_PATH = "../model/metrics.json"

def fit_models(X_train, y_train):
    """
    Train the candidate regression models.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        models (dict): Model name mapped to the fitted model.
    """
    models = {
        "linear_regression": LinearRegression(),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=12),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted model on the test set.
    Args:
        model: A fitted regression model.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing target values (in $100,000s).
    Returns:
        metrics (dict): RMSE converted to dollars, and R^2 score.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100_000
    return {
        "rmse_dollars": round(float(rmse), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }

def save_best_model(models, results):
    """
    Save the model with the highest R^2 score, along with all metrics.
    Args:
        models (dict): Model name mapped to the fitted model.
        results (dict): Model name mapped to its metrics.
    Returns:
        best_name (str): Name of the saved model.
    """
    best_name = max(results, key=lambda name: results[name]["r2"])
    joblib.dump(models[best_name], MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    return best_name

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = fit_models(X_train, y_train)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
        print(f"{name:<24} RMSE: ${results[name]['rmse_dollars']:,.0f}  R^2: {results[name]['r2']:.4f}")

    best_name = save_best_model(models, results)
    print(f"\nBest model: {best_name}, saved to {MODEL_PATH}")
