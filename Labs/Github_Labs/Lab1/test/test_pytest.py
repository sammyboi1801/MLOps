# Tests for my FastAPI housing app (Labs/API_Labs/FastAPI_Labs)
# TestClient calls the endpoints directly so the server doesn't need to be running
import sys
from pathlib import Path
import pytest

# the code is in the FastAPI lab folder so add it to the path
FASTAPI_SRC = Path(__file__).resolve().parents[3] / "API_Labs" / "FastAPI_Labs" / "src"
sys.path.insert(0, str(FASTAPI_SRC))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# A real district near Berkeley, its actual median value is $452,600
BERKELEY = {
    "median_income": 8.3252,
    "house_age": 41,
    "avg_rooms": 6.98,
    "avg_bedrooms": 1.02,
    "population": 322,
    "avg_occupancy": 2.56,
    "latitude": 37.88,
    "longitude": -122.23,
}

def make_house(**changes):
    # copy of the Berkeley house with some values changed
    return {**BERKELEY, **changes}


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "HistGradientBoostingRegressor"}

def test_predict_returns_price():
    response = client.post("/predict", json=BERKELEY)
    assert response.status_code == 200
    price = response.json()["predicted_price"]
    assert isinstance(price, float)
    # prices in the dataset are between about $15k and $500k
    assert 10_000 < price < 600_000

def test_predict_berkeley_close_to_actual():
    price = client.post("/predict", json=BERKELEY).json()["predicted_price"]
    assert abs(price - 452_600) < 100_000

def test_higher_income_predicts_higher_price():
    # same house, only income changes, so price should go up
    low = client.post("/predict", json=make_house(median_income=2.0)).json()["predicted_price"]
    high = client.post("/predict", json=make_house(median_income=10.0)).json()["predicted_price"]
    assert high > low

@pytest.mark.parametrize("field, value, message", [
    ("latitude", 45, "less than or equal to 42"),
    ("longitude", -100, "less than or equal to -114"),
    ("median_income", -1, "greater than 0"),
    ("house_age", -5, "greater than or equal to 0"),
    ("avg_bedrooms", 9, "cannot be greater than avg_rooms"),
    ("median_income", "abc", "valid number"),
])
def test_predict_rejects_bad_input(field, value, message):
    response = client.post("/predict", json=make_house(**{field: value}))
    assert response.status_code == 422
    assert message in response.json()["detail"][0]["msg"]

def test_predict_rejects_missing_field():
    house = dict(BERKELEY)
    del house["population"]
    response = client.post("/predict", json=house)
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Field required"

def test_batch_matches_single_predictions():
    houses = [BERKELEY, make_house(median_income=2.0, latitude=34.05, longitude=-118.24)]
    batch = client.post("/predict/batch", json={"houses": houses}).json()["predictions"]
    singles = [client.post("/predict", json=h).json()["predicted_price"] for h in houses]
    assert batch == singles

@pytest.mark.parametrize("size, expected_status", [
    (0, 422),
    (1, 200),
    (100, 200),
    (101, 422),
])
def test_batch_size_limits(size, expected_status):
    response = client.post("/predict/batch", json={"houses": [BERKELEY] * size})
    assert response.status_code == expected_status
    if expected_status == 200:
        assert len(response.json()["predictions"]) == size

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    info = response.json()
    assert info["model_type"] == "HistGradientBoostingRegressor"
    assert info["features"] == list(BERKELEY.keys())
    assert info["best_model"] == "hist_gradient_boosting"
