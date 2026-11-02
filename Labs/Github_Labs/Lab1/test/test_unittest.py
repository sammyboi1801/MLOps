# Tests for the data, training and model parts of my FastAPI lab
import sys
import json
import unittest
from pathlib import Path
import numpy as np

# the code is in the FastAPI lab folder so add it to the path
FASTAPI_SRC = Path(__file__).resolve().parents[3] / "API_Labs" / "FastAPI_Labs" / "src"
sys.path.insert(0, str(FASTAPI_SRC))

from data import load_data, split_data
from train import evaluate_model
from predict import predict_data, model, MODEL_PATH

METRICS_PATH = MODEL_PATH.parent / "metrics.json"

# Berkeley house, same feature order as training
BERKELEY = [8.3252, 41, 6.98, 1.02, 322, 2.56, 37.88, -122.23]


# fake model so I know exactly what evaluate_model should return
class FixedModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions


class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_data()

    def test_shape(self):
        self.assertEqual(self.X.shape, (20640, 8))
        self.assertEqual(self.y.shape, (20640,))

    def test_target_range(self):
        # prices are in $100,000s
        self.assertGreaterEqual(self.y.min(), 0.14)
        self.assertLessEqual(self.y.max(), 5.01)

    def test_split_sizes(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertAlmostEqual(len(X_test) / len(self.X), 0.3, places=2)

    def test_split_is_reproducible(self):
        first = split_data(self.X, self.y)
        second = split_data(self.X, self.y)
        np.testing.assert_array_equal(first[1], second[1])


class TestEvaluateModel(unittest.TestCase):

    def test_perfect_predictions(self):
        y = [1.0, 2.0, 3.0]
        metrics = evaluate_model(FixedModel(y), None, y)
        self.assertEqual(metrics["rmse_dollars"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_predicting_the_mean_gives_r2_zero(self):
        metrics = evaluate_model(FixedModel([2.0, 2.0, 2.0]), None, [1.0, 2.0, 3.0])
        self.assertEqual(metrics["r2"], 0.0)

    def test_rmse_is_in_dollars(self):
        # every prediction is off by 0.5 which is $50,000
        metrics = evaluate_model(FixedModel([1.5, 2.5, 3.5]), None, [1.0, 2.0, 3.0])
        self.assertEqual(metrics["rmse_dollars"], 50_000.0)


class TestPredict(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists())

    def test_prediction_is_in_dollars(self):
        price = predict_data([BERKELEY])[0]
        raw = model.predict([BERKELEY])[0]
        self.assertAlmostEqual(price, raw * 100_000)
        self.assertTrue(10_000 < price < 600_000)

    def test_one_prediction_per_row(self):
        self.assertEqual(len(predict_data([BERKELEY] * 3)), 3)


# if the model gets worse these fail and the build goes red
class TestModelQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_r2_above_threshold(self):
        best = self.metrics["best_model"]
        self.assertGreaterEqual(self.metrics["results"][best]["r2"], 0.80)

    def test_best_model_beats_baseline(self):
        results = self.metrics["results"]
        self.assertGreater(results["hist_gradient_boosting"]["r2"],
                           results["linear_regression"]["r2"])


if __name__ == '__main__':
    unittest.main()
