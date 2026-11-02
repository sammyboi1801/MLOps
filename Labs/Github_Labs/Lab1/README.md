# GitHub Actions Lab 1: Testing my FastAPI Housing API

[![Lab1 API Tests (pytest)](https://github.com/sammyboi1801/MLOps/actions/workflows/lab1_pytest.yml/badge.svg)](https://github.com/sammyboi1801/MLOps/actions/workflows/lab1_pytest.yml)
[![Lab1 Model Tests (unittest)](https://github.com/sammyboi1801/MLOps/actions/workflows/lab1_unittest.yml/badge.svg)](https://github.com/sammyboi1801/MLOps/actions/workflows/lab1_unittest.yml)

**Note:** this lab doesn't have its own source code. The tests here test my [FastAPI lab](../../API_Labs/FastAPI_Labs) (California housing price API). So the code being tested is in `Labs/API_Labs/FastAPI_Labs/src` and the tests and workflows are here.

The original lab had a small calculator (add, subtract, multiply) with pytest and unittest tests, and GitHub Actions that run the tests on every push. I wanted the tests to check something real, so I replaced the calculator with my housing API from the FastAPI lab. Now every time I push, GitHub runs tests on the actual API and model, and if I break something the build goes red.

## What I changed

1. **Testing real ML code instead of a calculator.** The tests import my FastAPI app, the training code and the saved model.
2. **pytest tests for the API.** I used FastAPI's `TestClient`, which calls the endpoints without starting a server. I used `@pytest.mark.parametrize` so one test function checks many bad inputs.
3. **unittest tests for data, training and the model.** These check the dataset, the train/test split, `evaluate_model()` and the predictions.
4. **Model quality check.** If the saved model's R² goes below 0.80, or it stops beating Linear Regression, the tests fail.
5. **Fixed the workflows.** The original workflow file had a typo (`run-nam`) and used `branches` and `branches-ignore` together which GitHub doesn't allow. It also used `upload-artifact@v2` which GitHub shut down, so it wouldn't run anymore. I updated everything to the current versions.
6. **Testing on 3 Python versions.** Both workflows run on Python 3.10, 3.11 and 3.12 at the same time using a matrix.
7. **Only run when needed.** The workflows only run when files in this lab or the FastAPI lab change, so editing some other lab doesn't start a test run. They can also be started manually from the Actions tab.
8. **pip caching** so the runs after the first one are faster.
9. **Removed other workflows from my fork.** My fork had workflows from other labs (Docker push, model retraining that commits to the repo every night, GCP connection) that would fail or spam commits without the instructor's secrets, so I deleted them and kept only mine.

## Folder structure

```
Labs/
├── Github_Labs/Lab1/
│   ├── test/
│   │   ├── test_pytest.py       # API tests
│   │   └── test_unittest.py     # data, training and model tests
│   ├── requirements.txt         # uses the FastAPI lab requirements + pytest
│   └── README.md
└── API_Labs/FastAPI_Labs/       # the code being tested
    ├── model/housing_model.pkl
    ├── model/metrics.json
    └── src/ (data.py, train.py, predict.py, main.py)

.github/workflows/
├── lab1_pytest.yml
└── lab1_unittest.yml
```

## Tests

### pytest (API) - `test_pytest.py`

| Test | What it checks |
|---|---|
| `test_health` | `/` returns healthy and the right model name |
| `test_predict_returns_price` | `/predict` returns a price that makes sense (between $10k and $600k) |
| `test_predict_berkeley_close_to_actual` | Berkeley example is within $100k of the real value ($452,600) |
| `test_higher_income_predicts_higher_price` | Same house with higher income gets a higher price |
| `test_predict_rejects_bad_input` | 6 bad inputs (latitude 45, longitude -100, negative income, negative age, more bedrooms than rooms, text instead of a number) all return 422 with the right message |
| `test_predict_rejects_missing_field` | Leaving out a field returns 422 |
| `test_batch_matches_single_predictions` | `/predict/batch` gives the same prices as calling `/predict` one by one |
| `test_batch_size_limits` | 0 and 101 houses are rejected, 1 and 100 are accepted |
| `test_model_info` | `/model-info` returns the model type, features and best model |

### unittest (data and model) - `test_unittest.py`

| Test class | What it checks |
|---|---|
| `TestData` | Dataset has 20,640 rows and 8 features, prices are in the expected range, split is 70/30 and gives the same result every time |
| `TestEvaluateModel` | Uses a fake model with known outputs to check the RMSE and R² math (perfect predictions give R² 1, predicting the mean gives R² 0, being off by 0.5 gives RMSE $50,000) |
| `TestPredict` | Model file exists, predictions come back in dollars, one prediction per row |
| `TestModelQuality` | R² is at least 0.80 and the best model beats Linear Regression |

## Workflows

Both workflows are in `.github/workflows/` (GitHub only looks there, not inside the lab folder).

- **lab1_pytest.yml:** installs the requirements, runs `test_pytest.py` and uploads the test report (`pytest-report.xml`). You can download it from the bottom of the run page under Artifacts.
- **lab1_unittest.yml:** installs the requirements and runs `test_unittest.py`.

Steps in each workflow:
1. Checkout the code
2. Set up Python (3.10, 3.11 and 3.12 run as separate jobs)
3. Install dependencies (with pip cache)
4. Run the tests

## How to run the tests locally

From the `Github_Labs/Lab1` folder. I used Python 3.11 because `scikit-learn==1.5.1` doesn't support 3.13.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt

pytest test/test_pytest.py -v
python -m unittest test.test_unittest -v
```

## Results

Locally all tests pass:
- pytest: 17 passed (10 test functions, but the parametrized ones count each case separately)
- unittest: 12 passed

## What I learned

- GitHub only runs workflows from `.github/workflows/` at the root of the repo. The original lab had copies of the workflow files inside the lab folder, which GitHub ignores.
- Forks have Actions turned off by default. When I looked at my fork there were 6 workflows from other labs waiting, and turning on Actions would have started all of them. I had to clean that up first.
- Testing with a fake model (`FixedModel`) was useful. I know exactly what the predictions are, so I can check the RMSE and R² numbers exactly instead of just checking that they "look ok".
- The "higher income gives higher price" test doesn't check an exact number, it checks that the model behaves in a way that makes sense. I think this kind of test is good for ML because exact predictions can change when you retrain.
- The quality check means a worse model can't get merged without someone noticing.
- The downside of my setup is that this lab depends on another folder. If I change the FastAPI lab and break it, these tests fail. But that's kind of the point of CI, it catches it right away. I made the workflows run when either folder changes because of this.
- I had to delete the empty `data/` folder from the original lab because it had the same name as `data.py` in my FastAPI lab and Python could end up importing the wrong one.
