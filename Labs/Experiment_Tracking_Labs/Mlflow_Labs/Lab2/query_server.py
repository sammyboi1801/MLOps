# step 4: send a few people to the served model and print what it predicts
# start the server first (see README):
#   mlflow models serve -m "models:/adult_income_classifier@champion" -p 5001 --env-manager local

import requests

from data import load_data, split_data

URL = "http://127.0.0.1:5001/invocations"

X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_data())
sample = X_test.head(8)

# some rows have missing occupation/workclass, NaN isn't valid json so send null instead
payload = {"dataframe_split": sample.astype(object).where(sample.notna(), None).to_dict(orient="split")}
resp = requests.post(URL, json=payload, timeout=30)
resp.raise_for_status()
preds = resp.json()["predictions"]

def label(v):
    return ">50K" if v == 1 else "<=50K"


print(f"{'age':>4} {'occupation':18s} {'hours':>5} {'actual':>7} {'predicted':>9}")
for (_, row), actual, pred in zip(sample.iterrows(), y_test.head(8), preds):
    print(f"{row['age']:>4} {str(row['occupation']).replace('nan', 'missing')[:18]:18s} {row['hours_per_week']:>5} "
          f"{label(actual):>7} {label(pred):>9}")
