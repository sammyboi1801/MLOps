import os

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "adult.csv")

# sqlite backend so the model registry works (file store doesn't support it)
TRACKING_URI = "sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/")
EXPERIMENT = "adult_income"
MODEL_NAME = "adult_income_classifier"

# newer mlflow saves sklearn models with skops instead of pickle, and skops refuses to
# load types it doesn't know. these are the ones my 3 pipelines need (found with
# skops.io.get_untrusted_types)
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree",
                 "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]

NUM_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT_COLS = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def load_data():
    # download once from openml, then just read the local csv
    if not os.path.exists(DATA_PATH):
        print("downloading adult dataset from openml...")
        df = fetch_openml("adult", version=2, as_frame=True).frame
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.replace("-", "_") for c in df.columns]

    # target: 1 if income is more than 50K
    df["income"] = (df["class"].str.strip() == ">50K").astype(int)

    # fnlwgt is a census sampling weight and education is the same as education_num
    df = df.drop(columns=["class", "fnlwgt", "education"])
    return df


def split_data(df):
    X = df[NUM_COLS + CAT_COLS]
    y = df["income"]

    # 60 / 20 / 20 split like the original lab, stratified since only ~24% are >50K
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=0.6, stratify=y, random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, stratify=y_rem, random_state=123)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_pipeline(model):
    # preprocessing lives inside the pipeline, so the logged model takes raw rows
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
    return Pipeline([("prep", prep), ("model", model)])
