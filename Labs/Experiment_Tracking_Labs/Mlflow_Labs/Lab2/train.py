# step 1: train 3 baseline models and log everything to mlflow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from data import TRUSTED_TYPES, load_data, make_pipeline, setup_mlflow, split_data


def evaluate(model, X, y, prefix):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_auc": roc_auc_score(y, proba),
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
    }


def log_plots(model, X, y):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax[0])
    ax[0].set_title("ROC curve (val)")
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax[1], colorbar=False)
    ax[1].set_title("confusion matrix (val)")
    fig.tight_layout()
    mlflow.log_figure(fig, "eval_plots.png")
    plt.close(fig)


def train_and_log(run_name, model, params, data):
    X_train, X_val, y_train, y_val = data
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", run_name)
        mlflow.log_params(params)

        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_val, y_val, "val")
        metrics.update(evaluate(pipe, X_train, y_train, "train"))
        mlflow.log_metrics(metrics)
        log_plots(pipe, X_val, y_val)

        # signature + example so the served model knows what input to expect
        signature = infer_signature(X_val, pipe.predict(X_val))
        mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                                 input_example=X_val.head(3),
                                 skops_trusted_types=TRUSTED_TYPES)

        print(f"{run_name:22s} val_auc={metrics['val_auc']:.4f}  val_f1={metrics['val_f1']:.4f}")


if __name__ == "__main__":
    setup_mlflow()
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}  positive rate={y_train.mean():.3f}")

    data = (X_train, X_val, y_train, y_val)

    lr_params = {"C": 1.0, "max_iter": 1000}
    train_and_log("logistic_regression", LogisticRegression(**lr_params), lr_params, data)

    rf_params = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2}
    train_and_log("random_forest", RandomForestClassifier(**rf_params, n_jobs=-1, random_state=123),
                  rf_params, data)

    xgb_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1}
    train_and_log("xgboost", XGBClassifier(**xgb_params, eval_metric="logloss", random_state=123),
                  xgb_params, data)
