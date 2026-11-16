# step 2: tune xgboost with optuna, each trial is a nested mlflow run

import sys

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from data import TRUSTED_TYPES, load_data, make_pipeline, setup_mlflow, split_data
from train import evaluate, log_plots

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25

setup_mlflow()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_data())


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        pipe = make_pipeline(XGBClassifier(**params, eval_metric="logloss", random_state=123))
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        mlflow.log_metric("val_auc", auc)
    return auc


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="xgboost_tuned"):
        mlflow.set_tag("model_type", "xgboost_tuned")
        mlflow.log_param("n_trials", N_TRIALS)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=123))
        study.optimize(objective, n_trials=N_TRIALS)
        print(f"best trial #{study.best_trial.number}  val_auc={study.best_value:.4f}")
        print("best params:", study.best_params)

        # refit with the best params and log it on the parent run
        best = make_pipeline(XGBClassifier(**study.best_params, eval_metric="logloss",
                                           random_state=123))
        best.fit(X_train, y_train)

        mlflow.log_params(study.best_params)
        metrics = evaluate(best, X_val, y_val, "val")
        metrics.update(evaluate(best, X_train, y_train, "train"))
        mlflow.log_metrics(metrics)
        log_plots(best, X_val, y_val)

        signature = infer_signature(X_val, best.predict(X_val))
        mlflow.sklearn.log_model(best, name="model", signature=signature,
                                 input_example=X_val.head(3),
                                 skops_trusted_types=TRUSTED_TYPES)
