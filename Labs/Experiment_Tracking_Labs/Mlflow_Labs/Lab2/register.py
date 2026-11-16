# step 3: pick the best run, register it, give it the "champion" alias, then test it

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import classification_report, roc_auc_score

from data import EXPERIMENT, HERE, MODEL_NAME, load_data, setup_mlflow, split_data

setup_mlflow()
client = MlflowClient()

# only the main runs have a model_type tag (optuna trials don't), so this skips the trials
runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                          filter_string="attributes.status = 'FINISHED'",
                          order_by=["metrics.val_auc DESC"])
runs = runs[runs["tags.model_type"].notna()]

print("leaderboard (validation AUC):")
print(runs[["tags.model_type", "metrics.val_auc", "metrics.val_f1", "run_id"]]
      .drop_duplicates("tags.model_type").to_string(index=False))

best = runs.iloc[0]
print(f"\nbest run: {best['tags.model_type']} ({best['run_id']})")

# register it as a new version
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
client.update_model_version(MODEL_NAME, mv.version,
                            description=f"{best['tags.model_type']}, val_auc={best['metrics.val_auc']:.4f}")
client.set_model_version_tag(MODEL_NAME, mv.version, "model_type", best["tags.model_type"])

# stages (Staging/Production) are deprecated in newer mlflow, so I used an alias instead
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"registered {MODEL_NAME} v{mv.version} and set alias 'champion'")

# load it back from the registry and check it on the test set (never used before this)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_data())
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
proba = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, proba)
print(f"\ntest AUC of champion: {test_auc:.4f}")
print(classification_report(y_test, (proba >= 0.5).astype(int), target_names=["<=50K", ">50K"]))

client.set_model_version_tag(MODEL_NAME, mv.version, "test_auc", f"{test_auc:.4f}")

# batch inference: save predictions for the test set (replaces the spark udf part of the lab)
out = X_test.copy()
out["actual"] = y_test.values
out["prob_over_50k"] = proba.round(4)
out["prediction"] = (proba >= 0.5).astype(int)
out_path = os.path.join(HERE, "outputs", "test_predictions.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out.to_csv(out_path, index=False)
print(f"saved batch predictions to {out_path}")
