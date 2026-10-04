# Import necessary libraries
import os
import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report
import joblib

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

if __name__ == '__main__':
    # Load the Breast Cancer dataset (30 tumor features; 0 = malignant, 1 = benign)
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Stratified split keeps the malignant/benign ratio equal in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Candidate models to compare
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "malignant_recall": recall_score(y_test, preds, pos_label=0),
            "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        }
        r = results[name]
        print(f"{name:<20} accuracy: {r['accuracy']:.4f}  "
              f"malignant recall: {r['malignant_recall']:.4f}  ROC AUC: {r['roc_auc']:.4f}")

    # Pick the best model: missing a malignant tumor is the costliest error,
    # so rank by malignant recall first, then accuracy as a tie-breaker
    best_name = max(results, key=lambda n: (results[n]["malignant_recall"], results[n]["accuracy"]))
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test), target_names=cancer.target_names))

    # Save the best model and its metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "cancer_model.pkl"))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Saved model and metrics to {OUTPUT_DIR}/")
