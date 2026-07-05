"""Train and calibrate the 30-day readmission model.

A logistic-regression baseline is compared against gradient boosting, both
class-weighted for the ~9% positive rate, using stratified 5-fold CV for
selection. The winner is wrapped in isotonic calibration so the predicted
probabilities can be read as risk scores rather than arbitrary ranks. The
fitted pipeline, the held-out test predictions and the sensitive attributes
are all persisted for the evaluation and equity-audit steps.
"""

import json
from pathlib import Path

import joblib
import mlflow
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data import build_modelling_frame
from src.features import build_preprocessor, split_features

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RANDOM_STATE = 42


def candidate_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "lgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def main():
    df = build_modelling_frame()
    X, y, sensitive = split_features(df)
    pre = build_preprocessor(X)

    X_tr, X_te, y_tr, y_te, _, s_te = train_test_split(
        X, y, sensitive, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        scores[name] = auc.mean()
        print(f"{name}: CV ROC-AUC {auc.mean():.4f} +/- {auc.std():.4f}")

    best = max(scores, key=scores.get)
    print(f"selected: {best}")

    final = Pipeline([("pre", pre), ("clf", candidate_models()[best])])
    calibrated = CalibratedClassifierCV(final, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)

    proba = calibrated.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, proba)
    test_pr = average_precision_score(y_te, proba)
    print(f"held-out ROC-AUC: {test_auc:.4f}")
    print(f"held-out PR-AUC:  {test_pr:.4f}")
    print(f"positive rate:    {y_te.mean():.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "readmission_model.joblib"
    joblib.dump(calibrated, model_path)
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(list(X.columns)))

    predictions = s_te.copy()
    predictions["y_true"] = y_te.values
    predictions["y_proba"] = proba
    predictions.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    # Log the run to MLflow (local ./mlruns) for experiment tracking.
    mlflow.set_experiment("readmission")
    with mlflow.start_run():
        mlflow.log_param("selected_model", best)
        mlflow.log_param("calibration", "isotonic")
        for name, score in scores.items():
            mlflow.log_metric(f"cv_roc_auc_{name}", score)
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("test_pr_auc", test_pr)
        mlflow.log_artifact(str(model_path))

    print(f"saved model and {len(predictions):,} test predictions to {MODELS_DIR}")


if __name__ == "__main__":
    main()
