"""SHAP explainability for the readmission model.

Calibration is monotonic, so it doesn't change which features drive the score.
This explains the underlying LightGBM ranker fit on the same training split,
and produces a beeswarm summary plus a mean-|SHAP| ranking of the top features.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.data import build_modelling_frame
from src.features import build_preprocessor, split_features
from src.train import RANDOM_STATE

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SAMPLE = 2000


def main():
    REPORTS.mkdir(exist_ok=True)
    df = build_modelling_frame()
    X, y, _ = split_features(df)
    pre = build_preprocessor(X)

    X_tr, X_te, y_tr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    x_train = pre.fit_transform(X_tr)
    names = pre.get_feature_names_out()

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(x_train, y_tr)

    x_test = pre.transform(X_te)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(x_test.shape[0], min(SAMPLE, x_test.shape[0]), replace=False)
    sample = x_test[idx]
    sample = sample.toarray() if hasattr(sample, "toarray") else sample
    sample_df = pd.DataFrame(sample, columns=names)

    values = shap.TreeExplainer(model).shap_values(sample_df)
    if isinstance(values, list):
        values = values[1]

    shap.summary_plot(values, sample_df, show=False, max_display=15)
    plt.gcf().savefig(REPORTS / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame(
        {"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.head(20).to_csv(REPORTS / "shap_importance.csv", index=False)
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
