"""Feature pipeline for the readmission model.

The cleaned frame is split into the model design matrix and a separate frame of
sensitive attributes used only for the fairness audit. race and gender are
deliberately held out of the model - a clinical risk score shouldn't take race
as an input - while age (clinically relevant) is retained. The audit then
checks whether disparities appear anyway through correlated features.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "readmit_30d"
SENSITIVE = ["race", "gender", "age_band"]

# The genuinely continuous columns. Everything else in the design matrix is
# treated as categorical, including the id-coded columns (admission_type_id
# etc.) which are lookup codes, not quantities.
NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age_years",
]


def _to_str_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing with an explicit category and coerce codes to strings."""
    return frame.fillna("Missing").astype(str)


def split_features(df: pd.DataFrame):
    """Return (X, y, sensitive); sensitive is kept only for auditing."""
    y = df[TARGET]
    sensitive = df[SENSITIVE].copy()
    X = df.drop(columns=[TARGET, *SENSITIVE])
    return X, y, sensitive


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in NUMERIC if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # min_frequency folds rare categories (e.g. uncommon medical specialties)
    # into a single bucket so the one-hot matrix stays manageable.
    categorical_pipe = Pipeline(
        [
            ("to_str", FunctionTransformer(_to_str_frame, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )
