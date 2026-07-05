import numpy as np
import pandas as pd

from src.data import _group_diagnosis
from src.features import SENSITIVE, TARGET, _to_str_frame, split_features


def test_group_diagnosis_categories():
    assert _group_diagnosis("250.83") == "Diabetes"
    assert _group_diagnosis("410") == "Circulatory"
    assert _group_diagnosis("486") == "Respiratory"
    assert _group_diagnosis("V45") == "Other"
    assert _group_diagnosis(np.nan) == "Missing"


def test_to_str_frame_fills_missing():
    df = pd.DataFrame({"a": [1, None], "b": ["x", None]})
    out = _to_str_frame(df)
    assert out.isna().sum().sum() == 0
    assert (out.loc[1] == "Missing").all()


def test_split_features_holds_out_sensitive():
    df = pd.DataFrame(
        {
            TARGET: [0, 1],
            "race": ["a", "b"],
            "gender": ["Male", "Female"],
            "age_band": ["[60-70)", "[70-80)"],
            "num_medications": [5, 9],
        }
    )
    X, y, sensitive = split_features(df)
    assert TARGET not in X.columns
    assert all(c not in X.columns for c in SENSITIVE)
    assert list(sensitive.columns) == SENSITIVE
    assert "num_medications" in X.columns
