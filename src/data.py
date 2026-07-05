"""Load and clean the UCI Diabetes 130-US Hospitals dataset (Strack et al.,
2014; UCI id 296) into a modelling frame for 30-day readmission.

The cleaning here mirrors what a careful clinical-analytics pipeline does
rather than the minimal "load the CSV" version: encounters where the patient
died or was discharged to hospice are removed (they cannot be readmitted),
repeat encounters for the same patient are collapsed to the first admission so
the same person can't land in both train and test, and the three ICD-9
diagnosis columns are grouped into broad clinical categories.
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_CACHE = DATA_DIR / "diabetes_raw.csv"

# discharge_disposition_id values meaning death or hospice - excluded because
# these patients cannot be readmitted.
TERMINAL_DISCHARGES = {11, 13, 14, 19, 20, 21}

# weight is missing for ~97% of encounters; payer_code is an insurance-type
# proxy for socioeconomic status that we don't want driving a clinical risk
# score (and is ~40% missing). The two id columns are used for de-duplication
# and then dropped.
DROP_COLUMNS = ["weight", "payer_code", "encounter_id", "patient_nbr"]

_AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def load_raw() -> pd.DataFrame:
    """Fetch features, target and ids from UCI, caching a combined CSV."""
    if RAW_CACHE.exists():
        return pd.read_csv(RAW_CACHE, low_memory=False)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    repo = fetch_ucirepo(id=296)
    raw = pd.concat([repo.data.ids, repo.data.features, repo.data.targets], axis=1)
    raw.to_csv(RAW_CACHE, index=False)
    return raw


def _group_diagnosis(code) -> str:
    """Map an ICD-9 diagnosis code to a broad clinical category."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    if code.startswith("250"):
        return "Diabetes"
    try:
        n = float(code)
    except ValueError:
        return "Other"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasm"
    return "Other"


def build_modelling_frame() -> pd.DataFrame:
    """Return a cleaned frame with the binary 30-day readmission target.

    race, gender and age_band are kept so the evaluation can measure
    performance across groups. The model sees them as ordinary features; the
    disparities are reported separately.
    """
    df = load_raw().replace("?", pd.NA)

    df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

    df = df[~df["discharge_disposition_id"].isin(TERMINAL_DISCHARGES)]

    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )

    df = df[df["gender"].isin(["Male", "Female"])]

    for col in ("diag_1", "diag_2", "diag_3"):
        df[col + "_grp"] = df[col].map(_group_diagnosis)

    # Keep unknown race explicit rather than dropping it - the missingness
    # itself matters for the equity read.
    df["race"] = df["race"].fillna("Unknown")

    df["age_band"] = df["age"]
    df["age_years"] = df["age"].map(_AGE_MIDPOINT)

    df = df.drop(
        columns=DROP_COLUMNS + ["readmitted", "diag_1", "diag_2", "diag_3", "age"]
    )

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    frame = build_modelling_frame()
    print(f"rows: {len(frame):,}  columns: {frame.shape[1]}")
    print(f"30-day readmission rate: {frame['readmit_30d'].mean():.3%}")
    print("\nreadmission rate by race:")
    print(frame.groupby("race")["readmit_30d"].agg(["mean", "size"]))
