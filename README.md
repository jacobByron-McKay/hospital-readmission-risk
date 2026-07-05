# Hospital 30-day readmission risk

An end-to-end, equity-aware machine-learning project on public hospital data:
predict whether a diabetic inpatient is readmitted within 30 days of discharge,
so a follow-up team can prioritise who to contact. Built to demonstrate a
careful senior workflow — leakage-aware preparation, calibration, explainability,
a fairness audit, and honest reporting of a hard problem — not to chase a
headline accuracy number.

## Results (held-out test set)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.654 |
| PR-AUC | 0.186 |
| Brier score | 0.079 |
| Lift at top-10% flagged | 2.6× |

Thirty-day readmission is only weakly predictable from administrative data — a
logistic-regression baseline and gradient boosting land within 0.001 of each
other, which says the problem is signal-limited, not model-limited. A
suspiciously high AUC on this dataset almost always means leakage. The work here
is in characterising that ceiling honestly, calibrating the score so it is still
usable for targeting (top-decile flagging finds readmissions at 2.6× the base
rate), and checking the model behaves equitably.

## What it demonstrates

- **Leakage-aware data prep** — removes encounters that can't be readmitted
  (death/hospice), and collapses repeat visits to each patient's first admission
  so no patient spans train and test.
- **Calibrated probabilities** — isotonic calibration so the output reads as a
  risk, and calibration holds within demographic groups.
- **Explainability** — SHAP shows the score leans on prior inpatient use, discharge
  destination, age and primary diagnosis — clinically sensible drivers.
- **Equity audit** — error rates and calibration across race and age, with the
  disparate-impact ratio and an explicit account of the calibration-vs-parity
  tension. See [`reports/model_card.md`](reports/model_card.md).
- **A deployable surface** — a FastAPI service, tests, and CI.

## Equity

Race and gender are deliberately **kept out of the model's inputs** but used as
the **groupings for the fairness audit** — a variable can be a prohibited
predictor and a required audit dimension at the same time. The audit finds the
model is well calibrated within each group; selection-rate differences largely
track genuine base-rate differences rather than bias. The model card documents
how this transfers to a New Zealand corrections/health setting (prioritised
ethnicity, Te Mana Raraunga governance, a systems-not-deficit framing).

## Layout

```
src/
  data.py       fetch + clean into a modelling frame
  features.py   preprocessing; holds sensitive attributes out of the model
  train.py      CV model selection + isotonic calibration
  evaluate.py   metrics, calibration, equity audit
  explain.py    SHAP feature attributions
  api.py        FastAPI /predict service
tests/          unit tests (pure, no network)
reports/        model card, calibration/equity/SHAP plots
```

## Running it

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
python -m src.train        # fetch, clean, select, calibrate, persist
python -m src.evaluate     # metrics, calibration, equity audit
python -m src.explain      # SHAP attributions
pytest -q
```

Serve the model:

```
uvicorn src.api:app --reload
# POST /predict  {"features": {"number_inpatient": 3, "time_in_hospital": 8, ...}}
```

Or build the self-contained container (trains during build, then serves):

```
docker build -t readmission . && docker run -p 8000:8000 readmission
```

## A note on tooling

This project was built with AI assistance. I used AI tools as a coding and
research assistant to accelerate the build — the same AI-augmented workflow I use
in my day-to-day analytics work (as I've used Copilot on my R NLP and tabular-ML
projects). I directed the design decisions and reviewed and validated the
results.

## Data

UCI Diabetes 130-US Hospitals (Strack et al., 2014) — 101,766 diabetic inpatient
encounters, 1999-2008. Downloaded automatically via `ucimlrepo`. Public,
de-identified research data. This is a demonstration, not a validated clinical
tool.
