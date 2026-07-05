# Model card: 30-day hospital readmission risk

## Summary

A calibrated classifier that estimates the probability a diabetic inpatient is
readmitted within 30 days of discharge, built on the UCI Diabetes 130-US
Hospitals dataset. It is a demonstration of a careful, senior end-to-end
workflow — leakage-aware data preparation, honest evaluation, probability
calibration, explainability and an equity audit — rather than a deployable
clinical tool.

The headline finding is deliberately not a high accuracy number. Thirty-day
readmission is only weakly predictable from administrative data, and a
logistic-regression baseline and gradient boosting land within 0.001 of each
other on ROC-AUC. The value is in characterising that ceiling honestly,
calibrating the score so it is still usable for targeting, and auditing whether
the model behaves equitably across groups.

## Intended use

- **Intended:** illustrating how a risk score could help a discharge follow-up
  team prioritise which patients to contact when they can only reach a fraction
  of discharges. The score ranks patients by risk; the team works down the list
  to their capacity.
- **Not intended:** any real clinical decision, resource-denial, or individual
  determination. The data is US hospital data from 1999-2008 and the model has
  not been validated for deployment.

## Data

- **Source:** Strack et al. (2014), UCI Machine Learning Repository (id 296);
  101,766 inpatient encounters for diabetic patients across 130 US hospitals,
  1999-2008.
- **Target:** readmission within 30 days of discharge (`readmitted == "<30"`),
  a 9.0% positive rate after cleaning.
- **Cleaning (the parts that matter):**
  - Encounters ending in death or discharge to hospice are removed — those
    patients cannot be readmitted, so leaving them in would bias the label.
  - Repeat encounters for the same patient are collapsed to the earliest
    admission, so no patient appears in both the training and test split. This
    is the main leakage guard and it drops the sample from 101,766 to 69,987.
  - The three free-text-ish ICD-9 diagnosis columns are grouped into broad
    clinical categories (circulatory, respiratory, diabetes, etc.).
  - `weight` (~97% missing) and `payer_code` (insurance type — a socioeconomic
    proxy we don't want driving a clinical score) are dropped.

## Features and a deliberate exclusion

The model uses admission characteristics, prior-utilisation counts, diagnosis
groups, medication and lab indicators, and age.

**Race and gender are deliberately held out of the model's inputs.** A clinical
risk score should not take race as a predictor. They are retained only as the
groupings for the equity audit — which is a different job for the variable
entirely, and the distinction is the whole point:

- *As a model input*, race/gender would let the model encode "being in group X →
  higher risk". That is what most clinical and government settings restrict.
- *As an audit grouping*, race/gender are essential — you cannot check whether a
  group is served inequitably without grouping by it. In many health systems
  this stratified reporting is mandated.

Age is retained as a predictor (clinically legitimate) and is also audited.

## Model and calibration

- A class-weighted logistic-regression baseline is compared against class-
  weighted LightGBM using stratified 5-fold cross-validation. LightGBM is
  selected, though the two are effectively tied.
- The selected model is wrapped in isotonic calibration (`CalibratedClassifierCV`)
  so the outputs read as probabilities rather than arbitrary ranks.

## Performance (held-out test set)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.654 |
| PR-AUC | 0.186 |
| Brier score | 0.079 |
| Base rate | 9.0% |

At an operating point that flags the top 10% highest-risk patients (an
intervention-capacity framing, since a 0.5 cutoff never triggers on a 9% base
rate):

| Metric | Value |
| --- | --- |
| Precision | 23% |
| Recall | 26% |
| Lift vs base rate | 2.6× |

Even a marginal model gives 2.6× lift — flagging the top decile finds
readmissions at 2.6 times the background rate, which is operationally useful for
targeting. Calibration is good: predicted risk tracks observed rate across the
range and within groups.

## Equity audit

Error rates and calibration were checked across race and age groups at the
top-decile operating point.

**By race** — the model is well calibrated within each group (predicted risk ≈
observed rate group-by-group), so it is not systematically over- or under-scoring
any group. Selection rates differ (disparate-impact ratio 0.68, below the 0.80
"four-fifths" rule), but this largely tracks genuine base-rate differences: the
group flagged least also has the lowest true readmission rate. This surfaces a
real tension — **calibration and demographic parity cannot both hold when base
rates differ.** This model prioritises calibration (correct for a risk score)
and accepts unequal selection rates as a consequence, rather than distorting
probabilities to equalise flagging. That is a defensible, stateable choice.

**By age** — disparity is larger (disparate-impact ratio 0.38) but legitimate:
older patients are flagged more because they genuinely readmit more, and age is
a valid clinical predictor. The contrast is the point — age disparity is
intended; a race disparity would warrant scrutiny (here it survives it).

**Caveat:** the smallest groups (Asian n≈96, "Other" n≈229 in the test set) have
noisy metrics — an isolated group AUC of 0.85 is sample-size noise, not signal,
and is excluded from the headline ratios.

## Carrying this to practice

The equity code is grouping-agnostic — it takes whatever grouping column it is
given — so the same methodology extends to ethnicity-stratified equity
monitoring in a New Zealand health context by swapping the US race categories for
prioritised ethnicity (Māori, Pacific, Other). In that setting, three things
carry over:

- The **input-vs-audit distinction**: keep ethnicity out of the model, use it to
  measure equity. This resolves the common "we aren't allowed to use ethnicity"
  concern — you aren't using it to predict, you are using it to check.
- **Systems, not deficit**: report where the *system* under-serves a group so
  support can be targeted, not "group X is higher risk".
- **Māori data sovereignty (Te Mana Raraunga)**: ethnicity data used with Māori
  governance, for Māori benefit. For a single headline "equity score", a rate
  ratio (group rate ÷ reference rate) reported alongside the absolute gap is the
  most interpretable summary; a lone scalar hides direction and mechanism.

## Limitations

- Signal-limited: administrative data caps discrimination near ROC-AUC 0.65-0.68
  regardless of model. Richer clinical features (labs over time, meds, social
  determinants) would be needed to move it materially.
- Old, US-specific data; not transportable to other settings without revalidation.
- A demonstration, not a validated or deployed clinical tool.

## Reproducing

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m src.train      # fetch, clean, select, calibrate, persist
python -m src.evaluate   # metrics, calibration, equity audit, plots
```
