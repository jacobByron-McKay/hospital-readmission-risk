# Walkthrough — how this project works, in plain English

This is a guided tour of the whole project. The goal is that after reading it
you understand *what* each piece does, *why* it's there, and can explain any of
it in an interview without notes. No syntax memorising — just the ideas and the
decisions. Read it top to bottom; it follows the order the code actually runs.

## The one-paragraph version

We take a public dataset of ~100,000 US hospital stays for diabetic patients and
predict which patients will be readmitted within 30 days of going home. A
follow-up team can only phone a fraction of discharged patients, so the model
ranks patients by risk and they work down the list. Along the way we do the
things that separate a toy notebook from senior work: we stop the model from
cheating, we make its probabilities honest, we explain what drives them, we
check it treats demographic groups fairly, and we wrap it so it can actually be
run and deployed.

## The shape of the whole thing

```
data.py      get the data, clean it, define the thing we're predicting
   |
features.py  turn raw columns into numbers a model can use
   |
train.py     try two models, pick one, make its probabilities trustworthy
   |
evaluate.py  measure how good it is, and audit it for fairness
explain.py   work out which features drive the predictions
   |
api.py       expose the model as a web service anyone can call
```

Everything else in the repo (`requirements.txt`, `Dockerfile`, `tests/`, CI) is
about making that reproducible and deployable.

---

## 1. Getting and cleaning the data — `src/data.py`

**What it does.** Downloads the dataset once, then cleans it into a tidy table
where each row is one hospital stay and there's a single column saying whether
that stay led to a readmission within 30 days.

**The thing we're predicting.** The raw data records readmission as one of three
values — within 30 days, after 30 days, or never. We collapse that to a simple
yes/no: readmitted within 30 days or not. Thirty days is the number hospitals
actually care about and get measured on, so that's the target
(`readmit_30d`). About 9% of stays are a "yes" — the event is rare, which
matters later.

**The important bit: stopping the model cheating.** This is where most tutorials
cut corners and we don't. Two guards:

- *Some patients can't be readmitted.* If a patient died or went to hospice,
  they will never come back — including them would poison the "no" group with
  cases that were never *able* to be a "yes". We drop those (`TERMINAL_DISCHARGES`).
- *The same patient appears many times.* A frequent-flyer patient might have ten
  stays in the data. If some of their stays land in the data we train on and
  others in the data we test on, the model can effectively memorise that person
  and look better than it really is. That's called **leakage** — the model
  "peeking" at something it won't have in real life. We fix it by keeping only
  each patient's *first* stay (`drop_duplicates` on the patient id). This alone
  drops the data from ~102k rows to ~70k, and it's the single most important
  correctness decision in the project.

**Tidying the rest.** The three diagnosis columns are raw medical codes (ICD-9)
with thousands of distinct values — too many to be useful — so we group them
into broad buckets like "circulatory", "respiratory", "diabetes"
(`_group_diagnosis`). We drop `weight` (missing for 97% of stays, so useless)
and `payer_code` (insurance type — a stand-in for wealth that we don't want a
*clinical* risk score leaning on).

> **If asked:** "How did you prevent data leakage?" → "The same patient recurs
> in the dataset, so I collapsed to each patient's first admission so no one
> spans train and test, and I removed encounters ending in death or hospice
> because those patients can't be readmitted."

---

## 2. Preparing the features — `src/features.py`

**What it does.** A model can't read words like "Cardiology" or a missing value
— it needs numbers. This file builds the recipe (a scikit-learn *pipeline*) that
turns the cleaned table into a numeric matrix, and it does it in a way that can
be re-applied to new data automatically.

**Two kinds of column, two treatments.**

- *Numbers* (length of stay, number of prior visits, age): filled in if missing
  and put on a common scale.
- *Categories* (diagnosis group, medical specialty, admission type): turned into
  yes/no columns using **one-hot encoding** — plainly, "Cardiology" becomes a
  column that's 1 for cardiology stays and 0 otherwise, and similarly for every
  category. Rare categories are folded into one "other" bucket so we don't
  create thousands of near-empty columns.

**The equity decision — this is the one to really understand.** Two columns,
`race` and `gender`, are deliberately *pulled out* and **not given to the
model** (`split_features` returns them separately). A clinical risk score
shouldn't take race as an input. But we keep them to one side, because we use
them later to *check* the model for fairness. The key idea: **the same variable
can be a forbidden input and a required checking-tool at the same time.** You
don't predict *from* race; you measure fairness *across* race. Age we do keep as
an input, because age genuinely and legitimately affects readmission risk.

> **If asked:** "Did you use protected attributes?" → "Not as model inputs — a
> clinical score shouldn't predict from race or gender. I held them out of the
> model but kept them to audit whether the model's errors and calibration are
> even across groups."

---

## 3. Training and choosing a model — `src/train.py`

**What it does.** Tries two different models, picks the better one fairly, then
makes its outputs trustworthy, and records the run.

**Two models on purpose.** A simple, transparent **logistic regression** as a
baseline, and **gradient boosting** (LightGBM) as the stronger, more flexible
option. Trying a simple model alongside a fancy one is deliberate: if the fancy
one barely beats the simple one, that tells you something real about the problem
(more on that below).

**Choosing fairly with cross-validation.** We don't just train once and eyeball
it. **Cross-validation** splits the training data into five parts, trains on four
and tests on the fifth, and rotates so every part gets a turn as the test. You
get five scores instead of one lucky-or-unlucky score, which is a far more
honest read. We use the *stratified* version so each split keeps the same 9%
readmission rate (otherwise a split might get almost no "yes" cases).

**Handling the 9% problem.** Because only 9% of cases are "yes", a lazy model
could say "no" every time and be 91% "accurate" while being useless. We tell both
models to weight the rare "yes" cases more heavily (`class_weight="balanced"`) so
they actually try to find them.

**Making the probabilities honest — calibration.** A model can be good at
*ranking* patients (riskier ones score higher) while its actual numbers are
nonsense — it might say "80%" for a group that only gets readmitted 30% of the
time. **Calibration** fixes the numbers so that when the model says 10%, about
10% of those patients really are readmitted. We wrap the chosen model in
*isotonic* calibration (`CalibratedClassifierCV`). This matters because we want
to hand clinicians a *risk*, not just a ranking.

**Recording the run — MLflow.** `mlflow` logs the settings and scores of each
run to a local folder (`mlruns/`), like a lab notebook. If you change something
next week and the score moves, you have a record of what you did. It's the
standard tool for this, so it's here both for good practice and because
employers look for it.

> **If asked:** "What's calibration and why bother?" → "A model can rank well but
> output meaningless probabilities. Calibration makes the numbers trustworthy —
> when it says 10%, roughly 10% really are readmitted — which is essential if a
> human is going to act on the probability."

---

## 4. Measuring it, and the fairness audit — `src/evaluate.py`

**What it does.** Scores the model on the held-out test set (data it never saw),
and — the part you care most about — audits whether it behaves fairly across
groups.

**The headline scores, plainly.**

- **ROC-AUC ≈ 0.65.** Think of this as "if you pick one patient who *was*
  readmitted and one who *wasn't*, how often does the model score the readmitted
  one higher?" 0.5 is a coin flip, 1.0 is perfect. 0.65 is modest.
- **Why 0.65 is the honest answer, not a failure.** Both the simple and the fancy
  model land within a whisker of each other. When a simple and a complex model
  agree, the ceiling is the *data*, not the model — 30-day readmission just isn't
  very predictable from admin data, and the published literature agrees (~0.64–
  0.68). A suspiciously high score on this dataset almost always means leakage.
  Being able to say "this is the real ceiling and here's how I know" is a senior
  signal.
- **Making it useful anyway — lift.** We flag the top 10% highest-risk patients
  (a realistic "how many can the follow-up team actually phone" framing). Among
  those, readmissions turn up at **2.6× the background rate** — that's the *lift*.
  So even a modest model meaningfully focuses a limited team.

**The fairness audit (`equity_table`).** For each group (by race, then by age) we
compute, at that top-10% cutoff: how often the group is flagged, how many real
cases we catch (recall), the false-alarm rate, and whether the predicted risk
matches the real rate (calibration). Two ideas do the heavy lifting:

- **Calibration within groups.** For every race group, the average predicted
  risk lines up with the actual readmission rate. So the model isn't
  systematically over- or under-scoring any group — the most important fairness
  property for a risk score.
- **The disparate-impact ratio.** The "four-fifths rule": if the least-flagged
  group is flagged at less than 80% the rate of the most-flagged group, that's a
  flag. By race it's 0.68 (below 0.80). *But* — and this is the nuance to be able
  to explain — that mostly reflects genuinely different underlying rates, not
  bias: the least-flagged group also has the lowest real readmission rate. This
  exposes a real tension: **you cannot have both equal flagging and honest
  probabilities when the groups genuinely differ.** We chose honest probabilities
  (right for a clinical score) and are transparent about the trade-off. By age
  the ratio is lower still (0.38), but that's *legitimate* — older patients
  genuinely readmit more, and age is a fair thing to use.

**Why this is the crown jewel.** This is exactly the "equity measure" your own
work wants. The audit code doesn't care *what* the grouping is — swap the US race
categories for Māori / Pacific / Other and the same method runs. The model card
spells out that transfer, including keeping ethnicity out of the model but in the
audit, framing findings as "where the system under-serves a group" rather than
"this group is higher-risk", and Māori data-sovereignty governance.

> **If asked:** "Is the model fair?" → "It's well-calibrated within every group,
> so it's not mis-scoring anyone. Flagging rates differ, but that tracks genuine
> differences in readmission rate — and calibration and equal-flagging can't both
> hold when base rates differ, so I chose calibration and documented the
> trade-off."

---

## 5. Explaining the model — `src/explain.py`

**What it does.** Opens the black box: for the predictions, works out how much
each feature pushed the risk up or down, using **SHAP**.

**SHAP in plain terms.** Imagine the model's output as a score being built up
from a baseline. SHAP fairly divides the credit (or blame) among the features
for each individual prediction — "this patient's risk is high *mostly because* of
their many prior inpatient visits, *and a bit because of* their age". Average
those across everyone and you see the model's overall drivers.

**What it found — and why that's reassuring.** The top drivers are prior
inpatient visits, where the patient was discharged to, age, length of stay, and
having a circulatory primary diagnosis. These are exactly the things clinical
common sense (and the literature) would expect. A model whose explanations make
sense is one you can trust; if the top driver had been something weird, that
would have hinted at leakage or a bug.

> **If asked:** "How do you explain the model?" → "SHAP values — they attribute
> each prediction to its features. The drivers are prior utilisation, discharge
> destination and age, which are clinically sensible, so the model is leaning on
> real signal."

---

## 6. Serving it — `src/api.py`

**What it does.** Wraps the trained model in a small web service (**FastAPI**) so
another system could send a patient's details and get back a risk score — this is
the difference between a model that sits in a notebook and one that could
actually be used.

**How it works.** On start-up it loads the saved model and the list of columns it
expects. A request sends whatever patient fields it has as JSON to `/predict`;
anything missing is left blank and handled by the pipeline. It returns a
calibrated probability and a simple band (low / elevated / high). There's also a
`/health` check so a deployment system can tell the service is alive. We tested
it on a low-risk and a high-risk example and it returned 0.07 and 0.23
respectively — sensible and clearly separated.

---

## 7. Making it reproducible and deployable

The last layer is what makes it engineering rather than a script:

- **`requirements.txt`** pins the exact library versions, so anyone can recreate
  the environment.
- **`tests/`** are small automated checks (the diagnosis grouping is correct,
  missing values are handled, the sensitive columns really are held out). If a
  future change breaks something, the tests catch it.
- **CI (`.github/workflows/ci.yml`)** runs those tests and a code-style check
  automatically on every push to GitHub — so the repo advertises that it's tested
  and clean.
- **`Dockerfile`** packages the whole thing into a container that trains the
  model and serves the API, so it runs identically on any machine.
- **`.gitignore`** keeps the virtual environment, data, model files and caches
  out of the repository — and, importantly, would keep any secret out too.

---

## The story to tell about this project

If you get one minute to describe it: *"I built an end-to-end 30-day readmission
risk model on public hospital data. The interesting part isn't the accuracy —
30-day readmission is only weakly predictable from admin data, and I show that
honestly rather than chasing a leaky high score. The value is in the rigour: I
prevented patient-level leakage, calibrated the probabilities so they're
trustworthy, explained the drivers with SHAP, and — the part I care about — built
a fairness audit that checks calibration and error rates across demographic
groups, with an explicit account of the calibration-versus-equal-treatment
trade-off. The whole thing is tested, containerised and served behind an API, and
the fairness method transfers directly to measuring equity for Māori and Pacific
populations in my own work."*

That paragraph is honest, senior, and covers correctness, communication,
ethics/governance, and engineering — the things the roles you're targeting
actually screen for.

## Mini-glossary

- **Target / label:** the thing we're predicting (here, readmitted within 30 days).
- **Feature:** an input the model uses to predict (age, prior visits, etc.).
- **Leakage:** the model accidentally getting information it wouldn't have in real
  life, making it look better than it is.
- **One-hot encoding:** turning a category column into a set of 0/1 columns.
- **Cross-validation:** rotating which slice of data is used for testing, to get a
  stable, honest score.
- **Class imbalance:** when one outcome is rare (9% here), which needs handling or
  the model ignores it.
- **Calibration:** making the predicted probabilities mean what they say.
- **ROC-AUC:** ranking ability — chance a real "yes" scores above a real "no".
- **Lift:** how much better than random your flagged group is at containing real
  cases.
- **Disparate impact / four-fifths rule:** a fairness check on whether groups are
  selected at similar rates.
- **SHAP:** a method that attributes each prediction to its features.
- **API:** a way for other software to call your model over the web.
- **CI:** automation that tests your code on every change.
