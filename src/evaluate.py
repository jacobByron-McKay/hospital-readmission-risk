"""Evaluate the held-out predictions and run the equity audit.

Everything here works from the saved test predictions (risk score, outcome and
sensitive attributes), so the evaluation is independent of how the model was
fitted. It reports discrimination (ROC/PR-AUC), calibration, a decision at a
realistic operating point, and - the part that matters most for a clinical
score - whether error rates and calibration hold up across race and age groups.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "models" / "test_predictions.csv"
REPORTS = ROOT / "reports"

# Intervention capacity: a follow-up team can only chase a fraction of
# discharges, so the score is used to flag the highest-risk slice rather than
# at a 0.5 cutoff that would never trigger on a 9% base rate.
CAPACITY = 0.10

# Groups smaller than this give unstable rates; they are still reported but
# excluded from the headline disparate-impact ratio.
MIN_GROUP = 200


def overall_metrics(df: pd.DataFrame) -> dict:
    return {
        "roc_auc": roc_auc_score(df["y_true"], df["y_proba"]),
        "pr_auc": average_precision_score(df["y_true"], df["y_proba"]),
        "brier": brier_score_loss(df["y_true"], df["y_proba"]),
        "base_rate": df["y_true"].mean(),
    }


def operating_point(df: pd.DataFrame, capacity: float) -> dict:
    threshold = df["y_proba"].quantile(1 - capacity)
    flagged = df["y_proba"] >= threshold
    tp = int(((flagged) & (df["y_true"] == 1)).sum())
    precision = tp / max(flagged.sum(), 1)
    recall = tp / max((df["y_true"] == 1).sum(), 1)
    lift = precision / df["y_true"].mean()
    return {
        "threshold": threshold,
        "flagged_rate": flagged.mean(),
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def equity_table(df: pd.DataFrame, group_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group_col):
        flagged = g["y_proba"] >= threshold
        pos = g["y_true"] == 1
        tp = int((flagged & pos).sum())
        fp = int((flagged & ~pos).sum())
        rows.append(
            {
                "group": name,
                "n": len(g),
                "base_rate": g["y_true"].mean(),
                "mean_pred": g["y_proba"].mean(),
                "flagged_rate": flagged.mean(),
                "recall": tp / max(pos.sum(), 1),
                "fpr": fp / max((~pos).sum(), 1),
                "precision": tp / max(flagged.sum(), 1),
                "auc": roc_auc_score(g["y_true"], g["y_proba"])
                if g["y_true"].nunique() == 2
                else np.nan,
            }
        )
    table = pd.DataFrame(rows).sort_values("n", ascending=False)
    stable = table[table["n"] >= MIN_GROUP]["flagged_rate"]
    table.attrs["disparate_impact"] = stable.min() / stable.max()
    return table


def plot_calibration(df: pd.DataFrame, path: Path):
    frac_pos, mean_pred = calibration_curve(df["y_true"], df["y_proba"], n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.5], [0, 0.5], "--", color="grey", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("observed readmission rate")
    ax.set_title("Calibration (isotonic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_group_flagging(table: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["group"].astype(str), table["flagged_rate"])
    ax.axhline(table["flagged_rate"].max() * 0.8, ls="--", color="red",
               label="80% of highest group")
    ax.set_ylabel("flagged rate")
    ax.set_title(title)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(PREDICTIONS)

    metrics = overall_metrics(df)
    op = operating_point(df, CAPACITY)
    print("overall:", {k: round(v, 4) for k, v in metrics.items()})
    print(f"operating point (top {CAPACITY:.0%} flagged):",
          {k: round(v, 4) for k, v in op.items()})

    plot_calibration(df, REPORTS / "calibration.png")

    for col, title, fname in [
        ("race", "Flagged rate by race", "equity_race.png"),
        ("age_band", "Flagged rate by age band", "equity_age.png"),
    ]:
        table = equity_table(df, col, op["threshold"])
        table.to_csv(REPORTS / f"equity_{col}.csv", index=False)
        plot_group_flagging(table, title, REPORTS / fname)
        print(f"\nequity by {col} (disparate-impact ratio "
              f"{table.attrs['disparate_impact']:.2f}):")
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
