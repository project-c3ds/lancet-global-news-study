"""Evaluate distilled-model predictions against the 150-article human gold standard.

Reads `validation/sample_150_eval_merged.csv`, which has `human_<label>` and
`model_<label>` columns side by side, and prints the exact-match rate and
per-label classification metrics used in the appendix.

Note: earlier versions of the merged CSV carried `agree_<label>` columns that
were computed at annotation time and contained bugs for rows where the model
output failed to parse (the agree flag was set to False even when both human
and model labels were False). This script recomputes agreement directly from
the `human_*` and `model_*` columns and ignores the stored `agree_*` values.

Usage:
    python classification/eval_human.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

LABELS = [
    "climate_change",
    "health",
    "health_effects_of_climate_change",
    "health_effects_of_extreme_weather",
]

CSV_PATH = Path(__file__).resolve().parent / "validation" / "sample_150_eval_merged.csv"


def _to_bool(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} articles from {CSV_PATH.name}")

    for label in LABELS:
        for prefix in ("human", "model"):
            df[f"{prefix}_{label}"] = df[f"{prefix}_{label}"].apply(_to_bool)

    cols = [f"{p}_{l}" for p in ("human", "model") for l in LABELS]
    clean = df.dropna(subset=cols).copy()
    print(f"Rows with both human + model labels: {len(clean)}\n")

    all_labels = LABELS + ["none"]
    y_true, y_pred = [], []
    for _, row in clean.iterrows():
        true_vals = [bool(row[f"human_{l}"]) for l in LABELS]
        pred_vals = [bool(row[f"model_{l}"]) for l in LABELS]
        y_true.append(true_vals + [not any(true_vals)])
        y_pred.append(pred_vals + [not any(pred_vals)])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    exact = int(sum(np.array_equal(t, p) for t, p in zip(y_true, y_pred)))
    print(f"Exact match: {exact}/{len(clean)} ({exact / len(clean) * 100:.1f}%)\n")

    print("=== PER LABEL ===")
    for i, label in enumerate(all_labels):
        print(f"--- {label} ---")
        print(
            classification_report(
                y_true[:, i], y_pred[:, i],
                target_names=["False", "True"], digits=3, zero_division=0,
            )
        )

    print("=== MULTILABEL ===")
    print(
        classification_report(
            y_true, y_pred,
            target_names=all_labels, digits=3, zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
