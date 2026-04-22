"""Evaluate distilled-model predictions against held-out frontier (Claude) annotations.

Produces the distillation-fidelity numbers reported in the appendix
(91.4% exact match; per-label F1 CC 0.918, H 0.858, HECC 0.877, HEEW 0.872).
Handles both CoT (regex-parsed) and structured output predictions. Parse
failures are tracked separately and not silently defaulted.

Usage:
    python classification/eval_distillation.py /tmp/test_predictions_v2.jsonl --mode cot
    python classification/eval_distillation.py /tmp/test_predictions_structured.jsonl --mode structured
"""

import argparse
import json
import re

import numpy as np
from sklearn.metrics import classification_report

LABELS = [
    "climate_change",
    "health",
    "health_effects_of_climate_change",
    "health_effects_of_extreme_weather",
]


def parse_cot_labels(response):
    """Parse labels from CoT YAML response. Returns None for unparseable labels.

    Strips <think>...</think> reasoning first, then parses the YAML output.
    """
    # Remove think block to only parse the YAML output
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    result = {}
    for key in LABELS:
        match = re.search(rf"{key}:\s*\n\s*label:\s*(true|false)", cleaned, re.IGNORECASE)
        if match:
            result[key] = match.group(1).lower() == "true"
        else:
            result[key] = None  # explicitly mark as parse failure
    return result


def parse_structured_labels(pred):
    """Extract labels from structured output prediction."""
    return {k: pred.get(k) for k in LABELS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Predictions JSONL file")
    parser.add_argument("--mode", choices=["cot", "structured"], default="cot")
    args = parser.parse_args()

    results = []
    with open(args.input) as f:
        for line in f:
            results.append(json.loads(line))

    # Separate errors from valid
    if args.mode == "cot":
        valid = [r for r in results if "error" not in r]
        api_errors = len(results) - len(valid)
        for r in valid:
            r["pred"] = parse_cot_labels(r["response"])
    else:
        valid = [r for r in results if "error" not in r.get("pred", {})]
        api_errors = len(results) - len(valid)
        for r in valid:
            r["pred"] = parse_structured_labels(r["pred"])

    # Identify parse failures (any label is None)
    parse_failures = [r for r in valid if any(r["pred"][k] is None for k in LABELS)]

    print(f"Total: {len(results)}")
    print(f"API errors: {api_errors}")
    print(f"Parse failures: {len(parse_failures)} (penalized as wrong)")
    print(f"Evaluated: {len(valid)}")
    print()

    if parse_failures:
        print("Parse failure details:")
        for r in parse_failures[:5]:
            missing = [k for k in LABELS if r["pred"][k] is None]
            print(f"  Missing: {missing}")
        print()

    if not valid:
        print("No predictions to evaluate.")
        return

    # Build arrays — parse failures get pred=opposite of true (worst case penalty)
    all_labels = LABELS + ["none"]
    y_true = []
    y_pred = []
    for r in valid:
        # For missing labels, penalize: predict opposite of ground truth
        pred = {}
        for k in LABELS:
            if r["pred"][k] is None:
                pred[k] = not r["true_labels"][k]  # guaranteed wrong
            else:
                pred[k] = r["pred"][k]
        true_none = not any(r["true_labels"][k] for k in LABELS)
        pred_none = not any(pred[k] for k in LABELS)
        y_true.append([r["true_labels"][k] for k in LABELS] + [true_none])
        y_pred.append([pred[k] for k in LABELS] + [pred_none])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    exact = sum(1 for t, p in zip(y_true, y_pred) if np.array_equal(t, p))
    print(f"Exact match: {exact}/{len(valid)} ({exact / len(valid) * 100:.1f}%)")
    print()

    # Per-label reports
    print("=== PER LABEL ===")
    for i, label in enumerate(all_labels):
        print(f"--- {label} ---")
        print(classification_report(
            y_true[:, i], y_pred[:, i],
            target_names=["False", "True"], digits=3, zero_division=0,
        ))

    # Multilabel report
    print("=== MULTILABEL ===")
    print(classification_report(
        y_true, y_pred,
        target_names=all_labels, digits=3, zero_division=0,
    ))


if __name__ == "__main__":
    main()
