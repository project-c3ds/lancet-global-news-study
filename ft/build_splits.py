"""Split sonnet_ft_v2.jsonl into train/eval/test with stratified sampling.

Stratifies on the composite label combo (4 binary labels → string key)
to ensure proportional representation of rare classes in each split.

Split ratio: 80/20 train+eval/test, then 90/10 train/eval.
Final: ~72% train, ~8% eval, ~20% test.

Usage:
    python ft/build_splits.py
"""

import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

INPUT_PATH = Path("ft/data/sonnet_ft_v2.jsonl")
OUTPUT_DIR = Path("ft/data")

LABELS = ["climate_change", "health", "health_effects_of_climate_change", "health_effects_of_extreme_weather"]


def extract_combo(row):
    """Extract label combo string from assistant message YAML."""
    text = row["messages"][2]["content"]
    combo = []
    for label in LABELS:
        if f"{label}:\n  label: true" in text:
            combo.append("T")
        else:
            combo.append("F")
    return "-".join(combo)


def main():
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    combos = [extract_combo(r) for r in rows]

    print(f"Total: {len(rows)}")
    print(f"\nCombo distribution:")
    for k, v in sorted(Counter(combos).items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # First split: 80% train+eval, 20% test
    traineval_rows, test_rows, traineval_combos, _ = train_test_split(
        rows, combos, test_size=0.2, random_state=42, stratify=combos
    )

    # Second split: 90% train, 10% eval (of the 80%)
    train_rows, eval_rows = train_test_split(
        traineval_rows, test_size=0.1, random_state=42, stratify=traineval_combos
    )

    splits = {"train": train_rows, "eval": eval_rows, "test": test_rows}

    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        split_combos = Counter(extract_combo(r) for r in data)
        print(f"\n{name}: {len(data)} rows → {path}")
        for k, v in sorted(split_combos.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
