"""Convert RECoT JSONL into OpenAI messages format for SFT fine-tuning.

Reads recot.jsonl and produces:
  - train.jsonl  (90% with RECoT reasoning)
  - eval.jsonl   (10% held-out split)

If recot.jsonl is missing the 'content' field (older runs), use --data-dir to
rejoin with the original annotated CSVs by article_id.

Usage:
    python -m recot.prepare_splits --input data/ft/2026-03-29/recot.jsonl
    python -m recot.prepare_splits --input data/ft/2026-03-29/recot.jsonl --data-dir results/labeled/annotated
    python -m recot.prepare_splits --input data/ft/2026-03-29/recot.jsonl --eval-ratio 0.1
"""

import argparse
import csv
import json
import os
import sys

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prompts.base_prompt import slim_system_instruction, cot_trigger


def load_content_map(data_dir: str) -> dict:
    """Build article_id -> content map from annotated CSVs."""
    content_map = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                aid = str(row.get("article_id", ""))
                content_map[aid] = row.get("content", "")
    return content_map


def build_sft_record(text: str, response: str) -> dict:
    """Convert raw RECoT record to OpenAI SFT chat messages format."""
    return {
        "messages": [
            {"role": "system", "content": slim_system_instruction.strip()},
            {"role": "user", "content": f"### Article:\n{text}\n\n{cot_trigger}"},
            {"role": "assistant", "content": response},
        ]
    }


def write_jsonl(records: list, path: str):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {path} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser(description="Convert RECoT to SFT messages format")
    parser.add_argument("--input", required=True, help="Input recot.jsonl path")
    parser.add_argument("--data-dir", help="Annotated CSV dir to rejoin content by article_id (for older recot files missing content)")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input)

    raw = []
    with open(args.input) as f:
        for line in f:
            raw.append(json.loads(line))
    print(f"Loaded {len(raw)} RECoT records")

    # Rejoin content from source CSVs if needed
    content_map = None
    has_content = raw[0].get("content") if raw else None
    if not has_content and args.data_dir:
        content_map = load_content_map(args.data_dir)
        print(f"Loaded content for {len(content_map)} articles from {args.data_dir}")
    elif not has_content and not args.data_dir:
        print("WARNING: recot.jsonl missing 'content' field. Use --data-dir to rejoin with source CSVs.")

    records = []
    skipped = 0
    for row in raw:
        content = row.get("content", "")
        if not content and content_map:
            content = content_map.get(str(row.get("article_id", "")), "")
        if not content:
            skipped += 1
            continue
        text = f"{row.get('title', '')}\n\n{content}"
        records.append(build_sft_record(text, row["response"]))

    if skipped:
        print(f"Skipped {skipped} records with no content")

    languages = [row.get("language", "unknown") for row in raw if row.get("content") or content_map]
    languages = languages[:len(records)]
    train_idx, eval_idx = train_test_split(
        range(len(records)),
        test_size=args.eval_ratio,
        random_state=args.seed,
        stratify=languages,
    )

    print(f"\nSplit: {len(train_idx)} train / {len(eval_idx)} eval")
    print("Writing:")
    write_jsonl([records[i] for i in train_idx], os.path.join(output_dir, "train.jsonl"))
    write_jsonl([records[i] for i in eval_idx], os.path.join(output_dir, "eval.jsonl"))
    print("\nDone.")


if __name__ == "__main__":
    main()
