"""Build OpenAI fine-tuning JSONL from Sonnet classification results.

Reads articles and their Sonnet classifications, then outputs OpenAI
fine-tuning format (system/user/assistant messages) to ft/data/.

Usage:
    python ft/build_openai_ft_data.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts.base_prompt import system_instruction, user_template

MAX_CHARS = 10000

ARTICLES_PATH = ROOT / "data" / "annotation_sample_5k.jsonl"
CLASSIFICATIONS_PATH = ROOT / "data" / "classifications" / "annotation_sample_sonnet.jsonl"
OUTPUT_PATH = ROOT / "ft" / "data" / "sonnet_ft.jsonl"

LABELS = ["climate_change", "health", "health_effects_of_climate_change"]


def main():
    # Load articles keyed by id
    articles = {}
    with open(ARTICLES_PATH) as f:
        for line in f:
            a = json.loads(line)
            articles[a["id"]] = a

    # Load classifications, filter to valid ones
    classifications = []
    skipped = 0
    with open(CLASSIFICATIONS_PATH) as f:
        for line in f:
            c = json.loads(line)
            if "error" in c or not all(k in c for k in LABELS) or "raw_response" not in c:
                skipped += 1
                continue
            classifications.append(c)

    print(f"Articles: {len(articles)}")
    print(f"Valid classifications: {len(classifications)}, skipped: {skipped}")

    # Build messages
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0

    with open(OUTPUT_PATH, "w") as out:
        for c in classifications:
            article = articles.get(c["id"])
            if article is None:
                missing += 1
                continue

            text = f"{article.get('title', '') or ''}\n\n{article.get('content', '') or ''}"
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "..."
            user_content = user_template.format(text=text)

            row = {
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": c["raw_response"]},
                ]
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written} to {OUTPUT_PATH}")
    if missing:
        print(f"Missing articles: {missing}")


if __name__ == "__main__":
    main()
