"""Classify all articles from SQLite DB via Modal vLLM classifier endpoint.

Usage:
    python classify_all.py
    python classify_all.py --concurrency 100 --limit 1000
    python classify_all.py --no-resume  # start from scratch
"""

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(__file__))
from prompts.base_prompt import slim_system_instruction, cot_trigger


class ArticleClassification(BaseModel):
    climate_change: bool
    health: bool
    health_effects_of_climate_change: bool
    justification: str  # one line only


MODEL_NAME = "iRanadheer/lancet_qwen35_4b_full_merged"
MODAL_URL = "https://exec3ds--lancet-vllm-serve.modal.run/v1"
DB_PATH = "data/articles.db"
OUTPUT_PATH = Path("data/classifications.jsonl")


def get_client():
    return OpenAI(
        base_url=MODAL_URL,
        api_key=os.environ.get("HF_TOKEN", "not-needed"),
        timeout=30,
    )


def load_all_articles(db_path, limit=None):
    conn = sqlite3.connect(db_path)
    query = "SELECT id, title, content, language, country, source_uri FROM articles WHERE content IS NOT NULL AND content != ''"
    if limit:
        query += f" LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return rows


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(row["id"])
            except Exception:
                pass
    return done


def truncate_text(text, max_chars=8000):
    """Truncate text to roughly fit within token limits. ~4 chars per token."""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=True)
def call_api(client, text, system_prompt):
    return client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"### Article:\n{text}\n\nClassify this article. Provide a one-line justification."},
        ],
        response_format=ArticleClassification,
        max_tokens=750,
        temperature=0.01,
    )


def process_one(client, row):
    article_id, title, content, language, country, source = row
    text = truncate_text(f"{title or ''}\n\n{content or ''}")
    try:
        response = call_api(client, text, slim_system_instruction.strip())
        parsed = response.choices[0].message.parsed
        return {
            "id": article_id,
            "title": title or "",
            "language": language or "",
            "country": country or "",
            "source": source or "",
            "climate_change": parsed.climate_change,
            "health": parsed.health,
            "health_effects_of_climate_change": parsed.health_effects_of_climate_change,
            "justification": parsed.justification,
        }
    except Exception as e:
        return {
            "id": article_id,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Classify all articles from SQLite")
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10000, help="Articles to load from DB at a time")
    parser.add_argument("--limit", type=int, help="Limit total articles")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading all articles from {DB_PATH}...")
    articles = load_all_articles(DB_PATH, limit=args.limit)
    print(f"Loaded {len(articles):,} articles into memory")

    done_ids = set()
    if not args.no_resume:
        done_ids = load_done_ids(OUTPUT_PATH)
        if done_ids:
            articles = [r for r in articles if r[0] not in done_ids]
            print(f"Resuming: {len(done_ids):,} already done, {len(articles):,} remaining")

    if not articles:
        print("Nothing to classify.")
        return

    t0 = time.time()
    total_classified = 0
    total_errors = 0
    client = get_client()

    with open(OUTPUT_PATH, "a") as out, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_one, client, row): row[0] for row in articles}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            result = fut.result()
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if "error" in result:
                total_errors += 1
            else:
                total_classified += 1

            if total_classified % 1000 == 0:
                out.flush()

    elapsed = time.time() - t0
    print(f"\n{total_classified:,} classified in {elapsed/60:.1f}m ({total_classified/elapsed:.1f} art/s)")
    print(f"Errors: {total_errors}")

    elapsed = time.time() - t0
    print(f"\nDone. {total_classified:,} classified in {elapsed/60:.1f}m ({total_classified/elapsed:.1f} art/s)")
    print(f"Errors: {total_errors}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
