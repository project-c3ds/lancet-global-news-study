"""Classify climate articles using vLLM server (HTTP).

Reads parquet files from data/climate_by_year/, classifies each article
using the fine-tuned Qwen 3.5 4B model, and saves results as JSONL.

Usage:
    python serve/classify_offline.py
    python serve/classify_offline.py --year 2024
    python serve/classify_offline.py --concurrency 100 --batch-size 32
    python serve/classify_offline.py --limit 1000
    python serve/classify_offline.py --dry-run
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.base_prompt import slim_system_instruction, cot_trigger

MODEL_NAME = "lancet-classify"
VLLM_URL = "http://localhost:8000/v1"
INPUT_DIR = Path("data/climate_by_year")
OUTPUT_DIR = Path("data/classifications")


class ArticleClassification(BaseModel):
    climate_change: bool
    health: bool
    health_effects_of_climate_change: bool
    reasoning: str


def get_client():
    return OpenAI(
        base_url=VLLM_URL,
        api_key=os.environ.get("HF_TOKEN", "not-needed"),
        timeout=120,
    )


def classify_article(client, article_id, title, content, max_retries=3):
    """Classify a single article. Returns (id, result_dict) or (id, None) on failure."""
    text = f"### Article:\n{title}\n\n{content}"[:20000]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": slim_system_instruction.strip()},
                    {"role": "user", "content": f"{text}\n\n{cot_trigger}"},
                ],
                response_format=ArticleClassification,
                max_tokens=500,
                temperature=0.01,
            )
            parsed = response.choices[0].message.parsed
            result = parsed.model_dump()
            result["id"] = article_id
            return article_id, result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return article_id, None


def load_done_ids(output_path):
    """Load already-classified article IDs from JSONL output."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(row["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description="Classify climate articles")
    parser.add_argument("--year", type=int, help="Classify only this year")
    parser.add_argument("--concurrency", type=int, default=100, help="Concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit total articles")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Test connection and show plan")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find input files
    if args.year:
        files = [INPUT_DIR / f"{args.year}.parquet"]
        if not files[0].exists():
            print(f"File not found: {files[0]}")
            return
    else:
        files = sorted(INPUT_DIR.glob("*.parquet"))

    if not files:
        print("No parquet files found.")
        return

    # Show plan
    total = 0
    for f in files:
        df = pd.read_parquet(f, columns=["id"])
        total += len(df)
    print(f"Found {len(files)} files, {total:,} total articles")

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Concurrency: {args.concurrency}")
        print(f"\nTesting connection to {VLLM_URL} ...")
        try:
            client = get_client()
            models = client.models.list()
            print(f"OK: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"FAILED: {e}")
        return

    client = get_client()
    t0 = time.time()
    total_classified = 0
    total_remaining = 0

    for filepath in files:
        year = filepath.stem
        output_path = OUTPUT_DIR / f"{year}.jsonl"

        df = pd.read_parquet(filepath)
        print(f"\n=== {year}: {len(df):,} articles ===")

        # Resume support
        done_ids = set()
        if not args.no_resume:
            done_ids = load_done_ids(output_path)
            if done_ids:
                print(f"  Resuming: {len(done_ids):,} already done")

        # Filter to remaining
        remaining = df[~df["id"].isin(done_ids)]
        if args.limit:
            remaining = remaining.head(args.limit - total_classified)

        if len(remaining) == 0:
            print(f"  All done for {year}")
            continue

        total_remaining += len(remaining)
        print(f"  To classify: {len(remaining):,}")

        # Open output file for appending
        out_f = open(output_path, "a")

        # Submit all articles concurrently
        completed = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {}
            for _, row in remaining.iterrows():
                title = row.get("title", "") or ""
                content = row.get("content", "") or ""
                fut = pool.submit(classify_article, client, row["id"], title, content)
                futures[fut] = row["id"]

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Year {year}"):
                article_id, result = fut.result()
                if result:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed += 1
                else:
                    failed += 1

                total_classified += 1

        out_f.close()

        elapsed = time.time() - t0
        rate = total_classified / elapsed if elapsed > 0 else 0
        print(f"  Completed: {completed:,}, Failed: {failed:,}")
        print(f"  Rate: {rate:.1f} articles/s")

    elapsed = time.time() - t0
    print(f"\nDone. {total_classified:,} articles in {elapsed/60:.1f}m ({total_classified/elapsed:.1f} art/s)")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
