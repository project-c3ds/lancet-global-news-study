"""Classify climate articles using vLLM server with async OpenAI client.

Reads parquet files from data/climate_by_year/, classifies each article
using the fine-tuned Qwen 3.5 4B model, and saves results as JSONL.

Usage:
    python serve/classify_offline.py
    python serve/classify_offline.py --year 2024
    python serve/classify_offline.py --concurrency 500
    python serve/classify_offline.py --limit 1000
    python serve/classify_offline.py --dry-run

    # Point at Modal:
    VLLM_URL=https://exec3ds--lancet-classify-serve.modal.run/v1 python serve/classify_offline.py
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.base_prompt import slim_system_instruction

MODEL_NAME = os.environ.get("MODEL_NAME", "lancet-classify")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
INPUT_DIR = Path("data/climate_by_year")
OUTPUT_DIR = Path("data/classifications")
FAILED_IDS_PATH = OUTPUT_DIR / "failed_ids.txt"

SYSTEM_PROMPT = slim_system_instruction.strip()


class ArticleClassification(BaseModel):
    climate_change: bool
    health: bool
    health_effects_of_climate_change: bool


def truncate_text(text, max_chars=7000):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def get_client():
    return AsyncOpenAI(
        base_url=VLLM_URL,
        api_key=os.environ.get("HF_TOKEN", "not-needed"),
        timeout=120,
    )


async def classify_article(client, article_id, title, content):
    text = truncate_text(f"{title or ''}\n\n{content or ''}")
    for attempt in range(3):
            try:
                response = await client.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"### Article:\n{text}\n\nClassify this article."},
                    ],
                    response_format=ArticleClassification,
                    max_tokens=200,
                    temperature=0.01,
                )
                parsed = response.choices[0].message.parsed
                return {
                    "id": article_id,
                    "climate_change": parsed.climate_change,
                    "health": parsed.health,
                    "health_effects_of_climate_change": parsed.health_effects_of_climate_change,
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    with open(FAILED_IDS_PATH, "a") as f:
                        f.write(f"{article_id}\t{str(e)[:100]}\n")
                    return {"id": article_id, "error": str(e)}


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def classify_year(client, filepath, args, t0, total_classified):
    year = filepath.stem
    output_path = OUTPUT_DIR / f"{year}.jsonl"

    df = pd.read_parquet(filepath)
    print(f"\n=== {year}: {len(df):,} articles ===")

    done_ids = set()
    if not args.no_resume:
        done_ids = load_done_ids(output_path)
        if done_ids:
            print(f"  Resuming: {len(done_ids):,} already done")

    remaining = df[~df["id"].isin(done_ids)]
    if args.limit:
        remaining = remaining.head(args.limit - total_classified)

    if len(remaining) == 0:
        print(f"  All done for {year}")
        return 0, 0

    print(f"  To classify: {len(remaining):,}", flush=True)

    rows = list(remaining.itertuples(index=False))
    completed = 0
    failed = 0
    out_f = open(output_path, "a")
    pbar = tqdm(total=len(rows), desc=f"Year {year}")

    async def worker(queue):
        nonlocal completed, failed
        while True:
            row = await queue.get()
            if row is None:
                queue.task_done()
                break
            try:
                result = await classify_article(
                    client,
                    row.id,
                    getattr(row, "title", "") or "",
                    getattr(row, "content", "") or "",
                )
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if "error" in result:
                    failed += 1
                else:
                    completed += 1
                if (completed + failed) % 100 == 0:
                    out_f.flush()
                pbar.update(1)
            except Exception:
                failed += 1
                pbar.update(1)
            queue.task_done()

    queue = asyncio.Queue(maxsize=args.concurrency * 2)

    # Start workers
    workers = [asyncio.create_task(worker(queue)) for _ in range(args.concurrency)]

    # Feed the queue
    for row in rows:
        await queue.put(row)

    # Signal workers to stop
    for _ in range(args.concurrency):
        await queue.put(None)

    await asyncio.gather(*workers)
    pbar.close()
    out_f.close()

    elapsed = time.time() - t0
    rate = (total_classified + completed) / elapsed if elapsed > 0 else 0
    print(f"  Completed: {completed:,}, Failed: {failed:,}, Rate: {rate:.1f} art/s")
    return completed, failed


async def async_main():
    parser = argparse.ArgumentParser(description="Classify climate articles")
    parser.add_argument("--year", type=int, help="Classify only this year")
    parser.add_argument("--concurrency", type=int, default=500, help="Concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit total articles")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Test connection and show plan")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    total = 0
    for f in files:
        df = pd.read_parquet(f, columns=["id"])
        total += len(df)
    print(f"Found {len(files)} files, {total:,} total articles")

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Server: {VLLM_URL}")
        print(f"Model: {MODEL_NAME}")
        print(f"Concurrency: {args.concurrency}")
        print(f"\nTesting connection...")
        try:
            client = get_client()
            models = await client.models.list()
            print(f"OK: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"FAILED: {e}")
        return

    client = get_client()
    t0 = time.time()
    total_classified = 0
    total_errors = 0

    for filepath in files:
        completed, failed = await classify_year(client, filepath, args, t0, total_classified)
        total_classified += completed
        total_errors += failed

    elapsed = time.time() - t0
    print(f"\nDone. {total_classified:,} classified in {elapsed/60:.1f}m ({total_classified/elapsed:.1f} art/s)")
    print(f"Errors: {total_errors:,}")
    print(f"Output: {OUTPUT_DIR}/")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
