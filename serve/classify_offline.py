"""Classify climate articles using vLLM server (HTTP) with async IO.

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

import aiohttp
import pandas as pd
from dotenv import load_dotenv
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

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ArticleClassification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "climate_change": {"type": "boolean"},
                "health": {"type": "boolean"},
                "health_effects_of_climate_change": {"type": "boolean"},
            },
            "required": ["climate_change", "health", "health_effects_of_climate_change"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = slim_system_instruction.strip()


def truncate_text(text, max_chars=12000):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


async def classify_article(session, semaphore, article_id, title, content, url, api_key):
    text = truncate_text(f"{title or ''}\n\n{content or ''}")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"### Article:\n{text}\n\nClassify this article."},
        ],
        "response_format": RESPONSE_SCHEMA,
        "max_tokens": 200,
        "temperature": 0.01,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    f"{url}/chat/completions", json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        raise Exception(f"HTTP {resp.status}: {err[:200]}")
                    data = await resp.json()
                    content_str = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content_str)
                    parsed["id"] = article_id
                    return parsed
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
                row = json.loads(line)
                done.add(row["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def classify_year(filepath, args, t0, total_classified):
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

    api_key = os.environ.get("HF_TOKEN", "not-needed")
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)

    completed = 0
    failed = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for row in remaining.itertuples(index=False):
            task = classify_article(
                session, semaphore,
                row.id,
                getattr(row, "title", "") or "",
                getattr(row, "content", "") or "",
                VLLM_URL, api_key,
            )
            tasks.append(task)

        out_f = open(output_path, "a")
        async for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Year {year}"):
            result = await result
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if "error" in result:
                failed += 1
            else:
                completed += 1
            if completed % 1000 == 0:
                out_f.flush()

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
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{VLLM_URL}/models") as resp:
                    data = await resp.json()
                    print(f"OK: {[m['id'] for m in data['data']]}")
        except Exception as e:
            print(f"FAILED: {e}")
        return

    t0 = time.time()
    total_classified = 0
    total_errors = 0

    for filepath in files:
        completed, failed = await classify_year(filepath, args, t0, total_classified)
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
