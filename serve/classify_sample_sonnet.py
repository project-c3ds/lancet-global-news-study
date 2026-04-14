"""Classify annotation sample using Sonnet 4.6 with chain-of-thought reasoning.

Produces <think> reasoning followed by YAML classification output.
Uses tenacity for retries and asyncio semaphore for parallel requests.

Usage:
    python serve/classify_sample_sonnet.py
    python serve/classify_sample_sonnet.py --input data/annotation_sample_5k.jsonl
    python serve/classify_sample_sonnet.py --concurrency 50
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.base_prompt import system_instruction, user_template

DEFAULT_INPUT = Path("data/annotation_sample_5k.jsonl")
DEFAULT_OUTPUT = Path("data/classifications/annotation_sample_sonnet.jsonl")

SYSTEM_PROMPT = system_instruction
USER_TEMPLATE = user_template


def truncate_text(text, max_chars=10000):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def parse_response(raw):
    """Extract thinking and YAML labels from model response."""
    result = {}

    thinking_match = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", raw, re.DOTALL)
    if thinking_match:
        result["reasoning"] = thinking_match.group(1).strip()

    for key in ["climate_change", "health", "health_effects_of_climate_change"]:
        match = re.search(rf"{key}:\s*\n\s*label:\s*(true|false)", raw, re.IGNORECASE)
        if match:
            result[key] = match.group(1).lower() == "true"
        else:
            match = re.search(rf"{key}:\s*(true|false)", raw, re.IGNORECASE)
            if match:
                result[key] = match.group(1).lower() == "true"

    return result


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def call_api(client, text):
    """Single API call with tenacity retry."""
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
    )
    return response.content[0].text


async def classify_article(client, article_id, title, content, semaphore):
    text = truncate_text(f"{title or ''}\n\n{content or ''}")
    async with semaphore:
        try:
            raw = await call_api(client, text)
            parsed = parse_response(raw)
            return {
                "id": article_id,
                "raw_response": raw,
                **parsed,
            }
        except Exception as e:
            return {"id": article_id, "error": str(e)}


async def async_main():
    parser = argparse.ArgumentParser(description="Classify with Sonnet 4.6 + CoT")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    args = parser.parse_args()

    import anthropic
    client = anthropic.AsyncAnthropic()

    # Load articles
    articles = []
    with open(args.input) as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles from {args.input}")

    # Resume support
    done_ids = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(done_ids)} already done")

    remaining = [a for a in articles if a["id"] not in done_ids]
    if not remaining:
        print("All done.")
        return

    print(f"To classify: {len(remaining)}, concurrency: {args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    completed = 0
    failed = 0
    parse_errors = 0

    out_f = open(args.output, "a")
    labels = ["climate_change", "health", "health_effects_of_climate_change"]

    from tqdm.asyncio import tqdm

    async def process_one(article):
        nonlocal completed, failed, parse_errors
        result = await classify_article(
            client,
            article["id"],
            article.get("title", ""),
            article.get("content", ""),
            semaphore,
        )
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if "error" in result:
            failed += 1
        elif not all(l in result for l in labels):
            parse_errors += 1
        else:
            completed += 1
        if (completed + failed + parse_errors) % 100 == 0:
            out_f.flush()

    tasks = [asyncio.create_task(process_one(a)) for a in remaining]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Classifying"):
        await coro

    out_f.flush()
    out_f.close()

    elapsed = time.time() - t0
    total_done = completed + failed + parse_errors
    rate = total_done / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s ({rate:.1f} art/s)")
    print(f"  Completed:    {completed:,}")
    print(f"  Parse errors: {parse_errors:,}")
    print(f"  API errors:   {failed:,}")

    # Distribution
    results = []
    with open(args.output) as f:
        for line in f:
            r = json.loads(line)
            if all(l in r for l in labels):
                results.append(r)

    n = len(results)
    if n > 0:
        cc = sum(1 for r in results if r["climate_change"])
        h = sum(1 for r in results if r["health"])
        ch = sum(1 for r in results if r["health_effects_of_climate_change"])
        print(f"\n--- Distribution (n={n}) ---")
        print(f"Climate Change:       {cc} ({cc/n*100:.1f}%)")
        print(f"Health:               {h} ({h/n*100:.1f}%)")
        print(f"Health Effects of CC: {ch} ({ch/n*100:.1f}%)")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
