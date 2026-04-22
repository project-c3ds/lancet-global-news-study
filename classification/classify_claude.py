"""Classify sample articles using Claude via LiteLLM.

Reads a sample JSONL of articles and writes Claude's labels + reasoning as JSONL.

Usage:
    python classification/classify_claude.py
    python classification/classify_claude.py --concurrency 10
    python classification/classify_claude.py --model claude-sonnet-4-20250514
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts.base_prompt import slim_system_instruction

INPUT_PATH = Path("data/classifications/sample_500_with_articles.jsonl")
OUTPUT_PATH = Path("data/classifications/sample_500_claude.jsonl")

SYSTEM_PROMPT = slim_system_instruction.strip()


class ArticleClassification(BaseModel):
    reasoning: str
    climate_change: bool
    health: bool
    health_effects_of_climate_change: bool


def truncate_text(text, max_chars=10000):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


async def classify_article(article_id, title, content, model, semaphore):
    import litellm

    text = truncate_text(f"{title or ''}\n\n{content or ''}")
    async with semaphore:
        for attempt in range(3):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"### Article:\n{text}\n\nClassify this article. Respond with JSON matching this schema: {{\"reasoning\": \"<brief reasoning>\", \"climate_change\": bool, \"health\": bool, \"health_effects_of_climate_change\": bool}}"},
                    ],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "ArticleClassification",
                        "strict": True,
                        "schema": ArticleClassification.model_json_schema(),
                    }},
                    max_tokens=500,
                    temperature=0,
                )
                parsed = json.loads(response.choices[0].message.content)
                # Validate with pydantic
                result = ArticleClassification(**parsed)
                return {
                    "id": article_id,
                    "climate_change": result.climate_change,
                    "health": result.health,
                    "health_effects_of_climate_change": result.health_effects_of_climate_change,
                    "reasoning": result.reasoning,
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"id": article_id, "error": str(e)}


async def async_main():
    parser = argparse.ArgumentParser(description="Classify sample with Claude")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514", help="LiteLLM model string")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    args = parser.parse_args()

    # Load sample
    articles = []
    with open(INPUT_PATH) as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles from {INPUT_PATH}")

    # Resume support
    done_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
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

    print(f"To classify: {len(remaining)}, model: {args.model}, concurrency: {args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    completed = 0
    failed = 0

    out_f = open(OUTPUT_PATH, "a")

    from tqdm.asyncio import tqdm

    async def process_one(article):
        nonlocal completed, failed
        result = await classify_article(
            article["id"],
            article.get("title", ""),
            article.get("content", ""),
            args.model,
            semaphore,
        )
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if "error" in result:
            failed += 1
        else:
            completed += 1
        if (completed + failed) % 50 == 0:
            out_f.flush()

    tasks = [asyncio.create_task(process_one(a)) for a in remaining]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Classifying"):
        await coro

    out_f.flush()
    out_f.close()

    elapsed = time.time() - t0
    print(f"\nDone. {completed} classified, {failed} failed in {elapsed:.1f}s ({completed/elapsed:.1f} art/s)")
    print(f"Output: {OUTPUT_PATH}")

    # Show distribution
    results = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)

    n = len(results)
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
