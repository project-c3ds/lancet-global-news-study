#!/usr/bin/env python3
"""Collect articles from NewsAPI for specific bad-quality scrapai sources.

Downloads articles to data/newsapi_bad_low_sources/ for later ingestion
via ingest_newsapi_sources.py.

Usage:
    python collect_bad_sources.py
    python collect_bad_sources.py --source telegraf.rs
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter, ReturnInfo, ArticleInfoFlags

load_dotenv()

OUTPUT_DIR = Path("data/newsapi_bad_low_sources")

# Sources to collect: (source_uri, newsapi_uri)
# newsapi_uri is usually the same as source_uri
SOURCES = [
    "udn.com",
    "telegraf.rs",
    "vecernji.hr",
    "independent.ie",
    "chinadaily.com.cn",
    "gazzetta.it",
    "scmp.com",
    "standardmedia.co.ke",
    "globes.co.il",
    "lematin.ma",
]


def collect_source(er, uri, output_path):
    """Collect all articles for a source and write to JSONL."""
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            return existing, "skipped"

    q = QueryArticlesIter(
        sourceUri=uri,
        dateStart="2020-01-01",
        dateEnd="2026-03-28",
    )

    ri = ReturnInfo(articleInfo=ArticleInfoFlags(
        body=True, title=True, concepts=False, categories=False,
        links=False, image=False, videos=False, extractedDates=False,
        socialScore=False, sentiment=False, location=False,
        duplicateList=False, originalArticle=False, storyUri=False,
    ))

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for art in q.execQuery(er, returnInfo=ri, sortBy="date"):
            record = {
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "body": art.get("body", ""),
                "lang": art.get("lang", ""),
                "dateTime": art.get("dateTime", ""),
                "source_uri": art.get("source", {}).get("uri", ""),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count, "done"


def main():
    parser = argparse.ArgumentParser(description="Collect bad scrapai sources from NewsAPI")
    parser.add_argument("--source", help="Collect a single source")
    args = parser.parse_args()

    api_key = os.environ["NEWSAPI_KEY"]
    er = EventRegistry(apiKey=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = SOURCES
    if args.source:
        sources = [s for s in sources if s == args.source]

    usage_before = er.getUsageInfo()
    print(f"Tokens available: {usage_before['availableTokens']:,}")
    print(f"Sources to collect: {len(sources)}")
    print()

    for i, uri in enumerate(sources, 1):
        output_path = OUTPUT_DIR / f"{uri}.jsonl"
        t0 = time.time()

        count, status = collect_source(er, uri, output_path)
        elapsed = time.time() - t0

        if status == "skipped":
            print(f"  [{i}/{len(sources)}] {uri}: {count:,} articles (skipped)")
        else:
            print(f"  [{i}/{len(sources)}] {uri}: {count:,} articles ({elapsed:.0f}s)")

    usage_after = er.getUsageInfo()
    tokens_used = usage_after["usedTokens"] - usage_before["usedTokens"]
    print(f"\nDone. Tokens used: {tokens_used:,}")
    print(f"Tokens remaining: {usage_after['availableTokens']:,}")


if __name__ == "__main__":
    main()
