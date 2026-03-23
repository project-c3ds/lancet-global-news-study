#!/usr/bin/env python3
"""Collect articles from NewsAPI for sources under 10k articles.

Usage:
    python collect_newsapi.py                  # all sources under 10k
    python collect_newsapi.py --source lanacion.com.ar  # single source
    python collect_newsapi.py --limit 5        # first 5 sources only
"""

import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter, ReturnInfo, ArticleInfoFlags

load_dotenv()

SOURCES_CSV = Path("data/sources/sources_issues.csv")
OUTPUT_DIR = Path("data/newsapi_articles")
MAX_COUNT = 10000


def load_sources(source_filter=None, limit=None, max_count=None):
    """Load newsapi sources to collect."""
    sources = []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["collection_method"] != "newsapi":
                continue
            if not row.get("newsapi_uri"):
                continue
            count = int(row.get("newsapi_count") or 0)
            if max_count and count >= max_count:
                continue
            if source_filter and row["newsapi_uri"] != source_filter:
                continue
            sources.append(row)
    if limit:
        sources = sources[:limit]
    return sources


def collect_source(er, uri, output_path):
    """Collect all articles for a source and write to JSONL."""
    # Resume: count existing articles
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            return existing, "skipped"

    q = QueryArticlesIter(
        sourceUri=uri,
        dateStart="2020-01-01",
        dateEnd="2026-03-23",
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
    parser = argparse.ArgumentParser(description="Collect articles from NewsAPI")
    parser.add_argument("--source", help="Collect a single source by URI")
    parser.add_argument("--limit", type=int, help="Limit number of sources")
    parser.add_argument("--max-count", type=int, help="Only collect sources under this article count")
    args = parser.parse_args()

    api_key = os.environ["NEWSAPI_KEY"]
    er = EventRegistry(apiKey=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_sources(args.source, args.limit, args.max_count)
    if not sources:
        print("No sources found.")
        return

    usage_before = er.getUsageInfo()
    print(f"Tokens available: {usage_before['availableTokens']:,}")
    print(f"Tokens used so far: {usage_before['usedTokens']}")
    print(f"Sources to collect: {len(sources)}")
    print()

    for i, row in enumerate(sources, 1):
        uri = row["newsapi_uri"]
        name = row["name"]
        output_path = OUTPUT_DIR / f"{uri}.jsonl"

        count, status = collect_source(er, uri, output_path)

        if status == "skipped":
            print(f"  [{i}/{len(sources)}] {name} ({uri}): {count} articles (skipped)")
        else:
            print(f"  [{i}/{len(sources)}] {name} ({uri}): {count} articles")

    usage_after = er.getUsageInfo()
    total_tokens = usage_after["usedTokens"] - usage_before["usedTokens"]
    print(f"\nDone. Tokens used: {total_tokens}")
    print(f"Tokens remaining: {usage_after['availableTokens']}")


if __name__ == "__main__":
    main()
