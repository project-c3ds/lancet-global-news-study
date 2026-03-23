#!/usr/bin/env python3
"""Check NewsAPI (eventregistry) for article availability of missing/low-count sources.

Usage:
    python check_newsapi_sources.py                # all 167 sources
    python check_newsapi_sources.py --limit 1      # dry run with 1 source
"""

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter

load_dotenv()

SOURCES_CSV = Path("data/sources/source_status.csv")
OUTPUT_CSV = Path("data/sources/source_status.csv")
NEWSAPI_CSV = Path("data/sources/newsapi_sources.csv")


def load_sources(limit=None):
    """Load sources that need checking (not collected or < 1000 articles)."""
    sources = []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["collected"] == "No" or int(row["article_count"]) < 1000:
                sources.append(row)
    if limit:
        sources = sources[:limit]
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of sources to check")
    args = parser.parse_args()

    api_key = os.environ["NEWSAPI_KEY"]
    er = EventRegistry(apiKey=api_key)

    targets = load_sources(args.limit)
    print(f"Checking {len(targets)} sources...")

    # Build a set of target source names for quick lookup
    target_names = {row["name"] for row in targets}

    # Query NewsAPI for each target — use domain from URL, fall back to name
    results = {}
    for i, row in enumerate(targets, 1):
        name = row["name"]
        # Extract domain from website_url for more reliable matching
        from urllib.parse import urlparse
        domain = urlparse(row["website_url"].strip().rstrip("/")).hostname or ""
        domain = domain.removeprefix("www.")
        uri = er.getSourceUri(domain) if domain else None
        if not uri:
            uri = er.getSourceUri(name)

        if uri:
            q = QueryArticlesIter(sourceUri=uri, dateStart="2020-01-01")
            count = q.count(er)
            results[name] = {"newsapi_uri": uri, "newsapi_count": count}
            print(f"  [{i}/{len(targets)}] {name} -> {uri} ({count:,} articles)")
        else:
            results[name] = {"newsapi_uri": "", "newsapi_count": 0}
            print(f"  [{i}/{len(targets)}] {name} -> NOT FOUND")

    # Write newsapi_sources.csv with just the queried results
    with open(NEWSAPI_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "country", "name", "website_url", "reason", "article_count",
            "newsapi_uri", "newsapi_count",
        ])
        writer.writeheader()
        for row in targets:
            r = results[row["name"]]
            reason = "not_collected" if row["collected"] == "No" else "low_count"
            writer.writerow({
                "country": row["country"],
                "name": row["name"],
                "website_url": row["website_url"],
                "reason": reason,
                "article_count": row["article_count"],
                "newsapi_uri": r["newsapi_uri"],
                "newsapi_count": r["newsapi_count"],
            })
    print(f"Wrote {NEWSAPI_CSV}")

    # Re-read full CSV and add newsapi columns
    all_rows = []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for col in ["newsapi_uri", "newsapi_count"]:
            if col not in fieldnames:
                fieldnames.append(col)
        for row in reader:
            if row["name"] in results:
                row["newsapi_uri"] = results[row["name"]]["newsapi_uri"]
                row["newsapi_count"] = results[row["name"]]["newsapi_count"]
            else:
                row.setdefault("newsapi_uri", "")
                row.setdefault("newsapi_count", "")
            all_rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Updated {OUTPUT_CSV}")
    resolved = sum(1 for r in results.values() if r["newsapi_uri"])
    print(f"Resolved: {resolved}/{len(results)}")


if __name__ == "__main__":
    main()
