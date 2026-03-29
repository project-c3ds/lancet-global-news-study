#!/usr/bin/env python3
"""Check NewsAPI counts for scrapai-only sources with < 5k articles.

Usage:
    python check_low_count_sources.py
"""

import csv
import os
import sqlite3
import time

from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter

load_dotenv()

DB_PATH = "data/articles.db"


def get_sources():
    """Get independent scrapai-only sources with 50-5000 articles."""
    db = sqlite3.connect(DB_PATH)
    rows = db.execute('''
        SELECT source_uri, COUNT(*) as cnt
        FROM articles
        WHERE collection_method = 'scrapai'
        AND source_uri NOT IN (
            SELECT DISTINCT source_uri FROM articles WHERE collection_method = 'newsapi'
        )
        GROUP BY source_uri
        HAVING cnt < 5000 AND cnt >= 50
        ORDER BY cnt ASC
    ''').fetchall()

    all_sources = set(r[0] for r in db.execute(
        'SELECT DISTINCT source_uri FROM articles'
    ).fetchall())
    db.close()

    filtered = []
    for source, count in rows:
        if ":" in source:
            continue
        parts = source.split(".")
        is_subdomain = False
        if len(parts) > 2:
            parent = ".".join(parts[1:])
            if parent in all_sources:
                is_subdomain = True
            if len(parts) > 3:
                grandparent = ".".join(parts[2:])
                if grandparent in all_sources:
                    is_subdomain = True
        if not is_subdomain:
            filtered.append((source, count))
    return filtered


def main():
    er = EventRegistry(apiKey=os.environ["NEWSAPI_KEY"])
    sources = get_sources()

    print(f"Checking {len(sources)} sources on NewsAPI...\n")
    print(f"  {'source':40s} {'scrapai':>8s} {'newsapi':>8s}  {'action'}")
    print("  " + "-" * 75)

    results = []
    for source, scrapai_count in sources:
        uri = er.getSourceUri(source)
        time.sleep(0.2)

        newsapi_count = 0
        if uri:
            q = QueryArticlesIter(sourceUri=uri, dateStart="2020-01-01")
            newsapi_count = q.count(er)
            time.sleep(0.2)

        action = ""
        if newsapi_count > scrapai_count:
            action = "UPGRADE"
        elif newsapi_count > 0:
            action = "available"
        else:
            action = "not found"

        print(f"  {source:40s} {scrapai_count:>8,} {newsapi_count:>8,}  {action}")

        results.append({
            "source": source,
            "scrapai_count": scrapai_count,
            "newsapi_uri": uri or "",
            "newsapi_count": newsapi_count,
            "action": action,
        })

    # Save
    outpath = "results/low_count_newsapi_check.csv"
    os.makedirs("results", exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    upgradeable = [r for r in results if r["action"] == "UPGRADE"]
    available = [r for r in results if r["newsapi_count"] > 0]
    print(f"\nAvailable: {len(available)}/{len(sources)}")
    print(f"Upgradeable (newsapi > scrapai): {len(upgradeable)}")
    print(f"Saved to {outpath}")

    usage = er.getUsageInfo()
    print(f"Tokens remaining: {usage['availableTokens']:,}")


if __name__ == "__main__":
    main()
