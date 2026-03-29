#!/usr/bin/env python3
"""Add country column to SQLite articles table based on top10 CSV and sources_issues CSV.

Usage:
    python add_country.py
"""

import csv
import sqlite3
import time
from urllib.parse import urlparse

DB_PATH = "data/articles.db"


def build_country_mapping():
    mapping = {}

    # From top10
    with open("data/top10_per_country.csv") as f:
        for row in csv.DictReader(f):
            url = row["website_url"].strip()
            if not url.startswith("http"):
                url = "http://" + url
            domain = urlparse(url).netloc.lower().replace("www.", "").rstrip("/")
            mapping[domain] = row["country"]

    # From sources_issues
    with open("data/sources/sources_issues.csv") as f:
        for row in csv.DictReader(f):
            uri = row.get("newsapi_uri", "").strip()
            url = row.get("website_url", "").strip()
            country = row["country"]
            if uri:
                mapping[uri] = country
            if url:
                if not url.startswith("http"):
                    url = "http://" + url
                domain = urlparse(url).netloc.lower().replace("www.", "").rstrip("/")
                mapping[domain] = country

    # Manual fixes
    mapping["fijitimes.com.fj"] = "Fiji"

    return mapping


def main():
    mapping = build_country_mapping()
    print(f"Country mapping: {len(mapping)} domains")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # Add column if needed
    cols = [r[1] for r in conn.execute("PRAGMA table_info(articles)")]
    if "country" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN country TEXT")
        print("Added country column")
    else:
        print("country column already exists")

    # Get sources and counts
    sources = conn.execute(
        "SELECT source_uri, count(*) FROM articles GROUP BY source_uri ORDER BY source_uri"
    ).fetchall()

    t0 = time.time()
    updated = 0
    total_rows = 0

    for i, (src, count) in enumerate(sources, 1):
        country = mapping.get(src)
        if not country:
            parts = src.split(".")
            if len(parts) > 2:
                parent = ".".join(parts[1:])
                country = mapping.get(parent)

        if country:
            conn.execute("UPDATE articles SET country = ? WHERE source_uri = ?", (country, src))
            updated += 1
            total_rows += count
        else:
            print(f"  No country for: {src} ({count} articles)")

        if i % 50 == 0:
            conn.commit()
            elapsed = time.time() - t0
            print(f"  [{i}/{len(sources)}] {total_rows:,} rows updated ({elapsed:.0f}s)")

    conn.commit()

    # Create index on country
    print("Creating index on country...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_country ON articles(country)")
    conn.commit()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Updated {updated}/{len(sources)} sources ({total_rows:,} rows)")

    # Verify
    null_count = conn.execute("SELECT count(*) FROM articles WHERE country IS NULL").fetchone()[0]
    print(f"Articles with no country: {null_count}")

    print("\nTop 10 countries:")
    for row in conn.execute(
        "SELECT country, count(*) as n FROM articles GROUP BY country ORDER BY n DESC LIMIT 10"
    ):
        print(f"  {row[0]}: {row[1]:,}")

    conn.close()


if __name__ == "__main__":
    main()
