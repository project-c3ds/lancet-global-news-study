#!/usr/bin/env python3
"""Ingest newsapi_bad_low_sources into SQLite, assign country, move to newsapi_articles.

For each JSONL file in data/newsapi_bad_low_sources/:
  1. Insert articles into SQLite with collection_method='newsapi'
  2. Deduplicate: remove scrapai articles with the same URL
  3. Assign country from CSV mappings + manual overrides
  4. Move the JSONL file to data/newsapi_articles/

Merges ltn.com.tw subdomains (3c, ec, ent, health, news, sports) into
a single source_uri 'ltn.com.tw'.

Usage:
    python ingest_newsapi_sources.py
    python ingest_newsapi_sources.py --source aif.ru        # single source
    python ingest_newsapi_sources.py --dry-run               # preview only
"""

import argparse
import csv
import json
import shutil
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path("data")
BAD_LOW_DIR = DATA_DIR / "newsapi_bad_low_sources"
NEWSAPI_DIR = DATA_DIR / "newsapi_articles"
DB_PATH = DATA_DIR / "articles.db"

INSERT_SQL = """
INSERT INTO articles (url, title, content, source_uri, language, published_date,
                      extracted_at, author, collection_method, original_id, country)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Subdomains that merge into ltn.com.tw
LTN_SUBDOMAINS = {
    "3c.ltn.com.tw", "ec.ltn.com.tw", "ent.ltn.com.tw",
    "health.ltn.com.tw", "news.ltn.com.tw", "sports.ltn.com.tw",
}

# Manual country mappings for sources missing from CSVs
MANUAL_COUNTRY = {
    "3c.ltn.com.tw": "Taiwan",
    "ec.ltn.com.tw": "Taiwan",
    "ent.ltn.com.tw": "Taiwan",
    "health.ltn.com.tw": "Taiwan",
    "news.ltn.com.tw": "Taiwan",
    "sports.ltn.com.tw": "Taiwan",
    "ltn.com.tw": "Taiwan",
    "almasryalyoum.com": "Egypt",
    "economx.hu": "Hungary",
    "irishmirror.ie": "Ireland",
    "mgronline.com": "Thailand",
    "news.mingpao.com": "Hong Kong",
    "sowetan.co.za": "South Africa",
}


def build_country_mapping():
    """Build source_uri -> country mapping from CSVs + manual overrides."""
    mapping = {}

    # From top10_per_country.csv
    top10_path = DATA_DIR / "top10_per_country.csv"
    if top10_path.exists():
        with open(top10_path) as f:
            for row in csv.DictReader(f):
                url = row["website_url"].strip()
                if not url.startswith("http"):
                    url = "http://" + url
                domain = urlparse(url).netloc.lower().replace("www.", "").rstrip("/")
                mapping[domain] = row["country"]

    # From sources_issues.csv
    issues_path = DATA_DIR / "sources" / "sources_issues.csv"
    if issues_path.exists():
        with open(issues_path) as f:
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

    # Manual overrides
    mapping.update(MANUAL_COUNTRY)

    return mapping


def resolve_country(source_uri, country_map):
    """Look up country for a source_uri, trying parent domain as fallback."""
    country = country_map.get(source_uri)
    if not country:
        parts = source_uri.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[1:])
            country = country_map.get(parent)
    return country


def resolve_source_uri(source_uri):
    """Normalize source_uri, merging ltn.com.tw subdomains."""
    if source_uri in LTN_SUBDOMAINS:
        return "ltn.com.tw"
    return source_uri


def ingest_file(conn, filepath, country_map, dry_run=False):
    """Ingest a single JSONL file into SQLite. Returns (inserted, deduped)."""
    raw_source_uri = filepath.stem  # e.g. "aif.ru" from "aif.ru.jsonl"
    source_uri = resolve_source_uri(raw_source_uri)
    country = resolve_country(raw_source_uri, country_map)

    if not country:
        print(f"  WARNING: no country mapping for {raw_source_uri}")

    # Read articles
    articles = []
    line_num = 0
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                art = json.loads(line)
            except json.JSONDecodeError:
                continue
            line_num += 1
            title = (art.get("title") or "").strip()
            body = (art.get("body") or "").strip()
            if not title and not body:
                continue
            articles.append((line_num, art, title, body))

    if not articles:
        print(f"  {raw_source_uri}: no valid articles, skipping")
        return 0, 0

    if dry_run:
        print(f"  {raw_source_uri} -> {source_uri} ({country}): {len(articles):,} articles [DRY RUN]")
        return len(articles), 0

    cursor = conn.cursor()

    # Collect URLs for dedup check
    new_urls = {(art.get("url") or "").strip() for _, art, _, _ in articles}
    new_urls.discard("")

    # Find and remove scrapai duplicates by URL
    deduped = 0
    if new_urls:
        placeholders = ",".join("?" for _ in new_urls)
        deduped = cursor.execute(
            f"SELECT count(*) FROM articles WHERE url IN ({placeholders}) "
            f"AND collection_method = 'scrapai'",
            list(new_urls),
        ).fetchone()[0]

        if deduped > 0:
            cursor.execute(
                f"DELETE FROM articles WHERE url IN ({placeholders}) "
                f"AND collection_method = 'scrapai'",
                list(new_urls),
            )

    # Insert new articles
    batch = []
    for line_num, art, title, body in articles:
        url = (art.get("url") or "").strip()
        lang = (art.get("lang") or "").strip() or None

        batch.append((
            url,
            title,
            body,
            source_uri,
            lang,
            art.get("dateTime") or None,
            None,  # extracted_at
            None,  # author
            "newsapi",
            line_num,
            country,
        ))

        if len(batch) >= 5000:
            cursor.executemany(INSERT_SQL, batch)
            batch = []

    if batch:
        cursor.executemany(INSERT_SQL, batch)

    conn.commit()
    return len(articles), deduped


def move_file(filepath, source_uri):
    """Move JSONL to newsapi_articles/, merging ltn.com.tw subdomains."""
    NEWSAPI_DIR.mkdir(parents=True, exist_ok=True)
    raw_source = filepath.stem

    if raw_source in LTN_SUBDOMAINS:
        # Append to merged ltn.com.tw.jsonl
        dest = NEWSAPI_DIR / "ltn.com.tw.jsonl"
        with open(filepath, "r") as src, open(dest, "a") as dst:
            for line in src:
                dst.write(line)
        filepath.unlink()
        return dest
    else:
        dest = NEWSAPI_DIR / filepath.name
        shutil.move(str(filepath), str(dest))
        return dest


def main():
    parser = argparse.ArgumentParser(description="Ingest newsapi_bad_low_sources into SQLite")
    parser.add_argument("--source", help="Process a single source (e.g., aif.ru)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    if not BAD_LOW_DIR.exists():
        print(f"Source directory not found: {BAD_LOW_DIR}")
        return

    country_map = build_country_mapping()
    print(f"Country mappings: {len(country_map)} domains")

    # Discover files
    files = sorted(BAD_LOW_DIR.glob("*.jsonl"))
    if args.source:
        files = [f for f in files if f.stem == args.source]
    if not files:
        print("No files to process")
        return

    print(f"Files to ingest: {len(files)}")

    conn = None
    if not args.dry_run:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-512000")

    t_start = time.time()
    total_inserted = 0
    total_deduped = 0

    for i, filepath in enumerate(files, 1):
        inserted, deduped = ingest_file(conn, filepath, country_map, dry_run=args.dry_run)
        total_inserted += inserted
        total_deduped += deduped

        source_uri = resolve_source_uri(filepath.stem)
        country = resolve_country(filepath.stem, country_map)
        dedup_msg = f", deduped {deduped} scrapai" if deduped > 0 else ""
        print(f"  [{i}/{len(files)}] {filepath.stem} -> {source_uri} ({country}): "
              f"+{inserted:,}{dedup_msg}")

        # Move file after successful ingest
        if not args.dry_run and inserted > 0:
            dest = move_file(filepath, source_uri)
            print(f"    moved -> {dest}")

    elapsed = time.time() - t_start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Inserted: {total_inserted:,}")
    print(f"  Scrapai deduped: {total_deduped:,}")

    if conn:
        # Show updated counts
        row = conn.execute(
            "SELECT count(*) FROM articles WHERE collection_method = 'newsapi'"
        ).fetchone()
        print(f"  Total newsapi articles in DB: {row[0]:,}")
        conn.close()


if __name__ == "__main__":
    main()
