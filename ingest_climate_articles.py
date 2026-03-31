#!/usr/bin/env python3
"""Ingest collected climate articles into climate.db.

Reads from data/newsapi_articles_{year}/*_climate.jsonl and inserts into SQLite.

Usage:
    python ingest_climate_articles.py --year 2025
    python ingest_climate_articles.py --year 2024 --source nytimes.com
    python ingest_climate_articles.py --year 2025 --dry-run
"""

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "climate.db"
ARTICLES_DB = DATA_DIR / "articles.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT,
    title TEXT,
    content TEXT,
    source_uri TEXT,
    language TEXT,
    published_date TEXT,
    extracted_at TEXT,
    collection_method TEXT,
    country TEXT
);
CREATE INDEX IF NOT EXISTS idx_source_uri ON articles(source_uri);
CREATE INDEX IF NOT EXISTS idx_country ON articles(country);
CREATE INDEX IF NOT EXISTS idx_language ON articles(language);
CREATE INDEX IF NOT EXISTS idx_published_date ON articles(published_date);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO articles (id, url, title, content, source_uri, language, published_date, extracted_at, collection_method, country)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def build_country_mapping():
    """Build source_uri -> country mapping from articles.db."""
    conn = sqlite3.connect(str(ARTICLES_DB))
    rows = conn.execute("""
        SELECT source_uri, country, COUNT(*) as n
        FROM articles
        WHERE country IS NOT NULL AND country != ''
        GROUP BY source_uri, country
        ORDER BY n DESC
    """).fetchall()
    conn.close()

    # Take the most common country per source
    mapping = {}
    for source, country, _ in rows:
        if source not in mapping:
            mapping[source] = country
    return mapping


def init_db():
    """Create climate.db and schema if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ingest_file(conn, filepath, year, country_map, dry_run=False):
    """Ingest a single JSONL file. Returns (inserted, skipped)."""
    # Parse source_uri from filename: {source_uri}_climate.jsonl
    stem = filepath.stem  # e.g., "nytimes.com_climate"
    source_uri = stem.replace("_climate", "")
    country = country_map.get(source_uri, "")
    now = datetime.utcnow().isoformat() + "Z"

    articles = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                art = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = (art.get("title") or "").strip()
            body = (art.get("body") or "").strip()
            if not title and not body:
                continue

            article_id = f"newsapi_{source_uri}_{year}_{i:06d}"

            articles.append((
                article_id,
                (art.get("url") or "").strip(),
                title,
                body,
                source_uri,
                (art.get("lang") or "").strip(),
                (art.get("dateTime") or "").strip(),
                now,
                "newsapi",
                country,
            ))

    if dry_run:
        return len(articles), 0

    cursor = conn.cursor()
    before = cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    cursor.executemany(INSERT_SQL, articles)
    conn.commit()
    after = cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    inserted = after - before

    return inserted, len(articles) - inserted


def main():
    parser = argparse.ArgumentParser(description="Ingest climate articles into climate.db")
    parser.add_argument("--year", type=int, required=True, help="Year to ingest")
    parser.add_argument("--source", help="Single source to ingest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = DATA_DIR / f"newsapi_articles_{args.year}"
    if not input_dir.exists():
        log.error(f"Directory not found: {input_dir}")
        return

    files = sorted(input_dir.glob("*_climate.jsonl"))
    if args.source:
        files = [f for f in files if f.stem.replace("_climate", "") == args.source]

    if not files:
        log.error("No files found")
        return

    log.info(f"Building country mapping from articles.db...")
    country_map = build_country_mapping()
    log.info(f"  {len(country_map)} source -> country mappings")

    conn = None
    if not args.dry_run:
        conn = init_db()

    log.info(f"Ingesting {len(files)} files from {input_dir} {'(DRY RUN)' if args.dry_run else ''}")

    t0 = time.time()
    total_inserted = 0
    total_skipped = 0

    for i, filepath in enumerate(files, 1):
        source_uri = filepath.stem.replace("_climate", "")
        country = country_map.get(source_uri, "???")

        inserted, skipped = ingest_file(conn, filepath, args.year, country_map, dry_run=args.dry_run)
        total_inserted += inserted
        total_skipped += skipped

        label = f"+{inserted:,}" if not args.dry_run else f"{inserted:,} articles"
        skip_label = f" ({skipped:,} skipped)" if skipped > 0 else ""
        log.info(f"  [{i}/{len(files)}] {source_uri:<30} {country:<20} {label}{skip_label}")

    elapsed = time.time() - t0

    if conn:
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
    else:
        total = total_inserted

    log.info(f"\nDone in {elapsed:.1f}s")
    log.info(f"  Inserted: {total_inserted:,}")
    log.info(f"  Skipped:  {total_skipped:,}")
    log.info(f"  Total in DB: {total:,}")


if __name__ == "__main__":
    main()
