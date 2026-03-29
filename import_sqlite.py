#!/usr/bin/env python3
"""Import all articles into a SQLite database with a unified schema.

Imports from two sources:
  - world_news_premium*/ crawl files (collection_method='scrapai')
  - newsapi_articles/ JSONL files (collection_method='newsapi')

Usage:
    python import_sqlite.py                          # import everything
    python import_sqlite.py --only scrapai           # only premium crawls
    python import_sqlite.py --only newsapi           # only newsapi
    python import_sqlite.py --db data/articles.db    # custom DB path
"""

import argparse
import gzip
import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path("data")
PREMIUM_FOLDERS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]
NEWSAPI_DIR = DATA_DIR / "newsapi_articles"
DEFAULT_DB = DATA_DIR / "articles.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    source_uri TEXT NOT NULL,
    language TEXT,
    published_date TEXT,
    extracted_at TEXT,
    author TEXT,
    collection_method TEXT NOT NULL,
    original_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_source_uri ON articles(source_uri);
CREATE INDEX IF NOT EXISTS idx_language ON articles(language);
CREATE INDEX IF NOT EXISTS idx_lang_source ON articles(language, source_uri);
CREATE INDEX IF NOT EXISTS idx_collection_method ON articles(collection_method);
CREATE INDEX IF NOT EXISTS idx_url ON articles(url);
"""

INSERT_SQL = """
INSERT INTO articles (url, title, content, source_uri, language, published_date,
                      extracted_at, author, collection_method, original_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Source -> language mapping from metadata CSV
def build_source_language_map():
    meta_path = DATA_DIR / "top10_per_country.csv"
    lang_map = {}
    if not meta_path.exists():
        return lang_map
    import csv
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("website_url", "")
                   .replace("http://", "").replace("https://", "")
                   .replace("www.", "").rstrip("/"))
            lang = row.get("language", "").strip()
            if url and lang:
                lang_map[url] = lang
    return lang_map


def domain_from_url(url):
    """Extract domain from URL, removing www. prefix."""
    if not url:
        return ""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def import_scrapai(conn, lang_map):
    """Import articles from world_news_premium crawl files."""
    cursor = conn.cursor()
    total = 0
    sources_done = 0

    # Discover all source directories
    source_dirs = []
    for folder in PREMIUM_FOLDERS:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            continue
        for source_dir in sorted(folder_path.iterdir()):
            crawl_dir = source_dir / "crawls"
            if crawl_dir.is_dir():
                files = sorted(crawl_dir.glob("*.jsonl.gz"))
                if files:
                    source_dirs.append((source_dir.name, files))

    print(f"  Found {len(source_dirs)} scrapai sources")
    t_start = time.time()

    for source_name, crawl_files in source_dirs:
        # Derive source_uri from folder name (e.g., "kompas_com" -> "kompas.com")
        # This avoids subdomain issues from URL extraction
        folder_domain = source_name.replace("_", ".")
        language = lang_map.get(folder_domain, "")
        line_num = 0
        batch = []

        for crawl_file in crawl_files:
            with gzip.open(crawl_file, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        art = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = (art.get("title") or "").strip()
                    content = (art.get("content") or "").strip()
                    if not title and not content:
                        continue

                    line_num += 1
                    url = (art.get("url") or "").strip()

                    batch.append((
                        url,
                        title,
                        content,
                        folder_domain,
                        language or None,
                        art.get("published_date") or None,
                        art.get("extracted_at") or None,
                        art.get("author") or None,
                        "scrapai",
                        line_num,
                    ))

                    if len(batch) >= 5000:
                        cursor.executemany(INSERT_SQL, batch)
                        total += len(batch)
                        batch = []

        if batch:
            cursor.executemany(INSERT_SQL, batch)
            total += len(batch)

        sources_done += 1
        if sources_done % 25 == 0 or sources_done == len(source_dirs):
            conn.commit()
            elapsed = time.time() - t_start
            rate = total / elapsed if elapsed > 0 else 0
            print(
                f"    [{sources_done}/{len(source_dirs)}] "
                f"{total:,} articles | {rate:,.0f}/s",
                flush=True,
            )

    conn.commit()
    return total


def import_newsapi(conn, lang_map):
    """Import articles from newsapi_articles/ JSONL files."""
    cursor = conn.cursor()
    total = 0

    if not NEWSAPI_DIR.exists():
        print("  newsapi_articles/ not found, skipping")
        return 0

    jsonl_files = sorted(NEWSAPI_DIR.glob("*.jsonl"))
    print(f"  Found {len(jsonl_files)} newsapi files")
    t_start = time.time()

    for file_idx, jsonl_file in enumerate(jsonl_files):
        line_num = 0
        batch = []

        with open(jsonl_file, "r", encoding="utf-8") as fh:
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

                source_uri = (art.get("source_uri") or "").strip()
                url = (art.get("url") or "").strip()
                if not source_uri:
                    source_uri = domain_from_url(url)

                # newsapi lang codes are ISO 639-3 (e.g., 'spa', 'eng')
                # Store as-is; can normalise later
                lang = (art.get("lang") or "").strip() or None

                # Language mapping: try source_uri in lang_map
                language = lang_map.get(source_uri, "") or lang

                batch.append((
                    url,
                    title,
                    body,
                    source_uri,
                    language or None,
                    art.get("dateTime") or None,
                    None,  # no extracted_at
                    None,  # no author
                    "newsapi",
                    line_num,
                ))

                if len(batch) >= 5000:
                    cursor.executemany(INSERT_SQL, batch)
                    total += len(batch)
                    batch = []

        if batch:
            cursor.executemany(INSERT_SQL, batch)
            total += len(batch)

        if (file_idx + 1) % 20 == 0 or file_idx + 1 == len(jsonl_files):
            conn.commit()
            elapsed = time.time() - t_start
            rate = total / elapsed if elapsed > 0 else 0
            print(
                f"    [{file_idx+1}/{len(jsonl_files)}] "
                f"{total:,} articles | {rate:,.0f}/s",
                flush=True,
            )

    conn.commit()
    return total


def main():
    parser = argparse.ArgumentParser(description="Import articles into SQLite")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Database path")
    parser.add_argument("--only", choices=["scrapai", "newsapi"],
                        help="Import only one source type")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-512000")  # 512MB cache
    conn.execute("PRAGMA temp_store=MEMORY")

    # Create schema
    conn.executescript(SCHEMA)

    lang_map = build_source_language_map()
    print(f"Language mappings: {len(lang_map)} sources")

    t_total = time.time()
    grand_total = 0

    if args.only != "newsapi":
        print("\nImporting scrapai (world_news_premium)...")
        count = import_scrapai(conn, lang_map)
        grand_total += count
        print(f"  Scrapai total: {count:,}")

    if args.only != "scrapai":
        print("\nImporting newsapi...")
        count = import_newsapi(conn, lang_map)
        grand_total += count
        print(f"  Newsapi total: {count:,}")

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f}m — {grand_total:,} articles total")

    # Show stats
    cursor = conn.cursor()
    cursor.execute("SELECT collection_method, COUNT(*) FROM articles GROUP BY collection_method")
    print("\nBy collection method:")
    for method, count in cursor.fetchall():
        print(f"  {method:15s} {count:>10,}")

    cursor.execute("""
        SELECT language, COUNT(*) FROM articles
        WHERE language IS NOT NULL
        GROUP BY language ORDER BY COUNT(*) DESC LIMIT 15
    """)
    print("\nTop languages:")
    for lang, count in cursor.fetchall():
        print(f"  {lang:20s} {count:>10,}")

    cursor.execute("SELECT COUNT(DISTINCT source_uri) FROM articles")
    print(f"\nUnique sources: {cursor.fetchone()[0]:,}")

    # DB size
    conn.close()
    size_gb = db_path.stat().st_size / (1024**3)
    print(f"Database size: {size_gb:.1f}GB")


if __name__ == "__main__":
    main()
