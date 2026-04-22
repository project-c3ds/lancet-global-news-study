#!/usr/bin/env python3
"""Ingest climate-relevant articles from articles.db (scrapai) into climate.db.

Searches title + content for climate keywords in the article's language + English.
Uses word boundary matching for Latin-script languages, substring for CJK/Thai.

Usage:
    python collection/ingest_scrapai_climate.py
    python collection/ingest_scrapai_climate.py --language English
    python collection/ingest_scrapai_climate.py --source eltiempo.com
    python collection/ingest_scrapai_climate.py --dry-run
"""

import argparse
import json
import logging
import re
import sqlite3
import time
from pathlib import Path

DATA_DIR = Path("data")
ARTICLES_DB = DATA_DIR / "articles.db"
CLIMATE_DB = DATA_DIR / "climate.db"
TRANSLATIONS_JSON = DATA_DIR / "keywords" / "keyword_translations.json"

BATCH_SIZE = 10000

# Languages where word boundaries don't apply (no spaces between words)
CJK_LANGUAGES = {"zho", "jpn", "tha", "kor"}

LANG_TO_ISO = {
    "Albanian": "sqi", "Arabic": "ara", "Bengali": "ben", "Bulgarian": "bul",
    "Chinese": "zho", "Croatian": "hrv", "Czech": "ces", "Danish": "dan",
    "Dhivehi": "div", "Dutch": "nld", "English": "eng", "Finnish": "fin",
    "French": "fra", "Ganda": "lug", "German": "deu", "Greek": "ell",
    "Hebrew": "heb", "Hungarian": "hun", "Indonesian": "ind", "Irish": "gle",
    "Italian": "ita", "Japanese": "jpn", "Korean": "kor", "Malay": "msa",
    "Malayalam": "mal", "Norwegian": "nob", "Persian": "fas", "Polish": "pol",
    "Portuguese": "por", "Romanian": "ron", "Russian": "rus", "Serbian": "srp",
    "Slovak": "slk", "Spanish": "spa", "Swahili": "swa", "Swedish": "swe",
    "Thai": "tha", "Turkish": "tur", "Ukrainian": "ukr", "Urdu": "urd",
    "Vietnamese": "vie",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_translations():
    with open(TRANSLATIONS_JSON) as f:
        return json.load(f)


def build_keyword_matcher(keywords, is_cjk=False):
    """Build a function that checks if any keyword matches in text.

    For Latin-script: uses word boundary regex.
    For CJK/Thai: uses case-insensitive substring.
    """
    if is_cjk:
        keywords_lower = [k.lower() for k in keywords]

        def matcher(text):
            text_lower = text.lower()
            for kw in keywords_lower:
                if kw in text_lower:
                    return True
            return False
        return matcher
    else:
        # Build a single compiled regex with alternation for speed
        # Escape keywords and join with | , wrap in word boundaries
        escaped = [re.escape(k) for k in keywords]
        pattern = re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', re.IGNORECASE)

        def matcher(text):
            return pattern.search(text) is not None
        return matcher


def build_matchers_by_language(translations):
    """Build keyword matcher per language (local + english)."""
    eng_keywords = translations["climate"].get("eng", [])
    matchers = {}

    for lang_full, iso in LANG_TO_ISO.items():
        local_keywords = translations["climate"].get(iso, [])

        # Combine english + local, deduplicate
        seen = set()
        combined = []
        for kw in eng_keywords + local_keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                combined.append(kw)

        if not combined:
            combined = eng_keywords  # fallback to english only

        is_cjk = iso in CJK_LANGUAGES
        matchers[lang_full] = build_keyword_matcher(combined, is_cjk=is_cjk)

    # Fallback matcher for unknown languages (english only)
    matchers[None] = build_keyword_matcher(eng_keywords, is_cjk=False)
    matchers[""] = matchers[None]

    return matchers


def init_climate_db():
    conn = sqlite3.connect(str(CLIMATE_DB))
    conn.executescript("""
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
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_existing_ids(climate_conn):
    """Get set of IDs already in climate.db for resume support."""
    rows = climate_conn.execute("SELECT id FROM articles").fetchall()
    return set(r[0] for r in rows)


def main():
    parser = argparse.ArgumentParser(description="Ingest scrapai climate articles into climate.db")
    parser.add_argument("--language", help="Only process one language (e.g., English)")
    parser.add_argument("--source", help="Only process one source")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log.info("Loading keyword translations...")
    translations = load_translations()

    log.info("Building keyword matchers per language...")
    matchers = build_matchers_by_language(translations)
    log.info(f"  {len(matchers)} language matchers ready")

    # Connect to source DB
    src_conn = sqlite3.connect(str(ARTICLES_DB))

    # Build query
    query = """
        SELECT id, url, title, content, source_uri, language, published_date,
               extracted_at, collection_method, country
        FROM articles
        WHERE content IS NOT NULL AND content != ''
    """
    params = []
    if args.language:
        query += " AND language = ?"
        params.append(args.language)
    if args.source:
        query += " AND source_uri = ?"
        params.append(args.source)

    # Count total
    count_query = query.replace("SELECT id, url, title, content, source_uri, language, published_date,\n               extracted_at, collection_method, country", "SELECT COUNT(*)")
    total = src_conn.execute(count_query, params).fetchone()[0]
    log.info(f"Total articles to scan: {total:,}")

    # Init climate DB
    climate_conn = None
    existing_ids = set()
    if not args.dry_run:
        climate_conn = init_climate_db()
        existing_ids = get_existing_ids(climate_conn)
        if existing_ids:
            log.info(f"  {len(existing_ids):,} articles already in climate.db")

    t0 = time.time()
    scanned = 0
    matched = 0
    inserted = 0
    skipped = 0
    lang_counts = {}
    batch = []

    cursor = src_conn.execute(query, params)

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        for row in rows:
            orig_id, url, title, content, source_uri, language, pub_date, extracted_at, coll_method, country = row
            scanned += 1

            # Build article ID
            year = (pub_date or "0000")[:4]
            article_id = f"{coll_method}_{source_uri}_{year}_{orig_id}"

            if article_id in existing_ids:
                skipped += 1
                continue

            # Get matcher for this language
            matcher = matchers.get(language, matchers[None])

            # Search title + content
            text = f"{title or ''} {content or ''}"
            if matcher(text):
                matched += 1
                iso = LANG_TO_ISO.get(language, "")
                lang_counts[language] = lang_counts.get(language, 0) + 1

                if not args.dry_run:
                    batch.append((
                        article_id, url, title, content, source_uri,
                        iso, pub_date, extracted_at, "scrapai", country,
                    ))

                    if len(batch) >= 5000:
                        climate_conn.executemany(
                            "INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?,?,?)",
                            batch,
                        )
                        climate_conn.commit()
                        inserted += len(batch)
                        batch = []

        # Progress
        elapsed = time.time() - t0
        rate = scanned / elapsed if elapsed > 0 else 0
        pct = scanned / total * 100 if total > 0 else 0
        eta = (total - scanned) / rate / 60 if rate > 0 else 0
        log.info(
            f"  {scanned:,}/{total:,} ({pct:.1f}%) scanned | "
            f"{matched:,} matched ({matched/scanned*100:.1f}%) | "
            f"{rate:.0f}/s | ETA {eta:.1f}m"
        )

    # Flush remaining batch
    if batch and climate_conn:
        climate_conn.executemany(
            "INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        climate_conn.commit()
        inserted += len(batch)

    src_conn.close()

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed/60:.1f}m")
    log.info(f"  Scanned:  {scanned:,}")
    log.info(f"  Matched:  {matched:,} ({matched/scanned*100:.1f}%)")
    log.info(f"  Skipped:  {skipped:,} (already in DB)")
    if not args.dry_run:
        total_in_db = climate_conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        log.info(f"  Inserted: {inserted:,}")
        log.info(f"  Total in climate.db: {total_in_db:,}")
        climate_conn.close()

    log.info(f"\nMatches by language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / matched * 100 if matched > 0 else 0
        print(f"  {str(lang or 'unknown'):<20} {count:>8,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
