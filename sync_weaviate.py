#!/usr/bin/env python3
"""Rebuild Weaviate NewsArticles from SQLite + pre-computed embeddings.

Drops and recreates the collection with SQ (scalar quantization) to reduce
memory usage, then imports only articles that exist in the cleaned SQLite DB.

Usage:
    python sync_weaviate.py                     # full rebuild
    python sync_weaviate.py --source abc.es     # single source (append, no drop)
    python sync_weaviate.py --dry-run           # show what would be imported
"""

import argparse
import gzip
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import AdditionalConfig, Timeout

DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SQLITE_DB = DATA_DIR / "articles.db"
COLLECTION_NAME = "NewsArticles"
BATCH_SIZE = 200
PREMIUM_GLOBS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def get_sqlite_sources(conn):
    """Return {source_uri: {collection_method: set_of_original_ids}}."""
    cur = conn.execute(
        "SELECT source_uri, collection_method, original_id FROM articles"
    )
    sources = {}
    for uri, method, oid in cur:
        sources.setdefault(uri, {}).setdefault(method, set()).add(oid)
    return sources


def get_valid_ids(conn, source_uri, collection_method):
    """Return set of original_ids for a source+method in SQLite."""
    cur = conn.execute(
        "SELECT original_id FROM articles WHERE source_uri = ? AND collection_method = ?",
        (source_uri, collection_method),
    )
    return {row[0] for row in cur}


def get_published_dates_from_sqlite(conn, source_uri, collection_method):
    """Return {original_id: published_date} for scrapai articles."""
    cur = conn.execute(
        "SELECT original_id, published_date FROM articles "
        "WHERE source_uri = ? AND collection_method = ? AND published_date IS NOT NULL",
        (source_uri, collection_method),
    )
    return {row[0]: row[1] for row in cur}


# ---------------------------------------------------------------------------
# Embedding file helpers
# ---------------------------------------------------------------------------

def open_embedding_file(filepath):
    if str(filepath).endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8")
    return open(filepath, "r", encoding="utf-8")


def find_embedding_file(source_underscore, prefix=""):
    """Find .jsonl or .jsonl.gz for a source, with optional prefix."""
    name = f"{prefix}{source_underscore}"
    # Prefer .jsonl.gz (smaller), fall back to .jsonl
    for ext in (".jsonl.gz", ".jsonl"):
        path = EMBEDDINGS_DIR / f"{name}{ext}"
        if path.exists():
            return path
    return None


def load_published_dates_from_crawl(source_underscore):
    """Load published_date by line number from raw crawl files."""
    dates = {}
    for folder in PREMIUM_GLOBS:
        crawl_dir = DATA_DIR / folder / source_underscore / "crawls"
        if not crawl_dir.is_dir():
            continue
        line_num = 0
        for f in sorted(crawl_dir.glob("*.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line_num += 1
                    try:
                        art = json.loads(line)
                        pd = art.get("published_date")
                        if pd:
                            dates[line_num] = pd
                    except json.JSONDecodeError:
                        continue
        if dates:
            return dates
    return dates


# ---------------------------------------------------------------------------
# Weaviate collection
# ---------------------------------------------------------------------------

def create_collection(client):
    """Create NewsArticles with SQ quantization for reduced memory."""
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            quantizer=Configure.VectorIndex.Quantizer.sq(
                training_limit=100_000,
            ),
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="source_uri", data_type=DataType.TEXT),
            Property(name="collection_method", data_type=DataType.TEXT),
            Property(name="language", data_type=DataType.TEXT),
            Property(name="extracted_at", data_type=DataType.DATE),
            Property(name="published_date", data_type=DataType.DATE),
            Property(name="article_id", data_type=DataType.INT),
        ],
    )
    log.info(f"Created collection '{COLLECTION_NAME}' with SQ quantization.")


def drop_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        log.info(f"Dropped collection '{COLLECTION_NAME}'.")


# ---------------------------------------------------------------------------
# Import logic
# ---------------------------------------------------------------------------

def fmt_date(d):
    """Ensure RFC3339 format for Weaviate date fields."""
    if not d:
        return None
    if "T" not in d:
        d = d.replace(" ", "T")
    if not d.endswith("Z") and "+" not in d:
        d += "Z"
    return d


def import_embedding_file(collection, emb_path, valid_ids, source_uri,
                          source_underscore, collection_method, pub_dates):
    """Import articles from one embedding file, filtering by valid_ids.

    Returns (imported, skipped, failed, embedded_ids) where embedded_ids
    is the set of valid article IDs that were found in the embedding file.
    """
    count = 0
    skipped = 0
    embedded_ids = set()

    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        with open_embedding_file(emb_path) as fh:
            for line in fh:
                rec = json.loads(line)
                article_id = rec["id"]

                if article_id not in valid_ids:
                    skipped += 1
                    continue

                embedded_ids.add(article_id)

                vector = rec["embedding"]

                # published_date: from embedding (newsapi) or crawl/sqlite
                pub_date = rec.get("published_date") or pub_dates.get(article_id)

                props = {
                    "title": rec.get("title", ""),
                    "url": rec.get("url", ""),
                    "source": source_underscore,
                    "source_uri": source_uri,
                    "collection_method": collection_method,
                    "article_id": article_id,
                }

                lang = rec.get("language")
                if lang:
                    props["language"] = lang

                extracted = rec.get("extracted_at")
                if extracted:
                    props["extracted_at"] = fmt_date(extracted)

                if pub_date:
                    props["published_date"] = fmt_date(pub_date)

                batch.add_object(properties=props, vector=vector)
                count += 1

    failed = collection.batch.failed_objects
    n_failed = len(failed) if failed else 0
    return count, skipped, n_failed, embedded_ids


def discover_sources_to_import(conn):
    """Build list of (source_uri, collection_method, emb_path) tuples."""
    cur = conn.execute(
        "SELECT DISTINCT source_uri, collection_method FROM articles ORDER BY source_uri"
    )
    tasks = []
    for source_uri, method in cur:
        source_underscore = source_uri.replace(".", "_").replace("-", "-")
        if method == "newsapi":
            emb_path = find_embedding_file(source_underscore, prefix="newsapi_")
        else:
            emb_path = find_embedding_file(source_underscore)
        tasks.append((source_uri, method, source_underscore, emb_path))
    return tasks


COVERAGE_REPORT = Path("results/embedding_coverage.csv")


def _write_coverage_report(coverage_gaps, missing_sources, total_imported, conn):
    """Write a CSV of sources with incomplete embedding coverage."""
    COVERAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)

    # Count total SQLite articles
    cur = conn.execute("SELECT count(*) FROM articles")
    total_sqlite = cur.fetchone()[0]

    total_missing_articles = sum(len(m) for _, _, _, _, m in coverage_gaps)
    total_no_file = sum(
        len(get_valid_ids(conn, s, m)) for s, m in missing_sources
    )

    log.info(f"\n--- Coverage Report ---")
    log.info(f"SQLite articles: {total_sqlite:,}")
    log.info(f"Imported to Weaviate: {total_imported:,}")
    log.info(f"Partial coverage (have file, missing some IDs): {len(coverage_gaps)} sources, {total_missing_articles:,} articles")
    log.info(f"No embedding file at all: {len(missing_sources)} sources, {total_no_file:,} articles")
    log.info(f"Report written to {COVERAGE_REPORT}")

    with open(COVERAGE_REPORT, "w") as f:
        f.write("source_uri,collection_method,sqlite_count,embedded_count,missing_count,coverage_pct,status\n")

        # Sources with no embedding file
        for source_uri, method in missing_sources:
            n = len(get_valid_ids(conn, source_uri, method))
            f.write(f"{source_uri},{method},{n},0,{n},0.0,no_embedding_file\n")

        # Sources with partial coverage
        for source_uri, method, sqlite_n, emb_n, missing_ids in sorted(
            coverage_gaps, key=lambda x: len(x[4]), reverse=True
        ):
            pct = emb_n / sqlite_n * 100 if sqlite_n else 100
            f.write(f"{source_uri},{method},{sqlite_n},{emb_n},{len(missing_ids)},{pct:.1f},partial\n")


def main():
    parser = argparse.ArgumentParser(description="Rebuild Weaviate from SQLite + embeddings")
    parser.add_argument("--source", help="Sync a single source_uri (e.g., abc.es). Appends without dropping.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without making changes")
    parser.add_argument("--no-drop", action="store_true", help="Don't drop the collection first")
    args = parser.parse_args()

    # Connect to SQLite
    conn = sqlite3.connect(str(SQLITE_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # Discover what needs importing
    tasks = discover_sources_to_import(conn)

    if args.source:
        tasks = [t for t in tasks if t[0] == args.source]
        if not tasks:
            log.error(f"Source '{args.source}' not found in SQLite.")
            return

    # Stats
    has_embeddings = [(s, m, su, p) for s, m, su, p in tasks if p is not None]
    missing = [(s, m) for s, m, su, p in tasks if p is None]

    log.info(f"SQLite source+method combos: {len(tasks)}")
    log.info(f"  With embeddings: {len(has_embeddings)}")
    log.info(f"  Missing embeddings: {len(missing)}")
    if missing:
        for s, m in missing:
            n = len(get_valid_ids(conn, s, m))
            log.info(f"    {s} ({m}): {n} articles — no embedding file")

    if args.dry_run:
        log.info("\nDry run — no changes made.")
        total = 0
        for source_uri, method, source_underscore, emb_path in has_embeddings:
            valid_ids = get_valid_ids(conn, source_uri, method)
            log.info(f"  Would import {source_uri} ({method}): up to {len(valid_ids)} articles from {emb_path.name}")
            total += len(valid_ids)
        log.info(f"  Total: up to {total:,} articles")
        conn.close()
        return

    # Connect to Weaviate
    client = weaviate.connect_to_local(
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=300, insert=300),
        ),
    )

    try:
        # Drop and recreate (unless single source or --no-drop)
        if not args.source and not args.no_drop:
            drop_collection(client)
            create_collection(client)
        elif not client.collections.exists(COLLECTION_NAME):
            create_collection(client)

        collection = client.collections.get(COLLECTION_NAME)

        total_imported = 0
        total_skipped = 0
        total_failed = 0
        coverage_gaps = []  # (source_uri, method, sqlite_count, embedded_count, missing_ids)
        t_start = time.time()

        for i, (source_uri, method, source_underscore, emb_path) in enumerate(has_embeddings, 1):
            t0 = time.time()

            # Get valid article IDs from SQLite
            valid_ids = get_valid_ids(conn, source_uri, method)

            # Get published dates for scrapai (newsapi has them in embedding file)
            pub_dates = {}
            if method == "scrapai":
                pub_dates = get_published_dates_from_sqlite(conn, source_uri, method)

            count, skipped, failed, embedded_ids = import_embedding_file(
                collection, emb_path, valid_ids,
                source_uri, source_underscore, method, pub_dates,
            )

            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            total_imported += count
            total_skipped += skipped
            total_failed += failed

            # Track coverage gaps
            missing_ids = valid_ids - embedded_ids
            if missing_ids:
                pct = len(embedded_ids) / len(valid_ids) * 100 if valid_ids else 100
                coverage_gaps.append((source_uri, method, len(valid_ids), len(embedded_ids), missing_ids))

            status = (
                f"[{i}/{len(has_embeddings)}] {source_uri} ({method}): "
                f"{count:,} imported, {skipped:,} filtered out "
                f"({elapsed:.1f}s, {rate:.0f}/s)"
            )
            if missing_ids:
                pct = len(embedded_ids) / len(valid_ids) * 100
                status += f" | coverage {pct:.1f}% ({len(missing_ids)} missing)"
            if failed:
                status += f" ⚠ {failed} failed"
            log.info(status)

        elapsed = time.time() - t_start
        log.info(
            f"\nDone in {elapsed/60:.1f}m. "
            f"Imported: {total_imported:,}, filtered: {total_skipped:,}, failed: {total_failed:,}"
        )

        # Write coverage report
        _write_coverage_report(coverage_gaps, missing, total_imported, conn)

    finally:
        client.close()
        conn.close()


if __name__ == "__main__":
    main()
