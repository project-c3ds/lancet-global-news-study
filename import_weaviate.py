#!/usr/bin/env python3
"""Import pre-computed embeddings into local Weaviate, joining with raw data for published_date.

Usage:
    python import_weaviate.py                      # all sources
    python import_weaviate.py --source 20min_ch    # single source
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType

DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PREMIUM_GLOBS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]
COLLECTION_NAME = "NewsArticles"
BATCH_SIZE = 200


def get_raw_crawl_files(source_name):
    """Find raw crawl .jsonl.gz files for a source across premium folders."""
    for folder in PREMIUM_GLOBS:
        crawl_dir = DATA_DIR / folder / source_name / "crawls"
        if crawl_dir.is_dir():
            files = sorted(crawl_dir.glob("*.jsonl.gz"))
            if files:
                return files
    return []


def load_published_dates(crawl_files):
    """Load published_date by line number from raw crawl files."""
    dates = {}
    line_num = 0
    for f in crawl_files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line_num += 1
                art = json.loads(line)
                pd = art.get("published_date")
                if pd:
                    dates[line_num] = pd
    return dates


def open_embedding_file(filepath):
    """Open a .jsonl or .jsonl.gz file transparently."""
    if filepath.suffix == ".gz":
        return gzip.open(filepath, "rt", encoding="utf-8")
    return open(filepath, "r")


def discover_embedding_sources(source_filter=None):
    """Find all .jsonl or .jsonl.gz embedding files."""
    sources = []
    seen = set()
    for pattern in ("*.jsonl.gz", "*.jsonl"):
        for f in sorted(EMBEDDINGS_DIR.glob(pattern)):
            name = f.name.replace(".jsonl.gz", "").replace(".jsonl", "")
            if name in seen:
                continue
            seen.add(name)
            if source_filter and name != source_filter:
                continue
            sources.append((name, f))
    return sources


def create_collection(client):
    """Create the NewsArticles collection if it doesn't exist."""
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="extracted_at", data_type=DataType.DATE),
            Property(name="published_date", data_type=DataType.DATE),
            Property(name="article_id", data_type=DataType.INT),
        ],
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def import_source(collection, source_name, embedding_file):
    """Import a single source's embeddings into Weaviate."""
    # Load published dates from raw crawl
    crawl_files = get_raw_crawl_files(source_name)
    dates = load_published_dates(crawl_files) if crawl_files else {}

    count = 0
    failed = 0
    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        with open_embedding_file(embedding_file) as fh:
            for line in fh:
                rec = json.loads(line)
                article_id = rec["id"]
                vector = rec["embedding"]

                # Format dates as RFC3339
                extracted = rec.get("extracted_at", "")
                if extracted and "T" not in extracted:
                    extracted = extracted.replace(" ", "T") + "Z"

                pub_date = dates.get(article_id, "")
                if pub_date and "T" not in pub_date:
                    pub_date = pub_date.replace(" ", "T") + "Z"

                props = {
                    "title": rec.get("title", ""),
                    "url": rec.get("url", ""),
                    "source": rec.get("source", ""),
                    "article_id": article_id,
                }
                if extracted:
                    props["extracted_at"] = extracted
                if pub_date:
                    props["published_date"] = pub_date

                batch.add_object(properties=props, vector=vector)
                count += 1

    failed = collection.batch.failed_objects
    return count, len(failed) if failed else 0


def get_imported_counts(client):
    """Get per-source article counts already in Weaviate."""
    result = client.graphql_raw_query(
        '{Aggregate{NewsArticles(groupBy:"source"){groupedBy{value}meta{count}}}}'
    )
    counts = {}
    for entry in result.aggregate.get("NewsArticles", []):
        src = entry["groupedBy"]["value"]
        cnt = entry["meta"]["count"]
        counts[src] = cnt
    return counts


def count_embedding_lines(filepath):
    """Count lines in an embedding JSONL or JSONL.gz file."""
    count = 0
    with open_embedding_file(filepath) as fh:
        for _ in fh:
            count += 1
    return count


def compress_embedding_file(filepath):
    """Gzip a .jsonl file in place, removing the original."""
    gz_path = filepath.with_suffix(".jsonl.gz")
    with open(filepath, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
    os.remove(filepath)
    return gz_path


def delete_source(collection, source_name):
    """Delete all objects for a given source."""
    from weaviate.classes.query import Filter
    deleted = 0
    while True:
        result = collection.query.fetch_objects(
            filters=Filter.by_property("source").equal(source_name),
            limit=10000,
            return_properties=["source"],
        )
        if not result.objects:
            break
        ids = [o.uuid for o in result.objects]
        for uid in ids:
            collection.data.delete_by_id(uid)
        deleted += len(ids)
    return deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Import a single source")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-import all sources (skip resume logic)")
    parser.add_argument("--fix-duplicates", action="store_true",
                        help="Delete and re-import sources with wrong counts")
    parser.add_argument("--compress", action="store_true",
                        help="Gzip each .jsonl file after successful import to free disk space")
    args = parser.parse_args()

    client = weaviate.connect_to_local()
    try:
        create_collection(client)
        collection = client.collections.get(COLLECTION_NAME)

        sources = discover_embedding_sources(args.source)
        print(f"Found {len(sources)} source(s) to import.")

        # Resume logic: get existing counts
        imported_counts = {}
        if not args.no_resume:
            print("Checking existing imports for resume...")
            imported_counts = get_imported_counts(client)
            print(f"  {len(imported_counts)} sources already in Weaviate.")

        total_imported = 0
        total_skipped = 0
        total_failed = 0
        start = time.time()

        for i, (name, filepath) in enumerate(sources, 1):
            expected = count_embedding_lines(filepath)
            existing = imported_counts.get(name, 0)

            # Skip if already fully imported
            if not args.no_resume and existing == expected:
                if args.compress and filepath.suffix != ".gz":
                    gz = compress_embedding_file(filepath)
                    print(f"  [{i}/{len(sources)}] {name}: already imported, compressed -> {gz.name}")
                total_skipped += 1
                continue

            # Handle duplicates or partial imports
            if existing > 0:
                if args.fix_duplicates or existing != expected:
                    print(f"  [{i}/{len(sources)}] {name}: clearing {existing} existing (expected {expected})...")
                    delete_source(collection, name)
                elif not args.no_resume:
                    # Count mismatch but no --fix-duplicates: skip with warning
                    print(f"  [{i}/{len(sources)}] {name}: SKIP (has {existing}, expected {expected}, use --fix-duplicates)")
                    total_skipped += 1
                    continue

            t0 = time.time()
            count, failed = import_source(collection, name, filepath)
            elapsed = time.time() - t0
            total_imported += count
            total_failed += failed
            status = f"  [{i}/{len(sources)}] {name}: {count} articles ({elapsed:.1f}s)"
            if failed:
                status += f" ({failed} failed)"
            print(status)

            # Compress after successful import
            if args.compress and failed == 0 and filepath.suffix != ".gz":
                gz = compress_embedding_file(filepath)
                print(f"    compressed -> {gz.name}")

        elapsed = time.time() - start
        print(f"\nDone. Imported {total_imported}, skipped {total_skipped}, failed {total_failed}. ({elapsed:.0f}s)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
