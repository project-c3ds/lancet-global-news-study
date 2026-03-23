#!/usr/bin/env python3
"""Import pre-computed embeddings into local Weaviate, joining with raw data for published_date.

Usage:
    python import_weaviate.py                      # all sources
    python import_weaviate.py --source 20min_ch    # single source
"""

import argparse
import gzip
import json
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


def discover_embedding_sources(source_filter=None):
    """Find all .jsonl embedding files."""
    sources = []
    for f in sorted(EMBEDDINGS_DIR.glob("*.jsonl")):
        name = f.stem
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
        with open(embedding_file, "r") as fh:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Import a single source")
    args = parser.parse_args()

    client = weaviate.connect_to_local()
    try:
        create_collection(client)
        collection = client.collections.get(COLLECTION_NAME)

        sources = discover_embedding_sources(args.source)
        print(f"Found {len(sources)} source(s) to import.")

        total_imported = 0
        total_failed = 0
        start = time.time()

        for i, (name, filepath) in enumerate(sources, 1):
            t0 = time.time()
            count, failed = import_source(collection, name, filepath)
            elapsed = time.time() - t0
            total_imported += count
            total_failed += failed
            status = f"  [{i}/{len(sources)}] {name}: {count} articles ({elapsed:.1f}s)"
            if failed:
                status += f" ({failed} failed)"
            print(status)

        elapsed = time.time() - start
        print(f"\nDone. Imported {total_imported} articles in {elapsed:.0f}s. Failed: {total_failed}.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
