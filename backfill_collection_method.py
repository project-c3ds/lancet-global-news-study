#!/usr/bin/env python3
"""Backfill collection_method for all Weaviate articles.

For non-overlapping sources, sets scrapai or newsapi directly.
For overlapping sources, uses article_id matching against newsapi
embedding files to distinguish.

Usage:
    python backfill_collection_method.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

EMBEDDINGS_DIR = Path("data/embeddings")


def build_newsapi_lookup():
    """Build {source_underscore: set(article_ids)} from newsapi embedding files."""
    lookup = {}
    for f in sorted(EMBEDDINGS_DIR.glob("newsapi_*.jsonl")):
        # newsapi_abc_es.jsonl -> abc_es
        source = f.stem.replace("newsapi_", "")
        ids = set()
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    ids.add(rec["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        lookup[source] = ids
    return lookup


def update_batch(col, batch):
    """Update a batch of (uuid, collection_method) pairs."""
    for uuid, cm in batch:
        col.data.update(uuid=uuid, properties={"collection_method": cm})
    return len(batch)


def main():
    print("Building newsapi article_id lookup...", flush=True)
    newsapi_lookup = build_newsapi_lookup()
    newsapi_sources = set(newsapi_lookup.keys())
    print(f"  {len(newsapi_sources)} newsapi sources, "
          f"{sum(len(v) for v in newsapi_lookup.values()):,} article IDs")

    # Build set of scrapai-only sources (underscore format)
    scrapai_sources = set()
    for f in EMBEDDINGS_DIR.glob("*.jsonl"):
        if not f.stem.startswith("newsapi_"):
            scrapai_sources.add(f.stem)
    for f in EMBEDDINGS_DIR.glob("*.jsonl.gz"):
        stem = f.stem.replace(".jsonl", "")
        if not stem.startswith("newsapi_"):
            scrapai_sources.add(stem)

    overlap = scrapai_sources & newsapi_sources
    scrapai_only = scrapai_sources - newsapi_sources
    newsapi_only = newsapi_sources - scrapai_sources

    print(f"  Scrapai only: {len(scrapai_only)}, Newsapi only: {len(newsapi_only)}, Overlap: {len(overlap)}")

    client = weaviate.connect_to_local(
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=300, insert=300)),
    )
    try:
        col = client.collections.get("NewsArticles")

        t_start = time.time()
        scanned = 0
        updated = 0
        skipped = 0
        pending = []
        futures = []

        executor = ThreadPoolExecutor(max_workers=8)
        try:
            for obj in col.iterator(return_properties=["source", "article_id", "collection_method"]):
                scanned += 1

                # Skip if already set
                if obj.properties.get("collection_method"):
                    skipped += 1
                else:
                    source = obj.properties.get("source", "")
                    # Convert dot format to underscore for lookup
                    source_key = source.replace(".", "_")
                    article_id = obj.properties.get("article_id")

                    if source_key in scrapai_only:
                        cm = "scrapai"
                    elif source_key in newsapi_only:
                        cm = "newsapi"
                    elif source_key in overlap:
                        # Check if this article_id is in the newsapi set
                        if article_id in newsapi_lookup.get(source_key, set()):
                            cm = "newsapi"
                        else:
                            cm = "scrapai"
                    else:
                        # Unknown source — default to scrapai
                        cm = "scrapai"

                    pending.append((obj.uuid, cm))
                    updated += 1

                    if len(pending) >= 100:
                        futures.append(executor.submit(update_batch, col, pending))
                        pending = []

                        if len(futures) > 32:
                            done = [f for f in futures if f.done()]
                            for f in done:
                                f.result()
                                futures.remove(f)

                if scanned % 100_000 == 0:
                    elapsed = time.time() - t_start
                    rate = updated / elapsed if elapsed > 0 else 0
                    print(
                        f"  {scanned:,} scanned | {updated:,} updated | "
                        f"{skipped:,} already set | {rate:.0f}/s",
                        flush=True,
                    )

            if pending:
                futures.append(executor.submit(update_batch, col, pending))

            for f in futures:
                f.result()

        finally:
            executor.shutdown(wait=True)

        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed / 60:.1f}m")
        print(f"  Scanned: {scanned:,}")
        print(f"  Updated: {updated:,}")
        print(f"  Already set: {skipped:,}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
