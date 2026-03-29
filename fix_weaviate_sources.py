#!/usr/bin/env python3
"""Set source property to URL domain for ALL articles in Weaviate.

Usage:
    python fix_weaviate_sources.py
    python fix_weaviate_sources.py --workers 8
"""

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

COLLECTION_NAME = "NewsArticles"


def source_from_url(url):
    if not url:
        return ""
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return domain.replace(".", "_").replace("/", "_")
    except Exception:
        return ""


def update_batch(col, batch):
    for uuid, new_source in batch:
        col.data.update(uuid=uuid, properties={"source": new_source})
    return len(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    client = weaviate.connect_to_local(
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=300, insert=300)),
    )
    try:
        col = client.collections.get(COLLECTION_NAME)

        t_start = time.time()
        scanned = 0
        updated = 0
        no_url = 0
        source_counts = Counter()
        pending = []
        futures = []

        print(f"Setting source = URL domain for all articles ({args.workers} workers)...", flush=True)

        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            for obj in col.iterator(return_properties=["url"]):
                scanned += 1
                url = obj.properties.get("url", "")
                new_source = source_from_url(url)

                if not new_source:
                    no_url += 1
                    continue

                source_counts[new_source] += 1
                pending.append((obj.uuid, new_source))
                updated += 1

                if len(pending) >= args.batch_size:
                    futures.append(executor.submit(update_batch, col, pending))
                    pending = []

                    # Drain completed futures to bound memory
                    if len(futures) > args.workers * 4:
                        done = [f for f in futures if f.done()]
                        for f in done:
                            f.result()
                            futures.remove(f)

                if scanned % 100_000 == 0:
                    elapsed = time.time() - t_start
                    rate = updated / elapsed
                    eta = (6_100_000 - updated) / rate / 60 if rate > 0 else 0
                    print(
                        f"  {scanned:,} scanned | {updated:,} updated | "
                        f"{rate:.0f}/s | ~{eta:.0f}m remaining",
                        flush=True,
                    )

            if pending:
                futures.append(executor.submit(update_batch, col, pending))

            for f in futures:
                f.result()

        finally:
            executor.shutdown(wait=True)

        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed/60:.1f}m ({elapsed:.0f}s)", flush=True)
        print(f"  Updated: {updated:,}")
        print(f"  No URL: {no_url:,}")
        print(f"  Unique sources: {len(source_counts):,}")
        print(f"\nTop sources:")
        for src, count in source_counts.most_common(20):
            print(f"  {src:50s} {count:>8,}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
