#!/usr/bin/env python3
"""Embed newsapi articles and import into Weaviate.

Reads from data/newsapi_articles/*.jsonl, embeds via vLLM,
writes embeddings to data/embeddings/newsapi_{source}.jsonl,
then imports into Weaviate.

Usage:
    python embed_newsapi.py                          # embed + import all
    python embed_newsapi.py --source abc.es          # single source
    python embed_newsapi.py --embed-only             # embed without Weaviate import
    python embed_newsapi.py --import-only            # import existing embeddings to Weaviate
    python embed_newsapi.py --batch-size 80          # custom batch size
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VLLM_URL = "http://localhost:8000/v1/embeddings"

DATA_DIR = Path("data")
NEWSAPI_DIR = DATA_DIR / "newsapi_articles"
OUTPUT_DIR = DATA_DIR / "embeddings"
LOG_FILE = OUTPUT_DIR / "newsapi_progress.log"

BATCH_SIZE = 80


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )
    return logging.getLogger(__name__)


log = setup_logging()


def discover_sources(source_filter=None):
    """Find all newsapi JSONL files, keyed by source_uri."""
    if not NEWSAPI_DIR.exists():
        return []
    sources = []
    for f in sorted(NEWSAPI_DIR.glob("*.jsonl")):
        # Filename is like abc.es.jsonl -> source_uri = abc.es
        source_uri = f.stem
        if source_filter and source_uri != source_filter:
            continue
        sources.append((source_uri, f))
    return sources


def load_articles(filepath):
    """Yield (line_num, article_dict, text_to_embed) from a newsapi JSONL."""
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
            yield line_num, art, f"{title}\n{body}"


def count_articles(filepath):
    count = 0
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                art = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (art.get("title") or "").strip()
            body = (art.get("body") or "").strip()
            if title or body:
                count += 1
    return count


def count_existing(output_path):
    """Count valid lines in output file."""
    if not output_path.exists():
        return 0
    count = 0
    with open(output_path) as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    break
    return count


def embed_batch(texts, session):
    """Send texts to vLLM and return embeddings."""
    resp = session.post(VLLM_URL, json={
        "model": MODEL_NAME,
        "input": texts,
    })
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def embed_source(source_uri, filepath, batch_size):
    """Embed all articles for a single newsapi source."""
    # Output uses newsapi_ prefix to avoid collision with premium sources
    output_path = OUTPUT_DIR / f"newsapi_{source_uri.replace('.', '_')}.jsonl"

    total = count_articles(filepath)
    existing = count_existing(output_path)

    if existing >= total and total > 0:
        return source_uri, total, 0, "skipped"

    log.info(f"  {source_uri}: {total} total, {existing} existing")

    t0 = time.time()
    session = requests.Session()
    mode = "a" if existing > 0 else "w"
    start_from = existing
    processed = 0
    last_log_time = t0

    batch_buf = []

    with open(output_path, mode) as out:
        skip_count = 0
        for line_num, art, text in load_articles(filepath):
            if skip_count < start_from:
                skip_count += 1
                continue

            batch_buf.append((line_num, art, text))

            if len(batch_buf) >= batch_size:
                texts = [t for _, _, t in batch_buf]
                embeddings = embed_batch(texts, session)

                for (ln, a, _), emb in zip(batch_buf, embeddings):
                    record = {
                        "id": ln,
                        "url": a.get("url", ""),
                        "title": a.get("title", ""),
                        "source": source_uri,
                        "published_date": a.get("dateTime", ""),
                        "language": a.get("lang", ""),
                        "embedding": emb,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                processed += len(batch_buf)
                batch_buf = []

                now = time.time()
                if now - last_log_time >= 10:
                    elapsed = now - t0
                    rate = processed / elapsed
                    done = start_from + processed
                    pct = done / total * 100 if total > 0 else 0
                    eta = (total - done) / rate / 60 if rate > 0 else 0
                    log.info(
                        f"  {source_uri}: {done}/{total} ({pct:.1f}%) | "
                        f"{rate:.1f} art/s | ETA {eta:.1f}m"
                    )
                    last_log_time = now

        # Flush remaining
        if batch_buf:
            texts = [t for _, _, t in batch_buf]
            embeddings = embed_batch(texts, session)
            for (ln, a, _), emb in zip(batch_buf, embeddings):
                record = {
                    "id": ln,
                    "url": a.get("url", ""),
                    "title": a.get("title", ""),
                    "source": source_uri,
                    "published_date": a.get("dateTime", ""),
                    "language": a.get("lang", ""),
                    "embedding": emb,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += len(batch_buf)

    elapsed = time.time() - t0
    return source_uri, processed, elapsed, "done"


def import_to_weaviate(source_uri):
    """Import embeddings for a newsapi source into Weaviate."""
    import weaviate
    from weaviate.classes.init import AdditionalConfig, Timeout

    emb_path = OUTPUT_DIR / f"newsapi_{source_uri.replace('.', '_')}.jsonl"
    if not emb_path.exists():
        log.warning(f"  {source_uri}: no embedding file found at {emb_path}")
        return 0

    client = weaviate.connect_to_local(
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=120, insert=120)),
    )
    try:
        collection = client.collections.get("NewsArticles")

        count = 0
        with collection.batch.fixed_size(batch_size=200) as batch:
            with open(emb_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    vector = rec["embedding"]

                    pub_date = rec.get("published_date", "")
                    if pub_date and "T" not in pub_date:
                        pub_date = pub_date.replace(" ", "T") + "Z"

                    props = {
                        "title": rec.get("title", ""),
                        "url": rec.get("url", ""),
                        "source": source_uri,
                        "article_id": rec["id"],
                        "collection_method": "newsapi",
                    }
                    if pub_date:
                        props["published_date"] = pub_date

                    batch.add_object(properties=props, vector=vector)
                    count += 1

        failed = collection.batch.failed_objects
        if failed:
            log.warning(f"  {source_uri}: {len(failed)} failed imports")

        return count
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Embed newsapi articles and import to Weaviate")
    parser.add_argument("--source", help="Process a single source (e.g., abc.es)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--embed-only", action="store_true", help="Skip Weaviate import")
    parser.add_argument("--import-only", action="store_true", help="Skip embedding, import existing")
    parser.add_argument("--workers", type=int, default=1, help="Parallel sources for embedding")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(args.source)
    if not sources:
        log.info("No newsapi sources found.")
        return

    log.info(f"Found {len(sources)} newsapi source(s)")

    # Check vLLM if we need to embed
    if not args.import_only:
        try:
            requests.get("http://localhost:8000/health", timeout=5)
            log.info("vLLM server is running")
        except requests.ConnectionError:
            log.error("vLLM server not running on localhost:8000. Start it or use --import-only.")
            return

    t_start = time.time()
    total_embedded = 0
    total_imported = 0

    for i, (source_uri, filepath) in enumerate(sources, 1):
        # Embed
        if not args.import_only:
            name, count, elapsed, status = embed_source(source_uri, filepath, args.batch_size)
            total_embedded += count
            if status == "skipped":
                log.info(f"[{i}/{len(sources)}] {source_uri}: {count} articles (skipped)")
            else:
                rate = count / elapsed if elapsed > 0 else 0
                log.info(f"[{i}/{len(sources)}] {source_uri}: embedded {count} in {elapsed:.1f}s ({rate:.0f}/s)")

        # Import to Weaviate
        if not args.embed_only:
            count = import_to_weaviate(source_uri)
            total_imported += count
            log.info(f"  {source_uri}: imported {count} to Weaviate")

    elapsed = time.time() - t_start
    log.info(f"\nDone in {elapsed/60:.1f}m")
    if not args.import_only:
        log.info(f"  Embedded: {total_embedded:,}")
    if not args.embed_only:
        log.info(f"  Imported to Weaviate: {total_imported:,}")


if __name__ == "__main__":
    main()
