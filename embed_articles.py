#!/usr/bin/env python3
"""Embed articles from premium news folders using Qwen3-Embedding-0.6B via vLLM.

Requires vLLM server running:
    vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --port 8000

Usage:
    python embed_articles.py                        # all premium sources
    python embed_articles.py --source scmp_com      # single source
    python embed_articles.py --workers 4            # 4 sources in parallel
    python embed_articles.py --batch-size 256       # custom batch size
"""

import argparse
import gzip
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VLLM_URL = "http://localhost:8000/v1/embeddings"
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "embeddings"
LOG_FILE = OUTPUT_DIR / "progress.log"
BATCH_SIZE = 256
PREMIUM_GLOBS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]


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
    """Find all source directories with crawl files."""
    sources = []
    for folder in PREMIUM_GLOBS:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            continue
        for source_dir in sorted(folder_path.iterdir()):
            crawl_dir = source_dir / "crawls"
            if not crawl_dir.is_dir():
                continue
            if source_filter and source_dir.name != source_filter:
                continue
            files = sorted(crawl_dir.glob("*.jsonl.gz"))
            if files:
                sources.append((source_dir.name, files))
    return sources


def load_articles(files):
    """Load articles from JSONL files, yielding (line_number, article, text) tuples."""
    line_num = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                art = json.loads(line)
                line_num += 1
                title = (art.get("title") or "").strip()
                content = (art.get("content") or "").strip()
                if not title and not content:
                    continue
                yield line_num, art, f"{title}\n{content}"


def embed_batch(texts, session):
    """Send a batch of texts to vLLM and return embeddings."""
    resp = session.post(VLLM_URL, json={
        "model": MODEL_NAME,
        "input": texts,
    })
    resp.raise_for_status()
    data = resp.json()["data"]
    # Sort by index to maintain order
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]


CHUNK_SIZE = 1000  # process this many articles at a time to limit RAM


def embed_source(source_name, files, batch_size):
    """Embed all articles for a single source and write output JSONL."""
    output_path = OUTPUT_DIR / f"{source_name}.jsonl"

    articles = list(load_articles(files))
    total = len(articles)

    # Resume: skip already-written articles
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
    if existing >= total and total > 0:
        return source_name, total, 0, "skipped"

    t0 = time.time()
    session = requests.Session()

    # Append mode for resume support
    mode = "a" if existing > 0 else "w"
    start_from = existing

    with open(output_path, mode) as out:
        for start in range(start_from, total, CHUNK_SIZE):
            chunk = articles[start : start + CHUNK_SIZE]

            for i in range(0, len(chunk), batch_size):
                batch = chunk[i : i + batch_size]
                texts = [text for _, _, text in batch]
                embeddings = embed_batch(texts, session)

                for (line_num, art, _), emb in zip(batch, embeddings):
                    record = {
                        "id": line_num,
                        "url": art.get("url", ""),
                        "title": art.get("title", ""),
                        "source": art.get("source", ""),
                        "extracted_at": art.get("extracted_at", ""),
                        "embedding": emb,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

    elapsed = time.time() - t0
    processed = total - start_from
    return source_name, processed, elapsed, "done"


def main():
    parser = argparse.ArgumentParser(description="Embed premium news articles")
    parser.add_argument("--source", help="Process a single source by name")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of sources to process in parallel")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for embedding (default: 256)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check vLLM server is up
    try:
        requests.get("http://localhost:8000/health", timeout=5)
    except requests.ConnectionError:
        log.error("vLLM server not running on localhost:8000")
        return

    sources = discover_sources(args.source)
    if not sources:
        log.info("No sources found.")
        return

    log.info(f"Found {len(sources)} sources to process (workers={args.workers}, batch_size={args.batch_size})")

    done_count = 0
    total_articles = 0
    t_start = time.time()

    if args.workers <= 1:
        for name, files in sources:
            name, count, elapsed, status = embed_source(name, files, args.batch_size)
            done_count += 1
            total_articles += count
            if status == "skipped":
                log.info(f"[{done_count}/{len(sources)}] {name}: {count} articles (skipped)")
            else:
                rate = count / elapsed if elapsed > 0 else 0
                log.info(f"[{done_count}/{len(sources)}] {name}: {count} articles in {elapsed:.1f}s ({rate:.0f}/s)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(embed_source, name, files, args.batch_size): name
                for name, files in sources
            }
            for future in as_completed(futures):
                done_count += 1
                name, count, elapsed, status = future.result()
                total_articles += count
                if status == "skipped":
                    log.info(f"[{done_count}/{len(sources)}] {name}: {count} articles (skipped)")
                else:
                    rate = count / elapsed if elapsed > 0 else 0
                    log.info(f"[{done_count}/{len(sources)}] {name}: {count} articles in {elapsed:.1f}s ({rate:.0f}/s)")

    total_elapsed = time.time() - t_start
    log.info(f"Done. {total_articles:,} articles across {done_count} sources in {total_elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
