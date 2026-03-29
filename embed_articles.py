#!/usr/bin/env python3
"""Embed articles from premium news folders using Qwen3-Embedding-0.6B.

Supports two backends:
  - vllm (default): local vLLM server
  - hf: HuggingFace Inference Endpoint (TEI)

Usage:
    python embed_articles.py                                    # local vLLM
    python embed_articles.py --backend hf --hf-url URL          # HuggingFace TEI
    python embed_articles.py --source scmp_com                  # single source
    python embed_articles.py --workers 4                        # 4 sources in parallel
    python embed_articles.py --batch-size 256                   # custom batch size
"""

import argparse
import gzip
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VLLM_URL = "http://localhost:8000/v1/embeddings"

# Backend config (set via args)
BACKEND = "vllm"
HF_URL = "https://nk3aoahfg5ektayg.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = None
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
    """Send a batch of texts to the configured backend and return embeddings."""
    if BACKEND == "hf":
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        resp = session.post(HF_URL, headers=headers, json={
            "inputs": texts,
            "truncate": True,
        })
        resp.raise_for_status()
        return resp.json()
    else:
        resp = session.post(VLLM_URL, json={
            "model": MODEL_NAME,
            "input": texts,
        })
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [d["embedding"] for d in data]


CHUNK_SIZE = 1000  # process this many articles at a time to limit RAM


def count_articles(files):
    """Count total articles without loading them into memory."""
    count = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                art = json.loads(line)
                title = (art.get("title") or "").strip()
                content = (art.get("content") or "").strip()
                if title or content:
                    count += 1
    return count


def count_existing(output_path):
    """Count valid lines in output file and trim corrupt trailing line."""
    if not output_path.exists():
        return 0
    # Check last line for corruption
    last_line = ""
    with open(output_path, "rb") as f:
        # Seek to end, read last line
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return 0
        pos = size - 1
        while pos > 0:
            f.seek(pos)
            if f.read(1) == b"\n" and pos < size - 1:
                break
            pos -= 1
        last_line = f.read().decode("utf-8").strip()
    if last_line:
        try:
            json.loads(last_line)
        except json.JSONDecodeError:
            # Trim corrupt last line — use readline() instead of
            # iteration so f.tell() works
            with open(output_path, "r") as f:
                valid_lines = 0
                good_pos = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        try:
                            json.loads(line)
                            valid_lines += 1
                            good_pos = f.tell()
                        except json.JSONDecodeError:
                            break
            with open(output_path, "r+") as f:
                f.seek(good_pos)
                f.truncate()
            return valid_lines
    # Count lines
    count = 0
    with open(output_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def embed_source(source_name, files, batch_size, concurrency=1):
    """Embed all articles for a single source and write output JSONL."""
    output_path = OUTPUT_DIR / f"{source_name}.jsonl"

    log.info(f"  {source_name}: counting articles...")
    total = count_articles(files)
    existing = count_existing(output_path)
    log.info(f"  {source_name}: {total} total, {existing} existing")

    if existing >= total and total > 0:
        return source_name, total, 0, "skipped"

    t0 = time.time()
    session = requests.Session()

    mode = "a" if existing > 0 else "w"
    start_from = existing
    processed = 0
    skipped = 0
    last_log_time = t0

    def collect_batches():
        """Stream articles into batches."""
        batch_buf = []
        skip_count = 0
        for line_num, art, text in load_articles(files):
            if skip_count < start_from:
                skip_count += 1
                continue
            batch_buf.append((line_num, art, text))
            if len(batch_buf) >= batch_size:
                yield batch_buf
                batch_buf = []
        if batch_buf:
            yield batch_buf

    def process_batch(batch):
        """Embed a single batch and return (batch_meta, embeddings)."""
        texts = [t for _, _, t in batch]
        embeddings = embed_batch(texts, session)
        return batch, embeddings

    with open(output_path, mode) as out:
        if concurrency <= 1:
            # Sequential — same as before
            for batch in collect_batches():
                batch_t0 = time.time()
                batch, embeddings = process_batch(batch)
                batch_elapsed = time.time() - batch_t0

                for (ln, a, _), emb in zip(batch, embeddings):
                    record = {
                        "id": ln,
                        "url": a.get("url", ""),
                        "title": a.get("title", ""),
                        "source": a.get("source", ""),
                        "extracted_at": a.get("extracted_at", ""),
                        "embedding": emb,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                processed += len(batch)

                now = time.time()
                if now - last_log_time >= 10:
                    overall_elapsed = now - t0
                    avg_rate = processed / overall_elapsed
                    batch_rate = len(batch) / batch_elapsed if batch_elapsed > 0 else 0
                    done = start_from + processed
                    pct = done / total * 100 if total > 0 else 0
                    eta_secs = (total - done) / avg_rate if avg_rate > 0 else 0
                    eta_min = eta_secs / 60
                    log.info(
                        f"  {source_name}: {done}/{total} ({pct:.1f}%) | "
                        f"avg {avg_rate:.1f} art/s | batch {batch_rate:.1f} art/s | "
                        f"ETA {eta_min:.1f}m"
                    )
                    last_log_time = now
        else:
            # Concurrent batches — submit up to `concurrency` batches at once
            # Write results in submission order to maintain consistency
            from concurrent.futures import ThreadPoolExecutor, as_completed
            pool = ThreadPoolExecutor(max_workers=concurrency)
            futures = []
            batch_iter = collect_batches()

            # Fill the pipeline
            for batch in batch_iter:
                futures.append(pool.submit(process_batch, batch))
                if len(futures) >= concurrency:
                    break

            while futures:
                # Wait for the FIRST submitted (preserves order)
                future = futures.pop(0)
                batch_t0 = time.time()
                batch, embeddings = future.result()
                batch_elapsed = time.time() - batch_t0

                for (ln, a, _), emb in zip(batch, embeddings):
                    record = {
                        "id": ln,
                        "url": a.get("url", ""),
                        "title": a.get("title", ""),
                        "source": a.get("source", ""),
                        "extracted_at": a.get("extracted_at", ""),
                        "embedding": emb,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                processed += len(batch)

                # Submit next batch to keep pipeline full
                try:
                    next_batch = next(batch_iter)
                    futures.append(pool.submit(process_batch, next_batch))
                except StopIteration:
                    pass

                now = time.time()
                if now - last_log_time >= 10:
                    overall_elapsed = now - t0
                    avg_rate = processed / overall_elapsed
                    batch_rate = len(batch) / batch_elapsed if batch_elapsed > 0 else 0
                    done = start_from + processed
                    pct = done / total * 100 if total > 0 else 0
                    eta_secs = (total - done) / avg_rate if avg_rate > 0 else 0
                    eta_min = eta_secs / 60
                    log.info(
                        f"  {source_name}: {done}/{total} ({pct:.1f}%) | "
                        f"avg {avg_rate:.1f} art/s | batch {batch_rate:.1f} art/s | "
                        f"ETA {eta_min:.1f}m"
                    )
                    last_log_time = now

            pool.shutdown(wait=False)

    elapsed = time.time() - t0
    return source_name, processed, elapsed, "done"


def main():
    parser = argparse.ArgumentParser(description="Embed premium news articles")
    parser.add_argument("--source", help="Process a single source by name")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of sources to process in parallel")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for embedding (default: 256)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent batches per source (default: 1, use higher for HF)")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                        help="Embedding backend: vllm (local) or hf (HuggingFace TEI)")
    parser.add_argument("--hf-url", help="HuggingFace endpoint URL")
    parser.add_argument("--hf-token", help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    global BACKEND, HF_URL, HF_TOKEN
    BACKEND = args.backend

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if BACKEND == "hf":
        HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            log.error("--hf-url and --hf-token (or HF_TOKEN env var) required for hf backend")
            return
        try:
            resp = requests.get(HF_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
            log.info(f"HuggingFace endpoint reachable (status {resp.status_code})")
        except requests.ConnectionError:
            log.error(f"Cannot reach HuggingFace endpoint: {HF_URL}")
            return
    else:
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
            name, count, elapsed, status = embed_source(name, files, args.batch_size, args.concurrency)
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
