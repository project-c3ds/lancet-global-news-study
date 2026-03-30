"""Embed all articles from SQLite DB via Modal vLLM endpoint.

Usage:
    python embed_all.py
    python embed_all.py --concurrency 200 --batch-size 128
    python embed_all.py --limit 10000  # test run
    python embed_all.py --resume       # skip already embedded
"""

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODAL_URL = "http://localhost:8000/v1"
DB_PATH = "data/articles.db"
OUTPUT_DIR = Path("data/embeddings_np")


def get_client():
    return OpenAI(
        base_url=MODAL_URL,
        api_key=os.environ.get("HF_TOKEN", "not-needed"),
        timeout=30,
    )


def count_articles(db_path):
    """Count total embeddable articles."""
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    return count


def load_shard(db_path, limit, offset):
    """Load a shard of articles from SQLite. Returns list of (id, title, content)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, title, content FROM articles LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    rows = [(id, title, content) for id, title, content in rows if content]
    conn.close()
    return rows


def load_done_ids(output_dir):
    """Load set of already-embedded article IDs from existing shards."""
    done = set()
    if not output_dir.exists():
        return done
    for f in output_dir.glob("ids_*.npy"):
        ids = np.load(f)
        done.update(ids.tolist())
    return done


def embed_batch(client, texts, max_retries=3):
    """Embed a batch of texts. Returns list of embeddings. Retries on failure."""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=MODEL_NAME,
                input=texts,
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n  Retry {attempt+1}/{max_retries} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                print(f"\n  FAILED after {max_retries} retries: {e}")
                raise


def process_chunk(client, chunk, batch_size):
    """Embed a chunk of articles in sequential batches. Returns (ids, embeddings) as numpy arrays."""
    ids = []
    all_embeddings = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i : i + batch_size]
        texts = [f"{title}\n{content}" for _, title, content in batch]
        batch_ids = [row[0] for row in batch]
        try:
            embeddings = embed_batch(client, texts)
            ids.extend(batch_ids)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"\n  Skipping batch: {e}")
    return np.array(ids, dtype=np.int64), np.array(all_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Embed all articles from SQLite")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--batch-size", type=int, default=64, help="Texts per request")
    parser.add_argument("--shard-size", type=int, default=50000, help="Articles per output shard")
    parser.add_argument("--limit", type=int, help="Limit total articles (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch, ignore existing embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Test connection and show plan without embedding")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count total articles
    print("Counting articles...", flush=True)
    total_articles = count_articles(DB_PATH)
    if args.limit:
        total_articles = min(total_articles, args.limit)
    print(f"Total articles in DB: {total_articles:,}")

    # Resume support
    done_ids = set()
    existing_shards = 0
    if not args.no_resume:
        done_ids = load_done_ids(OUTPUT_DIR)
        existing_shards = len(list(OUTPUT_DIR.glob("emb_*.npy")))
        if done_ids:
            print(f"Resuming: {len(done_ids):,} already done")

    if len(done_ids) >= total_articles:
        print("Nothing to embed.")
        return

    remaining = total_articles - len(done_ids)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Remaining to embed: {remaining:,}")
        print(f"Shards to process: {(remaining + args.shard_size - 1) // args.shard_size}")
        print(f"Batch size: {args.batch_size}, Concurrency: {args.concurrency}")
        # Test server connection
        print(f"\nTesting connection to {MODAL_URL} ...")
        try:
            client = get_client()
            resp = client.embeddings.create(model=MODEL_NAME, input="test")
            dim = len(resp.data[0].embedding)
            print(f"OK: model={MODEL_NAME}, dim={dim}")
        except Exception as e:
            print(f"FAILED: {e}")
        return

    # Process in shards, streaming from SQLite
    total_shards = (total_articles + args.shard_size - 1) // args.shard_size
    t0 = time.time()
    total_embedded = 0

    for shard_idx in range(total_shards):
        offset = shard_idx * args.shard_size
        shard = load_shard(DB_PATH, args.shard_size, offset)

        # Skip already done
        if done_ids:
            shard = [row for row in shard if row[0] not in done_ids]

        if not shard:
            print(f"  Skipping shard {shard_idx}/{total_shards} (all done)")
            continue

        shard_num = existing_shards

        print(f"\nShard {shard_num} ({len(shard):,} articles)")

        # Split shard into chunks, each worker processes multiple batches sequentially
        chunk_size = args.batch_size * 10
        chunks = [shard[i : i + chunk_size] for i in range(0, len(shard), chunk_size)]
        client = get_client()

        all_ids = []
        all_embs = []

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(process_chunk, client, chunk, args.batch_size): i
                for i, chunk in enumerate(chunks)
            }

            for fut in tqdm(as_completed(futures), total=len(chunks), desc=f"Shard {shard_num}"):
                ids, embs = fut.result()
                all_ids.append(ids)
                all_embs.append(embs)

        # Save shard
        ids_arr = np.concatenate(all_ids)
        embs_arr = np.concatenate(all_embs)
        np.save(OUTPUT_DIR / f"ids_{shard_num:06d}.npy", ids_arr)
        np.save(OUTPUT_DIR / f"emb_{shard_num:06d}.npy", embs_arr)

        total_embedded += len(ids_arr)
        existing_shards += 1
        elapsed = time.time() - t0
        rate = total_embedded / elapsed if elapsed > 0 else 0
        eta = (remaining - total_embedded) / rate if rate > 0 else 0
        print(f"  Saved: {len(ids_arr):,} embeddings ({embs_arr.shape})")
        print(f"  Total: {total_embedded:,}/{remaining:,} | {rate:.0f} emb/s | ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"\nDone. {total_embedded:,} embeddings in {elapsed/60:.1f}m ({total_embedded/elapsed:.0f} emb/s)")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
