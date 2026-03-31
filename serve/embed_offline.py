"""Offline embedding using vLLM LLM class — no HTTP server needed.

Usage:
    python embed_offline.py
    python embed_offline.py --batch-size 2048
    python embed_offline.py --limit 10000
    python embed_offline.py --no-resume
"""

import argparse
import multiprocessing as mp
import sqlite3
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, TokensPrompt

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DB_PATH = "data/articles.db"
OUTPUT_DIR = Path("data/embeddings_np")


def count_articles(db_path):
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    return count


def load_shard(db_path, limit, offset):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, title, content FROM articles LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    rows = [(id, title, content) for id, title, content in rows if content]
    conn.close()
    return rows


def load_done_ids(output_dir):
    done = set()
    if not output_dir.exists():
        return done
    files = sorted(output_dir.glob("ids_*.npy"))
    print(f"Loading {len(files)} id shards...", flush=True)
    arrays = [np.load(f) for f in files]
    if arrays:
        done = set(np.concatenate(arrays).tolist())
    print(f"Loaded {len(done):,} done IDs", flush=True)
    return done


_tokenizer = None

def _init_tokenizer(model_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

def _tokenize_chunk(args):
    """Worker function for parallel tokenization."""
    texts, max_tokens = args
    encoded = _tokenizer(texts, truncation=True, max_length=max_tokens, add_special_tokens=True)
    return encoded["input_ids"]


def parallel_tokenize(texts, model_name, max_tokens=8000, num_workers=20):
    """Tokenize texts in parallel across CPU cores."""
    chunk_size = max(1, len(texts) // num_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    args = [(chunk, max_tokens) for chunk in chunks]
    with mp.Pool(num_workers, initializer=_init_tokenizer, initargs=(model_name,)) as pool:
        results = pool.map(_tokenize_chunk, args)
    all_ids = []
    for chunk_ids in results:
        all_ids.extend(chunk_ids)
    return all_ids


def main():
    parser = argparse.ArgumentParser(description="Offline embed all articles")
    parser.add_argument("--batch-size", type=int, default=2048, help="Articles per vLLM embed call")
    parser.add_argument("--shard-size", type=int, default=50000, help="Articles per output shard")
    parser.add_argument("--limit", type=int, help="Limit total articles")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Test without embedding")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        existing_files = list(OUTPUT_DIR.glob("emb_*.npy"))
        if existing_files:
            max_num = max(int(f.stem.split("_")[1]) for f in existing_files)
            existing_shards = max_num + 1
        else:
            existing_shards = 0
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
        print(f"Batch size: {args.batch_size}")
        return

    # Initialize vLLM offline
    print("Loading vLLM model...", flush=True)
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        max_model_len=8192,
        max_num_seqs=4096,
        max_num_batched_tokens=524288,
        gpu_memory_utilization=0.90,
        runner="pooling",
        pooler_config={"pooling_type": "MEAN", "enable_chunked_processing": True, "max_embed_len": 32768},
        trust_remote_code=True,
    )
    print("Model loaded.", flush=True)

    # Process in shards
    total_shards = (total_articles + args.shard_size - 1) // args.shard_size
    t0 = time.time()
    total_embedded = 0

    for shard_idx in range(total_shards):
        offset = shard_idx * args.shard_size
        shard = load_shard(DB_PATH, args.shard_size, offset)

        if done_ids:
            shard = [row for row in shard if row[0] not in done_ids]

        if not shard:
            print(f"  Skipping shard {shard_idx}/{total_shards} (all done)")
            continue

        shard_num = existing_shards
        print(f"\nShard {shard_num} ({len(shard):,} articles)")

        texts = [f"{title}\n{content}" for _, title, content in shard]
        ids = [row[0] for row in shard]

        # Pre-tokenize in parallel across CPUs (truncates to 8000 tokens)
        print(f"  Tokenizing {len(texts):,} texts across 20 cores...", flush=True)
        tok_t0 = time.time()
        token_ids_list = parallel_tokenize(texts, MODEL_NAME, max_tokens=8000, num_workers=20)
        print(f"  Tokenized in {time.time() - tok_t0:.1f}s", flush=True)

        # Build prompts with pre-tokenized IDs
        prompts = [TokensPrompt(prompt_token_ids=tids) for tids in token_ids_list]

        # Embed in batches via vLLM offline
        all_embs = []
        all_valid_ids = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Shard {shard_num}"):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_ids = ids[i : i + args.batch_size]
            try:
                outputs = llm.embed(batch_prompts)
                batch_embs = [o.outputs.embedding for o in outputs]
                all_embs.extend(batch_embs)
                all_valid_ids.extend(batch_ids)
            except Exception as e:
                print(f"\n  Skipping batch {i//args.batch_size}: {e}")
                with open(OUTPUT_DIR / "failed_ids.txt", "a") as f:
                    for bid in batch_ids:
                        f.write(f"{bid}\n")

        # Save shard
        ids_arr = np.array(all_valid_ids, dtype=np.int64)
        embs_arr = np.array(all_embs, dtype=np.float32)
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
