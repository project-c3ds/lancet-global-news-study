#!/usr/bin/env python3
"""Compute keyword similarity scores for all Weaviate articles and store in SQLite.

Iterates the Weaviate NewsArticles collection in batches, computes cosine
similarity against climate and health keyword embeddings (English), and
writes the max-sim scores + best keywords into the SQLite articles table.

Resume-friendly: skips articles that already have scores.

Usage:
    python score_articles.py                    # score all unscored articles
    python score_articles.py --batch-size 10000 # larger batches
    python score_articles.py --reset            # clear existing scores and re-run
"""

import argparse
import sqlite3
import time

import numpy as np

from weaviate_utils import (
    KEYWORD_EMBEDDINGS_DIR,
    connect,
    cosine_similarity_batch,
    get_collection,
    load_keyword_embeddings,
)

DB_PATH = "data/articles.db"
BATCH_SIZE = 5000

# Properties needed from Weaviate to join back to SQLite
WEAVIATE_PROPS = ["source_uri", "collection_method", "article_id"]


def add_score_columns(conn):
    """Add score columns to articles table if they don't exist."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(articles)")}
    new_cols = {
        "climate_max_sim": "REAL",
        "climate_best_keyword": "TEXT",
        "health_max_sim": "REAL",
        "health_best_keyword": "TEXT",
    }
    for col, dtype in new_cols.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {dtype}")
            print(f"  Added column: {col}")
    conn.commit()


def build_scored_set(conn):
    """Return set of (source_uri, collection_method, original_id) already scored."""
    rows = conn.execute(
        "SELECT source_uri, collection_method, original_id "
        "FROM articles WHERE climate_max_sim IS NOT NULL"
    ).fetchall()
    return set(rows)


def reset_scores(conn):
    """Clear all existing scores."""
    conn.execute(
        "UPDATE articles SET climate_max_sim=NULL, climate_best_keyword=NULL, "
        "health_max_sim=NULL, health_best_keyword=NULL"
    )
    conn.commit()
    print("Cleared all existing scores.")


def iterate_collection_for_scoring(collection, batch_size, scored_set):
    """Yield (properties_list, vectors_array) in batches, skipping already-scored articles."""
    meta_buf = []
    vec_buf = []
    skipped = 0

    for obj in collection.iterator(
        include_vector=True,
        return_properties=WEAVIATE_PROPS,
    ):
        source_uri = obj.properties.get("source_uri", "")
        method = obj.properties.get("collection_method", "")
        art_id = obj.properties.get("article_id")

        # Skip if already scored
        if (source_uri, method, art_id) in scored_set:
            skipped += 1
            continue

        vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
        if vec is None:
            continue

        meta_buf.append(obj.properties)
        vec_buf.append(vec)

        if len(meta_buf) >= batch_size:
            yield meta_buf, np.array(vec_buf, dtype=np.float32), skipped
            meta_buf = []
            vec_buf = []
            skipped = 0

    if meta_buf:
        yield meta_buf, np.array(vec_buf, dtype=np.float32), skipped


def score_batch(embeddings, climate_emb, health_emb, climate_kw, health_kw):
    """Compute max similarity scores for a batch of article embeddings."""
    climate_sims = cosine_similarity_batch(embeddings, climate_emb)
    health_sims = cosine_similarity_batch(embeddings, health_emb)

    results = []
    for i in range(len(embeddings)):
        c_idx = int(climate_sims[i].argmax())
        h_idx = int(health_sims[i].argmax())
        results.append((
            float(climate_sims[i, c_idx]),
            climate_kw[c_idx],
            float(health_sims[i, h_idx]),
            health_kw[h_idx],
        ))
    return results


def update_sqlite(conn, meta_list, scores):
    """Batch update SQLite with computed scores."""
    cursor = conn.cursor()
    updates = []
    for props, (c_sim, c_kw, h_sim, h_kw) in zip(meta_list, scores):
        updates.append((
            c_sim, c_kw, h_sim, h_kw,
            props["source_uri"],
            props["collection_method"],
            props["article_id"],
        ))

    cursor.executemany(
        "UPDATE articles "
        "SET climate_max_sim=?, climate_best_keyword=?, "
        "    health_max_sim=?, health_best_keyword=? "
        "WHERE source_uri=? AND collection_method=? AND original_id=?",
        updates,
    )
    conn.commit()
    return cursor.rowcount


def main():
    parser = argparse.ArgumentParser(description="Score articles with keyword similarity")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reset", action="store_true", help="Clear existing scores first")
    args = parser.parse_args()

    # Load keyword embeddings
    climate_kw, climate_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "climate_eng.jsonl"
    )
    health_kw, health_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "health_eng.jsonl"
    )
    print(f"Keywords: {len(climate_kw)} climate, {len(health_kw)} health")

    # Setup SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-512000")

    add_score_columns(conn)

    if args.reset:
        reset_scores(conn)

    # Build set of already-scored articles for resume
    print("Loading scored article set for resume...")
    scored_set = build_scored_set(conn)
    print(f"  Already scored: {len(scored_set):,}")

    # Create index for fast updates if not exists
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_method_origid "
        "ON articles(source_uri, collection_method, original_id)"
    )
    conn.commit()

    # Connect to Weaviate
    client = connect()
    try:
        collection = get_collection(client)

        t_start = time.time()
        total_scored = 0
        total_skipped = 0
        batch_num = 0

        for meta_list, vectors, skipped in iterate_collection_for_scoring(
            collection, args.batch_size, scored_set
        ):
            batch_num += 1
            total_skipped += skipped

            # Compute scores
            scores = score_batch(vectors, climate_emb, health_emb, climate_kw, health_kw)

            # Write to SQLite
            update_sqlite(conn, meta_list, scores)
            total_scored += len(meta_list)

            elapsed = time.time() - t_start
            rate = total_scored / elapsed if elapsed > 0 else 0
            print(
                f"  Batch {batch_num}: scored {len(meta_list):,} | "
                f"total {total_scored:,} | skipped {total_skipped:,} | "
                f"{rate:,.0f} art/s | {elapsed:.0f}s",
                flush=True,
            )

    finally:
        client.close()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f}m")
    print(f"  Scored: {total_scored:,}")
    print(f"  Skipped (already scored): {total_skipped:,}")

    # Summary stats
    row = conn.execute(
        "SELECT count(*) FROM articles WHERE climate_max_sim IS NOT NULL"
    ).fetchone()
    print(f"  Total articles with scores: {row[0]:,}")

    row = conn.execute("SELECT count(*) FROM articles").fetchone()
    print(f"  Total articles in DB: {row[0]:,}")

    conn.close()


if __name__ == "__main__":
    main()
