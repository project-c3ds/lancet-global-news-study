#!/usr/bin/env python3
"""Build a stratified labeled dataset for classifier training and evaluation.

Samples articles from Weaviate with full text from SQLite, stratified
across score ranges to produce a balanced dataset for human/LLM labeling.

Single-pass design: scans Weaviate once, scores all articles, and buckets
them by language before stratified sampling.

Usage:
    # Extract articles for a language, ready for labeling
    python build_labeled_dataset.py extract --lang English --n 300

    # Extract for all major languages (single Weaviate scan)
    python build_labeled_dataset.py extract --lang all --n 250
"""

import argparse
import csv
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from weaviate_utils import (
    KEYWORD_EMBEDDINGS_DIR,
    connect,
    get_collection,
    load_keyword_embeddings,
    build_source_language_map,
    resolve_source,
    normalize_language,
    score_articles,
    iterate_collection,
)

DB_PATH = Path("data/articles.db")
RESULTS_DIR = Path("results")

MIN_CONTENT_LEN = 500  # Minimum content length for labeled dataset


def build_valid_article_set(db_conn):
    """Build set of (source_uri, collection_method, original_id) for articles
    that exist in the cleaned SQLite DB with sufficient content length."""
    cursor = db_conn.cursor()
    rows = cursor.execute(
        "SELECT source_uri, collection_method, original_id "
        "FROM articles WHERE length(content) >= ?",
        (MIN_CONTENT_LEN,),
    ).fetchall()
    return set(rows)


def score_and_bucket_all(collection, lang_map, target_languages, climate_kw,
                         climate_emb, health_kw, health_emb, valid_articles):
    """Single pass: scan all Weaviate articles, score them, bucket by language.

    Only includes articles that exist in the cleaned SQLite DB (valid_articles set).
    Returns dict of {language: [scored_article, ...]}
    """
    # Build reverse map: source_name -> language
    source_to_lang = {}
    for source_name, raw_lang in lang_map.items():
        lang = normalize_language(raw_lang)
        if lang in target_languages:
            source_to_lang[source_name] = lang

    by_language = defaultdict(list)
    total_scanned = 0
    total_matched = 0
    total_filtered = 0

    for meta_batch, emb_batch in iterate_collection(collection):
        scores = score_articles(
            emb_batch, climate_emb, health_emb, climate_kw, health_kw
        )
        for meta, sc in zip(meta_batch, scores):
            total_scanned += 1
            source = resolve_source(meta, lang_map)
            lang = source_to_lang.get(source)
            if lang is None:
                continue

            # Validate against cleaned SQLite
            key = (
                meta.get("source_uri", ""),
                meta.get("collection_method", ""),
                meta.get("article_id"),
            )
            if key not in valid_articles:
                total_filtered += 1
                continue

            total_matched += 1
            by_language[lang].append({
                "source": source,
                "source_uri": key[0],
                "collection_method": key[1],
                "article_id": key[2],
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "climate_max_sim": sc["climate_max_sim"],
                "health_max_sim": sc["health_max_sim"],
                "climate_best_keyword": sc["climate_best_keyword"],
                "health_best_keyword": sc["health_best_keyword"],
            })

        if total_scanned % 500_000 < len(meta_batch):
            print(f"    Scanned {total_scanned:,}, matched {total_matched:,}, "
                  f"filtered {total_filtered:,}...")

    print(f"  Done scanning. {total_scanned:,} total, {total_matched:,} matched, "
          f"{total_filtered:,} filtered (not in cleaned DB).")
    for lang in sorted(by_language):
        print(f"    {lang}: {len(by_language[lang]):,}")

    return by_language


def stratified_sample(all_scored, n_articles, seed=42):
    """Sample articles into 5 balanced buckets by score range.

    Buckets:
      1. high_climate  — Top 5% climate score (climate-only candidates)
      2. high_health   — Top 5% health score (health-only candidates)
      3. high_both     — Top 10% in both scores (climate+health / impact candidates)
      4. low_both      — Bottom 50% in both (negative examples)
      5. borderline    — Near the 85th percentile boundary (hard cases)

    Any remaining quota is filled with random articles for diversity.
    """
    random.seed(seed)
    np.random.seed(seed)

    per_bucket = n_articles // 5

    climate_scores = np.array([a["climate_max_sim"] for a in all_scored])
    health_scores = np.array([a["health_max_sim"] for a in all_scored])

    c_p50, c_p85, c_p90, c_p95 = np.percentile(climate_scores, [50, 85, 90, 95])
    h_p50, h_p85, h_p90, h_p95 = np.percentile(health_scores, [50, 85, 90, 95])

    print(f"    Percentiles — climate: p50={c_p50:.3f} p85={c_p85:.3f} p90={c_p90:.3f} p95={c_p95:.3f}")
    print(f"    Percentiles — health:  p50={h_p50:.3f} p85={h_p85:.3f} p90={h_p90:.3f} p95={h_p95:.3f}")

    high_c = sorted(
        [a for a in all_scored if a["climate_max_sim"] > c_p95],
        key=lambda x: -x["climate_max_sim"],
    )
    high_h = sorted(
        [a for a in all_scored if a["health_max_sim"] > h_p95],
        key=lambda x: -x["health_max_sim"],
    )
    high_both = sorted(
        [a for a in all_scored
         if a["climate_max_sim"] > c_p90 and a["health_max_sim"] > h_p90],
        key=lambda x: -(x["climate_max_sim"] + x["health_max_sim"]),
    )
    low_both = sorted(
        [a for a in all_scored
         if a["climate_max_sim"] < c_p50 and a["health_max_sim"] < h_p50],
        key=lambda x: x["climate_max_sim"] + x["health_max_sim"],
    )
    borderline = [
        a for a in all_scored
        if abs(a["climate_max_sim"] - c_p85) < 0.02
        or abs(a["health_max_sim"] - h_p85) < 0.02
    ]

    print(f"    Pool sizes — high_climate: {len(high_c)}, high_health: {len(high_h)}, "
          f"high_both: {len(high_both)}, low_both: {len(low_both)}, borderline: {len(borderline)}")

    def sample_bucket(bucket, n):
        if len(bucket) <= n:
            return bucket
        return random.sample(bucket, n)

    sampled = set()
    selected = []

    def add_from(bucket, n, label):
        added = 0
        for a in sample_bucket(bucket, n * 2):
            key = (a["source_uri"], a["collection_method"], a["article_id"])
            if key not in sampled:
                sampled.add(key)
                a["sample_bucket"] = label
                selected.append(a)
                added += 1
                if added >= n:
                    break

    add_from(high_c, per_bucket, "high_climate")
    add_from(high_h, per_bucket, "high_health")
    add_from(high_both, per_bucket, "high_both")
    add_from(low_both, per_bucket, "low_both")
    add_from(borderline, per_bucket, "borderline")

    # Fill remaining from random
    remaining = n_articles - len(selected)
    if remaining > 0:
        pool = [a for a in all_scored
                if (a["source_uri"], a["collection_method"], a["article_id"]) not in sampled]
        add_from(pool, remaining, "random")

    return selected


def retrieve_full_text(selected, lang, conn):
    """Look up full article text from SQLite for the selected articles."""
    cursor = conn.cursor()
    results = []
    not_found = 0

    for i, article in enumerate(selected):
        source_uri = article["source_uri"]
        method = article["collection_method"]
        art_id = article["article_id"]

        row = cursor.execute(
            "SELECT content, country FROM articles "
            "WHERE source_uri = ? AND collection_method = ? AND original_id = ? "
            "LIMIT 1",
            (source_uri, method, art_id),
        ).fetchone()

        if row is None:
            not_found += 1
            continue

        content = (row[0] or "").strip()
        country = row[1] or ""
        if len(content) > 3000:
            content = content[:3000] + "..."

        results.append({
            "language": lang,
            "country": country,
            "source": article["source_uri"],
            "collection_method": method,
            "article_id": art_id,
            "url": article["url"],
            "title": article["title"],
            "content": content,
            "climate_max_sim": f"{article['climate_max_sim']:.4f}",
            "health_max_sim": f"{article['health_max_sim']:.4f}",
            "climate_best_keyword": article["climate_best_keyword"],
            "health_best_keyword": article["health_best_keyword"],
            "sample_bucket": article["sample_bucket"],
            # Labels — to be filled during labeling
            "climate": "",
            "health": "",
            "health_climate_impact": "",
            "climate_justification": "",
            "health_justification": "",
            "impact_justification": "",
        })

    if not_found:
        print(f"    WARNING: {not_found} articles not found in SQLite")

    return results


def main():
    parser = argparse.ArgumentParser(description="Build labeled dataset")
    parser.add_argument("command", choices=["extract"],
                        help="Command to run")
    parser.add_argument("--lang", required=True,
                        help="Language to extract (or 'all' for all major languages)")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of articles to extract per language (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="results/labeled",
                        help="Output directory (default: results/labeled)")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output filenames (e.g. '_100' -> to_label_english_100.csv)")
    args = parser.parse_args()

    # Load keyword embeddings
    print("Loading keyword embeddings...")
    climate_kw, climate_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "climate_eng.jsonl"
    )
    health_kw, health_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "health_eng.jsonl"
    )

    lang_map = build_source_language_map()

    MAJOR_10 = {"English", "Spanish", "French", "Portuguese", "Arabic",
                "Chinese", "German", "Slovak", "Polish", "Italian"}

    if args.lang == "every":
        # All languages in one scan
        all_langs = set()
        for src, raw_lang in lang_map.items():
            l = normalize_language(raw_lang)
            if l:
                all_langs.add(l)
        languages = sorted(all_langs)
    elif args.lang == "all":
        languages = sorted(MAJOR_10)
    elif args.lang == "remaining":
        all_langs = set()
        for src, raw_lang in lang_map.items():
            l = normalize_language(raw_lang)
            if l:
                all_langs.add(l)
        languages = sorted(all_langs - MAJOR_10)
    else:
        languages = [args.lang]

    target_set = set(languages)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open SQLite for full text retrieval
    db_conn = sqlite3.connect(str(DB_PATH))
    db_conn.execute("PRAGMA cache_size=-256000")

    # Build valid article set from cleaned SQLite
    print(f"Building valid article set (content >= {MIN_CONTENT_LEN} chars)...")
    valid_articles = build_valid_article_set(db_conn)
    print(f"  {len(valid_articles):,} valid articles in SQLite")

    client = connect()
    try:
        collection = get_collection(client)

        # Single pass: score everything and bucket by language
        print(f"\nScoring all articles (single Weaviate scan, {len(target_set)} languages)...")
        by_language = score_and_bucket_all(
            collection, lang_map, target_set,
            climate_kw, climate_emb, health_kw, health_emb,
            valid_articles,
        )

        # Sample and retrieve full text per language
        for lang in languages:
            scored = by_language.get(lang, [])
            if not scored:
                print(f"\n  {lang}: no articles found, skipping")
                continue

            print(f"\n  {lang}: {len(scored):,} articles scored")

            selected = stratified_sample(scored, args.n, seed=args.seed)
            print(f"  Selected {len(selected)} articles for labeling")

            print(f"  Retrieving full text from SQLite...")
            results = retrieve_full_text(selected, lang, db_conn)
            print(f"  Retrieved full text for {len(results)} articles")

            if not results:
                continue

            output_path = output_dir / f"to_label_{lang.lower()}{args.suffix}.csv"
            fieldnames = list(results[0].keys())
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            print(f"  Saved to {output_path}")
            buckets = Counter(r["sample_bucket"] for r in results)
            for b, c in sorted(buckets.items()):
                print(f"    {b}: {c}")

    finally:
        client.close()
        db_conn.close()


if __name__ == "__main__":
    main()
