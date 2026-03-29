#!/usr/bin/env python3
"""Classify articles as climate, health, climate+health, or neither.

Reads article embeddings from Weaviate and computes cosine similarity
against English keyword embeddings, with per-language thresholds.

Modes:
  classify  - Full run: score all articles, output classifications
  sample    - Inspect articles near threshold boundaries for calibration

Usage:
    python classify_articles.py classify
    python classify_articles.py classify --output results/classifications.csv
    python classify_articles.py sample --lang English --category climate
    python classify_articles.py sample --lang Portuguese --category health --n 20
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from weaviate_utils import (
    KEYWORD_EMBEDDINGS_DIR,
    RESULTS_DIR,
    connect,
    get_collection,
    load_keyword_embeddings,
    build_source_language_map,
    normalize_language,
    get_thresholds,
    get_sources_for_language,
    resolve_source,
    score_articles,
    classify_single,
    iterate_collection,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_classify(args):
    """Full classification run over all articles in Weaviate."""
    print("Loading keyword embeddings...")
    climate_kw, climate_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "climate_eng.jsonl"
    )
    health_kw, health_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "health_eng.jsonl"
    )
    print(f"  Climate: {len(climate_kw)} keywords, Health: {len(health_kw)} keywords")

    lang_map = build_source_language_map()
    print(f"  Source-language mappings: {len(lang_map)}")

    # Output setup
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source", "language", "article_id", "url", "title", "extracted_at",
        "climate_max_sim", "climate_best_keyword",
        "health_max_sim", "health_best_keyword",
        "climate_threshold", "health_threshold",
        "category",
    ]

    # Connect to Weaviate
    client = connect()
    try:
        collection = get_collection(client)

        t_start = time.time()
        total_articles = 0
        batch_count = 0
        category_counts = defaultdict(int)
        lang_counts = defaultdict(int)

        with open(output_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for meta_batch, emb_batch in iterate_collection(collection):
                scores = score_articles(
                    emb_batch, climate_emb, health_emb, climate_kw, health_kw
                )

                for meta, sc in zip(meta_batch, scores):
                    source_name = resolve_source(meta, lang_map)

                    # Skip if --source filter is set and doesn't match
                    if args.source and source_name != args.source:
                        continue

                    raw_lang = lang_map.get(source_name, "")
                    language = normalize_language(raw_lang)
                    climate_t, health_t = get_thresholds(language)

                    category = classify_single(
                        sc["climate_max_sim"], sc["health_max_sim"],
                        climate_t, health_t,
                    )

                    # Format extracted_at
                    extracted = meta.get("extracted_at")
                    if extracted and hasattr(extracted, "isoformat"):
                        extracted = extracted.isoformat()
                    elif not extracted:
                        extracted = ""

                    row = {
                        "source": source_name,
                        "language": language or raw_lang or "unknown",
                        "article_id": meta.get("article_id", ""),
                        "url": meta.get("url", ""),
                        "title": meta.get("title", ""),
                        "extracted_at": extracted,
                        "climate_max_sim": f"{sc['climate_max_sim']:.4f}",
                        "climate_best_keyword": sc["climate_best_keyword"],
                        "health_max_sim": f"{sc['health_max_sim']:.4f}",
                        "health_best_keyword": sc["health_best_keyword"],
                        "climate_threshold": f"{climate_t:.2f}",
                        "health_threshold": f"{health_t:.2f}",
                        "category": category,
                    }

                    if category != "neither" or args.include_neither:
                        writer.writerow(row)

                    category_counts[category] += 1
                    lang_counts[language or "unknown"] += 1

                total_articles += len(meta_batch)
                batch_count += 1

                if batch_count % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = total_articles / elapsed if elapsed > 0 else 0
                    print(
                        f"  {total_articles:,} articles, {rate:,.0f}/s | "
                        f"climate={category_counts['climate']:,} "
                        f"health={category_counts['health']:,} "
                        f"both={category_counts['climate+health']:,} "
                        f"neither={category_counts['neither']:,}"
                    )

        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed:.1f}s — {total_articles:,} articles processed")
        print(f"\nCategory counts:")
        for cat in ["climate", "health", "climate+health", "neither"]:
            pct = category_counts[cat] / total_articles * 100 if total_articles else 0
            print(f"  {cat:<16} {category_counts[cat]:>8,}  ({pct:.1f}%)")
        print(f"\nTop languages:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {lang:<20} {count:>8,}")
        print(f"\nResults written to {output_path}")
        if not args.include_neither:
            matched = sum(v for k, v in category_counts.items() if k != "neither")
            print(f"  (filtered to {matched:,} matched articles; use --include-neither to include all)")

    finally:
        client.close()


def cmd_sample(args):
    """Sample articles near threshold boundaries for manual inspection."""
    climate_kw, climate_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "climate_eng.jsonl"
    )
    health_kw, health_emb = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "health_eng.jsonl"
    )

    lang_map = build_source_language_map()
    target_lang = args.lang
    source_list = get_sources_for_language(target_lang, lang_map)

    if not source_list:
        print(f"No sources found for language '{target_lang}'")
        available = sorted(set(
            normalize_language(v) or "unknown" for v in lang_map.values()
        ))
        print(f"Available languages: {available}")
        return

    climate_t, health_t = get_thresholds(target_lang)
    if args.category == "climate":
        threshold = climate_t
        score_key = "climate"
    else:
        threshold = health_t
        score_key = "health"

    window = args.window
    low = threshold - window
    high = threshold + window

    print(f"Sampling {args.category} articles in {target_lang}")
    print(f"  Threshold: {threshold:.4f}, window: [{low:.4f}, {high:.4f}]")
    print(f"  Sources: {len(source_list)}")
    print()

    # Connect to Weaviate and iterate over all articles, filtering to language
    source_set = set(source_list)
    client = connect()
    try:
        collection = get_collection(client)

        candidates = []
        total_scanned = 0

        for meta_batch, emb_batch in iterate_collection(collection):
            scores = score_articles(
                emb_batch, climate_emb, health_emb, climate_kw, health_kw
            )
            for meta, sc in zip(meta_batch, scores):
                source = resolve_source(meta, lang_map)
                if source not in source_set:
                    continue
                total_scanned += 1
                score = sc[f"{score_key}_max_sim"]
                if low <= score <= high:
                    candidates.append({
                        "source": source,
                        "score": score,
                        "best_keyword": sc[f"{score_key}_best_keyword"],
                        "title": meta.get("title", ""),
                        "url": meta.get("url", ""),
                        "climate_sim": sc["climate_max_sim"],
                        "health_sim": sc["health_max_sim"],
                        "climate_best_keyword": sc["climate_best_keyword"],
                        "health_best_keyword": sc["health_best_keyword"],
                    })

            if len(candidates) >= args.n * 5:
                break

    finally:
        client.close()

    print(f"Scanned {total_scanned:,} articles, found {len(candidates)} in window")
    print()

    if not candidates:
        print("No articles found in this window. Try widening with --window.")
        return

    # Sort by score and pick articles evenly from below and above threshold
    candidates.sort(key=lambda x: x["score"])
    below = [c for c in candidates if c["score"] < threshold]
    above = [c for c in candidates if c["score"] >= threshold]
    half_n = args.n // 2

    if len(below) > half_n:
        below = below[-half_n:]
    if len(above) > half_n:
        above = above[:half_n]

    sample = below + above

    # Display
    print(f"{'SCORE':>7} {'PASS':>5} {'C_SIM':>6} {'H_SIM':>6} {'KEYWORD':>25}  SOURCE / TITLE")
    print("-" * 120)
    for c in sample:
        passes = "YES" if c["score"] >= threshold else "no"
        print(
            f"{c['score']:>7.4f} {passes:>5} "
            f"{c['climate_sim']:>6.4f} {c['health_sim']:>6.4f} "
            f"{c['best_keyword']:>25}  "
            f"{c['source']}: {c['title'][:55]}"
        )

    print(f"\n--- Threshold: {threshold:.4f} ---")
    print(f"Articles above: {len(above)}, below: {len(below)} (in sample window)")

    # Save to CSV
    samples_dir = RESULTS_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_path = samples_dir / f"sample_{target_lang.lower()}_{args.category}_t{threshold:.2f}_w{window:.2f}.csv"

    fieldnames = [
        "language", "category", "threshold", "window",
        "score", "passes_threshold", "climate_sim", "health_sim",
        "best_keyword", "climate_best_keyword", "health_best_keyword",
        "source", "title", "url",
        "total_scanned", "total_in_window",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sample:
            writer.writerow({
                "language": target_lang,
                "category": args.category,
                "threshold": f"{threshold:.4f}",
                "window": f"{window:.4f}",
                "score": f"{c['score']:.4f}",
                "passes_threshold": "YES" if c["score"] >= threshold else "no",
                "climate_sim": f"{c['climate_sim']:.4f}",
                "health_sim": f"{c['health_sim']:.4f}",
                "best_keyword": c["best_keyword"],
                "climate_best_keyword": c["climate_best_keyword"],
                "health_best_keyword": c["health_best_keyword"],
                "source": c["source"],
                "title": c["title"],
                "url": c["url"],
                "total_scanned": total_scanned,
                "total_in_window": len(candidates),
            })
    print(f"\nSample saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Classify articles as climate, health, climate+health, or neither"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # classify
    p_classify = subparsers.add_parser("classify", help="Full classification run")
    p_classify.add_argument(
        "--output", default="results/classifications.csv",
        help="Output CSV path (default: results/classifications.csv)",
    )
    p_classify.add_argument("--source", help="Process a single source by name")
    p_classify.add_argument(
        "--include-neither", action="store_true",
        help="Include 'neither' articles in output (default: only matched)",
    )

    # sample
    p_sample = subparsers.add_parser("sample", help="Sample articles near thresholds")
    p_sample.add_argument(
        "--lang", required=True,
        help="Language to sample (e.g., English, Portuguese, Arabic)",
    )
    p_sample.add_argument(
        "--category", required=True, choices=["climate", "health"],
        help="Category threshold to inspect",
    )
    p_sample.add_argument(
        "--n", type=int, default=20,
        help="Number of articles to show (default: 20)",
    )
    p_sample.add_argument(
        "--window", type=float, default=0.03,
        help="Score window around threshold (default: 0.03)",
    )

    args = parser.parse_args()

    if args.command == "classify":
        cmd_classify(args)
    elif args.command == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
