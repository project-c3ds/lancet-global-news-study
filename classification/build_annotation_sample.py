"""Build a stratified sample of ~5K articles for human annotation.

Stratifies by BM25 score (3 bins), year (2021-2025), and language (top 10 + other),
with country diversity enforced within each stratum.

Usage:
    python build_annotation_sample.py
    python build_annotation_sample.py --target 5000 --floor 3 --country-cap 5
"""

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

DB_PATH = Path("data/climate.db")
OUTPUT_PATH = Path("data/annotation_sample_5k.jsonl")

TOP_LANGUAGES = ["eng", "spa", "zho", "fra", "deu", "tur", "tha", "hrv", "ara", "jpn"]
YEARS = [2021, 2022, 2023, 2024, 2025]


def get_bm25_bins(cur):
    """Compute tertile thresholds for bm25_avg across all scored articles."""
    cur.execute(
        "SELECT bm25_avg FROM articles WHERE bm25_avg IS NOT NULL "
        "AND CAST(SUBSTR(published_date, 1, 4) AS INTEGER) BETWEEN 2021 AND 2025"
    )
    scores = [row[0] for row in cur.fetchall()]
    scores.sort()
    t1 = np.percentile(scores, 33.3)
    t2 = np.percentile(scores, 66.6)
    print(f"BM25 tertile thresholds: low < {t1:.3f} <= medium < {t2:.3f} <= high")
    print(f"Total scored articles (2021-2025): {len(scores):,}")
    return t1, t2


def bm25_bin(score, t1, t2):
    if score < t1:
        return "low"
    elif score < t2:
        return "medium"
    else:
        return "high"


def lang_group(lang):
    if lang in TOP_LANGUAGES:
        return lang
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Build stratified annotation sample")
    parser.add_argument("--target", type=int, default=5000, help="Target sample size")
    parser.add_argument("--floor", type=int, default=3, help="Minimum articles per stratum")
    parser.add_argument("--country-cap", type=int, default=5, help="Max articles per country per stratum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Step 1: Compute BM25 bin thresholds
    print("Computing BM25 bin thresholds...")
    t1, t2 = get_bm25_bins(cur)

    # Step 2: Load all eligible articles into strata
    print("\nLoading articles into strata...")
    # strata[key] = list of (id, country) — key is (bm25_bin, year, lang_group)
    strata = defaultdict(list)
    article_ids = set()

    cur_iter = conn.cursor()
    cur_iter.execute(
        "SELECT id, language, country, published_date, bm25_avg "
        "FROM articles "
        "WHERE bm25_avg IS NOT NULL "
        "AND CAST(SUBSTR(published_date, 1, 4) AS INTEGER) BETWEEN 2021 AND 2025"
    )

    total = 0
    for row in tqdm(cur_iter, desc="Indexing"):
        year = int(row["published_date"][:4])
        if year not in YEARS:
            continue
        score = row["bm25_avg"]
        key = (bm25_bin(score, t1, t2), year, lang_group(row["language"] or ""))
        strata[key].append((row["id"], row["country"] or "unknown"))
        total += 1

    print(f"Total eligible: {total:,}")
    print(f"Non-empty strata: {len(strata)}")

    # Step 3: Allocate sample sizes per stratum
    # Floor allocation first, then distribute remainder proportionally
    non_empty = {k: v for k, v in strata.items() if len(v) > 0}
    n_strata = len(non_empty)
    floor_total = n_strata * args.floor
    remainder = max(0, args.target - floor_total)

    allocations = {}
    for key, articles in non_empty.items():
        prop_share = int(round(remainder * len(articles) / total))
        allocations[key] = min(len(articles), args.floor + prop_share)

    # Adjust to hit target
    current_total = sum(allocations.values())
    if current_total < args.target:
        # Add to largest strata
        deficit = args.target - current_total
        for key in sorted(allocations, key=lambda k: len(non_empty[k]), reverse=True):
            can_add = len(non_empty[key]) - allocations[key]
            add = min(can_add, deficit)
            allocations[key] += add
            deficit -= add
            if deficit <= 0:
                break

    print(f"Allocated: {sum(allocations.values()):,} articles across {n_strata} strata")

    # Step 4: Sample within each stratum with country diversity
    print("\nSampling with country diversity...")
    sampled_ids = []

    for key, n_sample in tqdm(allocations.items(), desc="Sampling"):
        articles = non_empty[key]
        random.shuffle(articles)

        # Group by country
        by_country = defaultdict(list)
        for art_id, country in articles:
            by_country[country].append(art_id)

        # Round-robin across countries, respecting country cap
        selected = []
        countries = list(by_country.keys())
        random.shuffle(countries)

        # Track how many picked per country
        country_count = defaultdict(int)
        round_idx = 0

        while len(selected) < n_sample:
            added_this_round = False
            for country in countries:
                if len(selected) >= n_sample:
                    break
                if country_count[country] >= args.country_cap:
                    continue
                pool = by_country[country]
                idx = country_count[country]
                if idx < len(pool):
                    selected.append(pool[idx])
                    country_count[country] += 1
                    added_this_round = True

            if not added_this_round:
                # All countries capped — lift cap for remaining
                for country in countries:
                    pool = by_country[country]
                    idx = country_count[country]
                    while idx < len(pool) and len(selected) < n_sample:
                        selected.append(pool[idx])
                        country_count[country] += 1
                        idx += 1
                break

            round_idx += 1

        sampled_ids.extend(selected)

    print(f"\nTotal sampled: {len(sampled_ids):,}")

    # Step 5: Fetch full articles and write output
    print("Fetching full articles and writing output...")
    sampled_set = set(sampled_ids)
    # Fetch in batches
    out_f = open(OUTPUT_PATH, "w")
    written = 0

    placeholders_batch = 500
    sampled_list = list(sampled_ids)
    for i in tqdm(range(0, len(sampled_list), placeholders_batch), desc="Writing"):
        batch = sampled_list[i : i + placeholders_batch]
        placeholders = ",".join(["?"] * len(batch))
        rows = cur.execute(
            f"SELECT id, url, title, content, source_uri, language, published_date, "
            f"country, bm25_climate, bm25_health, bm25_avg "
            f"FROM articles WHERE id IN ({placeholders})",
            batch,
        ).fetchall()

        for row in rows:
            record = dict(row)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    out_f.close()
    conn.close()

    print(f"\nDone. Wrote {written:,} articles to {OUTPUT_PATH}")

    # Summary stats
    print("\n=== Sample Summary ===")
    records = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    # By year
    from collections import Counter
    years = Counter(r["published_date"][:4] for r in records)
    print("\nBy year:")
    for y in sorted(years):
        print(f"  {y}: {years[y]:,}")

    # By language group
    langs = Counter(lang_group(r.get("language", "")) for r in records)
    print("\nBy language group:")
    for l, c in langs.most_common():
        print(f"  {l}: {c:,}")

    # By BM25 bin
    bins = Counter()
    for r in records:
        s = r.get("bm25_avg", 0) or 0
        bins[bm25_bin(s, t1, t2)] += 1
    print("\nBy BM25 bin:")
    for b_name in ["low", "medium", "high"]:
        print(f"  {b_name}: {bins[b_name]:,}")

    # Country diversity
    countries = Counter(r.get("country", "unknown") for r in records)
    print(f"\nUnique countries: {len(countries)}")
    print("Top 10 countries:")
    for c, n in countries.most_common(10):
        print(f"  {c}: {n:,}")


if __name__ == "__main__":
    main()
