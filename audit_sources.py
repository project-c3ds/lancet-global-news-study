#!/usr/bin/env python3
"""Audit scrapai source quality in SQLite.

Computes per-source quality metrics to identify broken scrapers,
boilerplate content, and other data quality issues.

Usage:
    python audit_sources.py
    python audit_sources.py --output results/source_quality_audit.csv
"""

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = "data/articles.db"


def audit_source(conn, source_uri):
    """Compute quality metrics for a single scrapai source."""
    rows = conn.execute(
        "SELECT title, content FROM articles "
        "WHERE source_uri = ? AND collection_method = 'scrapai'",
        (source_uri,),
    ).fetchall()

    n = len(rows)
    if n == 0:
        return None

    titles = []
    contents = []
    content_lengths = []

    for title, content in rows:
        t = (title or "").strip()
        c = (content or "").strip()
        titles.append(t)
        contents.append(c)
        content_lengths.append(len(c))

    content_lengths.sort()

    # --- Content length stats ---
    p10 = content_lengths[int(n * 0.10)] if n > 10 else content_lengths[0]
    p25 = content_lengths[int(n * 0.25)] if n > 4 else content_lengths[0]
    median = content_lengths[n // 2]
    p75 = content_lengths[int(n * 0.75)] if n > 4 else content_lengths[-1]

    # --- Empty / very short ---
    empty_count = sum(1 for l in content_lengths if l == 0)
    short_100 = sum(1 for l in content_lengths if l < 100)
    short_200 = sum(1 for l in content_lengths if l < 200)

    # --- Duplicate content ---
    content_counts = Counter(contents)
    # Articles whose exact content appears more than once
    dup_content = sum(count for text, count in content_counts.items() if count > 1 and text)
    # Most common content and its count
    most_common_content, most_common_count = content_counts.most_common(1)[0]

    # --- Duplicate titles ---
    title_counts = Counter(titles)
    dup_titles = sum(count for t, count in title_counts.items() if count > 1 and t)
    most_common_title, most_common_title_count = title_counts.most_common(1)[0]

    # --- Shared prefix (boilerplate header) ---
    # Sample up to 500 non-empty contents to find common prefix
    sample = [c for c in contents if len(c) > 50][:500]
    shared_prefix_len = 0
    if len(sample) >= 10:
        # Check how many share the same first 50 chars
        prefixes = Counter(c[:50] for c in sample)
        top_prefix, top_prefix_count = prefixes.most_common(1)[0]
        prefix_ratio = top_prefix_count / len(sample)
        if prefix_ratio > 0.3:
            # Find exact shared prefix length
            matching = [c for c in sample if c[:50] == top_prefix]
            if len(matching) >= 2:
                prefix = matching[0]
                for other in matching[1:]:
                    i = 0
                    while i < len(prefix) and i < len(other) and prefix[i] == other[i]:
                        i += 1
                    prefix = prefix[:i]
                shared_prefix_len = len(prefix)
    else:
        prefix_ratio = 0.0

    # --- Quality score (0-100, higher = better) ---
    # Penalize for: short content, duplicates, boilerplate
    penalties = 0
    penalties += min(40, (short_200 / n) * 80)         # up to 40 pts for short articles
    penalties += min(20, (dup_content / n) * 40)        # up to 20 pts for duplicate content
    penalties += min(15, (dup_titles / n) * 30)         # up to 15 pts for duplicate titles
    penalties += min(15, prefix_ratio * 30)             # up to 15 pts for shared prefix
    penalties += min(10, (empty_count / n) * 20)        # up to 10 pts for empty articles
    quality_score = max(0, round(100 - penalties))

    return {
        "source_uri": source_uri,
        "articles": n,
        "quality_score": quality_score,
        "median_len": median,
        "p10_len": p10,
        "p25_len": p25,
        "p75_len": p75,
        "empty_pct": round(empty_count / n * 100, 1),
        "short_100_pct": round(short_100 / n * 100, 1),
        "short_200_pct": round(short_200 / n * 100, 1),
        "dup_content_pct": round(dup_content / n * 100, 1),
        "dup_title_pct": round(dup_titles / n * 100, 1),
        "top_title": most_common_title[:60] if most_common_title else "",
        "top_title_count": most_common_title_count,
        "top_content_count": most_common_count,
        "shared_prefix_pct": round(prefix_ratio * 100, 1),
        "shared_prefix_len": shared_prefix_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit scrapai source quality")
    parser.add_argument("--output", default="results/source_quality_audit.csv")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA cache_size=-512000")

    # Get all scrapai sources
    sources = conn.execute(
        "SELECT source_uri, count(*) FROM articles "
        "WHERE collection_method = 'scrapai' "
        "GROUP BY source_uri ORDER BY source_uri"
    ).fetchall()

    print(f"Auditing {len(sources)} scrapai sources...")

    results = []
    for i, (source_uri, count) in enumerate(sources, 1):
        metrics = audit_source(conn, source_uri)
        if metrics:
            results.append(metrics)
        if i % 25 == 0 or i == len(sources):
            print(f"  [{i}/{len(sources)}]", flush=True)

    conn.close()

    # Sort by quality score ascending (worst first)
    results.sort(key=lambda x: x["quality_score"])

    # Write CSV
    import csv
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {args.output}")

    # Summary
    bad = [r for r in results if r["quality_score"] < 50]
    warn = [r for r in results if 50 <= r["quality_score"] < 75]
    ok = [r for r in results if r["quality_score"] >= 75]

    print(f"\nQuality breakdown:")
    print(f"  Bad (score < 50):    {len(bad)} sources")
    print(f"  Warning (50-74):     {len(warn)} sources")
    print(f"  OK (75+):            {len(ok)} sources")

    if bad:
        print(f"\nWorst sources:")
        for r in bad[:15]:
            print(f"  {r['source_uri']:35s} score={r['quality_score']:3d}  "
                  f"median={r['median_len']:5d}  short<200={r['short_200_pct']:5.1f}%  "
                  f"dup_content={r['dup_content_pct']:5.1f}%  dup_title={r['dup_title_pct']:5.1f}%")


if __name__ == "__main__":
    main()
