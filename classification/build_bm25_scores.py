"""Compute BM25 keyword scores for all articles in climate.db.

Two-pass approach:
  Pass 1: Compute per-language corpus stats (document frequency, avg doc length)
  Pass 2: Score each document against climate/health keyword queries, update DB

Usage:
    python build_bm25_scores.py
    python build_bm25_scores.py --batch-size 100000
"""

import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

# BM25 parameters
K1 = 1.5
B = 0.75

BATCH_SIZE = 50_000
DB_PATH = Path("data/climate.db")
KEYWORDS_PATH = Path("translations/keyword_translations.json")

# Languages needing substring matching (no word boundaries)
CJK_LANGS = {"zho", "jpn", "kor", "tha"}

# Fallback for languages not in keyword file
LANG_FALLBACK = {"cat": "spa", "bos": "hrv", "": "eng", "msa": "ind"}


def load_keywords():
    """Load and compile keyword patterns per language."""
    with open(KEYWORDS_PATH) as f:
        raw = json.load(f)

    keywords = {"climate": {}, "health": {}}
    for category in ["climate", "health"]:
        for lang, terms in raw[category].items():
            keywords[category][lang] = terms
    return keywords


def get_lang(lang, keywords_dict):
    """Resolve language to one that exists in keywords, or None."""
    if lang in keywords_dict:
        return lang
    fb = LANG_FALLBACK.get(lang)
    if fb and fb in keywords_dict:
        return fb
    return None


def count_hits(text, terms, is_cjk):
    """Count occurrences of each keyword term in text. Returns dict term->count."""
    hits = {}
    if is_cjk:
        for term in terms:
            c = text.count(term)
            if c > 0:
                hits[term] = c
    else:
        text_lower = text.lower()
        for term in terms:
            # Word-boundary match for non-CJK
            c = len(re.findall(r"\b" + re.escape(term.lower()) + r"\b", text_lower))
            if c > 0:
                hits[term] = c
    return hits


def doc_length(text, is_cjk):
    """Document length: character count for CJK, word count otherwise."""
    if is_cjk:
        return len(text)
    return len(text.split())


def bm25_score(hits, df, n_docs, avg_dl, dl):
    """Compute BM25 score for a single document given keyword hits."""
    score = 0.0
    for term, tf in hits.items():
        n = df.get(term, 0)
        idf = math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)
        tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avg_dl))
        score += idf * tf_norm
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    keywords = load_keywords()
    # Use climate keywords to determine available languages
    available_langs = set(keywords["climate"].keys())

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    # Add columns if they don't exist
    existing = {row[1] for row in cur.execute("PRAGMA table_info(articles)").fetchall()}
    for col in ["bm25_climate", "bm25_health", "bm25_avg"]:
        if col not in existing:
            cur.execute(f"ALTER TABLE articles ADD COLUMN {col} REAL")
    conn.commit()

    # Count total rows
    total = cur.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"Total articles: {total:,}")

    # =========================================================
    # PASS 1: Compute per-language corpus stats
    # =========================================================
    print("\n=== Pass 1: Computing corpus statistics ===")

    # Per-language stats
    # df[category][lang][term] = number of docs containing term
    df = {"climate": defaultdict(Counter), "health": defaultdict(Counter)}
    # doc count and total doc length per language
    lang_n = Counter()
    lang_total_dl = defaultdict(float)

    offset = 0
    pbar = tqdm(total=total, desc="Pass 1")
    while offset < total:
        rows = cur.execute(
            "SELECT id, title, content, language FROM articles LIMIT ? OFFSET ?",
            (args.batch_size, offset),
        ).fetchall()
        if not rows:
            break

        for row_id, title, content, lang in rows:
            text = (title or "") + " " + (content or "")
            resolved = get_lang(lang or "", keywords["climate"])
            if resolved is None:
                pbar.update(1)
                continue

            is_cjk = resolved in CJK_LANGS
            dl = doc_length(text, is_cjk)
            lang_n[resolved] += 1
            lang_total_dl[resolved] += dl

            for category in ["climate", "health"]:
                terms = keywords[category].get(resolved, [])
                if not terms:
                    continue
                hits = count_hits(text, terms, is_cjk)
                for term in hits:
                    df[category][resolved][term] += 1

            pbar.update(1)

        offset += len(rows)
    pbar.close()

    # Compute average doc length per language
    avg_dl = {}
    for lang in lang_n:
        avg_dl[lang] = lang_total_dl[lang] / lang_n[lang] if lang_n[lang] > 0 else 1.0

    print(f"\nLanguages with docs: {len(lang_n)}")
    for lang in sorted(lang_n, key=lang_n.get, reverse=True)[:10]:
        print(f"  {lang}: {lang_n[lang]:>8,} docs, avg_dl={avg_dl[lang]:.0f}")

    # =========================================================
    # PASS 2: Score and update
    # =========================================================
    print("\n=== Pass 2: Scoring and updating DB ===")

    offset = 0
    scored = 0
    skipped = 0
    pbar = tqdm(total=total, desc="Pass 2")
    while offset < total:
        rows = cur.execute(
            "SELECT id, title, content, language FROM articles LIMIT ? OFFSET ?",
            (args.batch_size, offset),
        ).fetchall()
        if not rows:
            break

        updates = []
        for row_id, title, content, lang in rows:
            text = (title or "") + " " + (content or "")
            resolved = get_lang(lang or "", keywords["climate"])
            if resolved is None:
                skipped += 1
                pbar.update(1)
                continue

            is_cjk = resolved in CJK_LANGS
            dl = doc_length(text, is_cjk)
            n_docs = lang_n[resolved]
            adl = avg_dl[resolved]

            scores = {}
            for category in ["climate", "health"]:
                terms = keywords[category].get(resolved, [])
                if not terms:
                    scores[category] = 0.0
                    continue
                hits = count_hits(text, terms, is_cjk)
                scores[category] = bm25_score(hits, df[category][resolved], n_docs, adl, dl)

            bm25_avg = (scores["climate"] + scores["health"]) / 2.0
            updates.append((scores["climate"], scores["health"], bm25_avg, row_id))
            scored += 1
            pbar.update(1)

        if updates:
            cur.executemany(
                "UPDATE articles SET bm25_climate=?, bm25_health=?, bm25_avg=? WHERE id=?",
                updates,
            )
            conn.commit()

        offset += len(rows)
    pbar.close()

    print(f"\nDone. Scored: {scored:,}, Skipped (no keywords): {skipped:,}")

    # Quick stats
    for col in ["bm25_climate", "bm25_health", "bm25_avg"]:
        row = cur.execute(
            f"SELECT MIN({col}), AVG({col}), MAX({col}) FROM articles WHERE {col} IS NOT NULL"
        ).fetchone()
        print(f"  {col}: min={row[0]:.3f}, avg={row[1]:.3f}, max={row[2]:.3f}")

    conn.close()


if __name__ == "__main__":
    main()
