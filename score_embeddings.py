"""Score articles using pre-computed embeddings and keyword cosine similarity.

Reads numpy embedding shards, computes max cosine similarity against
English climate/health keyword embeddings, applies per-language thresholds
with optional recall boost, and outputs matched articles with full text as JSONL.

Usage:
    python score_embeddings.py
    python score_embeddings.py --recall-boost 0.85
    python score_embeddings.py --recall-boost 0.90 --output data/scored_articles.jsonl
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

KEYWORD_DIR = Path("data/keyword_embeddings")
EMBEDDINGS_DIR = Path("data/embeddings_np")
DB_PATH = "data/articles.db"
DEFAULT_OUTPUT = Path("data/scored_articles.jsonl")

# Per-language thresholds from weaviate_utils.py (climate, health)
THRESHOLDS = {
    "English":    (0.59, 0.49),
    "Arabic":     (0.52, 0.49),
    "Chinese":    (0.44, 0.55),
    "Spanish":    (0.56, 0.54),
    "French":     (0.50, 0.49),
    "Portuguese": (0.55, 0.59),
    "German":     (0.58, 0.57),
    "Slovak":     (0.58, 0.60),
    "Bulgarian":  (0.35, 0.42),
    "Thai":       (0.42, 0.45),
    "Korean":     (0.40, 0.43),
    "Japanese":   (0.45, 0.46),
    "Polish":     (0.54, 0.57),
    "Indonesian": (0.53, 0.56),
    "Romanian":   (0.52, 0.57),
    "Czech":      (0.53, 0.58),
    "Hungarian":  (0.52, 0.54),
    "Turkish":    (0.49, 0.54),
    "Greek":      (0.49, 0.52),
    "Danish":     (0.56, 0.60),
    "Finnish":    (0.56, 0.60),
    "Italian":    (0.58, 0.60),
    "Croatian":   (0.58, 0.62),
    "Serbian":    (0.58, 0.62),
    "Swedish":    (0.62, 0.64),
}
DEFAULT_THRESHOLDS = (0.52, 0.54)


def get_thresholds(language, recall_boost):
    """Get (climate_threshold, health_threshold) for a language, with recall boost applied."""
    if language and language in THRESHOLDS:
        ct, ht = THRESHOLDS[language]
    else:
        ct, ht = DEFAULT_THRESHOLDS
    return ct * recall_boost, ht * recall_boost


def normalize_language(raw_lang):
    """Map raw language string to canonical name matching THRESHOLDS keys."""
    if not raw_lang:
        return None
    parts = [p.strip() for p in raw_lang.split(",")]
    for p in parts:
        if p in THRESHOLDS:
            return p
    return parts[0]


def load_keyword_embeddings(filepath):
    keywords = []
    embeddings = []
    with open(filepath) as f:
        for line in f:
            record = json.loads(line)
            keywords.append(record["keyword"])
            embeddings.append(record["embedding"])
    return keywords, np.array(embeddings, dtype=np.float32)


def cosine_similarity_batch(articles_emb, keyword_emb):
    a_norm = articles_emb / np.linalg.norm(articles_emb, axis=-1, keepdims=True)
    b_norm = keyword_emb / np.linalg.norm(keyword_emb, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def score_batch(embeddings, climate_emb, health_emb, climate_kw, health_kw):
    climate_sims = cosine_similarity_batch(embeddings, climate_emb)
    health_sims = cosine_similarity_batch(embeddings, health_emb)

    results = []
    for i in range(len(embeddings)):
        c_max_idx = int(climate_sims[i].argmax())
        h_max_idx = int(health_sims[i].argmax())
        results.append({
            "climate_max_sim": float(climate_sims[i, c_max_idx]),
            "climate_best_keyword": climate_kw[c_max_idx],
            "health_max_sim": float(health_sims[i, h_max_idx]),
            "health_best_keyword": health_kw[h_max_idx],
        })
    return results


def classify(climate_score, health_score, climate_t, health_t):
    is_climate = climate_score >= climate_t
    is_health = health_score >= health_t
    if is_climate and is_health:
        return "climate+health"
    elif is_climate:
        return "climate"
    elif is_health:
        return "health"
    else:
        return "neither"


def load_article_metadata(db_path, ids):
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, title, content, language, country, source_uri FROM articles WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    conn.close()
    return {row[0]: row for row in rows}


def build_id_language_map(db_path):
    """Build id -> language map for all articles."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, language FROM articles").fetchall()
    conn.close()
    return {row[0]: normalize_language(row[1]) for row in rows}


def main():
    parser = argparse.ArgumentParser(description="Score articles using keyword embeddings")
    parser.add_argument("--recall-boost", type=float, default=0.85,
                        help="Multiply thresholds by this factor to increase recall (default: 0.85)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-neither", action="store_true", help="Include articles below threshold")
    args = parser.parse_args()

    # Load keyword embeddings
    print("Loading keyword embeddings...")
    climate_kw, climate_emb = load_keyword_embeddings(KEYWORD_DIR / "climate_eng.jsonl")
    health_kw, health_emb = load_keyword_embeddings(KEYWORD_DIR / "health_eng.jsonl")
    print(f"  Climate: {len(climate_kw)} keywords, Health: {len(health_kw)} keywords")

    # Load language map
    print("Loading language map from SQLite...")
    lang_map = build_id_language_map(DB_PATH)
    print(f"  {len(lang_map):,} articles mapped")

    # Discover shards
    id_files = sorted(EMBEDDINGS_DIR.glob("ids_*.npy"))
    emb_files = sorted(EMBEDDINGS_DIR.glob("emb_*.npy"))
    assert len(id_files) == len(emb_files), f"Mismatch: {len(id_files)} id files vs {len(emb_files)} emb files"
    print(f"Found {len(id_files)} embedding shards")
    print(f"Recall boost: {args.recall_boost} (thresholds multiplied by this)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_scored = 0
    total_matched = 0
    category_counts = {"climate": 0, "health": 0, "climate+health": 0, "neither": 0}
    lang_counts = {}

    with open(output_path, "w") as out:
        for id_file, emb_file in tqdm(zip(id_files, emb_files), total=len(id_files), desc="Scoring"):
            ids = np.load(id_file)
            embeddings = np.load(emb_file)

            scores = score_batch(embeddings, climate_emb, health_emb, climate_kw, health_kw)

            matched_ids = []
            matched_scores = []
            for article_id, score in zip(ids, scores):
                lang = lang_map.get(int(article_id))
                ct, ht = get_thresholds(lang, args.recall_boost)
                cat = classify(score["climate_max_sim"], score["health_max_sim"], ct, ht)
                score["category"] = cat
                category_counts[cat] += 1
                if cat != "neither":
                    lang_counts[lang or "unknown"] = lang_counts.get(lang or "unknown", 0) + 1
                if cat != "neither" or args.include_neither:
                    matched_ids.append(int(article_id))
                    matched_scores.append(score)

            total_scored += len(ids)

            if not matched_ids:
                continue

            meta_map = load_article_metadata(DB_PATH, matched_ids)

            for article_id, score in zip(matched_ids, matched_scores):
                row = meta_map.get(article_id)
                if not row:
                    continue
                _, title, content, language, country, source = row
                record = {
                    "id": article_id,
                    "title": title or "",
                    "content": content or "",
                    "language": language or "",
                    "country": country or "",
                    "source": source or "",
                    "climate_max_sim": round(score["climate_max_sim"], 4),
                    "health_max_sim": round(score["health_max_sim"], 4),
                    "climate_best_keyword": score["climate_best_keyword"],
                    "health_best_keyword": score["health_best_keyword"],
                    "category": score["category"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_matched += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total scored: {total_scored:,}")
    print(f"Total matched: {total_matched:,} ({total_matched/total_scored*100:.1f}%)")
    print(f"\nCategory breakdown:")
    for cat in ["climate", "health", "climate+health", "neither"]:
        pct = category_counts[cat] / total_scored * 100 if total_scored > 0 else 0
        print(f"  {cat:<16} {category_counts[cat]:>10,}  ({pct:.1f}%)")
    print(f"\nTop languages (matched articles):")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {lang:<20} {count:>8,}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
