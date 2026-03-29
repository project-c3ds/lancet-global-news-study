#!/usr/bin/env python3
"""Shared utilities for Weaviate-based article classification and sampling."""

import csv
import json
from pathlib import Path

import numpy as np
import weaviate

COLLECTION_NAME = "NewsArticles"
KEYWORD_EMBEDDINGS_DIR = Path("data/keyword_embeddings")
SOURCE_METADATA = Path("data/top10_per_country.csv")
RESULTS_DIR = Path("results")

# Per-language thresholds (English max-sim).
# Calibrated in two stages:
#   1. Synthetic articles set initial bounds (calibrate_thresholds.py)
#   2. Real-data sampling adjusted for noise from cross-lingual keyword leakage
THRESHOLDS = {
    # --- Calibrated on synthetic + real-data sampling ---
    "English":    (0.59, 0.49),
    "Arabic":     (0.52, 0.49),
    "Chinese":    (0.44, 0.55),
    "Spanish":    (0.56, 0.54),
    "French":     (0.50, 0.49),
    "Portuguese": (0.55, 0.59),
    "German":     (0.58, 0.57),
    "Slovak":     (0.58, 0.60),
    # --- Non-Latin scripts ---
    "Bulgarian":  (0.35, 0.42),
    "Thai":       (0.42, 0.45),
    "Korean":     (0.40, 0.43),
    "Japanese":   (0.45, 0.46),
    # --- Latin-script, moderate noise ---
    "Polish":     (0.54, 0.57),
    "Indonesian": (0.53, 0.56),
    "Romanian":   (0.52, 0.57),
    "Czech":      (0.53, 0.58),
    "Hungarian":  (0.52, 0.54),
    "Turkish":    (0.49, 0.54),
    "Greek":      (0.49, 0.52),
    "Danish":     (0.56, 0.60),
    "Finnish":    (0.56, 0.60),
    # --- Latin-script, high noise ---
    "Italian":    (0.58, 0.60),
    "Croatian":   (0.58, 0.62),
    "Serbian":    (0.58, 0.62),
    # --- Scandinavian ---
    "Swedish":    (0.62, 0.64),
    "Norwegian":  (0.62, 0.64),
    # --- Other ---
    "Malay":      (0.50, 0.54),
    "Dutch":      (0.58, 0.60),
    "Flemish":    (0.58, 0.60),
}
DEFAULT_THRESHOLDS = (0.52, 0.54)


def connect():
    """Connect to local Weaviate instance."""
    return weaviate.connect_to_local()


def get_collection(client):
    """Get the NewsArticles collection."""
    return client.collections.get(COLLECTION_NAME)


def load_keyword_embeddings(filepath):
    """Load keyword embeddings from JSONL file."""
    keywords = []
    embeddings = []
    with open(filepath) as f:
        for line in f:
            record = json.loads(line)
            keywords.append(record["keyword"])
            embeddings.append(record["embedding"])
    return keywords, np.array(embeddings, dtype=np.float32)


def build_source_language_map():
    """Build source_name -> language mapping from metadata CSV."""
    lang_map = {}
    if not SOURCE_METADATA.exists():
        return lang_map
    with open(SOURCE_METADATA) as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("website_url", "")
                   .replace("http://", "").replace("https://", "")
                   .replace("www.", "").rstrip("/"))
            lang = row.get("language", "").strip()
            source_key = url.replace(".", "_").replace("/", "_")
            if source_key and lang:
                lang_map[source_key] = lang
    return lang_map


def source_from_url(url):
    """Derive source key from article URL (e.g., 'abc_es' from 'https://www.abc.es/...')."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain.replace(".", "_").replace("/", "_")
    except Exception:
        return ""


def resolve_source(properties, lang_map):
    """Get the actual source name for an article, falling back to URL-derived source.

    Many articles have source='sitemap_database_spider' from the crawler.
    For these, we derive the source from the URL domain.
    """
    source = properties.get("source", "")
    if source in lang_map:
        return source
    # Try URL-based derivation
    url_source = source_from_url(properties.get("url", ""))
    if url_source in lang_map:
        return url_source
    return source


def normalize_language(raw_lang):
    """Map raw language string to a canonical name matching THRESHOLDS keys."""
    if not raw_lang:
        return None
    parts = [p.strip() for p in raw_lang.split(",")]
    for p in parts:
        if p in THRESHOLDS:
            return p
    return parts[0]


def get_thresholds(language):
    """Get (climate_threshold, health_threshold) for a language."""
    if language and language in THRESHOLDS:
        return THRESHOLDS[language]
    return DEFAULT_THRESHOLDS


def get_sources_for_language(lang, lang_map):
    """Return list of source names for a given language."""
    sources = []
    for source, raw_lang in lang_map.items():
        if normalize_language(raw_lang) == lang:
            sources.append(source)
    return sources


def cosine_similarity_batch(articles_emb, keyword_emb):
    """Cosine similarity: (N, dim) x (K, dim) -> (N, K)."""
    a_norm = articles_emb / np.linalg.norm(articles_emb, axis=-1, keepdims=True)
    b_norm = keyword_emb / np.linalg.norm(keyword_emb, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def score_articles(embeddings, climate_emb, health_emb, climate_kw, health_kw):
    """Compute max similarity scores for a batch of article embeddings."""
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


def classify_single(climate_score, health_score, climate_t, health_t):
    """Classify an article based on scores and thresholds."""
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


def iterate_collection(collection, batch_size=5000):
    """Yield (properties_list, vectors_array) in batches from Weaviate.

    Uses the cursor-based iterator to stream through the entire collection.
    Vectors are accumulated into numpy arrays in batches for efficient
    similarity computation. Each batch is discarded after processing,
    keeping memory usage flat.

    Args:
        collection: Weaviate collection object
        batch_size: Number of articles per yielded batch
    """
    props = ["title", "url", "source", "source_uri", "collection_method",
             "extracted_at", "article_id"]
    meta_buf = []
    vec_buf = []

    for obj in collection.iterator(
        include_vector=True,
        return_properties=props,
    ):
        vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
        meta_buf.append(obj.properties)
        vec_buf.append(vec)

        if len(meta_buf) >= batch_size:
            yield meta_buf, np.array(vec_buf, dtype=np.float32)
            meta_buf = []
            vec_buf = []

    if meta_buf:
        yield meta_buf, np.array(vec_buf, dtype=np.float32)
