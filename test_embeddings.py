#!/usr/bin/env python3
"""Quick multilingual embedding quality test for qwen3-embedding:0.6b."""

import gzip
import json
from pathlib import Path

import numpy as np
import ollama

MODEL = "qwen3-embedding:0.6b"
DATA_DIR = Path("data")

# Hand-crafted parallel sentences (same meaning, different languages)
PARALLEL_SENTENCES = {
    "en": "Climate change is a serious threat to global health",
    "fr": "Le changement climatique est une menace sérieuse pour la santé mondiale",
    "de": "Der Klimawandel ist eine ernsthafte Bedrohung für die globale Gesundheit",
    "ar": "تغير المناخ يشكل تهديداً خطيراً للصحة العالمية",
    "ru": "Изменение климата представляет серьёзную угрозу для глобального здоровья",
    "zh": "气候变化对全球健康构成严重威胁",
}

# Sources to sample real articles from (one per language region)
SAMPLE_SOURCES = [
    "world_news_premium/scmp_com",        # English / HK
    "world_news_premium/lemonde_fr",      # French
    "world_news_premium/welt_de",         # German
    "world_news_premium_2/aif_ru",        # Russian
    "world_news_premium_2/alarabiya_net", # Arabic
    "world_news_premium/asahi_com",       # Japanese
]


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_first_article(source_path):
    """Load first article with non-empty content from a source."""
    crawl_dir = DATA_DIR / source_path / "crawls"
    if not crawl_dir.exists():
        return None
    for f in sorted(crawl_dir.glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                art = json.loads(line)
                if art.get("title") and art.get("content"):
                    return art
    return None


def print_sim_matrix(labels, embeddings):
    """Print a cosine similarity matrix."""
    n = len(labels)
    max_label = max(len(l) for l in labels)
    header = " " * (max_label + 2) + "  ".join(f"{l:>6}" for l in labels)
    print(header)
    for i in range(n):
        row = f"{labels[i]:<{max_label}}  " + "  ".join(
            f"{cosine_sim(embeddings[i], embeddings[j]):6.3f}" for j in range(n)
        )
        print(row)


def test_parallel_sentences():
    """Test: same meaning across languages should have high similarity."""
    print("=" * 60)
    print("TEST 1: Parallel sentences (same meaning, different languages)")
    print("=" * 60)
    langs = list(PARALLEL_SENTENCES.keys())
    texts = [PARALLEL_SENTENCES[l] for l in langs]
    result = ollama.embed(model=MODEL, input=texts)
    print()
    print_sim_matrix(langs, result.embeddings)
    print()


def test_real_articles():
    """Test: embed real articles from different language sources."""
    print("=" * 60)
    print("TEST 2: Real articles from different language sources")
    print("=" * 60)
    labels = []
    texts = []
    for src in SAMPLE_SOURCES:
        art = load_first_article(src)
        if art is None:
            print(f"  [skip] {src} - no articles found")
            continue
        name = src.split("/")[-1]
        labels.append(name)
        text = f"{art['title']}\n{art['content']}"
        # Truncate to ~2000 chars to keep things fast
        texts.append(text[:2000])
        print(f"  {name}: {art['title'][:80]}")

    print()
    result = ollama.embed(model=MODEL, input=texts)
    print(f"Embedding dim: {len(result.embeddings[0])}")
    print()
    print_sim_matrix(labels, result.embeddings)
    print()


if __name__ == "__main__":
    test_parallel_sentences()
    test_real_articles()
