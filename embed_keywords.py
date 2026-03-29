#!/usr/bin/env python3
"""Embed keyword lists via vLLM and save as JSONL for querying Weaviate."""

import json
from pathlib import Path
import requests

VLLM_URL = "http://localhost:8000/v1/embeddings"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
KEYWORDS_DIR = Path("data/keywords")
OUTPUT_DIR = Path("data/keyword_embeddings")


def load_keywords(filepath):
    keywords = []
    with open(filepath) as f:
        for line in f:
            kw = line.strip().strip('"')
            if kw:
                keywords.append(kw)
    return keywords


def embed(texts):
    resp = requests.post(VLLM_URL, json={"model": MODEL_NAME, "input": texts})
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for kw_file in sorted(KEYWORDS_DIR.glob("*.txt")):
        keywords = load_keywords(kw_file)
        print(f"{kw_file.name}: {len(keywords)} keywords")

        embeddings = embed(keywords)

        output_path = OUTPUT_DIR / f"{kw_file.stem}.jsonl"
        with open(output_path, "w") as f:
            for kw, emb in zip(keywords, embeddings):
                record = {"keyword": kw, "embedding": emb}
                f.write(json.dumps(record) + "\n")

        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
