#!/usr/bin/env python3
"""Calibrate similarity thresholds using synthetic articles with known labels.

Loads synthetic articles from data/synthetic_articles.json, embeds them with
the same model used for the article corpus, and computes max-similarity and
centroid-similarity against keyword embeddings.

Usage:
    python calibrate_thresholds.py
    python calibrate_thresholds.py --output calibration_report.csv
    python calibrate_thresholds.py --report calibration_findings.md
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import requests

VLLM_URL = "http://localhost:8000/v1/embeddings"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
KEYWORD_EMBEDDINGS_DIR = Path("data/keyword_embeddings")
SYNTHETIC_ARTICLES_PATH = Path("data/synthetic_articles.json")


def load_synthetic_articles(path=SYNTHETIC_ARTICLES_PATH):
    """Load synthetic articles from JSON file."""
    with open(path) as f:
        return json.load(f)


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


def load_multilingual_keyword_embeddings(category):
    """Load keyword embeddings for all available languages for a category.

    Returns dict: lang_code -> (keywords, embeddings_array)
    """
    result = {}
    for path in sorted(KEYWORD_EMBEDDINGS_DIR.glob(f"{category}_*.jsonl")):
        lang_code = path.stem.split("_", 1)[1]  # e.g. "eng", "por", "ara"
        kw, emb = load_keyword_embeddings(path)
        result[lang_code] = (kw, emb)
    return result


# Map article language codes to keyword file language codes
LANG_TO_KW_LANG = {
    "en": "eng",
    "ar": "ara",
    "zh": "cmn",
    "es": "spa",
    "fr": "fra",
    "pt": "por",
}


def embed_texts(texts):
    """Embed texts via the vLLM server."""
    resp = requests.post(VLLM_URL, json={"model": MODEL_NAME, "input": texts})
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return np.array([d["embedding"] for d in data], dtype=np.float32)


def cosine_similarity(a, b):
    """Cosine similarity between vector(s) a and vector(s) b."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def compute_threshold_stats(results, c_key, h_key, threshold, lang_filter=None):
    """Compute precision/recall at a given threshold, optionally filtered by language."""
    subset = results if lang_filter is None else [r for r in results if r["language"] == lang_filter]
    if not subset:
        return None

    tp_c = sum(1 for r in subset if r["label"] in ("climate", "climate+health") and r[c_key] >= threshold)
    fn_c = sum(1 for r in subset if r["label"] in ("climate", "climate+health") and r[c_key] < threshold)
    fp_c = sum(1 for r in subset if r["label"] in ("health", "neither") and r[c_key] >= threshold)
    tp_h = sum(1 for r in subset if r["label"] in ("health", "climate+health") and r[h_key] >= threshold)
    fn_h = sum(1 for r in subset if r["label"] in ("health", "climate+health") and r[h_key] < threshold)
    fp_h = sum(1 for r in subset if r["label"] in ("climate", "neither") and r[h_key] >= threshold)

    prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
    rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
    prec_h = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0
    rec_h = tp_h / (tp_h + fn_h) if (tp_h + fn_h) > 0 else 0

    return {
        "threshold": threshold,
        "climate_precision": prec_c, "climate_recall": rec_c,
        "climate_tp": tp_c, "climate_fp": fp_c, "climate_fn": fn_c,
        "health_precision": prec_h, "health_recall": rec_h,
        "health_tp": tp_h, "health_fp": fp_h, "health_fn": fn_h,
    }


def generate_markdown_report(results, output_path):
    """Write a markdown report of calibration findings."""
    lines = []
    w = lines.append

    has_matched = "climate_matched_max" in results[0]

    w("# Calibration Report: Keyword–Article Similarity Thresholds")
    w("")
    w("## Setup")
    w("")
    w(f"- **Model**: {MODEL_NAME}")
    w(f"- **Synthetic articles**: {len(results)}")
    w(f"- **Languages tested**: {', '.join(sorted(set(r['language'] for r in results)))}")
    w(f"- **Methods**: max-similarity (highest cosine sim to any keyword) and centroid (cosine sim to keyword centroid)")
    if has_matched:
        w("- **Keyword sets**: English-only vs. language-matched (English + native language keywords)")
    w("")

    # --- English vs Language-Matched comparison ---
    if has_matched:
        w("## English-Only vs. Language-Matched Keywords")
        w("")
        w("| Label | Lang | C_eng | C_matched | C_Δ | H_eng | H_matched | H_Δ | Title |")
        w("|-------|------|------:|----------:|----:|------:|----------:|----:|-------|")
        for r in results:
            c_delta = r["climate_matched_max"] - r["climate_max_sim"]
            h_delta = r["health_matched_max"] - r["health_max_sim"]
            title = r["title"][:40]
            w(f"| {r['label']} | {r['language']} "
              f"| {r['climate_max_sim']:.4f} | {r['climate_matched_max']:.4f} | {c_delta:+.3f} "
              f"| {r['health_max_sim']:.4f} | {r['health_matched_max']:.4f} | {h_delta:+.3f} "
              f"| {title} |")
        w("")

        # Improvement summary by language
        w("### Impact of Language-Matched Keywords by Language")
        w("")
        w("| Lang | Climate Δ (relevant) | Climate Δ (irrelevant) | Gap change | Health Δ (relevant) | Health Δ (irrelevant) | Gap change |")
        w("|------|---------------------:|-----------------------:|-----------:|--------------------:|----------------------:|-----------:|")
        for lang in sorted(set(r["language"] for r in results)):
            group = [r for r in results if r["language"] == lang]
            rel_c = [r for r in group if r["label"] in ("climate", "climate+health")]
            irr_c = [r for r in group if r["label"] in ("health", "neither")]
            rel_h = [r for r in group if r["label"] in ("health", "climate+health")]
            irr_h = [r for r in group if r["label"] in ("climate", "neither")]

            rc_delta = np.mean([r["climate_matched_max"] - r["climate_max_sim"] for r in rel_c]) if rel_c else 0
            ic_delta = np.mean([r["climate_matched_max"] - r["climate_max_sim"] for r in irr_c]) if irr_c else 0
            rh_delta = np.mean([r["health_matched_max"] - r["health_max_sim"] for r in rel_h]) if rel_h else 0
            ih_delta = np.mean([r["health_matched_max"] - r["health_max_sim"] for r in irr_h]) if irr_h else 0

            w(f"| {lang} | {rc_delta:+.4f} | {ic_delta:+.4f} | {rc_delta - ic_delta:+.4f} "
              f"| {rh_delta:+.4f} | {ih_delta:+.4f} | {rh_delta - ih_delta:+.4f} |")
        w("")

    # --- Per-article full scores ---
    w("## Per-Article Scores (English-Only)")
    w("")
    w("| Label | Lang | Length | Subtlety | C_max | C_cent | H_max | H_cent | Title |")
    w("|-------|------|--------|----------|------:|-------:|------:|-------:|-------|")
    for r in results:
        title = r["title"][:50]
        w(f"| {r['label']} | {r['language']} | {r['length']} | {r['subtlety']} "
          f"| {r['climate_max_sim']:.4f} | {r['climate_centroid_sim']:.4f} "
          f"| {r['health_max_sim']:.4f} | {r['health_centroid_sim']:.4f} "
          f"| {title} |")
    w("")

    # --- Summary by category ---
    w("## Summary by Category")
    w("")
    score_sets = [("English-Only", "climate_max_sim", "climate_centroid_sim", "health_max_sim", "health_centroid_sim")]
    if has_matched:
        score_sets.append(("Language-Matched", "climate_matched_max", "climate_matched_centroid", "health_matched_max", "health_matched_centroid"))

    for set_name, c_max_k, c_cent_k, h_max_k, h_cent_k in score_sets:
        w(f"### {set_name}")
        w("")
        for label in ["climate", "health", "climate+health", "neither"]:
            group = [r for r in results if r["label"] == label]
            if not group:
                continue
            w(f"#### {label.upper()} (n={len(group)})")
            w("")
            w("| Metric | Min | Mean | Max |")
            w("|--------|----:|-----:|----:|")
            for name, key in [
                ("Climate max-sim", c_max_k),
                ("Climate centroid", c_cent_k),
                ("Health max-sim", h_max_k),
                ("Health centroid", h_cent_k),
            ]:
                vals = [r[key] for r in group]
                w(f"| {name} | {min(vals):.4f} | {np.mean(vals):.4f} | {max(vals):.4f} |")
            w("")

    # --- Threshold analysis: English vs matched ---
    w("## Threshold Analysis")
    w("")
    threshold_configs = [
        ("English-Only Max-Sim", "climate_max_sim", "health_max_sim"),
        ("English-Only Centroid", "climate_centroid_sim", "health_centroid_sim"),
    ]
    if has_matched:
        threshold_configs.extend([
            ("Language-Matched Max-Sim", "climate_matched_max", "health_matched_max"),
            ("Language-Matched Centroid", "climate_matched_centroid", "health_matched_centroid"),
        ])

    for method_name, c_key, h_key in threshold_configs:
        w(f"### {method_name}")
        w("")
        w("| Threshold | C_Prec | C_Rec | C_TP | C_FP | C_FN | H_Prec | H_Rec | H_TP | H_FP | H_FN |")
        w("|----------:|-------:|------:|-----:|-----:|-----:|-------:|------:|-----:|-----:|-----:|")
        for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            s = compute_threshold_stats(results, c_key, h_key, t)
            w(f"| {t:.2f} | {s['climate_precision']:.2f} | {s['climate_recall']:.2f} "
              f"| {s['climate_tp']} | {s['climate_fp']} | {s['climate_fn']} "
              f"| {s['health_precision']:.2f} | {s['health_recall']:.2f} "
              f"| {s['health_tp']} | {s['health_fp']} | {s['health_fn']} |")
        w("")

    # --- Per-language gap analysis ---
    w("## Language Gap Analysis")
    w("")
    if has_matched:
        w("### Language-Matched Max-Sim")
        w("")
        w("| Lang | Climate: rel mean | Climate: irr mean | Gap | Health: rel mean | Health: irr mean | Gap |")
        w("|------|------------------:|------------------:|----:|-----------------:|-----------------:|----:|")
        for lang in sorted(set(r["language"] for r in results)):
            group = [r for r in results if r["language"] == lang]
            rel_c = [r["climate_matched_max"] for r in group if r["label"] in ("climate", "climate+health")]
            irr_c = [r["climate_matched_max"] for r in group if r["label"] in ("health", "neither")]
            rel_h = [r["health_matched_max"] for r in group if r["label"] in ("health", "climate+health")]
            irr_h = [r["health_matched_max"] for r in group if r["label"] in ("climate", "neither")]
            rc = np.mean(rel_c) if rel_c else 0
            ic = np.mean(irr_c) if irr_c else 0
            rh = np.mean(rel_h) if rel_h else 0
            ih = np.mean(irr_h) if irr_h else 0
            w(f"| {lang} | {rc:.4f} | {ic:.4f} | {rc - ic:.4f} | {rh:.4f} | {ih:.4f} | {rh - ih:.4f} |")
        w("")

    w("### English-Only Max-Sim")
    w("")
    w("| Lang | Climate: rel mean | Climate: irr mean | Gap | Health: rel mean | Health: irr mean | Gap |")
    w("|------|------------------:|------------------:|----:|-----------------:|-----------------:|----:|")
    for lang in sorted(set(r["language"] for r in results)):
        group = [r for r in results if r["language"] == lang]
        rel_c = [r["climate_max_sim"] for r in group if r["label"] in ("climate", "climate+health")]
        irr_c = [r["climate_max_sim"] for r in group if r["label"] in ("health", "neither")]
        rel_h = [r["health_max_sim"] for r in group if r["label"] in ("health", "climate+health")]
        irr_h = [r["health_max_sim"] for r in group if r["label"] in ("climate", "neither")]
        rc = np.mean(rel_c) if rel_c else 0
        ic = np.mean(irr_c) if irr_c else 0
        rh = np.mean(rel_h) if rel_h else 0
        ih = np.mean(irr_h) if irr_h else 0
        w(f"| {lang} | {rc:.4f} | {ic:.4f} | {rc - ic:.4f} | {rh:.4f} | {ih:.4f} | {rh - ih:.4f} |")
    w("")

    # --- Hard negatives ---
    w("## Hard Negatives")
    w("")
    hard = [r for r in results if r["label"] == "neither" and r["subtlety"] == "hard"]
    w("Articles designed to be confusing (metaphorical use of climate/health terms):")
    w("")
    if has_matched:
        for r in hard:
            w(f"- **{r['title'][:60]}** ({r['language']}): "
              f"C_eng={r['climate_max_sim']:.4f}, C_matched={r['climate_matched_max']:.4f}, "
              f"H_eng={r['health_max_sim']:.4f}, H_matched={r['health_matched_max']:.4f}")
    else:
        for r in hard:
            w(f"- **{r['title'][:60]}** ({r['language']}): "
              f"C_max={r['climate_max_sim']:.4f}, H_max={r['health_max_sim']:.4f}")
    w("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Calibrate similarity thresholds")
    parser.add_argument("--output", default="calibration_report.csv",
                        help="Output CSV path (default: calibration_report.csv)")
    parser.add_argument("--report", default="calibration_findings.md",
                        help="Output markdown report path (default: calibration_findings.md)")
    parser.add_argument("--articles", default=str(SYNTHETIC_ARTICLES_PATH),
                        help="Path to synthetic articles JSON")
    args = parser.parse_args()

    # Load synthetic articles
    articles = load_synthetic_articles(Path(args.articles))

    # Load keyword embeddings — English baseline
    print("Loading keyword embeddings...")
    climate_kw_eng, climate_emb_eng = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "climate_eng.jsonl"
    )
    health_kw_eng, health_emb_eng = load_keyword_embeddings(
        KEYWORD_EMBEDDINGS_DIR / "health_eng.jsonl"
    )

    # Load multilingual keyword embeddings
    climate_multi = load_multilingual_keyword_embeddings("climate")
    health_multi = load_multilingual_keyword_embeddings("health")
    print(f"  Climate keyword languages: {sorted(climate_multi.keys())}")
    print(f"  Health keyword languages: {sorted(health_multi.keys())}")

    # Compute English centroids
    climate_centroid_eng = climate_emb_eng.mean(axis=0, keepdims=True)
    health_centroid_eng = health_emb_eng.mean(axis=0, keepdims=True)

    # Embed synthetic articles
    print(f"Embedding {len(articles)} synthetic articles...")
    texts = [a["text"] for a in articles]
    article_emb = embed_texts(texts)

    # Compute English-only similarities
    print("Computing similarities...")
    climate_sims_eng = cosine_similarity(article_emb, climate_emb_eng)
    health_sims_eng = cosine_similarity(article_emb, health_emb_eng)
    climate_centroid_sims_eng = cosine_similarity(article_emb, climate_centroid_eng)
    health_centroid_sims_eng = cosine_similarity(article_emb, health_centroid_eng)

    # Compute language-matched similarities (English + native language keywords)
    # For each article, combine English keywords with the article's language keywords
    climate_matched_max = []
    health_matched_max = []
    climate_matched_cent = []
    health_matched_cent = []
    climate_matched_kw = []
    health_matched_kw = []

    for i, article in enumerate(articles):
        art_lang = article["language"]
        kw_lang = LANG_TO_KW_LANG.get(art_lang)

        # Start with English sims
        c_sims = list(climate_sims_eng[i])
        c_kws = list(climate_kw_eng)
        h_sims = list(health_sims_eng[i])
        h_kws = list(health_kw_eng)

        # Add native language keywords if available and not English
        if kw_lang and kw_lang != "eng":
            if kw_lang in climate_multi:
                native_kw, native_emb = climate_multi[kw_lang]
                native_sims = cosine_similarity(article_emb[i:i+1], native_emb)[0]
                c_sims.extend(native_sims)
                c_kws.extend(native_kw)
            if kw_lang in health_multi:
                native_kw, native_emb = health_multi[kw_lang]
                native_sims = cosine_similarity(article_emb[i:i+1], native_emb)[0]
                h_sims.extend(native_sims)
                h_kws.extend(native_kw)

        c_sims = np.array(c_sims)
        h_sims = np.array(h_sims)
        climate_matched_max.append(float(c_sims.max()))
        climate_matched_kw.append(c_kws[int(c_sims.argmax())])
        health_matched_max.append(float(h_sims.max()))
        health_matched_kw.append(h_kws[int(h_sims.argmax())])

        # Centroid of combined keywords
        if kw_lang and kw_lang != "eng" and kw_lang in climate_multi:
            combined_c = np.vstack([climate_emb_eng, climate_multi[kw_lang][1]])
        else:
            combined_c = climate_emb_eng
        if kw_lang and kw_lang != "eng" and kw_lang in health_multi:
            combined_h = np.vstack([health_emb_eng, health_multi[kw_lang][1]])
        else:
            combined_h = health_emb_eng

        c_cent = cosine_similarity(article_emb[i:i+1], combined_c.mean(axis=0, keepdims=True))
        h_cent = cosine_similarity(article_emb[i:i+1], combined_h.mean(axis=0, keepdims=True))
        climate_matched_cent.append(float(c_cent[0, 0]))
        health_matched_cent.append(float(h_cent[0, 0]))

    # Build results with both English-only and language-matched scores
    results = []
    for i, article in enumerate(articles):
        results.append({
            "label": article["label"],
            "language": article["language"],
            "length": article["length"],
            "subtlety": article["subtlety"],
            "title": article["title"],
            # English-only
            "climate_max_sim": round(float(climate_sims_eng[i].max()), 4),
            "climate_max_keyword": climate_kw_eng[int(climate_sims_eng[i].argmax())],
            "health_max_sim": round(float(health_sims_eng[i].max()), 4),
            "health_max_keyword": health_kw_eng[int(health_sims_eng[i].argmax())],
            "climate_centroid_sim": round(float(climate_centroid_sims_eng[i, 0]), 4),
            "health_centroid_sim": round(float(health_centroid_sims_eng[i, 0]), 4),
            # Language-matched (English + native)
            "climate_matched_max": round(climate_matched_max[i], 4),
            "climate_matched_keyword": climate_matched_kw[i],
            "health_matched_max": round(health_matched_max[i], 4),
            "health_matched_keyword": health_matched_kw[i],
            "climate_matched_centroid": round(climate_matched_cent[i], 4),
            "health_matched_centroid": round(health_matched_cent[i], 4),
        })

    # Print comparison table
    print(f"\n{'LABEL':<16} {'LANG':<5} {'LEN':<7} "
          f"{'C_eng':>6} {'C_match':>7} {'Δ':>6} {'H_eng':>6} {'H_match':>7} {'Δ':>6}  TITLE")
    print("-" * 130)
    for r in results:
        c_delta = r["climate_matched_max"] - r["climate_max_sim"]
        h_delta = r["health_matched_max"] - r["health_max_sim"]
        print(
            f"{r['label']:<16} {r['language']:<5} {r['length']:<7} "
            f"{r['climate_max_sim']:>6.4f} {r['climate_matched_max']:>7.4f} {c_delta:>+6.3f} "
            f"{r['health_max_sim']:>6.4f} {r['health_matched_max']:>7.4f} {h_delta:>+6.3f}  "
            f"{r['title'][:45]}"
        )

    # Save CSV
    csv_path = Path(args.output)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    # Generate markdown report
    md_path = Path(args.report)
    generate_markdown_report(results, md_path)
    print(f"Markdown report saved to {md_path}")


if __name__ == "__main__":
    main()
