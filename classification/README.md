# Classification

Multi-label article classification (CC, H, HECC, HEEW) via model distillation. See the appendix, "Methods to classify climate change and health related content".

> **Why HEEW?** HEEW (Health Effects of Extreme Weather) captures articles about health consequences of extreme weather events *without* a climate-change framing. It is **not** estimated as a prevalence — the corpus has no well-defined denominator of extreme-weather articles. It exists only so the classifier can route articles about extreme-weather health impacts away from HECC, improving HECC's precision.

## Pipeline

1. **Stratified annotation sample** — `build_annotation_sample.py`
   - ~5K articles, stratified across **3 BM25-score terciles × 5 years × 11 language groups = 165 strata**. The top 10 languages (eng, spa, zho, fra, deu, tur, tha, hrv, ara, jpn) form their own groups; everything else is pooled as "other". Allocation is floored-proportional (each non-empty stratum is guaranteed a minimum, the remainder allocated in proportion to stratum size), with a per-country cap within each stratum so dominant strata (e.g., high-BM25 English 2024) don't crowd out the smaller ones.
   - BM25 scores (`bm25_climate`, `bm25_health`, `bm25_avg`) are precomputed on the full corpus by `build_bm25_scores.py`: standard BM25 (k1=1.5, b=0.75) with per-language IDF and document-length normalisation, word-boundary matching for non-CJK languages and character-level substring matching for CJK (zho, jpn, kor, tha). Languages absent from the translated keyword file fall back to the nearest available (cat→spa, bos→hrv, msa→ind, empty→eng). Results are written back into `data/climate.db` as three columns on the `articles` table.

2. **Frontier annotation** — `classify_claude.py`
   - Uses Claude Sonnet 4.6 with the system prompt in `prompts/base_prompt.py` (Table 1 of the appendix).
   - Produces structured reasoning + YAML label assignments.

3. **Assemble the SFT dataset** — `build_sft_dataset.py`
   - Combines the raw articles (from step 1) with Claude's annotations (from step 2) into TRL's `{"messages": [system, user, assistant]}` jsonl format. The assistant turn is `<think>{reasoning}</think>` followed by a ```yaml``` block with the four labels.
   - `build_splits.py` then stratifies the combined file into train/eval/test.

4. **Fine-tune Qwen 3.5 9B** — `train_lancet_a100v2.py`
   - LoRA via Unsloth + TRL SFT. Trains on the full conversation sequence (system prompt, article, reasoning, labels) and publishes to the Hugging Face Hub.

5. **Production inference** — `classify_offline.py`
   - Serves the distilled model via vLLM (local or Modal) and classifies the full 2.2M-article corpus.

6. **Evaluation**
   - `eval_distillation.py` — distilled model vs. held-out frontier (Claude) annotations → **91.4% exact match**; per-label F1 CC 0.918, H 0.858, HECC 0.877, HEEW 0.872.
   - `eval_human.py` — distilled model vs. 150-article human gold standard (in `validation/`) → **88.5% exact match**; per-label F1 CC 0.892, H 0.889, HECC 0.778, HEEW 0.875.
