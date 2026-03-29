# Calibration Findings: Keyword–Article Similarity Thresholds

## Overview

This document records the full process of calibrating similarity thresholds for classifying news articles as **climate**, **health**, **climate+health**, or **neither** using cosine similarity between article embeddings and keyword embeddings.

**Goal**: Filter ~3.6M multilingual news articles down to a manageable set for downstream classification (expensive), maximizing recall while removing obvious non-matches.

## Pipeline Architecture

```
  Keywords (English)          Articles (multilingual)
        |                              |
  Embed via Qwen3-0.6B         Embed via Qwen3-0.6B
        |                              |
        v                              v
  Keyword embeddings           Article embeddings
  (data/keyword_embeddings/)   (data/embeddings/ + Weaviate)
        |                              |
        +--------> Cosine Sim <--------+
                      |
              Per-language thresholds
                      |
              climate / health / both / neither
```

- **Embedding model**: `Qwen/Qwen3-Embedding-0.6B` (1024-dim), served via local vLLM
- **Keyword source**: English keyword lists in `data/keywords/climate_eng.txt` (38 keywords) and `data/keywords/health_eng.txt` (44 keywords)
- **Similarity method**: Max-similarity — for each article, take the highest cosine similarity to any keyword in the category
- **Classification**: Article passes if max-sim ≥ per-language threshold for that category

## Corpus Summary

| Language | Sources | Articles | % of corpus | Thresholds calibrated |
|----------|--------:|---------:|------------:|:---------------------:|
| English | 79 | ~500k | ~14% | Yes |
| Spanish | 39 | ~200k | ~5% | Yes |
| French | 16 | ~60k | ~2% | Yes |
| Portuguese | 10 | ~50k | ~1% | Yes |
| Arabic | 8 | ~30k | ~1% | Yes |
| Chinese | 7 | ~25k | ~1% | Yes |
| Slovak | 4 | 701k | 19% | No (default) |
| German | 14 | 691k | 19% | No (default) |
| Polish | 5 | 197k | 5% | No (default) |
| Indonesian | 4 | 109k | 3% | No (default) |
| 27 other langs | 96 | ~1M | ~28% | No (default) |
| **Total** | **282** | **3,640,017** | | |

**Key finding**: Only 38% of articles are in calibrated languages. The remaining 62% — dominated by Slovak (19%) and German (19%) — use default thresholds and need calibration.

## Methodology

### Stage 1: Synthetic Article Calibration

**Script**: `calibrate_thresholds.py`
**Data**: `data/synthetic_articles.json` (58 articles)

We generated synthetic news articles across a matrix of:
- **Categories**: climate, health, climate+health, neither
- **Languages**: English, Arabic, Chinese, Spanish, French, Portuguese
- **Lengths**: short (~100 words), medium (~300 words), long (~800 words)
- **Subtlety**: direct (uses keyword terms explicitly) vs. indirect (discusses topic without exact keywords)
- **Hard negatives**: Articles using climate/health vocabulary metaphorically ("the race is heating up", "epidemic of burnout", "financial markets heat up")

Each synthetic article was embedded with the same Qwen3 model and scored against the English keyword sets using both max-similarity and centroid methods.

**To reproduce**:
```bash
python calibrate_thresholds.py
# Reads: data/synthetic_articles.json, data/keyword_embeddings/
# Outputs: calibration_report.csv, calibration_findings.md
```

### Stage 2: Multilingual Keyword Exploration

We also created translated keyword lists in Arabic, Chinese, Spanish, French, and Portuguese (`data/keywords/{climate,health}_{ara,cmn,spa,fra,por}.txt`) and tested a "language-matched" approach where each article is scored against English + native-language keywords.

**Finding**: Language-matched keywords boost scores for *all* articles in that language (relevant and irrelevant equally), so the discriminative gap does not improve — and for Portuguese and Spanish, it actually narrows to near-zero. This is because native-language keywords have closer vocabulary overlap with all text in that language, not just topically relevant text.

**Decision**: Use **English keywords only** with **per-language thresholds**. The cross-lingual embedding model provides enough signal, and per-language thresholds account for the varying score ranges.

The translated keyword files are retained in `data/keywords/` for potential future use but are not used in the classification pipeline.

### Stage 3: Real-Data Sampling

**Script**: `classify_articles.py sample`

We used the `sample` mode to inspect real articles near threshold boundaries. This revealed critical issues invisible in synthetic data:

**Key findings from real-data sampling**:

1. **"west nile" keyword** matched Nile-region geopolitics (Egypt, Iraq, Sudan) in non-English articles. In Portuguese alone, 5,372 of 8,028 initial health matches (67%) were false positives from this keyword. **Fix**: Changed to "west nile virus".

2. **"hay-fever" (hyphenated)** matched random tech/opinion columns at 0.60+ in Portuguese, while "hay fever" (space) caused zero false positives. The embedding model treats the hyphenated form differently. **Fix**: Dropped "hay-fever", kept "hay fever".

3. **"temperature rises"** matched inflation/interest rate articles in Portuguese and Spanish (e.g., "Índice de preços ao consumidor sobe").

4. **"extreme heat"** matched car and motorcycle reviews (e.g., "Bentley: conversível mais rápido do mundo").

5. **"carbon emission"** matched automotive articles (car reviews discussing fuel efficiency).

6. **Synthetic thresholds were too optimistic**: The synthetic Portuguese climate threshold of 0.47 let in thousands of sports articles from abola.pt. Real data showed the noise floor for Portuguese extends to ~0.53, with actual climate content starting at ~0.55.

**To reproduce sampling**:
```bash
# Inspect Portuguese climate articles near the threshold boundary
python classify_articles.py sample --lang Portuguese --category climate --n 20 --window 0.05

# Inspect Spanish health articles
python classify_articles.py sample --lang Spanish --category health --n 20
```

## Score Characteristics by Language

### English
- Climate relevant: 0.60–0.67 (direct), 0.60 (indirect)
- Climate irrelevant: 0.31–0.64 (hard negatives like "Heat Records at Australian Open" score 0.64)
- Health relevant: 0.50–0.67
- Health irrelevant: 0.28–0.56 ("Farmers Brace for Tough Season" scores 0.56)
- **Gap**: Good for climate (0.15), moderate for health (0.12)

### Arabic
- Climate relevant: 0.53–0.58
- Climate irrelevant: 0.28–0.42
- **Gap**: Excellent (0.23) — different script means irrelevant articles score very low against English keywords

### Chinese
- Similar to Arabic — good separation due to script distance
- Climate relevant: 0.45–0.56
- **Gap**: Good (0.22)

### Spanish
- Climate relevant: 0.54–0.62
- Climate irrelevant: 0.46–0.53 (noise from sports, economics)
- **Gap**: Narrow (0.09) — Romance vocabulary overlap with English keywords
- Real-data sampling showed noise extends to ~0.55

### French
- Climate relevant: 0.50–0.57
- Climate irrelevant: 0.39–0.46
- **Gap**: Moderate (0.15)

### Portuguese
- Climate relevant: 0.49–0.53 (synthetic), 0.55–0.67 (real data)
- Climate irrelevant: 0.35–0.46 (easy) up to 0.53 (noise from economics/sports)
- **Gap**: Near-zero in synthetic (0.03), but real data has a natural gap at ~0.55
- Health severely affected by "west nile" false positives until keyword fix

## Stage 4: Broad Language Score Distribution Analysis

After calibrating the initial 6 languages, we discovered that 62% of articles (2.25M) were in uncalibrated languages using default thresholds. We sampled 5,000 articles per language from 18 additional languages to measure score distributions and set appropriate thresholds.

**Method**: For each language, compute max-sim against English climate and health keywords for a sample of 5,000 articles. Measure what percentage of articles pass at various thresholds.

**Key finding — languages cluster into four groups by noise level**:

### Group 1: Non-Latin scripts (Bulgarian, Thai, Korean, Japanese)
- Scores rarely exceed 0.50. Like Arabic/Chinese, different scripts suppress cross-lingual similarity.
- Thresholds can be set low (~0.35–0.45) since almost no noise reaches those levels.
- Risk: genuine climate/health articles may also score low; recall depends on how well the model handles these scripts.

### Group 2: Latin-script, moderate noise (Polish, Indonesian, Romanian, Czech, Hungarian, Turkish, Greek, Danish)
- At T=0.50, health passes 5–16% of articles (mostly noise). At T=0.56, <1%.
- Climate is cleaner: <2% pass at T=0.50.
- Thresholds around 0.50–0.56 work well.

### Group 3: Latin-script, high noise (Italian, Croatian, Serbian, Finnish)
- Health at T=0.50 passes 38–63% — similar to the Portuguese/Spanish problem.
- Need thresholds of 0.58–0.60 for health.
- "casualties" keyword is a major noise driver in Croatian/Serbian (matches crime/accident reporting).

### Group 4: Scandinavian (Swedish, Norwegian)
- **87–89% of all articles** pass at T=0.50 for climate. **90–97%** for health.
- Extreme English vocabulary overlap (Germanic family).
- Even at T=0.60, 3–9% pass. Need T=0.60–0.62.
- At these thresholds, only articles with strong topical signal get through.

## Final Thresholds

Per-language thresholds for English max-sim. Defined in `classify_articles.py` in the `THRESHOLDS` dict.

### Calibrated via synthetic + real-data sampling

| Language | Climate T | Health T | Method | Articles |
|----------|----------:|---------:|--------|-------:|
| English | 0.59 | 0.49 | Synthetic + real-data sampling | 207k |
| Arabic | 0.52 | 0.49 | Synthetic + real-data sampling | ~30k |
| Chinese | 0.44 | 0.55 | Synthetic + real-data sampling | 3k |
| Spanish | 0.56 | 0.54 | Synthetic + real-data sampling | 1,027k |
| French | 0.50 | 0.49 | Synthetic + real-data sampling | 77k |
| Portuguese | 0.55 | 0.59 | Synthetic + real-data sampling | 71k |
| German | 0.58 | 0.57 | Real-data sampling | 691k |
| Slovak | 0.58 | 0.60 | Real-data sampling | 701k |
| Polish | 0.54 | 0.57 | Distribution + real-data sampling | 197k |
| Indonesian | 0.53 | 0.56 | Distribution + real-data sampling | 109k |
| Italian | 0.58 | 0.60 | Distribution + real-data sampling | 67k |
| Finnish | 0.56 | 0.60 | Distribution + real-data sampling | 64k |
| Romanian | 0.52 | 0.57 | Distribution + real-data sampling | 76k |
| Czech | 0.53 | 0.58 | Distribution + real-data sampling | 56k |
| Croatian | 0.58 | 0.62 | Distribution + real-data sampling | 39k |
| Swedish | 0.62 | 0.64 | Distribution + real-data sampling | 39k |
| Serbian | 0.58 | 0.62 | Distribution + real-data sampling | 32k |
| Norwegian | 0.62 | 0.64 | Distribution + real-data sampling | 27k |
| Danish | 0.56 | 0.60 | Distribution + real-data sampling | 18k |

### Set from score distribution analysis, then refined with real-data sampling

For the remaining 18+ languages, thresholds were set in two passes:

**Pass 1 — Score distribution analysis**: For each language, we sampled ~5,000 articles and computed their max-sim against English keywords. We measured what *percentage of all articles* exceed various thresholds (e.g., "at T=0.56, 0.5% of Polish articles pass on health"). Since the vast majority of articles in any newspaper are not about climate or health, a high pass rate implies the threshold is too low. This approach identifies the **noise floor** but does not verify that real climate/health articles score above the chosen threshold.

**Pass 2 — Real-data sampling**: For the 8 largest languages by article count (Polish, Italian, Indonesian, Finnish, Croatian, Norwegian, Swedish, plus re-checking Finnish), we ran `classify_articles.py sample` to inspect actual article titles at the threshold boundary. This revealed that Pass 1 thresholds were systematically too low — real-data inspection showed noise extending well above the distribution-based estimates. All thresholds were raised accordingly.

Detailed calibration of *every* language is expensive because it requires reading article titles (often in unfamiliar languages) and judging topical relevance. For the 28 languages covered here, we invested inspection time proportional to article volume: full sampling for languages with >30k articles, distribution-only for smaller languages. The remaining uncalibrated languages (Bengali, Urdu, Hebrew, Swahili, Malayalam, Ganda) have very few articles (<5k total) and use the default thresholds.

**Key findings from real-data sampling (Pass 2)**:

- **Polish** (197k articles): Climate at T=0.50 was entirely noise (Eurovision, tennis, motorcycle racing). Health at T=0.55 was recipes, gardening tips, and tariff news. Both raised.
- **Italian** (67k): Climate at T=0.56 was a genuine mix — some real articles (green urbanism, heat waves, electric cars) alongside car reviews. Health at T=0.58 was dominated by "west nile virus" matching geopolitics. Climate kept, health raised.
- **Indonesian** (109k): Both climate T=0.50 and health T=0.54 were pure noise — K-pop, recipes, religion. Both raised.
- **Finnish** (64k): Climate at T=0.54 had some real articles (fossil fuel emissions, August heat records) mixed with noise. Health at T=0.58 was "west nile virus" + Ukraine war coverage. Climate raised slightly, health raised more.
- **Croatian** (39k): Climate T=0.56 was all fashion/lifestyle noise. Health T=0.60 was fashion + "casualties" matching crime reporting. Both raised.
- **Norwegian** (27k): Climate at T=0.60 was a **reasonable mix** — California fires, European drought, Central American storms alongside some noise. This was the only language where the distribution-based threshold held up. Health T=0.62 still noisy (dead sheep, vandalism, cat deaths). Climate kept, health raised.
- **Swedish** (39k): Climate T=0.60 and health T=0.62 were both entirely noise — single-word topic pages ("Nick Cave", "Reumatism", "Alanya", "Franco Colapinto"). Swedish articles from Aftonbladet appear to be tag/topic index pages rather than full articles, which inflates scores. Both raised to 0.62/0.64.
- **Romanian** (76k): Climate T=0.52 was a genuine mix with real content present at the boundary (nuclear energy, water resources, hydrogen legislation, forests). Threshold kept. Health T=0.54 was "west nile virus" + Easter recipes. Raised to 0.57.
- **Czech** (56k): Climate T=0.52 was mostly weather forecasts and fashion with some real content (volcanic eruption, India heat). Raised slightly to 0.53. Health T=0.56 was olive oil reviews, animal cruelty, "west nile virus". Raised to 0.58.
- **Serbian** (32k): Climate T=0.58 was almost entirely daily weather forecasts ("Vremenska prognoza"), not climate change. Threshold kept (weather forecasts will pass but are borderline relevant). Health T=0.60 was car fires, North Korea missiles, traffic accidents. Raised to 0.62.
- **Danish** (18k): Climate T=0.56 was **excellent quality** — real climate policy articles (Klimarådet, Greta Thunberg, greenhouse gas reductions, waste sorting). Best boundary quality of any non-English language. Threshold kept. Health T=0.58 was "undernourishment" matching banking/finance. Raised to 0.60.

All sample files are saved in `results/samples/` as CSVs for reproducibility (22 files total, one per language+category). To re-run any sample:
```bash
python classify_articles.py sample --lang <Language> --category <climate|health> --n 20 --window 0.04
```

**Persistent noise drivers across languages**:

| Keyword | Problem | Languages affected |
|---------|---------|-------------------|
| "west nile virus" | Matches Nile-region geopolitics, Middle East news | Italian, Finnish, Norwegian, Polish, Croatian, Indonesian |
| "heatwave" | Matches any high-energy context (sports, fashion, music) | Swedish, Norwegian, Croatian, Finnish, Polish |
| "casualties" | Matches crime/accident reporting | Croatian, Serbian, Polish, Norwegian |
| "undernourishment" | Matches food prices, economic indicators, cooking | Polish, Finnish, Norwegian |
| "temperature rises" | Matches inflation, interest rates, weather forecasts | Polish, multiple others |
| "carbon emission" | Matches car reviews, automotive industry | Italian, Polish |

These keywords are retained because they do capture genuine content in English and some other languages, but they systematically inflate false positive rates for non-English sources.

**Final thresholds (all languages)**:

| Language | Climate T | Health T | Calibration method | Sample files |
|----------|----------:|---------:|-------------------:|:-------------|
| Bulgarian | 0.35 | 0.42 | Distribution only (non-Latin script) | — |
| Thai | 0.42 | 0.45 | Distribution only (non-Latin script) | — |
| Korean | 0.40 | 0.43 | Distribution only (non-Latin script) | — |
| Japanese | 0.45 | 0.46 | Distribution only (non-Latin script) | — |
| Polish | 0.54 | 0.57 | Distribution + real-data sampling | `sample_polish_*.csv` |
| Indonesian | 0.53 | 0.56 | Distribution + real-data sampling | `sample_indonesian_*.csv` |
| Romanian | 0.52 | 0.57 | Distribution + real-data sampling | `sample_romanian_*.csv` |
| Czech | 0.53 | 0.58 | Distribution + real-data sampling | `sample_czech_*.csv` |
| Hungarian | 0.52 | 0.54 | Distribution only | — |
| Turkish | 0.49 | 0.54 | Distribution only | — |
| Greek | 0.49 | 0.52 | Distribution only | — |
| Danish | 0.56 | 0.60 | Distribution + real-data sampling | `sample_danish_*.csv` |
| Finnish | 0.56 | 0.60 | Distribution + real-data sampling | `sample_finnish_*.csv` |
| Italian | 0.58 | 0.60 | Distribution + real-data sampling | `sample_italian_*.csv` |
| Croatian | 0.58 | 0.62 | Distribution + real-data sampling | `sample_croatian_*.csv` |
| Serbian | 0.58 | 0.62 | Distribution + real-data sampling | `sample_serbian_*.csv` |
| Swedish | 0.62 | 0.64 | Distribution + real-data sampling | `sample_swedish_*.csv` |
| Norwegian | 0.62 | 0.64 | Distribution + real-data sampling | `sample_norwegian_*.csv` |
| Malay | 0.50 | 0.54 | Distribution only | — |
| Dutch | 0.58 | 0.60 | Distribution only | — |
| Flemish | 0.58 | 0.60 | Distribution only | — |
| **Default** | **0.52** | **0.54** | — | — |

## Keyword Changes

Original keywords modified based on cross-lingual false positive analysis:

| Change | Reason |
|--------|--------|
| "west nile" → "west nile virus" | Matched Nile-region geopolitical articles (Egypt, Iraq, Sudan) in non-English sources |
| "hay-fever" dropped (kept "hay fever") | Hyphenated form produced spurious high-similarity matches (0.60+) with unrelated content; space form is clean |

## Approach Decisions

### Why English keywords only (not multilingual)?

We tested using English + native-language keywords per article. Results:

| Language | Gap (English-only) | Gap (Language-matched) | Change |
|----------|-------------------:|----------------------:|-------:|
| English | 0.151 | 0.151 | 0 |
| Arabic | 0.230 | 0.199 | -0.031 |
| Chinese | 0.218 | 0.173 | -0.045 |
| Spanish | 0.089 | 0.040 | -0.049 |
| French | 0.146 | 0.130 | -0.016 |
| Portuguese | 0.025 | 0.006 | -0.019 |

Native-language keywords boost irrelevant article scores equally or more than relevant ones, compressing the gap. English keywords with per-language thresholds provide better discrimination.

### Why max-similarity over centroid?

Both were tested. Centroid provides slightly better precision at the same recall, but max-similarity is more interpretable (you can see *which* keyword matched) and easier to debug. For a recall-optimized first pass feeding into an expensive classifier, max-similarity with per-keyword inspection was preferred.

### Why per-language thresholds over a global threshold?

The score range varies dramatically by language due to cross-lingual embedding distance. Arabic climate articles score 0.52–0.58 while English ones score 0.60–0.67. A global threshold either misses Arabic content or drowns in English noise.

## Known Limitations

1. **Non-Latin script languages** (Bulgarian, Thai, Korean) have very low scores overall. Thresholds are set near p99 of the score distribution, but recall for these languages is uncertain — real climate/health articles may also score below threshold due to weak cross-lingual signal. These languages should be spot-checked with `classify_articles.py sample` once a few real matches are found.

2. **Persistent cross-lingual keyword noise**: Several English keywords systematically produce false positives in non-English languages (see noise driver table above). "west nile virus" is the worst offender, matching any Nile-region or Middle East content. "heatwave" matches high-energy contexts (sports, fashion). "casualties" matches crime reporting. These keywords are retained because they capture genuine content in English and several other languages, but they inflate false positive rates broadly.

3. **Health thresholds are generally less precise than climate**. Health vocabulary ("disease", "deaths", "quality of life") overlaps broadly with general news. The downstream classifier must handle more noise for health.

4. **Synthetic calibration underestimated real-data noise**. The 58 synthetic articles were useful for initial bounds and for the English-vs-multilingual keyword decision, but all thresholds required upward adjustment after real-data sampling. Any new language calibration should go straight to `classify_articles.py sample`.

5. **Swedish source data quality**: Aftonbladet articles appear to be tag/topic index pages (single-word titles like "Nick Cave", "Alanya") rather than full articles. These inflate similarity scores because short generic text matches many keywords. This may affect classification quality for Swedish regardless of threshold.

6. **Distribution-only languages are less reliable**: Romanian, Czech, Hungarian, Turkish, Greek, Danish, Malay, and the non-Latin-script languages have thresholds set only from score distributions without article-level inspection. These should be treated as rough estimates.

## File Inventory

| File | Purpose |
|------|---------|
| `data/keywords/climate_eng.txt` | English climate keywords (38) — primary keyword set |
| `data/keywords/health_eng.txt` | English health keywords (44, after fixes) — primary keyword set |
| `data/keywords/{climate,health}_{ara,cmn,spa,fra,por}.txt` | Translated keywords (not used in pipeline, retained for reference) |
| `data/keyword_embeddings/*.jsonl` | Embedded keywords (regenerate with `embed_keywords.py`) |
| `data/synthetic_articles.json` | 58 synthetic articles with ground-truth labels |
| `calibration_report.csv` | Per-article similarity scores from synthetic calibration |
| `results/samples/*.csv` | Real-data threshold samples (14 files, one per language+category) |
| `calibrate_thresholds.py` | Synthetic calibration script |
| `classify_articles.py` | Main classification script (classify + sample modes) |
| `embed_keywords.py` | Embeds keyword text files → JSONL with vectors |
| `embed_articles.py` | Embeds articles from premium news folders → JSONL + Weaviate |

## Reproduction Steps

```bash
# 1. Ensure vLLM server is running with Qwen3-Embedding-0.6B on port 8000

# 2. Embed keywords (if keyword files changed)
python embed_keywords.py

# 3. Run synthetic calibration (optional — for understanding score ranges)
python calibrate_thresholds.py

# 4. Score distribution analysis for uncalibrated languages (optional)
# See Stage 4 methodology above — done via inline Python, not a standalone script

# 5. Sample real articles near thresholds (per language, per category)
# Samples are automatically saved to results/samples/
python classify_articles.py sample --lang English --category climate --n 20
python classify_articles.py sample --lang Portuguese --category health --n 20 --window 0.05
python classify_articles.py sample --lang Polish --category climate --n 20 --window 0.04
# ... repeat for each language/category pair
# Inspect output, adjust thresholds in classify_articles.py THRESHOLDS dict

# 6. Full classification run
python classify_articles.py classify --output results/classifications.csv

# 7. Inspect results
# Output CSV contains: source, language, article_id, url, title, scores, category
# Only matched articles (climate/health/both) are included by default
# Add --include-neither to output all articles
```
