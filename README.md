# Lancet Global News Study

Analysis of global news coverage across 300+ sources in 40+ languages, using semantic embeddings to study patterns in multilingual news content.

## Data

Article data is stored in S3 and downloaded locally to `data/`. The dataset is organized into five source groups:

- `world_news/` and `world_news_1/` — general news sources
- `world_news_premium/`, `world_news_premium_2/`, `world_news_premium_3/` — premium news sources (305 outlets total)

Each source contains gzipped JSONL crawl files under `{source}/crawls/crawl_DDMMYYYY.jsonl.gz`. Each article record has the following fields:

| Field | Description |
|---|---|
| `url` | Article URL |
| `title` | Article headline |
| `content` | Full article text |
| `author` | Author name (often null) |
| `published_date` | Publication date |
| `source` | Source identifier |
| `extracted_at` | Extraction timestamp |
| `metadata` | Nested object: top_image, keywords, summary, etc. |
| `spider_name` | Crawler identifier |
| `scraped_at` | Crawl timestamp (ISO format) |

## Architecture

```
  Raw articles (JSONL.gz)
        |
  embed_articles.py (Qwen3-Embedding-0.6B, 1024-dim)
        |
        v
  Weaviate (HNSW index, cosine distance)
        |
        +---> classify_articles.py (keyword similarity + per-language thresholds)
        |           |
        |           v
        |     classifications.csv (climate / health / both / neither)
        |
        +---> build_labeled_dataset.py (stratified sampling + full text)
                    |
                    v
              to_label_{lang}.csv (for human/LLM annotation)
                    |
                    v
              Fine-tuned classifier (climate, health, climate+health impact)
```

## Weaviate Vector Database

All article embeddings are stored in a local Weaviate instance for efficient similarity search and batch iteration.

### Schema: `NewsArticles`

| Property | Type | Description |
|---|---|---|
| `title` | text | Article headline |
| `url` | text | Article URL |
| `source` | text | Source identifier (maps to language via `top10_per_country.csv`) |
| `extracted_at` | date | Extraction timestamp |
| `published_date` | date | Publication date |
| `article_id` | int | Line number in source crawl files (1-indexed) |
| *(vector)* | float[1024] | Qwen3-Embedding-0.6B embedding |

### Setup

```bash
# Start Weaviate
docker compose up -d

# Import embeddings (resumes automatically)
.venv/bin/python import_weaviate.py

# Import a single source
.venv/bin/python import_weaviate.py --source 20min_ch
```

## Embedding Pipeline

The embedding pipeline processes all premium news sources using [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), a multilingual embedding model, via `sentence-transformers`. It produces 1024-dimensional embeddings of each article's combined title and content.

### How it works

1. Loads all articles from a source's JSONL files
2. Tokenizes each article and assigns it to a batch tier based on length:
   - Short articles (<=512 tokens): batch size 256
   - Medium articles (<=2048 tokens): batch size 64
   - Long articles (<=8192 tokens): batch size 8
3. Embeds each batch on GPU
4. Writes results to `data/embeddings/{source}.jsonl.gz`

### Running the pipeline

```bash
# Run embedding pipeline (all sources)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python embed_articles.py --workers 1

# Run a single source
.venv/bin/python embed_articles.py --source scmp_com
```

The pipeline supports resuming — completed sources are skipped on restart.

## Classification Pipeline

Articles are classified as **climate**, **health**, **climate+health**, or **neither** using cosine similarity between article embeddings and English keyword embeddings.

### How it works

1. **Keyword embeddings**: 38 climate keywords and 44 health keywords (English) are embedded with the same Qwen3 model (`embed_keywords.py`).

2. **Max-similarity scoring**: For each article, compute cosine similarity against all keyword embeddings. The article's score for a category is the maximum similarity to any keyword in that category.

3. **Per-language thresholds**: Each language has calibrated similarity thresholds for climate and health. An article passes if its max-sim exceeds the threshold for that language. Thresholds account for cross-lingual embedding distance — Arabic articles score lower against English keywords than English articles, so Arabic thresholds are lower.

4. **Weaviate iteration**: Articles are streamed from Weaviate in batches of 5000 using cursor-based pagination. Vectors are fetched, scored, and discarded — only classification results are kept in memory.

### Running classification

```bash
# Full classification run
.venv/bin/python classify_articles.py classify

# Single source
.venv/bin/python classify_articles.py classify --source reuters_com

# Include unmatched articles
.venv/bin/python classify_articles.py classify --include-neither

# Inspect threshold boundaries
.venv/bin/python classify_articles.py sample --lang Portuguese --category health --n 20
```

See `calibration_findings.md` for the full threshold calibration methodology and per-language details.

## Building the Labeled Dataset

`build_labeled_dataset.py` produces labeled datasets for training and evaluating a fine-tuned classifier. It extracts articles with full text, stratified by similarity score, ready for human or LLM annotation.

### Sampling Procedure

The sampling procedure is designed to produce a training set that covers the full decision space of the classifier, not just the easy cases. A naive random sample would be dominated by "neither" articles (>90% of the corpus), with very few climate+health examples. Instead, we use **score-stratified random sampling**: articles are sorted into buckets by their keyword similarity scores, then a random sample is drawn from each bucket.

#### Step 1: Score all articles for the target language

Every article in Weaviate is streamed in batches of 5,000. For each article, we compute:
- **climate_max_sim**: the maximum cosine similarity between the article's embedding and any of the 38 English climate keyword embeddings
- **health_max_sim**: the same against the 44 English health keyword embeddings

Only articles belonging to the target language (determined by source-to-language mapping) are retained. Vectors are discarded after scoring — only the metadata and two similarity scores are kept.

#### Step 2: Compute per-language percentile cutpoints

From the scored articles, we compute percentiles of the climate and health score distributions for this language:
- **p50** (median), **p85**, **p90**, **p95**

These cutpoints define the bucket boundaries. Because they are computed per-language, they adapt to each language's score distribution — e.g., Arabic articles score lower against English keywords than Spanish articles, so Arabic's p95 will be a lower absolute value.

#### Step 3: Assign articles to buckets

Each article is eligible for one or more buckets based on where its scores fall relative to the cutpoints:

| Bucket | Score criteria | What it captures |
|---|---|---|
| **high_climate** | climate_max_sim > p95 of climate scores | Top 5% by climate score. These are the articles most similar to climate keywords — likely genuine climate content, plus some false positives from keywords like "temperature rises" or "extreme heat" |
| **high_health** | health_max_sim > p95 of health scores | Top 5% by health score. Same logic for health — likely genuine plus noise from "casualties", "west nile virus" |
| **high_both** | climate_max_sim > p90 AND health_max_sim > p90 | Top 10% in *both* dimensions simultaneously. These are the rarest candidates — potential climate+health impact articles. Using p90 (not p95) because the intersection of two top-5% sets would yield very few articles |
| **low_both** | climate_max_sim < p50 AND health_max_sim < p50 | Bottom half in both dimensions. Clearly irrelevant to both climate and health — sports, entertainment, politics, etc. |
| **borderline** | climate_max_sim within +/-0.02 of p85, OR health_max_sim within +/-0.02 of p85 | Articles in a narrow band around the 85th percentile. These sit near the decision boundary where the keyword-similarity classifier is least confident |

#### Step 4: Random sample within each bucket

For a request of N articles per language (default 300), the quota is split evenly: ~N/5 per bucket (60 each for N=300).

Within each bucket, articles are selected by **uniform random sampling** (seeded for reproducibility, default seed=42). The procedure:

1. Randomly shuffle the eligible articles in the bucket (drawing 2x the quota to handle deduplication)
2. Pick articles one by one, skipping any already selected in a previous bucket (an article can be eligible for multiple buckets, e.g., high_climate and borderline)
3. Stop when the bucket quota is filled

Buckets are filled in order: high_climate, high_health, high_both, low_both, borderline. This means high_climate gets first pick of any articles that overlap multiple buckets.

If any bucket has fewer eligible articles than its quota (e.g., high_both may have very few articles), the shortfall is added to a final **random** bucket that samples uniformly from all remaining unselected articles.

#### Step 5: Retrieve full text

For each selected article, the full article text is retrieved from the original crawl files on disk (JSONL.gz) using the `source` and `article_id` fields. Content is truncated to 3,000 characters if longer.

#### Why this approach

1. **Why stratified, not random?** The classification task has severe class imbalance. Climate+health articles are <1% of the corpus. A random sample of 300 articles per language would contain 0-3 climate+health examples — far too few to train or evaluate on. Stratified sampling guarantees representation of all categories.

2. **Why score-based buckets rather than category-based?** We don't have ground-truth labels yet — that's what the labeled dataset is for. Similarity scores are our best proxy for category membership, so we stratify on scores. The percentile thresholds are computed per-language from the actual score distribution, so they adapt to each language's characteristics.

3. **Why random within buckets?** Within each score range, we want an unbiased sample. Sorting by score and taking the top/bottom would over-represent extreme cases. Random sampling gives a representative cross-section of each bucket.

4. **Why include borderline cases?** The keyword-similarity approach has known weaknesses: "temperature rises" matches inflation articles, "casualties" matches crime reporting, "heatwave" matches sports contexts. Borderline articles are where these false positives concentrate. Including them in the training set teaches the classifier to handle the cases that keyword matching cannot.

5. **Why include low-scoring negatives?** The classifier needs negative examples that are clearly irrelevant, not just hard negatives. The low_both bucket provides these. Without them, the model would only see articles near the decision boundary and might not learn the full score range.

6. **Why full text?** Title-only classification is unreliable for multilingual content. Many titles are ambiguous ("Temperature rising", "New wave hits coast") and only the body text reveals whether the article is about climate change or something else. The full text (truncated to 3,000 characters) provides enough context for accurate labeling.

#### Example: English with N=300

For illustration, suppose English has 900K articles in Weaviate. The score distribution might yield:

| Percentile | Climate score | Health score |
|---|---|---|
| p50 | 0.42 | 0.38 |
| p85 | 0.51 | 0.46 |
| p90 | 0.54 | 0.48 |
| p95 | 0.58 | 0.52 |

Then:
- **high_climate** (60 articles): random sample from ~45K articles with climate_max_sim > 0.58
- **high_health** (60 articles): random sample from ~45K articles with health_max_sim > 0.52
- **high_both** (60 articles): random sample from articles with climate > 0.54 AND health > 0.48
- **low_both** (60 articles): random sample from ~450K articles with both scores below median
- **borderline** (60 articles): random sample from articles with climate_max_sim in [0.49, 0.53] or health_max_sim in [0.44, 0.48]

#### Label schema

Each article in the output CSV has three binary label columns to be filled:

| Label | Question |
|---|---|
| `climate` | Is this article about climate change, global warming, or related environmental topics? |
| `health` | Is this article about human health, disease, healthcare, or public health? |
| `health_climate_impact` | Does this article discuss health effects caused by or linked to climate change? |

Each label has a corresponding `*_justification` column for the annotator's reasoning.

### Running extraction

```bash
# Extract 300 articles for English
.venv/bin/python build_labeled_dataset.py extract --lang English --n 300

# Extract for all major languages (300 each)
.venv/bin/python build_labeled_dataset.py extract --lang all --n 300

# Output goes to results/labeled/to_label_{lang}.csv
```

### Data flow

```
Weaviate (5.9M articles with vectors)
    |
    | cursor-based iteration, batched scoring
    | (vectors fetched and discarded per batch)
    v
Scored metadata (source, article_id, title, url, scores)
    |
    | stratified sampling across 5 buckets
    v
Selected articles (~300 per language)
    |
    | full text retrieval from crawl files
    v
results/labeled/to_label_{lang}.csv
    |
    | human or LLM annotation
    v
Labeled dataset for fine-tuning
```

## Scripts

| File | Purpose |
|---|---|
| `download_s3.py` | Download article data from S3 to `data/` |
| `embed_articles.py` | Embedding pipeline with tiered batching and GPU support |
| `embed_keywords.py` | Embed keyword text files to JSONL with vectors |
| `import_weaviate.py` | Import pre-computed embeddings into local Weaviate |
| `classify_articles.py` | Classify articles via keyword similarity (Weaviate-backed) |
| `build_labeled_dataset.py` | Build stratified labeled dataset for classifier training |
| `calibrate_thresholds.py` | Synthetic article threshold calibration |
| `weaviate_utils.py` | Shared Weaviate connection, iteration, and scoring utilities |
| `check_progress.py` | Monitor embedding pipeline progress |
| `test_embeddings.py` | Multilingual embedding quality test |
| `inspect_data.py` | Quick inspection of raw article data |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Start Weaviate
docker compose up -d
```

Requires a `.env` file with S3 credentials. GPU with CUDA support recommended for the embedding pipeline.
