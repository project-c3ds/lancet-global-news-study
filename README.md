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

Output records contain:

| Field | Description |
|---|---|
| `id` | Line number from the source JSONL |
| `url` | Article URL |
| `title` | Article headline |
| `source` | Source identifier |
| `extracted_at` | Extraction timestamp |
| `truncated` | Whether the article exceeded the 8192 token limit |
| `embedding` | 1024-dimensional float vector |

### Running the pipeline

```bash
# Download data from S3
.venv/bin/python download_s3.py

# Run embedding pipeline (all sources)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python embed_articles.py --workers 1

# Run a single source
.venv/bin/python embed_articles.py --source scmp_com

# Check progress
.venv/bin/python check_progress.py

# Tail the log
tail -f data/embeddings/progress.log
```

The pipeline supports resuming — completed sources are skipped on restart.

## Multilingual Quality Test

`test_embeddings.py` validates cross-lingual embedding quality with two tests:

1. **Parallel sentences** — the same sentence in English, French, German, Arabic, Russian, and Chinese. Similarity scores of 0.79-0.91 confirm the model captures meaning across languages.
2. **Real articles** — articles from different language sources. Unrelated content correctly shows low similarity (0.09-0.20).

```bash
.venv/bin/python test_embeddings.py
```

## Scripts

| File | Purpose |
|---|---|
| `download_s3.py` | Download article data from S3 to `data/` |
| `embed_articles.py` | Embedding pipeline with tiered batching and GPU support |
| `check_progress.py` | Monitor embedding pipeline progress |
| `test_embeddings.py` | Multilingual embedding quality test |
| `inspect_data.py` | Quick inspection of raw article data |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install sentence-transformers
```

Requires a `.env` file with S3 credentials (see `.env` for the template). GPU with CUDA support recommended for the embedding pipeline.
