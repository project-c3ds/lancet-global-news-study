# Lancet Global News Study

Code and data for **Indicator 5.1: Media Engagement with Health and Climate Change**, Lancet Countdown on Health and Climate Change 2026 (Coan, Malla, O'Neill).

The indicator tracks media coverage of climate change and health across 321 news sources in 75 countries and 40 languages, drawing on 2.2 million articles published between 1 January 2021 and 31 December 2025. Articles are classified with a fine-tuned multilingual language model, and prevalence is estimated using a Bayesian hierarchical model at the source level.

## Repository layout

Each top-level folder maps to one stage of the pipeline and has its own `README.md` with details.

| Folder | Stage | Contents |
|---|---|---|
| [`collection/`](collection/) | Article collection | NewsAPI + ScrapAI pipelines, SQLite ingestion |
| [`translations/`](translations/) | Keyword translations | English seed keywords + 45-language translations used to query sources and compute BM25 scores |
| [`classification/`](classification/) | Multi-label classification | BM25 scoring, stratified annotation sample, Claude frontier annotation, Qwen 3.5 9B distillation, production inference, human validation |
| [`estimation/`](estimation/) | Bayesian prevalence models | Hierarchical Group → Country → Source model (PyMC/NUTS), pre-aggregated per-analysis inputs |
| [`analysis/`](analysis/) | Master dataset, results, figures | Published master CSV (`corpus_monthly.csv`), posterior summaries, appendix figures and tables |
| `data/` | Inputs and build artifacts (gitignored) | SQLite, raw JSONL, country metadata |

## Reproducing the analysis

The full pipeline runs in four stages. The first two are infrastructure-heavy (API costs, GPU fine-tuning); the last two can be re-run in minutes from the committed master CSV (`analysis/corpus_monthly.csv`, ~1 MB).

```
collection/     →  data/climate.db
classification/ →  adds BM25 columns to climate.db, writes data/classifications_slim.db
analysis/build_master_dataset.py  →  analysis/corpus_monthly.csv          (committed)
estimation/estimate.sh            →  analysis/results/prevalence/
analysis/plot_yearly_trends.py    →  analysis/figures/
```

To regenerate the posterior summaries and the appendix figure from the committed master:

```bash
bash estimation/estimate.sh                       # ~30-60 min on CPU; writes analysis/results/prevalence/
.venv/bin/python analysis/plot_yearly_trends.py   # writes analysis/figures/yearly_trends.{pdf,png}
```

See each folder's `README.md` for the full upstream pipeline, including how to rebuild the master from raw inputs.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

An `.env` file is required for any step that hits external APIs (NewsAPI, Anthropic, Modal). The estimation and plotting steps need only the committed CSVs in `analysis/` and `estimation/estimation_inputs/`.

## Citation

Coan, T. G., Malla, R., & O'Neill, S. (2026). Media Engagement with Health and Climate Change.
