# Prevalence Estimation — Design Decisions, Limitations, and Mitigations

## Overview

We estimate country-level prevalence of climate, health, and climate-health-impact news coverage using a Bayesian hierarchical model. The pipeline is:

1. **Embed** all 3.5M articles using Qwen3-Embedding-0.6B
2. **Filter** using cosine similarity against English keyword embeddings with per-language thresholds (recall-boosted by 0.85)
3. **Classify** filtered articles using fine-tuned Qwen3.5-4B with structured output
4. **Estimate** country-level prevalence using hierarchical model with partial pooling

Two model implementations are available:
- `estimate_prevalence.py` — article-level Bernoulli (extensible to article-level covariates like date)
- `estimate_prevalence_binomial.py` — source-level Beta-Binomial (fast, equal source weighting)

Both support `--time yearly` or `--time quarterly` to add temporal effects. Without the flag, time is ignored and the model estimates marginal prevalence.

---

## Key Design Decisions

### Decision 1: Estimand — media landscape, not article pool

**Problem:** Countries have vastly different numbers of sources (2–5) and articles per source (2K–530K). A naive count would be dominated by high-volume sources.

**Decision:** The estimand is "what proportion of the news media landscape in country X covers climate/health?" — not "what proportion of articles." Each source is treated as one observation of the country's media environment.

**Implementation:** The Beta-Binomial model (`estimate_prevalence_binomial.py`) models each source's proportion as a draw from a country-level Beta distribution. Sources are weighted equally regardless of volume. Within-source sampling uncertainty is still captured (small sources contribute less information), but a source with 500K articles doesn't dominate a source with 5K articles.

**Alternative considered:** Volume-weighted Binomial model (sources weighted by article count). Rejected because a single high-volume source with unusual editorial focus would define the country estimate.

### Decision 2: Three-level hierarchy with UN regions

**Problem:** With only 2–5 sources per country, country-level parameters are weakly identified.

**Decision:** Use a three-level hierarchy: Region → Country → Source. The 7 UN Geoscheme regions (Africa, Asia, Europe, Latin America & Caribbean, North America, Oceania) provide partial pooling. Countries with few sources borrow strength from their region.

**Implementation:**
- `mu_region ~ Normal(mu_global, sigma_global)` — regional mean
- `alpha_country ~ Normal(mu_region, tau_region)` — country deviates from region
- `p_source ~ Beta(mu_country * κ_region, (1-mu_country) * κ_region)` — source deviates from country
- `κ_region` (concentration) is per-region, allowing different levels of source heterogeneity across regions

### Decision 3: Per-region concentration parameter

**Problem:** Source-level heterogeneity (how much sources within a country differ) likely varies by region. European media landscapes may be more homogeneous within countries than, e.g., Sub-Saharan African ones.

**Decision:** `kappa_region` is estimated per region rather than globally. With 7 regions each containing multiple countries × 2–5 sources, there's enough data to estimate regional concentration.

### Decision 4: Time effects as categorical random effects

**Problem:** We want to plot prevalence trends over time (yearly or quarterly).

**Decision:** Time is modeled as categorical random effects rather than linear slopes:
- `δ_time[t]` — global time effect (shared across regions)
- `γ_region_time[r,t]` — region-specific deviation from global trend

This captures non-linear patterns (e.g., spikes around COP events) rather than assuming a linear trend. Available via `--time yearly` or `--time quarterly` in both `estimate_prevalence.py` and `estimate_prevalence_binomial.py`.

**Without `--time`:** The model ignores time entirely and estimates marginal prevalence. This is the default.

**With `--time`:** Data is aggregated to source × time period. For the Beta-Binomial version, each source in each time period is one observation (~150 sources × 6 years = ~900 for yearly, ~3,750 for quarterly). The country mean on logit scale becomes `α_country[k] + δ_time[t] + γ_region_time[r,t]`, and source-level proportions are drawn from a Beta parameterized by this time-varying country mean.

**Alternative considered:** Random slopes per country (`γ_country[k] * year`). Rejected because it assumes linearity, while categorical effects allow free-form trends that are better for plotting and can capture event-driven spikes.

### Decision 5: Sufficient statistics for the primary model

**Problem:** 3.5M article-level Bernoulli observations make MCMC slow.

**Decision:** The primary model (`estimate_prevalence_binomial.py`) aggregates to ~150 source-level (k, n) pairs. This is mathematically equivalent to the article-level model for the current specification (no article-level covariates). The article-level version (`estimate_prevalence.py`) is retained for future extensions (e.g., adding publication date as a covariate).

### Decision 6: Embedding-based pre-filtering with recall boost

**Problem:** Classifying all 3.5M articles with an LLM is slow and expensive.

**Decision:** Pre-filter using cosine similarity against keyword embeddings, then classify only articles above threshold. Per-language thresholds (calibrated on synthetic + real data) are multiplied by a recall-boost factor (default 0.85) to err on the side of including borderline articles. The fine-tuned classifier makes the final decision.

**Implementation:** `score_embeddings.py` with `--recall-boost 0.85`.

---

## Known Limitations and Concerns

### 1. Classification error propagation

The model treats classifier output as ground truth. The Qwen3.5-4B classifier was fine-tuned on ~2K labeled examples and will have non-trivial error rates, particularly for the rare `health_effects_of_climate_change` label (2.2% of training data). Misclassification rates may vary systematically by language, which would bias country estimates.

**Potential mitigation:** A measurement error sub-model using per-language validation data. If we obtain a confusion matrix per language (from the labeled evaluation data across 20 languages), we could incorporate misclassification rates directly into the likelihood.

**Status:** Not yet implemented.

### 2. Source confounding with country

With only 2–5 sources per country, a single source with unusual editorial focus can heavily influence country estimates.

**Mitigation implemented:** Beta-Binomial model with equal source weighting. Each source is one observation of the country's media landscape. The concentration parameter `κ_region` controls how much sources can deviate within a country. Partial pooling via the regional hierarchy shrinks poorly-estimated countries toward regional means.

**Remaining concern:** With 2 sources per country, the country estimate is essentially the mean of 2 observations. The credible intervals will be wide, which is the correct behavior, but interpretation should acknowledge this.

### 3. Temporal confounding from source composition

Article volume varies substantially by year (380K in 2020 vs 1.18M in 2026). If source composition changed over time (new sources added, crawling intensity varying), temporal trends conflate real prevalence changes with source composition changes.

**Potential mitigation:** Restrict temporal analysis to sources present across all years, or add a source × time interaction.

**Status:** Not yet implemented. Should verify source consistency across years before interpreting temporal trends.

### 4. Country prevalence marginal over time

When using `--time`, `prevalence_country` comes from `invlogit(alpha_country)` only and does not incorporate time effects. It represents a baseline intercept, not mean prevalence across the study period.

**Potential mitigation:** Add a derived quantity that averages `invlogit(alpha_country + delta_time[t])` over time periods.

**Status:** Noted for future implementation.

### 5. Selection bias from embedding threshold

Only articles passing the cosine similarity filter are classified. If the filter systematically misses certain relevant articles (e.g., those discussing climate-health impacts without typical keywords, or articles in languages with poor keyword embedding coverage), prevalence estimates are conditional on passing the filter.

**Mitigation implemented:** Recall-boost factor (0.85) lowers all thresholds by 15%, casting a wider net. English keyword embeddings are used (Qwen3-Embedding is multilingual, so cross-lingual similarity works).

**Remaining concern:** The magnitude of filtering bias is unknown. Could be assessed by classifying a random sample of filtered-out articles to estimate the false-negative rate.

### 6. Partial 2026 data

2026 contains only ~3 months of data (Jan–Mar) but 1.18M articles, suggesting a collection ramp-up. Including 2026 in yearly trends may be misleading.

**Recommendation:** Exclude 2026 from yearly temporal analyses or restrict to complete years (2020–2025). For quarterly analyses, include only complete quarters.

---

## Running the analyses

```bash
# Step 1: Embed articles (on GPU)
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8000 --dtype auto --max-model-len 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90 --runner pooling \
    --pooler-config '{"pooling_type": "MEAN", "enable_chunked_processing": true, "max_embed_len": 32768}'
python embed_all.py --concurrency 20 --batch-size 256

# Step 2: Score and filter articles
python score_embeddings.py --recall-boost 0.85

# Step 3: Classify filtered articles (on GPU)
# Deploy classifier endpoint, then:
python classify_all.py --concurrency 100

# Step 4a: Estimate prevalence (source-level, fast)
python estimate_prevalence_binomial.py --input data/classifications.jsonl

# Step 4b: Estimate prevalence with time trends (article-level)
python estimate_prevalence.py --input data/classifications.jsonl --time yearly
python estimate_prevalence.py --input data/classifications.jsonl --time quarterly
```

## Output files

- `results/prevalence/prevalence_country_{category}.csv` — country-level estimates with 95% HDI
- `results/prevalence/prevalence_region_{category}.csv` — region-level estimates
- `results/prevalence/prevalence_time_{category}_{yearly|quarterly}.csv` — global trend
- `results/prevalence/prevalence_region_time_{category}_{yearly|quarterly}.csv` — per-region trends
- `results/prevalence/trace_{category}.nc` — full posterior (ArviZ NetCDF)
