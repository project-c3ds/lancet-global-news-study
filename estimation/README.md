# Estimation

Bayesian hierarchical prevalence estimation. See the appendix, "Statistical methods to estimate prevalence".

## Model

Three-level hierarchy: Group → Country → Source. Binomial likelihood at the source level (aggregates articles to sufficient statistics `(k_j, n_j)`, reducing hundreds of thousands of articles to a few hundred observations). Non-centred parameterisation for source-level random effects. NUTS sampler, 4 chains × 4,000 draws, target_accept=0.99.

## Scripts

- `estimate_prevalence_binomial.py` — single-model runner. Accepts either an article-level parquet/CSV or a **pre-aggregated** source-level CSV (with `k`, `n` columns) and skips aggregation in the latter case. CLI flags control grouping (`--group-by {region,climate_zone,hdi_category}`), time resolution (`--time {yearly,monthly}`), and the outcome label (`--category`). `--climate-filter` and `--min-articles` are no-ops against pre-aggregated inputs (applied upstream).
- `build_estimation_inputs.py` — derives the 4 per-analysis CSVs in `estimation_inputs/` from the published master dataset (`analysis/corpus_monthly.csv`). Pure `groupby.sum()` logic.
- `estimate.sh` — calls `build_estimation_inputs.py` (step 0), then runs the 4 Bayesian models.

## `estimation_inputs/` (committed)

Each file has the estimator's expected schema `(source, country, group, [time_period,] k, n)` and contains one row per source-level (or source×time) cell.

| File | Analysis | Rows | Size |
|---|---|---:|---:|
| `prev_hecc_cc_region_yearly.csv` | Model 1: HECC ⎮ CC, yearly × LC region | 1,258 | 51 KB |
| `prev_hecc_cc_climate_zone_monthly.csv` | Model 2: HECC ⎮ CC, monthly × climate zone | 12,377 | 611 KB |
| `prev_hecc_cc_hdi_category.csv` | Model 3: HECC ⎮ CC, HDI 2025 group | 294 | 11 KB |
| `prev_health_cc_region_yearly.csv` | Aux: Health ⎮ CC, yearly × LC region | 1,258 | 51 KB |

Region grouping is the **2026 Lancet Countdown LC Grouping** (Africa, Asia, Europe, Latin America, Northern America, Oceania, SIDS) — see `analysis/README.md` § *Country names and groupings* for the source-of-truth file (`backup/2026 Guidance_Country_Names_and_Groupings.csv`), Taiwan/Hong Kong overrides, and notes on the singleton Oceania group. HDI grouping is the **2025 HDI Group** (Low / Medium / High / Very High); rows with missing `hdi_2025` (Taiwan, Hong Kong) are dropped from the HDI input upstream.

HEEW is deliberately not estimated — it is used only at classification time to prevent articles about health consequences of extreme weather (without a climate-change framing) from being mislabelled as HECC. There is no well-defined denominator for a HEEW prevalence in this corpus.

All four files are regenerated from `analysis/corpus_monthly.csv` (the published master) by running `python estimation/build_estimation_inputs.py`. `estimate.sh` does this automatically before each sampling run, so a fresh clone can produce all results with:

```bash
bash estimation/estimate.sh
```

Outputs go to `analysis/results/prevalence/` as CSV summaries and NetCDF traces.
