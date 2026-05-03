# Analysis

Committed artifacts for Section 5.1 of the Lancet Countdown report: the **master dataset** that accompanies the paper, the pre-aggregated inputs derived from it, the figure-generation code, and the posterior summaries from estimation.

## The master dataset

**`corpus_monthly.csv`** — one row per `(source_uri, country, month)`, covering the 75 sample countries × 2021-2025. This is the data-release artifact that ships with the paper; every downstream estimation or plotting step derives from it.

Columns:

| Column | Type | Description |
|---|---|---|
| `source_uri` | str | |
| `country` | str | 75-country sample, `GB` remapped to `United Kingdom` |
| `iso3` | str | ISO 3166-1 alpha-3 country code |
| `lc_country_name` | str | formal country name from the 2026 Lancet Countdown guidance |
| `month` | str | `YYYY-MM`, 2021-01 through 2025-12 |
| `lc_region` | str | **primary region for analysis** — 2026 LC Grouping (Africa / Asia / Europe / Latin America / Northern America / Oceania / SIDS) |
| `who_region` | str | WHO Region; retained for sensitivity / traceability |
| `un_region` | str | legacy UN M49 broad region; retained for traceability |
| `climate_zone` | str | covariate (Northern temperate / Tropical / Southern temperate) |
| `hdi_2025` | str | **primary HDI grouping** — 2025 HDI Group (Very High / High / Medium / Low; blank for Taiwan and Hong Kong) |
| `hdi_category` | str | legacy HDI category; retained for traceability |
| `hdi_value` | float | covariate; NaN for Taiwan |
| `n_total` | int | all articles in the cell |
| `n_cc` | int | articles with `climate_change=True` |
| `k_hecc_cc` | int | articles with `climate_change=True` AND `health_effects_of_climate_change=True` |
| `k_health_cc` | int | articles with `climate_change=True` AND `health=True` |

About 14K rows, ~1 MB. All reported results in the paper can be reconstructed from this single file.

## Contents

### Master and build

- **`corpus_monthly.csv`** — the master dataset described above.
- **`build_master_dataset.py`** — rebuilds `corpus_monthly.csv` from the raw inputs:
    - `data/climate_articles_with_classifications.parquet` (unfiltered 2.2M-row article-level parquet)
    - `analysis/country_covariates.csv`

  Applies the paper's frame (country in 75-country sample, 2021-2025; remaps `GB` → `United Kingdom`). Run with `python analysis/build_master_dataset.py`.
- **`country_covariates.csv`** — per-country covariates for the 75 sample countries. Originally: UN region and subregion, climate zone, HDI value and category, Climate Risk Index ranks. As of the 2026 report, also carries the LC guidance columns (`lc_country_name`, `lc_region`, `who_region`, `hdi_2025`) merged on by `build_country_covariates.py` (see [Country names and groupings](#country-names-and-groupings) below). Used only at master-build time.
- **`build_country_covariates.py`** — merges the 2026 LC guidance file (`backup/2026 Guidance_Country_Names_and_Groupings.csv`) onto `country_covariates.csv` by ISO3. Idempotent: re-run if the guidance file changes. The guidance CSV lives under `backup/` (gitignored) because it is upstream input rather than a tracked artifact.

### Article-level dataset (external)

An article-level parquet (~95 MB, 759k rows — one row per article with full metadata and labels) can be rebuilt from raw inputs with `build_analysis_dataset.py`, or downloaded from:

- https://drive.google.com/file/d/1KuBkmBEOqbyJhhUhfhvrgEg1ldI3CiBq/view?usp=drive_link

This parquet is **not needed for reproducing the paper's numbers** — the master CSV above is sufficient. It is useful only for per-article exploration or rebuilding the master with a different filter.

- **`build_analysis_dataset.py`** — rebuilds the article-level parquet from `data/climate.db` + `data/classifications_slim.db` + `country_covariates.csv` (optional).

### Figures

- **`plot_inputs/yearly_trends_monthly.csv`** — 60-row monthly totals used by the appendix time-series figure (~1.5 KB).
- **`build_plot_inputs.py`** — rebuilds the above from the article-level parquet.
- **`plot_yearly_trends.py`** — renders the appendix figure (monthly CC volume + stacked HECC / Health share).
- **`figures/yearly_trends.{pdf,png}`** — output of `plot_yearly_trends.py`.

### Tables

- **`main_text_estimates.csv`** — Table 2 of the appendix. Hand-curated from the NetCDF traces produced by `estimation/estimate.sh`; each row is a quantity cited in the main text with its posterior median and 95% equal-tailed CrI.

### Posterior summaries (`results/prevalence/`)

Produced by `estimation/estimate.sh`. 22 CSVs in total, covering 4 Bayesian model runs:

| Run | Purpose (appendix) | `<base>` |
|---|---|---|
| 1 | Model 1 — HECC ⎮ CC, yearly × LC region | `health_effects_of_climate_change_region_yearly` |
| 2 | Model 2 — HECC ⎮ CC, monthly × climate zone | `health_effects_of_climate_change_climate_zone_monthly` |
| 3 | Model 3 — HECC ⎮ CC, HDI 2025 group (no time) | `health_effects_of_climate_change_hdi_category` |
| 4 | Auxiliary — Health ⎮ CC, yearly × LC region | `health_region_yearly` |

For each run, the estimation script writes six (or four, if no time dimension) summaries:

| Filename pattern | Contents |
|---|---|
| `prevalence_global_<base>.csv` | posterior of the overall mean prevalence (1 row) |
| `prevalence_group_<base>.csv` | posterior per top-level group |
| `prevalence_country_<base>.csv` | posterior per country |
| `prevalence_time_<base>.csv` | posterior per time period (time runs only) |
| `prevalence_group_time_<base>.csv` | posterior per group × time cell (time runs only) |
| `diagnostics_<base>.csv` | MCMC diagnostics per parameter (R-hat, ESS bulk/tail, MCSE) |

Each run also produces a `trace_<base>.nc` NetCDF with the full posterior samples. These are **not** committed (~4 GB total) but are regenerated by `estimate.sh`.

## Country names and groupings

Country names, ISO3 codes, and the regional / HDI groupings used in the 2026 report come from the official Lancet Countdown guidance file:

- **`backup/2026 Guidance_Country_Names_and_Groupings.csv`** — defines, for every country and territory, the formal name (`Country Name to use`), the **LC Grouping** (Africa / Asia / Europe / Latin America / Northern America / Oceania / SIDS), the **WHO Region**, and the **HDI Group 2025** (Low / Medium / High / Very High). Stored under `backup/` (gitignored) because it is upstream input distributed separately by the LC secretariat — request it from the LC working group if not present locally.

`analysis/build_country_covariates.py` joins this onto our 75-country covariates table by ISO3, producing four new columns on `country_covariates.csv`: `lc_country_name`, `lc_region`, `who_region`, `hdi_2025`. These then propagate to `corpus_monthly.csv` via `build_master_dataset.py` and into the estimation inputs via `estimation/build_estimation_inputs.py`.

The estimator's "region" analysis runs on `lc_region` (replacing the legacy `un_region`); the HDI analysis runs on `hdi_2025` (replacing the legacy `hdi_category`). The legacy columns are kept on the master CSV for traceability but are not used for any reported estimate.

### Manual overrides

Two cases are not derivable from the guidance file alone and are handled in `build_country_covariates.py`:

1. **Taiwan (TWN)** — not listed in the LC guidance. We retain it in the corpus with `lc_country_name = "Taiwan"`, `lc_region = "Asia"`, `who_region = "Western Pacific"`, and `hdi_2025` left blank (so Taiwan is excluded from the HDI analysis but kept in the LC region and climate-zone analyses, mirroring the legacy treatment).
2. **Hong Kong (HKG)** — listed in the guidance but with `HDI Group 2025 = #N/A` (HDI is not assigned to non-state territories). We set `hdi_2025` to blank, which drops Hong Kong from the HDI prevalence analysis. It remains in the LC region (`Asia`) and other analyses.

### Singleton Oceania

Under the LC Grouping, Fiji and Papua New Guinea move to **SIDS**, leaving **Australia as the sole "Oceania" country** in our sample. We keep Oceania as a singleton group rather than folding Australia into SIDS (Australia is neither small, an island in the relevant sense, nor a developing state, and folding it in would bias the SIDS estimate toward Australia's media patterns). The hierarchical model handles a group of size 1 without bias — the country-level posterior for Australia is well-identified, and the group-level posterior is wider but unbiased.

## Reproducing the results

From the project root, with the virtualenv active:

```bash
.venv/bin/python analysis/build_country_covariates.py   # merges LC guidance (only if guidance changed)
.venv/bin/python analysis/build_master_dataset.py       # rebuilds corpus_monthly.csv (only if guidance/covariates changed)
bash estimation/estimate.sh                    # ~30-60 min on CPU
.venv/bin/python analysis/plot_yearly_trends.py    # writes figures/yearly_trends.{pdf,png}
```

The first two commands are only needed if the LC guidance file or `country_covariates.csv` has changed; the committed `corpus_monthly.csv` is otherwise sufficient.

Neither step needs the article-level parquet — they both consume the committed small files (`corpus_monthly.csv`, the 4 `estimation_inputs/*.csv`, `plot_inputs/yearly_trends_monthly.csv`).
