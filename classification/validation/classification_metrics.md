# Classification Performance Metrics

N=148 (2 rows with missing model predictions excluded)

| Metric | Climate Change | Health | Health Effects of CC | Health Effects of EW |
|--------|---------------|--------|---------------------|---------------------|
| **Prevalence** | 46 (31.1%) | 40 (27.0%) | 11 (7.4%) | 9 (6.1%) |
| **Precision** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Recall** | 0.804 | 0.800 | 0.636 | 0.778 |
| **F1** | 0.892 | 0.889 | 0.778 | 0.875 |
| **Accuracy** | 0.939 | 0.946 | 0.973 | 0.986 |

**Exact Match (all 4 categories correct): 131/148 (0.885)**

Computed directly from `human_<label>` vs `model_<label>` columns in
`sample_150_eval_merged.csv`; see `../eval_human.py`.
