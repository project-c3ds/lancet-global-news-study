"""Estimate country-level prevalence using source-level Beta-Binomial model.

Input is a parquet file with pre-merged classification labels and metadata.

Without --time:
    p_source[j] ~ Beta(μ_country[k] * κ_r, (1-μ_country[k]) * κ_r)
    k[j] ~ Binomial(n[j], p_source[j])

With --time yearly, --time quarterly, or --time monthly:
    p_source_time[j,t] ~ Beta(μ_country_time[k,t] * κ_r, (1-μ_country_time[k,t]) * κ_r)
    logit(μ_country_time[k,t]) = α_country[k] + δ_time[t] + γ_group_time[g,t]
    k[j,t] ~ Binomial(n[j,t], p_source_time[j,t])

The top-level grouping (--group-by) can be:
    region        — LC Grouping (Africa, Asia, Europe, Latin America,
                    Northern America, Oceania, SIDS) per the 2026 Lancet
                    Countdown country/region guidance
    climate_zone  — Northern temperate, Tropical, Southern temperate
    hdi_category  — 2025 HDI Group (Very High, High, Medium, Low)

Usage:
    python estimate_prevalence_binomial.py --input analysis/analysis_data.parquet --climate-filter
    python estimate_prevalence_binomial.py --input analysis/analysis_data.parquet --climate-filter --time yearly
    python estimate_prevalence_binomial.py --input analysis/analysis_data.parquet --climate-filter --group-by climate_zone --time monthly
    python estimate_prevalence_binomial.py --input analysis/analysis_data.parquet --climate-filter --group-by hdi_category
"""

import argparse
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

OUTPUT_DIR = Path("analysis/results/prevalence")

# Mapping from --group-by value to the column name in the dataset
GROUP_COL_MAP = {
    "region": "lc_region",
    "climate_zone": "climate_zone",
    "hdi_category": "hdi_2025",
}


def load_dataset(path):
    """Load the estimation dataset from parquet or csv (extension-detected)."""
    p = str(path).lower()
    if p.endswith(".csv") or p.endswith(".csv.gz"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if "source_uri" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"source_uri": "source"})
    return df


def make_time_period(date_val, time_mode):
    """Convert a date value to a time period label."""
    if date_val is None or pd.isna(date_val):
        return None
    try:
        dt = pd.to_datetime(date_val)
    except Exception:
        return None
    if time_mode == "yearly":
        return str(dt.year)
    elif time_mode == "quarterly":
        return f"{dt.year}Q{dt.quarter}"
    elif time_mode == "monthly":
        return f"{dt.year}-{dt.month:02d}"
    return None


def is_pre_aggregated(df):
    """Pre-aggregated inputs carry k,n,group,country columns. Article-level
    parquets don't — they have a source_uri/source + the original label columns."""
    return {"k", "n", "group", "country"}.issubset(df.columns)


def prepare_data_from_aggregated(source_df, time_mode=None):
    """Skip re-aggregation when the input already has (k, n, group, country[, time_period])."""
    source_df = source_df.copy()
    # If the aggregate lacks min_articles filtering upstream, that's fine — they're already grouped.
    groups = sorted(source_df["group"].unique())
    countries = sorted(source_df["country"].unique())
    group_idx = {g: i for i, g in enumerate(groups)}
    country_idx = {c: i for i, c in enumerate(countries)}

    country_to_group = source_df.groupby("country")["group"].first().to_dict()
    country_group = np.array([group_idx[country_to_group[c]] for c in countries])

    source_df["group_id"] = source_df["group"].map(group_idx).values
    source_df["country_id"] = source_df["country"].map(country_idx).values

    n_obs = len(source_df)
    label = "source×time" if time_mode else "source-level"
    print(f"  Pre-aggregated input: {n_obs} {label} observations")

    n_articles = int(source_df["n"].sum())
    raw_prev = float(source_df["k"].sum() / n_articles) if n_articles else 0.0

    result = {
        "k": source_df["k"].values,
        "n": source_df["n"].values,
        "group_id": source_df["group_id"].values,
        "country_id": source_df["country_id"].values,
        "country_group": country_group,
        "groups": groups,
        "countries": countries,
        "n_groups": len(groups),
        "n_countries": len(countries),
        "n_obs": n_obs,
        "n_articles": n_articles,
        "raw_prevalence": raw_prev,
        "source_df": source_df,
    }

    if time_mode:
        time_periods = sorted(source_df["time_period"].unique())
        time_idx = {t: i for i, t in enumerate(time_periods)}
        source_df["time_id"] = source_df["time_period"].map(time_idx).values
        result["time_id"] = source_df["time_id"].values
        result["time_periods"] = time_periods
        result["n_times"] = len(time_periods)

    return result


def prepare_data(df, category, group_col, time_mode=None, min_articles=0):
    """Aggregate to source-level (or source×time) sufficient statistics."""
    col_name = GROUP_COL_MAP[group_col]

    df = df[df["country"].notna() & (df["country"] != "")].copy()
    df["group"] = df[col_name]
    df = df[df["group"].notna() & (df["group"] != "")].copy()
    df["y"] = df[category].astype(int)

    # Filter sources with too few articles
    if min_articles > 0:
        source_counts = df.groupby("source").size()
        keep_sources = source_counts[source_counts >= min_articles].index
        before = len(df)
        df = df[df["source"].isin(keep_sources)].copy()
        print(f"  Filtered to sources with >= {min_articles} articles: {df['source'].nunique()} sources, {len(df):,} articles (dropped {before - len(df):,})")

    # Time periods
    if time_mode:
        df["time_period"] = df["published_date"].apply(lambda x: make_time_period(x, time_mode))
        df = df[df["time_period"].notna()].copy()

    # Aggregate
    agg_cols = ["source", "country", "group"]
    if time_mode:
        agg_cols.append("time_period")

    source_df = df.groupby(agg_cols).agg(
        k=("y", "sum"),
        n=("y", "count"),
    ).reset_index()

    # Create integer indices
    groups = sorted(source_df["group"].unique())
    countries = sorted(source_df["country"].unique())

    group_idx = {g: i for i, g in enumerate(groups)}
    country_idx = {c: i for i, c in enumerate(countries)}

    # Build country -> group mapping from the data
    country_to_group = source_df.groupby("country")["group"].first().to_dict()
    country_group = np.array([group_idx[country_to_group[c]] for c in countries])

    source_df["group_id"] = source_df["group"].map(group_idx).values
    source_df["country_id"] = source_df["country"].map(country_idx).values

    n_obs = len(source_df)
    label = "source×time" if time_mode else "source-level"
    print(f"  Aggregated {len(df):,} articles -> {n_obs} {label} observations")

    result = {
        "k": source_df["k"].values,
        "n": source_df["n"].values,
        "group_id": source_df["group_id"].values,
        "country_id": source_df["country_id"].values,
        "country_group": country_group,
        "groups": groups,
        "countries": countries,
        "n_groups": len(groups),
        "n_countries": len(countries),
        "n_obs": n_obs,
        "n_articles": len(df),
        "raw_prevalence": df["y"].mean(),
        "source_df": source_df,
    }

    if time_mode:
        time_periods = sorted(source_df["time_period"].unique())
        time_idx = {t: i for i, t in enumerate(time_periods)}
        source_df["time_id"] = source_df["time_period"].map(time_idx).values
        result["time_id"] = source_df["time_id"].values
        result["time_periods"] = time_periods
        result["n_times"] = len(time_periods)

    return result


def build_model(data, time_mode=None, group_time=True, global_time=True, independent_groups=False):
    """Build PyMC hierarchical logit-Normal-Binomial model.

    Reparameterized from Beta-Binomial to logit-Normal for better NUTS geometry.
    Source-level proportions are modeled on the logit scale with a non-centered
    parameterization, eliminating the funnel between kappa and p_source.

    Time effect modes (when time_mode is set):
    - global_time=True, group_time=True (default): shared global trend + group deviations
    - global_time=True, group_time=False: shared global trend only
    - global_time=False, group_time=True: independent per-group time effects (no shared trend)

    If independent_groups=True, each group gets an independent prior on mu_group
    rather than pooling toward a shared mu_global. Countries still pool within groups.
    """
    with pm.Model() as model:
        if independent_groups:
            # Independent group priors — no shared global mean
            mu_group = pm.Normal("mu_group", mu=-2, sigma=2, shape=data["n_groups"])
            mu_global = pm.Deterministic("mu_global", mu_group.mean())
        else:
            # Hierarchical: groups pool toward shared global mean
            mu_global = pm.Normal("mu_global", mu=-2, sigma=2)
            sigma_global = pm.HalfNormal("sigma_global", sigma=1)
            mu_group = pm.Normal("mu_group", mu=mu_global, sigma=sigma_global, shape=data["n_groups"])

        # Country-within-group variance (always per-group)
        tau_group = pm.HalfNormal("tau_group", sigma=1, shape=data["n_groups"])

        # Country level (logit scale) — non-centered parameterization
        # to avoid Neal's funnel between alpha_country and tau_group, which
        # is acute for singleton groups (e.g. Oceania = Australia only).
        z_country = pm.Normal("z_country", mu=0, sigma=1, shape=data["n_countries"])
        alpha_country = pm.Deterministic(
            "alpha_country",
            mu_group[data["country_group"]] + tau_group[data["country_group"]] * z_country,
        )

        # Country mean on logit scale — add time effects if applicable
        logit_mu = alpha_country[data["country_id"]]

        if time_mode:
            n_t = data["n_times"]
            n_g = data["n_groups"]

            # Global time effect (optional)
            if global_time:
                sigma_time = pm.HalfNormal("sigma_time", sigma=0.5)
                delta_time = pm.Normal("delta_time", mu=0, sigma=sigma_time, shape=n_t)
                logit_mu = logit_mu + delta_time[data["time_id"]]

            # Per-group time effects (optional)
            if group_time:
                sigma_group_time = pm.HalfNormal("sigma_group_time", sigma=0.5)
                gamma_group_time = pm.Normal(
                    "gamma_group_time", mu=0, sigma=sigma_group_time, shape=(n_g, n_t)
                )
                logit_mu = logit_mu + gamma_group_time[data["group_id"], data["time_id"]]

            # Derived: global prevalence over time
            if global_time:
                pm.Deterministic("prevalence_time", pm.math.invlogit(mu_global + delta_time))

            # Derived: group-level prevalence over time
            if global_time and group_time:
                pm.Deterministic(
                    "prevalence_group_time",
                    pm.math.invlogit(mu_group[:, None] + delta_time[None, :] + gamma_group_time),
                )
            elif global_time and not group_time:
                pm.Deterministic(
                    "prevalence_group_time",
                    pm.math.invlogit(mu_group[:, None] + delta_time[None, :]),
                )
            elif not global_time and group_time:
                pm.Deterministic(
                    "prevalence_group_time",
                    pm.math.invlogit(mu_group[:, None] + gamma_group_time),
                )

        # Source-level heterogeneity (logit scale, per-group)
        sigma_source = pm.HalfNormal("sigma_source", sigma=1, shape=data["n_groups"])

        # Non-centered parameterization
        z_source = pm.Normal("z_source", mu=0, sigma=1, shape=data["n_obs"])
        logit_p_source = logit_mu + sigma_source[data["group_id"]] * z_source

        p_source = pm.Deterministic("p_source", pm.math.invlogit(logit_p_source))

        # Likelihood
        pm.Binomial("obs", n=data["n"], p=p_source, observed=data["k"])

        # Derived: marginal prevalence at each level
        pm.Deterministic("prevalence_global", pm.math.invlogit(mu_global))
        pm.Deterministic("prevalence_country", pm.math.invlogit(alpha_country))
        pm.Deterministic("prevalence_group", pm.math.invlogit(mu_group))

    return model


def run_model(data, time_mode=None, group_time=True, global_time=True, independent_groups=False, samples=2000, chains=4, target_accept=0.9, tune=1000):
    """Build and sample from the model."""
    model = build_model(data, time_mode=time_mode, group_time=group_time, global_time=global_time, independent_groups=independent_groups)
    with model:
        trace = pm.sample(
            draws=samples,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=42,
            progressbar=True,
        )
    return model, trace


def summarize_results(trace, data, category, group_col, time_mode=None):
    """Extract global, country, group, and time-level prevalence estimates."""
    prev_global = az.summary(trace, var_names=["prevalence_global"], hdi_prob=0.95, stat_focus="median")

    prev_country = az.summary(trace, var_names=["prevalence_country"], hdi_prob=0.95, stat_focus="median")
    prev_country.index = data["countries"]
    prev_country[group_col] = [
        data["groups"][data["country_group"][i]]
        for i in range(data["n_countries"])
    ]

    prev_group = az.summary(trace, var_names=["prevalence_group"], hdi_prob=0.95, stat_focus="median")
    prev_group.index = data["groups"]

    prev_time = None
    prev_group_time = None
    if time_mode:
        # prevalence_time only exists when global_time=True
        if "prevalence_time" in trace.posterior:
            prev_time = az.summary(trace, var_names=["prevalence_time"], hdi_prob=0.95, stat_focus="median")
            prev_time.index = data["time_periods"]

        if "prevalence_group_time" in trace.posterior:
            prev_group_time = az.summary(trace, var_names=["prevalence_group_time"], hdi_prob=0.95, stat_focus="median")
            idx = pd.MultiIndex.from_product(
                [data["groups"], data["time_periods"]], names=[group_col, "time_period"]
            )
            prev_group_time.index = idx

    return prev_global, prev_country, prev_group, prev_time, prev_group_time


def main():
    parser = argparse.ArgumentParser(description="Estimate country-level prevalence")
    parser.add_argument("--input", default="analysis/analysis_data.parquet", help="Estimation dataset parquet path")
    parser.add_argument("--category", default="all",
                        choices=["all", "health", "health_effects_of_climate_change", "health_effects_of_extreme_weather"],
                        help="Which category to model (default: health + hecc)")
    parser.add_argument("--group-by", dest="group_by", default="region",
                        choices=["region", "climate_zone", "hdi_category"],
                        help="Top-level grouping variable (default: region)")
    parser.add_argument("--time", choices=["yearly", "quarterly", "monthly"], default=None,
                        help="Add time effects (yearly, quarterly, or monthly)")
    parser.add_argument("--climate-filter", dest="climate_filter", action="store_true", default=False,
                        help="Filter to climate_change=True articles before estimation")
    parser.add_argument("--no-group-time", dest="group_time", action="store_false", default=True,
                        help="Omit group×time interaction (all groups share global time trend)")
    parser.add_argument("--no-global-time", dest="global_time", action="store_false", default=True,
                        help="Omit shared global time trend (each group gets independent time effects)")
    parser.add_argument("--independent-groups", dest="independent_groups", action="store_true", default=False,
                        help="Independent group priors (no pooling across groups, only within)")
    parser.add_argument("--min-articles", type=int, default=0, help="Minimum articles per source (default: 0, no filter)")
    parser.add_argument("--samples", type=int, default=2000, help="MCMC samples per chain")
    parser.add_argument("--tune", type=int, default=1000, help="NUTS warmup/tuning steps per chain (default: 1000)")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95, help="NUTS target accept rate (default: 0.95)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading dataset from {args.input}...")
    df = load_dataset(args.input)
    print(f"  {len(df):,} rows")

    pre_aggregated = is_pre_aggregated(df)
    if pre_aggregated:
        print("  Detected pre-aggregated input (k, n columns present) — skipping aggregation step")
        if args.climate_filter:
            print("  NOTE: --climate-filter is a no-op for pre-aggregated inputs (filter already applied upstream)")
        if args.min_articles:
            print("  NOTE: --min-articles is a no-op for pre-aggregated inputs")
        categories = [args.category]
    else:
        if args.climate_filter:
            df = df[df["climate_change"] == True].copy()
            print(f"  After climate_change filter: {len(df):,} articles")
        categories = ["health", "health_effects_of_climate_change"] if args.category == "all" else [args.category]

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"Group by: {args.group_by}")
        print(f"{'='*60}")

        if pre_aggregated:
            data = prepare_data_from_aggregated(df, time_mode=args.time)
        else:
            data = prepare_data(df, category, group_col=args.group_by, time_mode=args.time, min_articles=args.min_articles)
        print(f"  Articles: {data['n_articles']:,}")
        print(f"  Groups: {data['n_groups']} ({', '.join(data['groups'])})")
        print(f"  Countries: {data['n_countries']}, Observations: {data['n_obs']}")
        if args.time:
            print(f"  Time periods ({args.time}): {data['n_times']} ({data['time_periods'][0]} to {data['time_periods'][-1]})")
        print(f"  Prevalence (raw): {data['raw_prevalence']:.4f}")

        t0 = time.time()
        model, trace = run_model(data, time_mode=args.time, group_time=args.group_time, global_time=args.global_time, independent_groups=args.independent_groups, samples=args.samples, chains=args.chains, target_accept=args.target_accept, tune=args.tune)
        elapsed = time.time() - t0
        print(f"  Sampling done in {elapsed/60:.1f}m")

        prev_global, prev_country, prev_group, prev_time, prev_group_time = summarize_results(
            trace, data, category, group_col=args.group_by, time_mode=args.time
        )

        # Save — include group_col in filenames
        time_suffix = f"_{args.time}" if args.time else ""
        base = f"{category}_{args.group_by}{time_suffix}"
        prev_global.to_csv(OUTPUT_DIR / f"prevalence_global_{base}.csv")
        prev_group.to_csv(OUTPUT_DIR / f"prevalence_group_{base}.csv")
        prev_country.to_csv(OUTPUT_DIR / f"prevalence_country_{base}.csv")
        az.to_netcdf(trace, OUTPUT_DIR / f"trace_{base}.nc")

        if prev_time is not None:
            prev_time.to_csv(OUTPUT_DIR / f"prevalence_time_{base}.csv")
        if prev_group_time is not None:
            prev_group_time.to_csv(OUTPUT_DIR / f"prevalence_group_time_{base}.csv")

        # Save diagnostics
        diag = az.summary(trace, hdi_prob=0.95)
        diag.to_csv(OUTPUT_DIR / f"diagnostics_{base}.csv")
        n_divergences = int(trace.sample_stats["diverging"].sum())
        rhat_max = diag["r_hat"].max()
        ess_min = diag["ess_bulk"].min()
        print(f"\n  Diagnostics: {n_divergences} divergences, max r_hat={rhat_max:.3f}, min ESS={ess_min:.0f}")
        print(f"  Saved diagnostics to {OUTPUT_DIR}/diagnostics_{base}.csv")

        g = prev_global.iloc[0]
        print(f"\n  Global estimate: {g['median']:.4f} [{g['eti_2.5%']:.4f}, {g['eti_97.5%']:.4f}]")

        print(f"\n  Group estimates ({args.group_by}):")
        for group in prev_group.index:
            row = prev_group.loc[group]
            print(f"    {group:<35} {row['median']:.4f} [{row['eti_2.5%']:.4f}, {row['eti_97.5%']:.4f}]")

        if prev_time is not None:
            print(f"\n  Global trend ({args.time}):")
            for tp in prev_time.index:
                row = prev_time.loc[tp]
                print(f"    {tp:<10} {row['median']:.4f} [{row['eti_2.5%']:.4f}, {row['eti_97.5%']:.4f}]")

        print(f"\n  Top 5 countries:")
        top = prev_country.sort_values("median", ascending=False).head(5)
        for country in top.index:
            row = top.loc[country]
            print(f"    {country:<25} {row['median']:.4f} [{row['eti_2.5%']:.4f}, {row['eti_97.5%']:.4f}]  ({row[args.group_by]})")

        print(f"\n  Bottom 5 countries:")
        bottom = prev_country.sort_values("median", ascending=True).head(5)
        for country in bottom.index:
            row = bottom.loc[country]
            print(f"    {country:<25} {row['median']:.4f} [{row['eti_2.5%']:.4f}, {row['eti_97.5%']:.4f}]  ({row[args.group_by]})")

        print(f"\n  Saved to {OUTPUT_DIR}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
