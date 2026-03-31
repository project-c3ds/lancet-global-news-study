"""Estimate country-level prevalence using source-level Beta-Binomial model.

Without --time:
    p_source[j] ~ Beta(μ_country[k] * κ_r, (1-μ_country[k]) * κ_r)
    k[j] ~ Binomial(n[j], p_source[j])

With --time yearly or --time quarterly:
    p_source_time[j,t] ~ Beta(μ_country_time[k,t] * κ_r, (1-μ_country_time[k,t]) * κ_r)
    logit(μ_country_time[k,t]) = α_country[k] + δ_time[t] + γ_region_time[r,t]
    k[j,t] ~ Binomial(n[j,t], p_source_time[j,t])

Usage:
    python estimate_prevalence_binomial.py --input data/classifications.jsonl
    python estimate_prevalence_binomial.py --input data/classifications.jsonl --time yearly
    python estimate_prevalence_binomial.py --input data/classifications.jsonl --time quarterly
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from tqdm import tqdm

DB_PATH = "data/articles.db"
REGION_MAP_PATH = "data/country_regions.json"
OUTPUT_DIR = Path("results/prevalence")


def load_region_map(path):
    """Load region -> countries mapping, return country -> region dict."""
    with open(path) as f:
        region_to_countries = json.load(f)
    country_to_region = {}
    for region, countries in region_to_countries.items():
        for c in countries:
            country_to_region[c] = region
    return country_to_region


def load_classifications(path):
    """Load classified articles from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "error" not in row:
                records.append(row)
    return pd.DataFrame(records)


def load_source_metadata(db_path):
    """Load article id -> source, country, published_date from SQLite."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, source_uri, country, published_date FROM articles").fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["id", "source", "country", "published_date"])


def make_time_period(date_str, time_mode):
    """Convert a date string to a time period label."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        dt = pd.to_datetime(str(date_str)[:10])
    except Exception:
        return None
    if time_mode == "yearly":
        return str(dt.year)
    elif time_mode == "quarterly":
        return f"{dt.year}Q{dt.quarter}"
    return None


def prepare_data(df, country_to_region, category, time_mode=None):
    """Aggregate to source-level (or source×time) sufficient statistics."""
    df = df[df["country"].notna() & (df["country"] != "")].copy()
    df["region"] = df["country"].map(country_to_region)
    df = df[df["region"].notna()].copy()
    df["y"] = df[category].astype(int)

    # Time periods
    if time_mode:
        df["time_period"] = df["published_date"].apply(lambda x: make_time_period(x, time_mode))
        df = df[df["time_period"].notna()].copy()

    # Aggregate
    group_cols = ["source", "country", "region"]
    if time_mode:
        group_cols.append("time_period")

    source_df = df.groupby(group_cols).agg(
        k=("y", "sum"),
        n=("y", "count"),
    ).reset_index()

    # Create integer indices
    regions = sorted(source_df["region"].unique())
    countries = sorted(source_df["country"].unique())

    region_idx = {r: i for i, r in enumerate(regions)}
    country_idx = {c: i for i, c in enumerate(countries)}

    country_region = np.array([region_idx[country_to_region[c]] for c in countries])

    source_df["region_id"] = source_df["region"].map(region_idx).values
    source_df["country_id"] = source_df["country"].map(country_idx).values

    n_obs = len(source_df)
    label = "source×time" if time_mode else "source-level"
    print(f"  Aggregated {len(df):,} articles -> {n_obs} {label} observations")

    result = {
        "k": source_df["k"].values,
        "n": source_df["n"].values,
        "region_id": source_df["region_id"].values,
        "country_id": source_df["country_id"].values,
        "country_region": country_region,
        "regions": regions,
        "countries": countries,
        "n_regions": len(regions),
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


def build_model(data, time_mode=None):
    """Build PyMC hierarchical Beta-Binomial model.

    Each source's proportion is modeled as a draw from a country-level Beta
    distribution, so sources are weighted equally regardless of article volume.

    Without time: Hierarchy is Global → Region → Country → Source
    With time:    Country mean gets time effects:
                  logit(μ_country_time[k,t]) = α_country[k] + δ_time[t] + γ_region_time[r,t]
    """
    with pm.Model() as model:
        # Global mean prevalence (logit scale)
        mu_global = pm.Normal("mu_global", mu=-2, sigma=2)
        sigma_global = pm.HalfNormal("sigma_global", sigma=1)

        # Region level
        mu_region = pm.Normal("mu_region", mu=mu_global, sigma=sigma_global, shape=data["n_regions"])
        tau_region = pm.HalfNormal("tau_region", sigma=1, shape=data["n_regions"])

        # Country level (logit scale)
        alpha_country = pm.Normal(
            "alpha_country",
            mu=mu_region[data["country_region"]],
            sigma=tau_region[data["country_region"]],
            shape=data["n_countries"],
        )

        # Country mean on logit scale — add time effects if applicable
        logit_mu = alpha_country[data["country_id"]]

        if time_mode:
            n_t = data["n_times"]
            n_r = data["n_regions"]

            # Global time effect
            sigma_time = pm.HalfNormal("sigma_time", sigma=0.5)
            delta_time = pm.Normal("delta_time", mu=0, sigma=sigma_time, shape=n_t)

            # Region-specific deviation from global trend
            sigma_region_time = pm.HalfNormal("sigma_region_time", sigma=0.5)
            gamma_region_time = pm.Normal(
                "gamma_region_time", mu=0, sigma=sigma_region_time, shape=(n_r, n_t)
            )

            logit_mu = logit_mu + delta_time[data["time_id"]] + gamma_region_time[data["region_id"], data["time_id"]]

            # Derived: global and regional prevalence over time
            pm.Deterministic("prevalence_time", pm.math.invlogit(mu_global + delta_time))
            pm.Deterministic(
                "prevalence_region_time",
                pm.math.invlogit(mu_region[:, None] + delta_time[None, :] + gamma_region_time),
            )

        mu_obs = pm.math.invlogit(logit_mu)

        # Concentration parameter: per-region source heterogeneity
        kappa_region = pm.HalfNormal("kappa_region", sigma=50, shape=data["n_regions"])
        kappa_r = kappa_region[data["region_id"]]

        alpha_beta = mu_obs * kappa_r
        beta_beta = (1 - mu_obs) * kappa_r

        p_source = pm.Beta("p_source", alpha=alpha_beta, beta=beta_beta, shape=data["n_obs"])

        # Likelihood
        pm.Binomial("obs", n=data["n"], p=p_source, observed=data["k"])

        # Derived: marginal country and region prevalence (no time)
        pm.Deterministic("prevalence_country", pm.math.invlogit(alpha_country))
        pm.Deterministic("prevalence_region", pm.math.invlogit(mu_region))

    return model


def run_model(data, time_mode=None, samples=2000, chains=4, target_accept=0.9):
    """Build and sample from the model."""
    model = build_model(data, time_mode=time_mode)
    with model:
        trace = pm.sample(
            draws=samples,
            chains=chains,
            target_accept=target_accept,
            random_seed=42,
            progressbar=True,
        )
    return model, trace


def summarize_results(trace, data, category, time_mode=None):
    """Extract country, region, and time-level prevalence estimates."""
    prev_country = az.summary(trace, var_names=["prevalence_country"], hdi_prob=0.95)
    prev_country.index = data["countries"]
    prev_country["region"] = [
        data["regions"][data["country_region"][i]]
        for i in range(data["n_countries"])
    ]

    prev_region = az.summary(trace, var_names=["prevalence_region"], hdi_prob=0.95)
    prev_region.index = data["regions"]

    prev_time = None
    prev_region_time = None
    if time_mode:
        prev_time = az.summary(trace, var_names=["prevalence_time"], hdi_prob=0.95)
        prev_time.index = data["time_periods"]

        prev_region_time = az.summary(trace, var_names=["prevalence_region_time"], hdi_prob=0.95)
        idx = pd.MultiIndex.from_product(
            [data["regions"], data["time_periods"]], names=["region", "time_period"]
        )
        prev_region_time.index = idx

    return prev_country, prev_region, prev_time, prev_region_time


def main():
    parser = argparse.ArgumentParser(description="Estimate country-level prevalence")
    parser.add_argument("--input", required=True, help="Classifications JSONL path")
    parser.add_argument("--category", default="all",
                        choices=["all", "climate_change", "health", "health_effects_of_climate_change"],
                        help="Which category to model (default: all three)")
    parser.add_argument("--time", choices=["yearly", "quarterly"], default=None,
                        help="Add time effects (yearly or quarterly)")
    parser.add_argument("--samples", type=int, default=2000, help="MCMC samples per chain")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading classifications...")
    df = load_classifications(args.input)
    print(f"  {len(df):,} classified articles")

    print("Loading source metadata...")
    meta = load_source_metadata(DB_PATH)
    df = df.merge(meta[["id", "source", "country", "published_date"]], on="id", how="left", suffixes=("", "_db"))
    if "country_db" in df.columns:
        df["country"] = df["country"].fillna(df["country_db"])
    if "source_db" in df.columns:
        df["source"] = df["source"].fillna(df["source_db"])
    if "published_date_db" in df.columns:
        df["published_date"] = df["published_date"].fillna(df["published_date_db"])

    print("Loading region mapping...")
    country_to_region = load_region_map(REGION_MAP_PATH)

    categories = ["climate_change", "health", "health_effects_of_climate_change"] if args.category == "all" else [args.category]

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        data = prepare_data(df, country_to_region, category, time_mode=args.time)
        print(f"  Articles: {data['n_articles']:,}")
        print(f"  Regions: {data['n_regions']}, Countries: {data['n_countries']}, Observations: {data['n_obs']}")
        if args.time:
            print(f"  Time periods ({args.time}): {data['n_times']} ({data['time_periods'][0]} to {data['time_periods'][-1]})")
        print(f"  Prevalence (raw): {data['raw_prevalence']:.4f}")

        t0 = time.time()
        model, trace = run_model(data, time_mode=args.time, samples=args.samples, chains=args.chains)
        elapsed = time.time() - t0
        print(f"  Sampling done in {elapsed/60:.1f}m")

        prev_country, prev_region, prev_time, prev_region_time = summarize_results(
            trace, data, category, time_mode=args.time
        )

        # Save
        suffix = f"_{args.time}" if args.time else ""
        prev_country.to_csv(OUTPUT_DIR / f"prevalence_country_{category}{suffix}.csv")
        prev_region.to_csv(OUTPUT_DIR / f"prevalence_region_{category}{suffix}.csv")
        az.to_netcdf(trace, OUTPUT_DIR / f"trace_{category}{suffix}.nc")

        if prev_time is not None:
            prev_time.to_csv(OUTPUT_DIR / f"prevalence_time_{category}{suffix}.csv")
        if prev_region_time is not None:
            prev_region_time.to_csv(OUTPUT_DIR / f"prevalence_region_time_{category}{suffix}.csv")

        print(f"\n  Region estimates:")
        for region in prev_region.index:
            row = prev_region.loc[region]
            print(f"    {region:<35} {row['mean']:.4f} [{row['hdi_2.5%']:.4f}, {row['hdi_97.5%']:.4f}]")

        if prev_time is not None:
            print(f"\n  Global trend ({args.time}):")
            for tp in prev_time.index:
                row = prev_time.loc[tp]
                print(f"    {tp:<10} {row['mean']:.4f} [{row['hdi_2.5%']:.4f}, {row['hdi_97.5%']:.4f}]")

        print(f"\n  Top 10 countries:")
        top = prev_country.sort_values("mean", ascending=False).head(10)
        for country in top.index:
            row = top.loc[country]
            print(f"    {country:<25} {row['mean']:.4f} [{row['hdi_2.5%']:.4f}, {row['hdi_97.5%']:.4f}]  ({row['region']})")

        print(f"\n  Saved to {OUTPUT_DIR}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
