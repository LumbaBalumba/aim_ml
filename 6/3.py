#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deduplicate person records using Splink (DuckDB backend).

Input datasets are parquet files with (at least) these columns:
    surname, name, fathername, birthdate, gender, passport, inn

All are strings (recommended pre-cleaned in step 1/2).

Outputs:
    - clusters parquet for each dataset
    - deduped parquet for each dataset

Strategy:
    - Create/ensure a unique_id column.
    - Build Splink settings with tight multi-rule blocking.
    - Use NameComparison + DateOfBirthComparison + ExactMatch comparisons.
    - Unsupervised training (estimate u + EM).
    - Predict pairwise matches.
    - Cluster at a high threshold (default 0.95).
    - Keep one canonical record per cluster.

This is designed to be fast on ~2M rows using blocking.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on


EXPECTED_COLS = ["surname", "name", "fathername", "birthdate", "gender", "passport", "inn"]


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    return df


def ensure_string_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("")
        else:
            # If a column is missing, create it to keep the model consistent
            df[c] = ""
    return df


def ensure_unique_id(df: pd.DataFrame, id_col: str = "unique_id") -> pd.DataFrame:
    if id_col in df.columns:
        # Ensure it's string-like for Splink
        df[id_col] = df[id_col].astype("string")
        # Fill missing ids if any
        missing = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
        if missing.any():
            # deterministic fill
            df.loc[missing, id_col] = (
                pd.Series(range(len(df)), index=df.index)[missing].astype(str)
            )
    else:
        df[id_col] = pd.Series(range(len(df)), index=df.index).astype("string")
    return df


def build_settings(
    link_type: str = "dedupe_only",
    max_iterations: int = 20,
    em_convergence: float = 0.001,
):
    """
    Settings tuned for person dedupe with strong identifiers.
    Uses tight blocking to keep runtime low on millions of rows.
    """

    settings = SettingsCreator(
        link_type=link_type,
        blocking_rules_to_generate_predictions=[
            # Strong unique-ish identifiers first
            block_on("passport"),
            block_on("inn"),
            # Core identity blocks
            block_on("surname", "name", "birthdate"),
            block_on("surname", "birthdate"),
            block_on("name", "fathername", "birthdate"),
            block_on("surname", "name", "fathername"),
        ],
        comparisons=[
            # Names with TF adjustments help downweight very common names/surnames
            cl.NameComparison("surname").configure(term_frequency_adjustments=True),
            cl.NameComparison("name").configure(term_frequency_adjustments=True),
            cl.NameComparison("fathername"),
            # DOB comparison (string input is common in admin data)
            cl.DateOfBirthComparison(
                "birthdate",
                input_is_string=True,
                datetime_metrics=["day", "month", "year"],
                datetime_thresholds=[1, 1, 1],
            ),
            # High-value exact fields
            cl.ExactMatch("passport"),
            cl.ExactMatch("inn"),
            cl.ExactMatch("gender"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
        max_iterations=max_iterations,
        em_convergence=em_convergence,
    )

    return settings


def train_predict_cluster(
    df: pd.DataFrame,
    threshold: float,
    max_pairs_u: int,
):
    db_api = DuckDBAPI()

    settings = build_settings()
    linker = Linker(df, settings, db_api=db_api)

    # Estimate u using random sampling
    linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs_u)

    # EM training with progressively strong rules
    # These are chosen to be both informative and safe for large data
    em_rules = [
        block_on("surname", "name", "birthdate"),
        block_on("passport"),
        block_on("inn"),
        block_on("surname", "birthdate"),
    ]

    for rule in em_rules:
        linker.training.estimate_parameters_using_expectation_maximisation(rule)

    # Predict pairwise matches using the blocking rules in settings
    pairwise = linker.inference.predict()

    # Cluster
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        pairwise, threshold
    )

    # Convert to pandas
    clusters_df = clusters.as_pandas_dataframe()

    return clusters_df


def build_deduped_df(
    original: pd.DataFrame,
    clusters_df: pd.DataFrame,
    id_col: str = "unique_id",
):
    """
    Keep one canonical record per cluster.
    Strategy: take smallest unique_id lexicographically after casting to int when possible.
    """

    # Some Splink outputs call it "cluster_id"
    if "cluster_id" not in clusters_df.columns:
        raise RuntimeError("Expected 'cluster_id' in clusters output.")

    # Attach cluster ids to original
    merged = original.merge(clusters_df[[id_col, "cluster_id"]], on=id_col, how="left")

    # For singleton records that didn't appear in clusters table,
    # create a unique cluster id = unique_id
    merged["cluster_id"] = merged["cluster_id"].fillna(merged[id_col].astype(str))

    # Choose canonical per cluster
    # Try numeric sort of unique_id; fallback to string
    def safe_int(x):
        try:
            return int(str(x))
        except Exception:
            return None

    u_int = merged[id_col].map(safe_int)
    merged["_uid_int"] = u_int

    # If most ids are numeric, use numeric ordering
    if merged["_uid_int"].notna().mean() > 0.9:
        merged = merged.sort_values(["cluster_id", "_uid_int"], ascending=True)
    else:
        merged = merged.sort_values(["cluster_id", id_col], ascending=True)

    deduped = merged.groupby("cluster_id", as_index=False).head(1).copy()
    deduped = deduped.drop(columns=["_uid_int"])

    return deduped


def process_one_dataset(
    input_path: str,
    clusters_out: str,
    dedup_out: str,
    threshold: float,
    max_pairs_u: int,
):
    df = load_parquet(input_path)
    df = ensure_string_cols(df)
    df = ensure_unique_id(df, id_col="unique_id")

    clusters_df = train_predict_cluster(
        df=df,
        threshold=threshold,
        max_pairs_u=max_pairs_u,
    )

    # Save clusters
    Path(clusters_out).parent.mkdir(parents=True, exist_ok=True)
    clusters_df.to_parquet(clusters_out, index=False)

    # Build deduped
    deduped = build_deduped_df(df, clusters_df, id_col="unique_id")

    # Save deduped
    Path(dedup_out).parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(dedup_out, index=False)

    return len(df), len(deduped), len(clusters_df)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate person datasets using Splink (DuckDB backend)."
    )
    parser.add_argument("--df1", required=True, help="Path to df1 parquet")
    parser.add_argument("--df2", required=True, help="Path to df2 parquet")

    parser.add_argument("--out1-clusters", required=True, help="Output parquet for df1 clusters")
    parser.add_argument("--out2-clusters", required=True, help="Output parquet for df2 clusters")

    parser.add_argument("--out1-dedup", required=True, help="Output parquet for deduped df1")
    parser.add_argument("--out2-dedup", required=True, help="Output parquet for deduped df2")

    parser.add_argument("--threshold", type=float, default=0.95, help="Clustering threshold")
    parser.add_argument(
        "--max-pairs-u",
        type=int,
        default=5_000_000,
        help="Max random pairs for u estimation",
    )

    args = parser.parse_args()

    print("[INFO] Deduplicating df1...")
    n_in1, n_out1, n_cluster_rows1 = process_one_dataset(
        input_path=args.df1,
        clusters_out=args.out1_clusters,
        dedup_out=args.out1_dedup,
        threshold=args.threshold,
        max_pairs_u=args.max_pairs_u,
    )
    print(f"[INFO] df1: in={n_in1:,}  deduped={n_out1:,}  cluster_rows={n_cluster_rows1:,}")

    print("[INFO] Deduplicating df2...")
    n_in2, n_out2, n_cluster_rows2 = process_one_dataset(
        input_path=args.df2,
        clusters_out=args.out2_clusters,
        dedup_out=args.out2_dedup,
        threshold=args.threshold,
        max_pairs_u=args.max_pairs_u,
    )
    print(f"[INFO] df2: in={n_in2:,}  deduped={n_out2:,}  cluster_rows={n_cluster_rows2:,}")

    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
