"""
Feature Engineering Module

This module contains functions to create features for ultramarathon performance prediction
using Polars window functions for efficient cumulative calculations.
"""

import polars as pl
from pathlib import Path


def engineer_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create cumulative and rolling features for ultramarathon dataset using Polars window functions.

    This function generates features that capture:
    - Athlete progression over time (cumulative stats)
    - Race difficulty indicators
    - Recent performance trends
    - Athlete experience metrics
    - Uses .over() window functions for efficient computation

    Args:
        lf (pl.LazyFrame): Chronologically sorted lazy frame

    Returns:
        pl.LazyFrame: Dataset with engineered features
    """

    # Calculate cumulative features using window functions
    features_lf = (
        lf
        # Cumulative number of races per athlete (excluding current)
        .with_columns(
            [pl.int_range(0, pl.len()).over("Athlete ID").alias("cum_num_races")]
        )
        # Cumulative average pace (excluding current race)
        .with_columns(
            [
                (
                    pl.col("pace_min_per_km").cum_sum().over("Athlete ID")
                    / pl.int_range(1, pl.len() + 1).over("Athlete ID")
                )
                .shift(1)  # Maintains your "previous races only" logic
                .fill_null(-1)  # Handles the very first race
                .alias("cum_avg_pace")
            ]
        )
        # Cumulative best pace (excluding current race)
        .with_columns(
            [
                pl.col("pace_min_per_km")
                .cum_min()
                .over("Athlete ID")
                .shift(1)
                .fill_null(-1)
                .alias("cum_best_pace")
            ]
        )
        # Cumulative distance statistics (excluding current)
        .with_columns(
            [
                pl.col("Event distance_numeric")
                .cum_sum()
                .over("Athlete ID")
                .shift(1)
                .fill_null(-1)
                .alias("cum_total_distance"),
                pl.col("Event distance_numeric")
                .cum_min()
                .over("Athlete ID")
                .shift(1)
                .fill_null(-1)
                .alias("cum_shortest_distance"),
                pl.col("Event distance_numeric")
                .cum_max()
                .over("Athlete ID")
                .shift(1)
                .fill_null(-1)
                .alias("cum_longest_distance"),
            ]
        )
        # Cumulative average distance (handle division by zero)
        .with_columns(
            [
                pl.when(pl.col("cum_num_races") > 0)
                .then(pl.col("cum_total_distance") / pl.col("cum_num_races"))
                .otherwise(0.0)
                .alias("cum_avg_distance")
            ]
        )
        # Cumulative Western States finishes
        .with_columns(
            [
                pl.col("Event name")
                .str.contains("Western States")
                .cum_sum()
                .over("Athlete ID")
                .shift(1)
                .fill_null(-1)
                .cast(pl.Int64)
                .alias("cum_ws_finishes")
            ]
        )
        # Recent average distance (rolling 3 races)
        .with_columns(
            [
                pl.col("Event distance_numeric")
                .rolling_mean(window_size=3, min_samples=1)
                .over("Athlete ID")
                .shift(1)
                .fill_null(-1)
                .alias("recent_avg_distance")
            ]
        )
        # Distance gap from longest
        .with_columns(
            [
                (
                    pl.col("Event distance_numeric") - pl.col("cum_longest_distance")
                ).alias("distance_gap_from_longest")
            ]
        )
        # Athlete age at time of race
        .with_columns(
            [
                (pl.col("Year of event") - pl.col("Athlete year of birth")).alias(
                    "athlete_age"
                )
            ]
        )
    )

    # Ensure consistent data types to prevent upcasting/downcasting conflicts
    features_lf = features_lf.with_columns(
        [
            pl.col("cum_num_races").cast(pl.Int64),
            pl.col("cum_total_distance").cast(pl.Float64),
            pl.col("cum_shortest_distance").cast(pl.Float64),
            pl.col("cum_longest_distance").cast(pl.Float64),
            pl.col("cum_avg_distance").cast(pl.Float64),
            pl.col("cum_avg_pace").cast(pl.Float64),
            pl.col("cum_best_pace").cast(pl.Float64),
            pl.col("recent_avg_distance").cast(pl.Float64),
            pl.col("distance_gap_from_longest").cast(pl.Float64),
            pl.col("athlete_age").cast(pl.Int64),
        ]
    )

    return features_lf


def save_features(
    lf: pl.LazyFrame, output_path: str = "data/processed/final_features.parquet"
):
    """
    Save engineered features to parquet format.

    Args:
        lf (pl.LazyFrame): Lazy frame with engineered features
        output_path (str): Output path for parquet file
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save with streaming for memory efficiency
    lf.sink_parquet(output_path, compression="snappy")


def load_features(path: str = "data/processed/final_features.parquet") -> pl.LazyFrame:
    """
    Load previously engineered features from parquet file.

    Args:
        path (str): Path to features parquet file

    Returns:
        pl.LazyFrame: Loaded features data
    """
    return pl.scan_parquet(path)


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully")
    print("Use: engineer_features(lf) to create features")
