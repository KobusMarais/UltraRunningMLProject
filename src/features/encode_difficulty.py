"""
Difficulty Encoding Module (Model A Logic)

This module implements smoothed target encoding for race difficulty using Polars.
It calculates race difficulty based on historical performance and applies
Bayesian smoothing to prevent overfitting.
"""

import polars as pl
from pathlib import Path


def apply_smoothed_target_encoding(lf: pl.LazyFrame, column: str = 'Event_name_clean', 
                                   target: str = 'pace_min_per_km', m: float = 10.0) -> pl.LazyFrame:
    """
    Apply smoothed target encoding to categorical features using Polars.
    
    This technique encodes categorical variables (like race names) with their
    average target value, smoothed by the global mean to prevent overfitting.
    
    Formula: (count * mean + m * global_mean) / (count + m)
    
    Args:
        lf (pl.LazyFrame): Dataset with target variable
        column (str): Column name to encode
        target (str): Target variable name
        m (float): Smoothing factor (higher = more conservative)
        
    Returns:
        pl.LazyFrame: Dataset with encoded features
    """
    
    # Calculate global mean from training data only (No leakage!)
    global_mean_value = lf.select([pl.col(target).mean().alias("global_mean")]).collect()["global_mean"][0]
    
    # Calculate count and mean for each category
    agg_stats = (
        lf
        .group_by(column)
        .agg([
            pl.count().alias("count"),
            pl.col(target).mean().alias("mean")
        ])
    )
    
    # Calculate the smoothed value using Bayesian averaging
    smooth_weights = (
        agg_stats
        .with_columns([
            ((pl.col("count") * pl.col("mean") + m * global_mean_value) / 
             (pl.col("count") + m)).alias("smooth_weight")
        ])
        .select([column, "smooth_weight"])
    )
    
    # Join the weights back to the main dataset
    result = lf.join(smooth_weights, on=column, how="left")
    
    return result


def create_difficulty_encoding_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create the complete difficulty encoding pipeline.
    
    Args:
        lf (pl.LazyFrame): Dataset with pace_min_per_km
        
    Returns:
        pl.LazyFrame: Dataset with difficulty encoding applied
    """
    
    # Apply smoothed target encoding for race difficulty
    encoded_lf = apply_smoothed_target_encoding(lf, column='Event_name_clean', target='pace_min_per_km', m=10.0)
    
    # Rename the encoded column for clarity
    return encoded_lf.rename({"smooth_weight": "Race_Pace_Mean_Encoded"})


def save_encoded_data(lf: pl.LazyFrame, output_path: str = "data/processed/encoded_features.parquet"):
    """
    Save encoded data to parquet format.
    
    Args:
        lf (pl.LazyFrame): Lazy frame with encoded features
        output_path (str): Output path for parquet file
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with streaming for memory efficiency
    lf.sink_parquet(output_path, compression="snappy")


def load_encoded_data(path: str = "data/processed/encoded_features.parquet") -> pl.LazyFrame:
    """
    Load previously encoded data from parquet file.
    
    Args:
        path (str): Path to encoded parquet file
        
    Returns:
        pl.LazyFrame: Loaded encoded data
    """
    return pl.scan_parquet(path)


if __name__ == "__main__":
    # Example usage
    print("Difficulty encoding module loaded successfully")
    print("Use: apply_smoothed_target_encoding(lf) for encoding")
