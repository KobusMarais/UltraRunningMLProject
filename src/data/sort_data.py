"""
Chronological Sorting Module

This module contains functions to sort ultramarathon data chronologically
using Polars for out-of-core execution and accurate window functions.
"""

import polars as pl
from pathlib import Path


def sort_chronologically(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Sort data chronologically by Athlete ID and Year of event.
    
    This ensures accurate window functions in feature engineering by
    guaranteeing proper chronological ordering for each athlete.
    
    Args:
        lf (pl.LazyFrame): Lazy frame to sort
        
    Returns:
        pl.LazyFrame: Chronologically sorted lazy frame
    """
    return lf.sort(["Athlete ID", "Year of event", "Event name"])


def save_sorted_data(lf: pl.LazyFrame, output_path: str = "data/interim/sorted.parquet"):
    """
    Save sorted data to parquet format using streaming mode.
    
    Args:
        lf (pl.LazyFrame): Sorted lazy frame
        output_path (str): Output path for parquet file
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with streaming for memory efficiency
    lf.sink_parquet(output_path, compression="snappy")


def load_sorted_data(path: str = "data/interim/sorted.parquet") -> pl.LazyFrame:
    """
    Load previously sorted data from parquet file.
    
    Args:
        path (str): Path to sorted parquet file
        
    Returns:
        pl.LazyFrame: Loaded sorted data
    """
    return pl.scan_parquet(path)


if __name__ == "__main__":
    # Example usage
    print("Chronological sorting module loaded successfully")
    print("Use: sort_chronologically(lf) to sort your dataset")
