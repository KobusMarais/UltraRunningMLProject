"""
Data Loading Module

This module contains functions to load raw ultramarathon data from CSV files.
Uses Polars for memory-efficient loading of large datasets.
"""

import polars as pl
from pathlib import Path


def load_raw_data(path: str) -> pl.LazyFrame:
    """
    Load the raw ultramarathon dataset from a CSV file using Polars for memory efficiency.
    
    This function uses Polars lazy loading to handle large datasets that may not fit in memory.
    It defines explicit schema for better performance and memory usage.
    
    Args:
        path (str): File path to the CSV file containing ultramarathon data
        
    Returns:
        pl.LazyFrame: Lazy loaded DataFrame for memory-efficient processing
        
    Example:
        >>> lf = load_raw_data("data/raw/ultra_marathons.csv")
        >>> df = lf.collect()  # Only collect when needed
    """
    # Define schema for better memory usage and performance
    schema = {
        "Year of event": pl.Int32,
        "Event dates": pl.Utf8,
        "Event name": pl.Utf8,
        "Event distance/length": pl.Utf8,
        "Event number of finishers": pl.Int32,
        "Athlete performance": pl.Utf8,
        "Athlete club": pl.Utf8,
        "Athlete country": pl.Utf8,
        "Athlete year of birth": pl.Float64,
        "Athlete gender": pl.Utf8,
        "Athlete age category": pl.Utf8,
        "Athlete average speed": pl.Float64,
        "Athlete ID": pl.Int64
    }
    
    # Use lazy loading for memory efficiency
    return pl.scan_csv(
        path,
        schema=schema,
        null_values=["", "NULL", "null", "NA"],
        ignore_errors=True  # Skip problematic rows
    )


def save_interim_data(lf: pl.LazyFrame, output_path: str, streaming: bool = True):
    """
    Save lazy frame to parquet format for efficient storage.
    
    Args:
        lf (pl.LazyFrame): Lazy frame to save
        output_path (str): Path to save the parquet file
        streaming (bool): Use streaming mode for large datasets
    """
    lf.sink_parquet(output_path, compression="snappy")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        lf = load_raw_data(file_path)
        print(f"Loaded lazy frame from {file_path}")
        print("Use .collect() to materialize the data when needed")
    else:
        print("Usage: python load.py <path_to_csv>")
