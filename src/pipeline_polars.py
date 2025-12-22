"""
Polars-Based Pipeline Module

This module contains the main pipeline function that orchestrates the entire
ultramarathon pace prediction workflow using Polars for memory efficiency.
It combines all the individual Polars modules into a complete end-to-end pipeline.
"""

import polars as pl
from pathlib import Path

# Import all the Polars modules we created
from src.data.load import load_raw_data, save_interim_data
from src.data.clean import clean_data
from src.data.sort_data import sort_chronologically, save_sorted_data
from src.features.build_features import engineer_features, save_features
from src.features.encode_difficulty import create_difficulty_encoding_pipeline, save_encoded_data


def run_polars_pipeline(csv_path: str):
    """
    Full Polars-based ML pipeline for ultramarathon pace prediction.
    
    This function orchestrates the entire machine learning workflow using Polars:
    1. Load raw data with lazy evaluation
    2. Clean and preprocess data using vectorized operations
    3. Sort chronologically for accurate window functions
    4. Engineer features using Polars window functions
    5. Apply difficulty encoding with smoothed target encoding
    6. Save intermediate results for debugging and inspection
    
    Args:
        csv_path (str): Path to the raw CSV file
        
    Returns:
        pl.LazyFrame: Final dataset with all features engineered and encoded
    """
    
    print("=" * 60)
    print("ULTRAMARATHON PACE PREDICTION PIPELINE (POLARS)")
    print("=" * 60)
    
    # 1. Load raw data with lazy evaluation
    print("\n1. Loading raw data with Polars...")
    lf = load_raw_data(csv_path)
    print(f"   Lazy frame created - schema defined for memory efficiency")

    print("\n2. Sorting chronologically for window functions...")
    sorted_lf = sort_chronologically(lf)
    print(f"   Chronological sorting completed - saving to interim/sorted.parquet")
    save_sorted_data(sorted_lf, "data/interim/sorted.parquet")
    
    
    # 2. Clean data using vectorized operations
    print("\n3. Cleaning data with vectorized operations...")
    cleaned_lf = clean_data(sorted_lf)
    print(f"   Data cleaning completed - saving to interim/cleaned.parquet")
    save_interim_data(cleaned_lf, "data/interim/cleaned.parquet")
    
    # 3. Sort chronologically for accurate window functions

    # 4. Engineer features using Polars window functions
    print("\n4. Engineering features with window functions...")
    features_lf = engineer_features(cleaned_lf)
    print(f"   Feature engineering completed - saving to processed/final_features.parquet")
    save_features(features_lf, "data/processed/final_features.parquet")
    
    # 5. Apply difficulty encoding with smoothed target encoding
    encoded_lf = create_difficulty_encoding_pipeline(features_lf)
    save_encoded_data(encoded_lf, "data/processed/encoded_features.parquet")
    
    print("\n" + "=" * 60)
    print("POLARS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Intermediate files saved:")
    print("  - data/interim/cleaned.parquet")
    print("  - data/interim/sorted.parquet")
    print("  - data/processed/final_features.parquet")
    print("  - data/processed/encoded_features.parquet")
    
    return encoded_lf


def run_polars_pipeline_with_collection(csv_path: str):
    """
    Run the Polars pipeline and collect the final result for further processing.
    
    This version collects the final lazy frame into memory for use with pandas/scikit-learn.
    
    Args:
        csv_path (str): Path to the raw CSV file
        
    Returns:
        pl.DataFrame: Final dataset ready for modeling
    """
    
    # Run the pipeline
    final_lf = run_polars_pipeline(csv_path)
    
    # Collect the final result (this is where memory usage peaks)
    print("\n6. Collecting final dataset into memory...")
    print("   Note: This step loads data into memory for modeling")
    
    try:
        # Use streaming mode for better memory management
        final_df = final_lf.collect(streaming=True)
        print(f"   Collected {len(final_df)} rows with {len(final_df.columns)} columns")
        print(f"   Memory usage optimized with streaming collection")
        
        return final_df
        
    except Exception as e:
        print(f"   Streaming collection failed: {e}")
        print("   Falling back to regular collection...")
        final_df = final_lf.collect()
        print(f"   Collected {len(final_df)} rows with {len(final_df.columns)} columns")
        
        return final_df


def inspect_pipeline_results():
    """
    Inspect the results of the pipeline by loading intermediate files.
    
    Returns:
        dict: Dictionary of lazy frames for inspection
    """
    
    results = {}
    
    # Load intermediate results
    try:
        results['cleaned'] = pl.scan_parquet("data/interim/cleaned.parquet")
        print("✓ Cleaned data loaded")
    except:
        print("✗ Cleaned data not found")
    
    try:
        results['sorted'] = pl.scan_parquet("data/interim/sorted.parquet")
        print("✓ Sorted data loaded")
    except:
        print("✗ Sorted data not found")
    
    try:
        results['features'] = pl.scan_parquet("data/processed/final_features.parquet")
        print("✓ Features data loaded")
    except:
        print("✗ Features data not found")
    
    try:
        results['encoded'] = pl.scan_parquet("data/processed/encoded_features.parquet")
        print("✓ Encoded data loaded")
    except:
        print("✗ Encoded data not found")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Polars pipeline module loaded successfully")
    print("Use: run_polars_pipeline('path/to/data.csv') to run the pipeline")
    print("Use: run_polars_pipeline_with_collection('path/to/data.csv') to get results for modeling")
