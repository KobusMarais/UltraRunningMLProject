"""
ML Training Pipeline from Features File

This module loads pre-processed features from final_features.parquet
and runs the ML training pipeline. Useful for iterating on ML code
without re-running expensive data processing.
"""

import polars as pl
from pathlib import Path


def run_ml_from_features(features_file: str = "data/processed/final_features.parquet"):
    """
    Load processed features from parquet file and run ML training pipeline.

    This allows you to iterate on ML training code without re-running
    the complete data processing pipeline.

    Args:
        features_file (str): Path to the processed features parquet file

    Returns:
        dict: ML results containing model, predictions, and evaluation metrics
    """

    print("=" * 70)
    print("ML TRAINING FROM PROCESSED FEATURES")
    print("=" * 70)
    print(f"Loading features from: {features_file}")

    # Check if file exists
    if not Path(features_file).exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")

    # Load the processed features
    print("\n📂 Loading processed features...")
    try:
        processed_data = pl.read_parquet(features_file)
        print(f"   Loaded {len(processed_data)} rows with {len(processed_data.columns)} columns")
    except Exception as e:
        raise RuntimeError(f"Failed to load features file: {e}")

    # Import and run the ML pipeline
    from src.pipeline_ml import run_ml_pipeline

    print("\n🤖 Starting ML training pipeline...")
    results = run_ml_pipeline(processed_data)

    print("\n" + "=" * 70)
    print("🎉 ML TRAINING FROM FEATURES COMPLETED!")
    print("=" * 70)

    return results


def inspect_features_file(features_file: str = "data/processed/final_features.parquet"):
    """
    Inspect the contents of a features parquet file.

    Args:
        features_file (str): Path to the features file to inspect
    """

    print(f"Inspecting features file: {features_file}")

    if not Path(features_file).exists():
        print(f"❌ File not found: {features_file}")
        return

    try:
        df = pl.read_parquet(features_file)
        print(f"✅ Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        print("\n📊 Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")

        print("\n📈 Data types:")
        for col in df.columns[:10]:  # Show first 10 columns
            dtype = str(df[col].dtype)
            null_count = df[col].null_count()
            print(f"  {col:<25} {dtype:>12} (nulls: {null_count})")

        if len(df.columns) > 10:
            print(f"    ... and {len(df.columns) - 10} more columns")

        print(f"\n🔢 Sample values (first 5 rows, first 5 columns):")
        sample = df.head(5).select(df.columns[:5])
        print(sample)

    except Exception as e:
        print(f"❌ Error inspecting file: {e}")


if __name__ == "__main__":
    # Example usage
    print("ML Training from Features - Module loaded successfully")
    print("\nUsage examples:")
    print("python -c \"from src.pipeline_ml_from_features import run_ml_from_features; run_ml_from_features()\"")
    print("python -c \"from src.pipeline_ml_from_features import inspect_features_file; inspect_features_file()\"")
    print("\nAvailable functions:")
    print("- run_ml_from_features(): Train ML model from final_features.parquet")
    print("- inspect_features_file(): Inspect contents of features file")
