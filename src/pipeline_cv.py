"""
Cross-Validation Only Pipeline

This module contains the CV-only training pipeline that evaluates model performance
on training data via cross-validation without training a final model or evaluating
on test data. This allows for iterative model refinement.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# Import ML-related modules
from src.data.split import split_train_test
from src.models.train import train_with_cross_validation
from src.models.pipeline_utils import save_cv_results


def run_cv_pipeline(processed_data: pl.DataFrame, cv_folds: int = 5, output_dir: str = "training_results"):
    """
    Run cross-validation evaluation pipeline on training data only.

    This pipeline performs time-series cross-validation on the training set to assess
    model performance without training a final model or evaluating on test data.
    Useful for iterative model refinement and hyperparameter tuning.

    Args:
        processed_data (pl.DataFrame): Processed dataset from data pipeline
        cv_folds (int): Number of CV folds to use
        output_dir (str): Directory to save CV results

    Returns:
        dict: Cross-validation results
    """

    print("=" * 60)
    print("CROSS-VALIDATION EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Train/test split (but we'll only use training data for CV)
    print("\n🔄 PHASE 1: TRAIN/TEST SPLIT")
    print("-" * 40)
    df_train, df_test, feature_cols = split_train_test(processed_data)
    print(f"   Training set: {df_train.shape[0]} samples (used for CV)")
    print(f"   Test set: {df_test.shape[0]} samples (held out)")
    print(f"   Features: {len(feature_cols)}")

    # 2. Cross-validation evaluation on training set
    print("\n🔄 PHASE 2: CROSS-VALIDATION EVALUATION")
    print("-" * 40)
    print(f"Performing {cv_folds}-fold time-series cross-validation on training set...")

    # Convert to pandas for CV function
    df_train_pd = df_train.to_pandas()

    cv_results = train_with_cross_validation(df_train_pd, feature_cols, cv_folds=cv_folds)
    print(f"Average CV MAE:  {cv_results['avg_mae']:.4f} min/km")
    print(f"Average CV RMSE: {cv_results['avg_rmse']:.4f} min/km")

    # 3. Save CV results
    print("\n💾 PHASE 3: SAVE CV RESULTS")
    print("-" * 40)
    save_cv_results(cv_results, output_dir)

    print("\n" + "=" * 60)
    print("🎉 CV EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Use these CV metrics to refine your model.")
    print("When satisfied with performance, run the full training pipeline.")

    return cv_results


def load_processed_data_and_run_cv(csv_path: str = None, processed_df: pl.DataFrame = None,
                                   cv_folds: int = 5, output_dir: str = "training_results"):
    """
    Load processed data and run CV evaluation pipeline.

    Args:
        csv_path (str, optional): Path to raw CSV (will run data pipeline first)
        processed_df (pl.DataFrame, optional): Already processed dataframe
        cv_folds (int): Number of CV folds
        output_dir (str): Directory to save results

    Returns:
        dict: CV results
    """

    if processed_df is not None:
        # Use provided processed data
        return run_cv_pipeline(processed_df, cv_folds, output_dir)
    elif csv_path is not None:
        # Run data pipeline first, then CV
        from src.pipeline_data import run_polars_pipeline_with_collection
        processed_data = run_polars_pipeline_with_collection(csv_path)
        return run_cv_pipeline(processed_data, cv_folds, output_dir)
    else:
        raise ValueError("Must provide either csv_path or processed_df")


if __name__ == "__main__":
    # Example usage
    print("CV evaluation pipeline module loaded successfully")
    print("Use: run_cv_pipeline(processed_data) to evaluate model via CV")
    print("Use: load_processed_data_and_run_cv(csv_path='data.csv') to run full CV workflow")