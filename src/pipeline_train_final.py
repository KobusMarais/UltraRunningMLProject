"""
Full Training Pipeline

This module contains the full training pipeline that trains a final production model
on the complete training set and evaluates it on the test set (Western States 2022).
This should be run after satisfactory CV performance is achieved.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# Import ML-related modules
from src.data.split import split_train_test
from src.models.train import train_evaluate_lgbm
from src.models.pipeline_utils import prepare_data_for_modeling, save_model_results
from src.visualization.model_analysis import generate_model_analysis_charts


def run_final_training_pipeline(processed_data: pl.DataFrame, output_dir: str = "training_results"):
    """
    Run the full training pipeline: train final model and evaluate on test set.

    This pipeline trains a production-ready model on the complete training dataset
    and evaluates it on the held-out test set (Western States 2022). Should be run
    after CV evaluation shows satisfactory performance.

    Args:
        processed_data (pl.DataFrame): Processed dataset from data pipeline
        output_dir (str): Directory to save model and results

    Returns:
        dict: Training results containing model, predictions, and metrics
    """

    print("=" * 60)
    print("FULL MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Train/test split
    print("\n🔄 PHASE 1: TRAIN/TEST SPLIT")
    print("-" * 40)
    df_train, df_test, feature_cols = split_train_test(processed_data)
    print(f"   Training set: {df_train.shape[0]} samples")
    print(f"   Test set: {df_test.shape[0]} samples")
    print(f"   Features: {len(feature_cols)}")

    # 2. Data preparation
    print("\n🛠️  PHASE 2: DATA PREPARATION")
    print("-" * 40)
    X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_modeling(df_train, df_test, feature_cols)
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")

    # 3. Train final model
    print("\n🤖 PHASE 3: FINAL MODEL TRAINING")
    print("-" * 40)
    model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test)

    # 4. Generate analysis charts
    print("\n📊 PHASE 4: MODEL ANALYSIS CHARTS")
    print("-" * 40)
    from datetime import datetime
    import subprocess
    # Generate run_id similar to save_cv_results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd='.')
        git_commit = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        git_commit = "unknown"
    run_id = f"{timestamp}_final_training_{git_commit}"

    chart_results = generate_model_analysis_charts(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=X_train.columns.tolist(),
        run_id=run_id
    )

    # 5. Save model and results
    print("\n💾 PHASE 5: SAVE MODEL AND RESULTS")
    print("-" * 40)
    save_model_results(model, X_train, y_test, y_pred, feature_cols, output_dir)

    print("\n" + "=" * 60)
    print("🎉 FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Return comprehensive results
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_cols': feature_cols
    }

    return results


def load_processed_data_and_run_final_training(csv_path: str = None, processed_df: pl.DataFrame = None,
                                               output_dir: str = "training_results"):
    """
    Load processed data and run full training pipeline.

    Args:
        csv_path (str, optional): Path to raw CSV (will run data pipeline first)
        processed_df (pl.DataFrame, optional): Already processed dataframe
        output_dir (str): Directory to save results

    Returns:
        dict: Training results
    """

    if processed_df is not None:
        # Use provided processed data
        return run_final_training_pipeline(processed_df, output_dir)
    elif csv_path is not None:
        # Run data pipeline first, then training
        from src.pipeline_data import run_polars_pipeline_with_collection
        processed_data = run_polars_pipeline_with_collection(csv_path)
        return run_final_training_pipeline(processed_data, output_dir)
    else:
        raise ValueError("Must provide either csv_path or processed_df")


if __name__ == "__main__":
    # Example usage
    print("Full training pipeline module loaded successfully")
    print("Use: run_final_training_pipeline(processed_data) to train final model")
    print("Use: load_processed_data_and_run_final_training(csv_path='data.csv') to run full workflow")
