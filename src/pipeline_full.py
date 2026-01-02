"""
Full Ultramarathon ML Pipeline Module

This module combines the data processing and ML training pipelines
into a complete end-to-end workflow.
"""

import polars as pl
from pathlib import Path

# Import the separate pipeline modules
from src.pipeline_data import run_polars_pipeline_with_collection
from src.pipeline_ml import run_ml_pipeline
from src.pipeline_cv import run_cv_pipeline
from src.pipeline_train_final import run_final_training_pipeline


##TODO Rename as application entrypoint, not pipeline

def run_full_ml_pipeline(csv_path: str):
    """
    Complete end-to-end ML pipeline for ultramarathon pace prediction.

    This function runs the entire workflow:
    1. Data processing (load, clean, feature engineering)
    2. ML training and evaluation

    Args:
        csv_path (str): Path to the raw CSV file

    Returns:
        dict: Results containing processed data and model results
    """

    print("=" * 80)
    print("COMPLETE ULTRAMARATHON PACE PREDICTION PIPELINE")
    print("=" * 80)

    # 1. Run data processing pipeline
    print("\n📊 PHASE 1: DATA PROCESSING")
    print("-" * 40)
    final_df = run_polars_pipeline_with_collection(csv_path)

    # 2. Run ML training pipeline
    print("\n🤖 PHASE 2: ML TRAINING")
    print("-" * 40)
    ml_results = run_ml_pipeline(final_df)

    print("\n" + "=" * 80)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Results saved to training_results/ directory")

    # Combine results
    full_results = {
        'processed_data': final_df,
        **ml_results
    }

    return full_results


def run_data_only_pipeline(csv_path: str):
    """
    Run only the data processing pipeline.

    Args:
        csv_path (str): Path to the raw CSV file

    Returns:
        pl.DataFrame: Processed dataset ready for modeling
    """
    return run_polars_pipeline_with_collection(csv_path)


def run_cv_only_pipeline(csv_path: str, cv_folds: int = 5, output_dir: str = "training_results"):
    """
    Run cross-validation evaluation pipeline only.

    This pipeline processes data and performs CV evaluation on training data
    without training a final model. Useful for iterative model refinement.

    Args:
        csv_path (str): Path to the raw CSV file
        cv_folds (int): Number of CV folds
        output_dir (str): Directory to save CV results

    Returns:
        dict: CV results
    """

    print("=" * 80)
    print("CROSS-VALIDATION EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Run data processing pipeline
    print("\n📊 PHASE 1: DATA PROCESSING")
    print("-" * 40)
    final_df = run_polars_pipeline_with_collection(csv_path)

    # 2. Run CV evaluation pipeline
    print("\n🔄 PHASE 2: CV EVALUATION")
    print("-" * 40)
    cv_results = run_cv_pipeline(final_df, cv_folds, output_dir)

    print("\n" + "=" * 80)
    print("🎉 CV PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"CV results saved to {output_dir}/ directory")

    return cv_results


def run_final_training_only_pipeline(csv_path: str, output_dir: str = "training_results"):
    """
    Run final model training pipeline only.

    This pipeline processes data, trains a final model on the complete training set,
    and evaluates on the test set. Should be run after satisfactory CV performance.

    Args:
        csv_path (str): Path to the raw CSV file
        output_dir (str): Directory to save model and results

    Returns:
        dict: Training results
    """

    print("=" * 80)
    print("FINAL MODEL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Run data processing pipeline
    print("\n📊 PHASE 1: DATA PROCESSING")
    print("-" * 40)
    final_df = run_polars_pipeline_with_collection(csv_path)

    # 2. Run final training pipeline
    print("\n🤖 PHASE 2: FINAL TRAINING")
    print("-" * 40)
    training_results = run_final_training_pipeline(final_df, output_dir)

    print("\n" + "=" * 80)
    print("🎉 FINAL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Model and results saved to {output_dir}/ directory")

    return training_results


if __name__ == "__main__":
    # Example usage
    print("Full pipeline module loaded successfully")
    print("Use: run_full_ml_pipeline('path/to/data.csv') to run complete pipeline")
    print("Use: run_data_only_pipeline('path/to/data.csv') for data processing only")
    print("Use: run_ml_only_pipeline(processed_data) for combined ML training + CV")
    print("Use: run_cv_only_pipeline('path/to/data.csv') for CV evaluation only")
    print("Use: run_final_training_only_pipeline('path/to/data.csv') for final training only")
