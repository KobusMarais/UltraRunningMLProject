"""
ML Training Pipeline Module

This module contains the machine learning training and evaluation functions.
It assumes data has already been processed and is ready for modeling.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# Import ML-related modules
from src.data.split import split_train_test
from src.models.train import train_evaluate_lgbm, get_feature_importance, train_with_cross_validation
from src.models.pipeline_utils import prepare_data_for_modeling, save_cv_results, save_model_results
from src.evaluation.metrics import print_pace_metrics


def run_ml_pipeline(processed_data: pl.DataFrame):
    """
    Run the ML training pipeline on already processed data.

    This function assumes the data has been processed by the data pipeline
    and is ready for train/test splitting and model training.

    Args:
        processed_data (pl.DataFrame): Processed dataset from data pipeline

    Returns:
        dict: Results containing model, predictions, and evaluation metrics
    """

    print("=" * 60)
    print("ULTRAMARATHON ML TRAINING PIPELINE")
    print("=" * 60)

    # 1. Train/test split
    print("\n🔄 PHASE 1: TRAIN/TEST SPLIT")
    print("-" * 40)
    df_train, df_test, feature_cols = split_train_test(processed_data)
    print(f"   Training set: {df_train.shape[0]} samples")
    print(f"   Test set: {df_test.shape[0]} samples")
    print(f"   Features: {len(feature_cols)}")

    # 2. Prepare data for modeling
    print("\n🛠️  PHASE 2: DATA PREPARATION")
    print("-" * 40)
    # Convert to pandas for LightGBM (it doesn't support Polars directly)
    df_train_pd = df_train.to_pandas()
    df_test_pd = df_test.to_pandas()

    # 2.5. Cross-validation evaluation on training set (before encoding to prevent leakage)
    print("\n🔄 PHASE 2.5: CROSS-VALIDATION EVALUATION")
    print("-" * 40)
    print("Performing 5-fold cross-validation on training set for model refinement...")
    cv_results = train_with_cross_validation(df_train_pd, feature_cols, cv_folds=5)
    print(f"Average CV MAE:  {cv_results['avg_mae']:.4f}")
    print(f"Average CV RMSE: {cv_results['avg_rmse']:.4f}")
    print("Use these metrics to assess model performance on unseen training folds.")

    # Apply race difficulty encoding using pandas
    df_train_pd, df_test_pd = apply_smoothed_target_encoding(df_train_pd, df_test_pd)

    # Add encoded feature to feature columns
    feature_cols.append('Race_Pace_Mean_Encoded')

    # Select features
    X_train = df_train_pd[feature_cols].copy()
    X_test = df_test_pd[feature_cols].copy()

    # Define target
    y_train = df_train_pd['pace_min_per_km']
    y_test = df_test_pd['pace_min_per_km']

    # Handle categorical features for LightGBM
    categorical_features = ['Athlete gender']

    # Convert categorical features to category dtype and then to codes for LightGBM
    for cat_col in categorical_features:
        if cat_col in X_train.columns:
            # Combine train and test to ensure consistent encoding
            combined = pd.concat([X_train[[cat_col]], X_test[[cat_col]]], ignore_index=True)
            combined[cat_col] = combined[cat_col].astype('category')

            # Apply the same categories to train and test
            X_train[cat_col] = combined.iloc[:len(X_train)][cat_col]
            X_test[cat_col] = combined.iloc[len(X_train):][cat_col]

            # Convert to categorical codes (integers) for LightGBM
            X_train[cat_col] = X_train[cat_col].cat.codes
            X_test[cat_col] = X_test[cat_col].cat.codes

    # Clean infinities and missing values
    X_train.replace([float('inf'), -float('inf')], -1, inplace=True)
    X_test.replace([float('inf'), -float('inf')], -1, inplace=True)
    X_train.fillna(-1, inplace=True)
    X_test.fillna(-1, inplace=True)

    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")

    # Optional: Hyperparameter tuning (commented out by default for speed)
    # print("\n⚙️  PHASE 2.6: HYPERPARAMETER TUNING")
    # print("-" * 40)
    # print("Tuning hyperparameters using cross-validation...")
    # best_params = hyperparameter_tuning(X_train, y_train)
    # print(f"Best parameters found: {best_params}")
    # # Update params for final training (you can modify train_evaluate_lgbm to accept custom params)

    # 4. Train model
    print("\n🤖 PHASE 4: MODEL TRAINING")
    print("-" * 40)
    model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test)

    # 5. Detailed evaluation
    print("\n📈 PHASE 5: MODEL EVALUATION")
    print("-" * 40)
    print_pace_metrics(y_test.values, y_pred, "Ultra-Marathon Pace Predictor")

    # 6. Feature importance
    print("\n🎯 PHASE 6: FEATURE IMPORTANCE")
    print("-" * 40)
    importance_df = get_feature_importance(model, X_train.columns)
    print("Top 10 most important features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:>8.0f}")

    # 7. Save model and results
    print("\n💾 PHASE 7: SAVE RESULTS")
    print("-" * 40)
    import joblib
    import os
    os.makedirs("training_results", exist_ok=True)

    # Save CV results
    save_cv_results(cv_results)

    # Save model and other results
    save_model_results(model, X_train, y_test, y_pred, feature_cols)

    print("\n" + "=" * 60)
    print("🎉 ML TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Return comprehensive results
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': importance_df,
        'feature_cols': feature_cols,
        'categorical_features': categorical_features
    }

    return results


def load_processed_data_and_run_ml(csv_path: str = None, processed_df: pl.DataFrame = None):
    """
    Load processed data and run ML pipeline.

    Args:
        csv_path (str, optional): Path to raw CSV (will run data pipeline first)
        processed_df (pl.DataFrame, optional): Already processed dataframe

    Returns:
        dict: ML results
    """

    if processed_df is not None:
        # Use provided processed data
        return run_ml_pipeline(processed_df)
    elif csv_path is not None:
        # Run data pipeline first, then ML
        from src.pipeline_data import run_polars_pipeline_with_collection
        processed_data = run_polars_pipeline_with_collection(csv_path)
        return run_ml_pipeline(processed_data)
    else:
        raise ValueError("Must provide either csv_path or processed_df")


if __name__ == "__main__":
    # Example usage
    print("ML training pipeline module loaded successfully")
    print("Use: run_ml_pipeline(processed_data) to train model on processed data")
    print("Use: load_processed_data_and_run_ml(csv_path='data.csv') to run full workflow")
