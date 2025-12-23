"""
Pipeline Utilities

This module contains shared utility functions used across different ML pipeline components.
It includes data splitting, encoding, feature preparation, and result saving utilities.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import subprocess

# Import shared modules
from src.data.split import split_train_test


def apply_smoothed_target_encoding(train_df, test_df, column='Event_name_clean', target='pace_min_per_km', m=10):
    """
    Apply smoothed target encoding to categorical features.

    This technique encodes categorical variables (like race names) with their
    average target value, smoothed by the global mean to prevent overfitting.

    Args:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        column (str): Column name to encode
        target (str): Target variable name
        m (float): Smoothing factor (higher = more conservative)

    Returns:
        tuple: (train_df, test_df) with encoded features
    """
    # Calculate global mean from training data only (No leakage!)
    global_mean = train_df[target].mean()

    # Calculate count and mean for each category
    agg = train_df.groupby(column)[target].agg(['count', 'mean'])

    # Calculate the smoothed value using Bayesian averaging
    # Formula: (count * mean + m * global_mean) / (count + m)
    smooth_weights = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)

    # Map the weights back to the dataframes
    train_df['Race_Pace_Mean_Encoded'] = train_df[column].map(smooth_weights).fillna(global_mean)
    test_df['Race_Pace_Mean_Encoded'] = test_df[column].map(smooth_weights).fillna(global_mean)

    return train_df, test_df


def prepare_data_for_modeling(df_train, df_test, feature_cols):
    """
    Prepare train/test data for modeling by applying encoding and feature selection.

    Args:
        df_train (pl.DataFrame): Training dataframe
        df_test (pl.DataFrame): Test dataframe
        feature_cols (list): List of feature column names

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_cols)
    """
    # Convert to pandas for LightGBM
    df_train_pd = df_train.to_pandas()
    df_test_pd = df_test.to_pandas()

    # Apply race difficulty encoding
    df_train_pd, df_test_pd = apply_smoothed_target_encoding(df_train_pd, df_test_pd)

    # Add encoded feature to feature columns
    feature_cols = feature_cols + ['Race_Pace_Mean_Encoded']

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

    return X_train, X_test, y_train, y_test, feature_cols


def save_cv_results(cv_results, output_dir="training_results", pipeline_name="unknown"):
    """
    Save cross-validation results to file.

    Args:
        cv_results (dict): CV results from train_with_cross_validation
        output_dir (str): Directory to save results
        pipeline_name (str): Name of the pipeline that generated these results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get git commit hash if available
    git_commit = "unknown"
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:8]  # Short hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Git not available or not a git repo

    # Save latest results to cv_metrics.txt
    cv_metrics_path = f"{output_dir}/cv_metrics.txt"
    with open(cv_metrics_path, 'w') as f:
        f.write("Cross-Validation Results (Training Set)\n")
        f.write("=" * 40 + "\n")
        f.write(f"Average CV MAE:  {cv_results['avg_mae']:.4f} min/km\n")
        f.write(f"Average CV RMSE: {cv_results['avg_rmse']:.4f} min/km\n")
        f.write(f"CV MAE Std:     {np.std(cv_results['l1-mean']):.4f}\n")
        f.write(f"CV RMSE Std:    {np.std(cv_results['rmse-mean']):.4f}\n")
    print(f"   CV metrics saved to: {cv_metrics_path}")

    # Append to history file
    history_path = f"{output_dir}/cv_metrics_history.txt"
    is_new_file = not os.path.exists(history_path)

    with open(history_path, 'a') as f:
        if is_new_file:
            f.write("CV Metrics History\n")
            f.write("=" * 50 + "\n\n")

        f.write(f"Run: {timestamp} - Pipeline: {pipeline_name} - Git: {git_commit}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Average CV MAE:  {cv_results['avg_mae']:.4f} min/km\n")
        f.write(f"Average CV RMSE: {cv_results['avg_rmse']:.4f} min/km\n")
        f.write(f"CV MAE Std:     {np.std(cv_results['l1-mean']):.4f}\n")
        f.write(f"CV RMSE Std:    {np.std(cv_results['rmse-mean']):.4f}\n")
        f.write("---\n\n")

    print(f"   CV metrics history appended to: {history_path}")


def save_model_results(model, X_train, y_test, y_pred, feature_cols, output_dir="training_results"):
    """
    Save final model and evaluation results.

    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features for feature importance
        y_test (pd.Series): Test target
        y_pred (np.array): Predictions
        feature_cols (list): Feature column names
        output_dir (str): Directory to save results
    """
    import joblib
    import os
    from src.models.train import get_feature_importance
    from src.evaluation.metrics import print_pace_metrics

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = f"{output_dir}/lightgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"   Model saved to: {model_path}")

    # Feature importance
    importance_df = get_feature_importance(model, X_train.columns)
    importance_path = f"{output_dir}/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"   Feature importance saved to: {importance_path}")

    # Save predictions
    predictions_path = f"{output_dir}/test_predictions.csv"
    pred_df = pd.DataFrame({
        'actual_pace': y_test.values,
        'predicted_pace': y_pred,
        'error': y_test.values - y_pred
    })
    pred_df.to_csv(predictions_path, index=False)
    print(f"   Predictions saved to: {predictions_path}")

    # Print detailed evaluation
    print("\n📈 MODEL EVALUATION")
    print("-" * 40)
    print_pace_metrics(y_test.values, y_pred, "Ultra-Marathon Pace Predictor")

    print("\n🎯 TOP 10 FEATURES")
    print("-" * 40)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        feature_name = row['feature']
        importance_val = row['importance']
        print("  {:2d}. {:<25} {:8.0f}".format(i+1, feature_name, importance_val))
