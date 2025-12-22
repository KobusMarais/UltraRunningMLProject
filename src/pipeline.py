"""
Main Pipeline Module

This module contains the main pipeline function that orchestrates the entire
ultramarathon pace prediction workflow. It combines all the individual modules
into a complete end-to-end machine learning pipeline.
"""

import pandas as pd
import numpy as np

# Import all the modules we created
from src.data.load import load_raw_data
from src.data.clean import clean_data
from src.data.features import engineer_features
from src.data.split import split_train_test
from src.models.prepare import prepare_model_data
from src.models.train import train_evaluate_lgbm


def run_pipeline(raw_csv_path):
    """
    Full ML pipeline for ultramarathon pace prediction.
    
    This function orchestrates the entire machine learning workflow:
    1. Load raw data
    2. Clean and preprocess data
    3. Engineer features
    4. Split into train/test sets
    5. Prepare features for modeling
    6. Train and evaluate model
    
    Args:
        raw_csv_path (str): Path to the raw CSV file
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred)
            - model: Trained LightGBM model
            - X_train: Training features
            - X_test: Test features
            - y_train: Training target
            - y_test: Test target
            - y_pred: Model predictions
    """
    
    print("=" * 60)
    print("ULTRAMARATHON PACE PREDICTION PIPELINE")
    print("=" * 60)
    
    # 1. Load raw data
    print("\n1. Loading raw data...")
    df = load_raw_data(raw_csv_path)
    print(f"   Raw data shape: {df.shape}")
    
    # 2. Clean data
    print("\n2. Cleaning data...")
    df_clean = clean_data(df)
    print(f"   After cleaning: {df_clean.shape}")
    
    # 3. Feature engineering
    print("\n3. Engineering features...")
    df_features = engineer_features(df_clean)
    print(f"   After feature engineering: {df_features.shape}")
    
    # 4. Train/Test split
    print("\n4. Splitting into train/test sets...")
    df_train, df_test, feature_cols = split_train_test(df_features)
    print(f"   Train set: {df_train.shape}")
    print(f"   Test set:  {df_test.shape}")
    
    # 5. Prepare features & target
    print("\n5. Preparing features for modeling...")
    X_train, X_test, y_train, y_test = prepare_model_data(df_train, df_test, feature_cols)
    print(f"   Feature preparation complete")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    
    # 6. Train & evaluate model
    print("\n6. Training and evaluating model...")
    model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return model, X_train, X_test, y_train, y_test, y_pred


def run_pipeline_with_custom_params(raw_csv_path, model_params=None):
    """
    Run pipeline with custom model parameters.
    
    Args:
        raw_csv_path (str): Path to the raw CSV file
        model_params (dict): Custom LightGBM parameters
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred)
    """
    print("=" * 60)
    print("CUSTOM PARAMETER PIPELINE")
    print("=" * 60)
    
    # Load and preprocess data (same as main pipeline)
    df = load_raw_data(raw_csv_path)
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    df_train, df_test, feature_cols = split_train_test(df_features)
    X_train, X_test, y_train, y_test = prepare_model_data(df_train, df_test, feature_cols)
    
    # Train with custom parameters
    print("\nTraining with custom parameters...")
    if model_params:
        print(f"Custom parameters: {model_params}")
    model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test, params=model_params)
    
    return model, X_train, X_test, y_train, y_test, y_pred


def save_pipeline_results(model, X_train, X_test, y_train, y_test, y_pred, output_dir="./results/"):
    """
    Save pipeline results to files.
    
    Args:
        model: Trained model
        X_train, X_test, y_train, y_test: Data splits
        y_pred: Predictions
        output_dir (str): Directory to save results
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.booster_.save_model(f"{output_dir}model.txt")
    print(f"Model saved to {output_dir}model.txt")
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred
    })
    results_df.to_csv(f"{output_dir}predictions.csv", index=False)
    print(f"Predictions saved to {output_dir}predictions.csv")
    
    # Save feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(f"{output_dir}feature_importance.csv", index=False)
    print(f"Feature importance saved to {output_dir}feature_importance.csv")


if __name__ == "__main__":
    # Example usage
    print("Pipeline module loaded successfully")
    print("Use: run_pipeline('path/to/data.csv') to run the full pipeline")
    print("Use: run_pipeline_with_custom_params('path/to/data.csv', params) for custom parameters")
