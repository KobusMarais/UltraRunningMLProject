#!/usr/bin/env python3
"""
Training and Evaluation Pipeline Script

This script loads the pre-processed encoded_features.parquet file and runs
the complete training and evaluation pipeline using pandas/scikit-learn.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / 'src'))

# Import training/evaluation modules
from src.data.split import split_train_test
from src.models.prepare import prepare_model_data
from src.models.train import train_evaluate_lgbm


def run_training_evaluation(encoded_features_path):
    """
    Run training and evaluation on the pre-processed encoded features.
    
    Args:
        encoded_features_path (str): Path to the encoded_features.parquet file
        
    Returns:
        dict: Complete pipeline results including model, data, and metrics
    """
    
    print("=" * 80)
    print("TRAINING AND EVALUATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load encoded features
    print("\n📁 Step 1: Loading encoded features...")
    if not os.path.exists(encoded_features_path):
        print(f"❌ File not found: {encoded_features_path}")
        return None
    
    df = pd.read_parquet(encoded_features_path)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Step 2: Prepare data for machine learning
    print("\n🔧 Step 2: Preparing data for machine learning...")
    
    # Identify feature columns (exclude ID and target columns)
    exclude_cols = ['Athlete ID', 'Event name', 'Event dates', 'Athlete performance', 
                   'Athlete average speed', 'Athlete year of birth', 'Athlete age category']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Target column: pace_min_per_km")
    
    # Step 3: Train/Test split
    print("\n📊 Step 3: Splitting into train/test sets...")
    df_train, df_test, feature_cols = split_train_test(df)
    print(f"   Train set: {df_train.shape}")
    print(f"   Test set:  {df_test.shape}")
    
    # Step 4: Prepare features & target
    print("\n⚙️  Step 4: Preparing features for modeling...")
    X_train, X_test, y_train, y_test = prepare_model_data(df_train, df_test, feature_cols)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test shape:  {y_test.shape}")
    
    # Step 5: Train & evaluate model
    print("\n🤖 Step 5: Training and evaluating model...")
    model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test)
    
    # Step 6: Calculate additional metrics
    print("\n📈 Step 6: Calculating performance metrics...")
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Calculate additional metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / y_test)) * 100
    
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Step 7: Create results summary
    print("\n📋 Step 7: Creating results summary...")
    
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'residuals': residuals,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        },
        'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
        'dataset_info': {
            'total_rows': len(df),
            'total_features': len(feature_cols),
            'train_rows': len(df_train),
            'test_rows': len(df_test),
            'feature_cols': feature_cols
        }
    }
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total records processed: {results['dataset_info']['total_rows']:,}")
    print(f"   Total features created:  {results['dataset_info']['total_features']}")
    print(f"   Training records:        {results['dataset_info']['train_rows']:,}")
    print(f"   Test records:           {results['dataset_info']['test_rows']:,}")
    
    print(f"\n🤖 Model Performance:")
    print(f"   MAE:  {results['metrics']['mae']:.4f} min/km")
    print(f"   RMSE: {results['metrics']['rmse']:.4f} min/km")
    print(f"   MAPE: {results['metrics']['mape']:.2f}%")
    
    print(f"\n🔧 Top 10 Most Important Features:")
    feature_importance = results['feature_importance']
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature}: {importance}")
    
    print(f"\n🎉 Training and evaluation finished successfully!")
    print(f"💡 All results are available in the returned dictionary")
    
    return results


def save_results(results, output_dir="./training_results/"):
    """
    Save training and evaluation results to files.
    
    Args:
        results (dict): Pipeline results from run_training_evaluation
        output_dir (str): Directory to save results
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    results['model'].booster_.save_model(f"{output_dir}model.txt")
    print(f"✅ Model saved to {output_dir}model.txt")
    
    # Save predictions and actuals
    predictions_df = pd.DataFrame({
        'actual': results['y_test'],
        'predicted': results['y_pred'],
        'residuals': results['residuals']
    })
    predictions_df.to_csv(f"{output_dir}predictions.csv", index=False)
    print(f"✅ Predictions saved to {output_dir}predictions.csv")
    
    # Save feature importance
    feature_importance_df = pd.DataFrame([
        {'feature': feature, 'importance': importance}
        for feature, importance in results['feature_importance'].items()
    ]).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv(f"{output_dir}feature_importance.csv", index=False)
    print(f"✅ Feature importance saved to {output_dir}feature_importance.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(f"{output_dir}metrics.csv", index=False)
    print(f"✅ Metrics saved to {output_dir}metrics.csv")
    
    # Save dataset info
    dataset_info_df = pd.DataFrame([results['dataset_info']])
    dataset_info_df.to_csv(f"{output_dir}dataset_info.csv", index=False)
    print(f"✅ Dataset info saved to {output_dir}dataset_info.csv")
    
    print(f"\n📁 All results saved to {output_dir}")


def main():
    """Main function to run the training and evaluation pipeline."""
    
    # Path to the encoded features file
    encoded_features_path = "data/processed/encoded_features.parquet"
    
    print(f"📁 Using encoded features file: {encoded_features_path}")
    
    # Check if file exists
    if not os.path.exists(encoded_features_path):
        print(f"❌ File not found: {encoded_features_path}")
        print("Please run the Polars pipeline first to create this file.")
        return
    
    file_size = os.path.getsize(encoded_features_path) / (1024 * 1024)  # Size in MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    try:
        # Run the training and evaluation pipeline
        results = run_training_evaluation(encoded_features_path)
        
        if results is None:
            return
        
        # Save results
        print(f"\n💾 Saving results...")
        save_results(results)
        
        print(f"\n🎉 Training and evaluation completed successfully!")
        print(f"💡 Check the training_results/ directory for all outputs")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Training and evaluation completed successfully!")
        print(f"   Model: {type(results['model']).__name__}")
        print(f"   Test MAE: {results['metrics']['mae']:.4f}")
        print(f"   Test RMSE: {results['metrics']['rmse']:.4f}")
        print(f"   Test MAPE: {results['metrics']['mape']:.2f}%")
    else:
        print(f"\n❌ Training and evaluation failed!")
        sys.exit(1)
