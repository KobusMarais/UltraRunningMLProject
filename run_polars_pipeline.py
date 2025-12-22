#!/usr/bin/env python3
"""
Polars Pipeline Runner Script

This script runs the complete ultramarathon pace prediction pipeline using Polars
for memory efficiency with the actual data file in the raw data folder.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / 'src'))

# Import Polars pipeline modules
from src.pipeline_polars import run_polars_pipeline_with_collection

# Import training/evaluation modules
from src.data.split import split_train_test
from src.models.prepare import prepare_model_data
from src.models.train import train_evaluate_lgbm


def run_complete_polars_pipeline(csv_path):
    """
    Run the complete pipeline with Polars for data processing and pandas for ML.
    
    This function:
    1. Uses Polars for memory-efficient data loading, cleaning, and feature engineering
    2. Converts the final dataset to pandas for machine learning
    3. Runs train/test split, model training, and evaluation
    4. Returns all results for analysis
    
    Args:
        csv_path (str): Path to the raw CSV file
        
    Returns:
        dict: Complete pipeline results including model, data, and metrics
    """
    
    print("=" * 80)
    print("COMPLETE ULTRAMARATHON PACE PREDICTION PIPELINE (POLARS + PANDAS)")
    print("=" * 80)
    
    # Step 1: Run Polars pipeline for data processing
    print("\n🚀 Step 1: Running Polars data processing pipeline...")
    final_df = run_polars_pipeline_with_collection(csv_path)
    
    print(f"✅ Polars pipeline completed: {len(final_df)} rows, {len(final_df.columns)} columns")
    
    # Step 2: Prepare data for machine learning
    print("\n🔧 Step 2: Preparing data for machine learning...")
    
    # Identify feature columns (exclude ID and target columns)
    exclude_cols = ['Athlete ID', 'Event name', 'Event dates', 'Athlete performance', 
                   'Athlete average speed', 'Athlete year of birth', 'Athlete age category']
    feature_cols = [col for col in final_df.columns if col not in exclude_cols]
    
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Target column: pace_min_per_km")
    
    # Create a copy for ML processing
    ml_df = final_df.copy()
    
    # Step 3: Train/Test split
    print("\n📊 Step 3: Splitting into train/test sets...")
    df_train, df_test, feature_cols = split_train_test(ml_df)
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
            'total_rows': len(final_df),
            'total_features': len(feature_cols),
            'train_rows': len(df_train),
            'test_rows': len(df_test),
            'feature_cols': feature_cols
        }
    }
    
    # Print final summary
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE COMPLETED SUCCESSFULLY!")
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
    
    print(f"\n🎉 Complete pipeline finished successfully!")
    print(f"💡 All results are available in the returned dictionary")
    
    return results


def save_complete_results(results, output_dir="./complete_results/"):
    """
    Save complete pipeline results to files.
    
    Args:
        results (dict): Pipeline results from run_complete_polars_pipeline
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
    """Run the complete Polars pipeline with the actual data."""
    
    print("=" * 80)
    print("ULTRAMARATHON PACE PREDICTION PIPELINE (POLARS)")
    print("=" * 80)
    
    # Find the data file
    data_dir = Path("data/raw")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/raw/")
        print("Please ensure your data file is in the data/raw/ directory")
        return
    
    # Use the first CSV file found
    data_file = csv_files[0]
    print(f"📁 Using data file: {data_file}")
    
    # Check if file exists and has reasonable size
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    file_size = data_file.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"📊 File size: {file_size:.2f} MB")
    print("🚀 Using Polars for memory-efficient processing...")
    
    # Run the complete pipeline
    try:
        print("\n🚀 Starting complete pipeline...")
        results = run_complete_polars_pipeline(str(data_file))
        
        # Save results
        print(f"\n💾 Saving results...")
        save_complete_results(results)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"💡 Check the complete_results/ directory for all outputs")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
