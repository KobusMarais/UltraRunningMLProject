#!/usr/bin/env python3
"""
Pipeline Runner Script

This script runs the complete ultramarathon pace prediction pipeline
using the actual data file in the raw data folder.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / 'src'))

# Import pipeline modules
from src.pipeline import run_pipeline
from src.evaluation.metrics import print_pace_metrics
from src.visualization.eda import plot_model_performance, plot_feature_importance
from src.models.train import get_feature_importance


def main():
    """Run the complete pipeline with the actual data."""
    
    print("=" * 80)
    print("ULTRAMARATHON PACE PREDICTION PIPELINE")
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
    
    # Run the pipeline
    try:
        print("\n🚀 Starting pipeline...")
        model, X_train, X_test, y_train, y_test, y_pred = run_pipeline(str(data_file))
        
        # Print final results
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print detailed metrics
        print_pace_metrics(y_test, y_pred, "Ultra-Marathon Pace Predictor")
        
        # Get feature importance
        print("\n📈 Feature Importance Analysis:")
        importance_df = get_feature_importance(model, X_train.columns)
        print("\nTop 10 most important features:")
        print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        try:
            plot_model_performance(y_test, y_pred, "Ultra-Marathon Pace Predictor")
            plot_feature_importance(importance_df, top_n=15)
            print("✅ Visualizations created successfully")
        except Exception as viz_error:
            print(f"⚠️  Visualization error (non-critical): {viz_error}")
        
        # Save results
        print("\n💾 Saving results...")
        try:
            # Create results directory
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Save model
            model.booster_.save_model(str(results_dir / "model.txt"))
            print(f"✅ Model saved to {results_dir / 'model.txt'}")
            
            # Save predictions
            results_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'error': y_test - y_pred
            })
            results_df.to_csv(results_dir / "predictions.csv", index=False)
            print(f"✅ Predictions saved to {results_dir / 'predictions.csv'}")
            
            # Save feature importance
            importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
            print(f"✅ Feature importance saved to {results_dir / 'feature_importance.csv'}")
            
        except Exception as save_error:
            print(f"⚠️  Save error (non-critical): {save_error}")
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"📈 Model performance: R² = {np.corrcoef(y_test, y_pred)[0,1]**2:.4f}")
        print(f"🎯 Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} min/km")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check that your CSV file has the required columns")
        print("2. Ensure sufficient memory is available for large datasets")
        print("3. Verify all dependencies are installed (see requirements.txt)")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
