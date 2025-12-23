"""
Model Analysis Visualization Module

This module contains functions to create feature importance and residual analysis
charts for trained machine learning models. Charts are saved with run IDs for
easy linking to performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os


def plot_feature_importance(model, feature_names, run_id=None, save_path="reports/figures/", top_n=15):
    """
    Create and save feature importance chart.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        run_id (str): Run identifier for chart naming
        save_path (str): Directory to save chart
        top_n (int): Number of top features to display

    Returns:
        str: Path to saved chart file
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Create horizontal bar plot
    ax = sns.barplot(data=importance_df, y='feature', x='importance',
                     palette='viridis', orient='h')

    # Customize the plot
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)

    # Add value labels on bars
    for i, v in enumerate(importance_df['importance']):
        ax.text(v + max(importance_df['importance']) * 0.01, i,
                f'{v:.0f}', va='center', fontsize=10)

    plt.tight_layout()

    # Generate filename
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feature_importance_{run_id}.png"
    filepath = Path(save_path) / filename

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Feature importance chart saved to: {filepath}")
    return str(filepath)


def plot_residual_analysis(model, X_test, y_test, feature_names, run_id=None, save_path="reports/figures/"):
    """
    Create comprehensive residual analysis charts.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test values
        feature_names (list): Feature names (for correlation analysis)
        run_id (str): Run identifier for chart naming
        save_path (str): Directory to save chart

    Returns:
        str: Path to saved chart file
    """
    # Get predictions
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
    sns.set_style("whitegrid")

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20, color='blue', edgecolors='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 0].set_ylabel('Residuals', fontsize=12)
    axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals Distribution
    sns.histplot(residuals, bins=50, kde=True, ax=axes[0, 1], color='green', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Residuals Distribution', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q Plot for normality
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    axes[1, 0].scatter(osm, osr, alpha=0.6, s=20, color='purple', edgecolors='black', linewidth=0.5)
    axes[1, 0].plot(osm, slope*osm + intercept, color='red', linewidth=2)
    axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[1, 0].set_ylabel('Sample Quantiles', fontsize=12)
    axes[1, 0].set_title(f'Q-Q Plot (R² = {r:.3f})', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals vs Most Important Feature
    # Find the most important feature
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        most_important_idx = np.argmax(importance)
        most_important_feature = feature_names[most_important_idx]

        if most_important_feature in X_test.columns:
            feature_values = X_test[most_important_feature]
            axes[1, 1].scatter(feature_values, residuals, alpha=0.6, s=20, color='orange', edgecolors='black', linewidth=0.5)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel(f'{most_important_feature}', fontsize=12)
            axes[1, 1].set_ylabel('Residuals', fontsize=12)
            axes[1, 1].set_title(f'Residuals vs {most_important_feature}', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate filename
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"residual_analysis_{run_id}.png"
    filepath = Path(save_path) / filename

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Residual analysis chart saved to: {filepath}")
    return str(filepath)


def generate_model_analysis_charts(model, X_test, y_test, feature_names, run_id=None, save_path="reports/figures/"):
    """
    Generate both feature importance and residual analysis charts.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test values
        feature_names (list): List of feature names
        run_id (str): Run identifier for chart naming
        save_path (str): Directory to save charts

    Returns:
        dict: Paths to saved chart files
    """
    print("Generating model analysis charts...")

    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Feature importance chart
    try:
        feature_chart_path = plot_feature_importance(model, feature_names, run_id, save_path)
    except Exception as e:
        print(f"Error generating feature importance chart: {e}")
        feature_chart_path = None

    # Residual analysis chart
    try:
        residual_chart_path = plot_residual_analysis(model, X_test, y_test, feature_names, run_id, save_path)
    except Exception as e:
        print(f"Error generating residual analysis chart: {e}")
        residual_chart_path = None

    print(f"Model analysis charts completed for run: {run_id}")

    return {
        'feature_importance_chart': feature_chart_path,
        'residual_analysis_chart': residual_chart_path,
        'run_id': run_id
    }


if __name__ == "__main__":
    # Example usage
    print("Model analysis visualization module loaded successfully")
    print("Use: generate_model_analysis_charts(model, X_test, y_test, feature_names, run_id)")
    print("Charts will be saved to reports/figures/ with run_id in filename")
