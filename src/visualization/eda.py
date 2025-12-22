"""
EDA Visualization Module

This module contains functions to create exploratory data analysis visualizations
for ultramarathon data. It includes plots for data distribution, feature relationships,
and model performance evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_data_overview(df):
    """
    Create an overview of the dataset structure and basic statistics.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dataset Overview', fontsize=16)
    
    # 1. Missing values heatmap
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    axes[0, 0].bar(range(len(missing_pct)), missing_pct)
    axes[0, 0].set_title('Missing Values (%)')
    axes[0, 0].set_xlabel('Features')
    axes[0, 0].set_ylabel('Missing %')
    
    # 2. Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Data Types Distribution')
    
    # 3. Numeric features distribution
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols[0]].hist(bins=30, ax=axes[0, 2])
        axes[0, 2].set_title(f'Distribution of {numeric_cols[0]}')
        axes[0, 2].set_xlabel(numeric_cols[0])
    
    # 4. Categorical features count
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        value_counts = df[cat_cols[0]].value_counts().head(10)
        axes[1, 0].bar(range(len(value_counts)), value_counts.values)
        axes[1, 0].set_title(f'Top 10 {cat_cols[0]} Values')
        axes[1, 0].set_xticks(range(len(value_counts)))
        axes[1, 0].set_xticklabels(value_counts.index, rotation=45)
    
    # 5. Dataset size over time (if date column exists)
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        try:
            df[date_cols[0]].value_counts().sort_index().plot(ax=axes[1, 1])
            axes[1, 1].set_title(f'Data Points Over Time ({date_cols[0]})')
            axes[1, 1].tick_params(axis='x', rotation=45)
        except:
            axes[1, 1].text(0.5, 0.5, 'Date parsing failed', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Date Analysis')
    
    # 6. Memory usage
    memory_usage = df.memory_usage(deep=True).sort_values(ascending=False)
    axes[1, 2].bar(range(min(10, len(memory_usage))), memory_usage.head(10))
    axes[1, 2].set_title('Memory Usage by Column (Top 10)')
    axes[1, 2].set_ylabel('Memory (bytes)')
    
    plt.tight_layout()
    plt.show()


def plot_pace_distribution(df, target_col='pace_min_per_km'):
    """
    Plot the distribution of pace values.
    
    Args:
        df (pd.DataFrame): Dataset with pace column
        target_col (str): Name of pace column
    """
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pace Distribution Analysis', fontsize=16)
    
    # 1. Histogram
    df[target_col].hist(bins=50, ax=axes[0, 0], alpha=0.7)
    axes[0, 0].set_title('Pace Distribution (Histogram)')
    axes[0, 0].set_xlabel('Pace (min/km)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Box plot
    df[target_col].plot(kind='box', ax=axes[0, 1])
    axes[0, 1].set_title('Pace Distribution (Box Plot)')
    axes[0, 1].set_ylabel('Pace (min/km)')
    
    # 3. Violin plot by gender
    if 'Athlete gender' in df.columns:
        sns.violinplot(data=df, x='Athlete gender', y=target_col, ax=axes[1, 0])
        axes[1, 0].set_title('Pace by Gender')
        axes[1, 0].set_ylabel('Pace (min/km)')
    
    # 4. Pace vs Distance
    if 'Event distance_numeric' in df.columns:
        axes[1, 1].scatter(df['Event distance_numeric'], df[target_col], alpha=0.1, s=1)
        axes[1, 1].set_title('Pace vs Distance')
        axes[1, 1].set_xlabel('Distance (km)')
        axes[1, 1].set_ylabel('Pace (min/km)')
    
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(df, target_col='pace_min_per_km'):
    """
    Plot correlation matrix for numeric features.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Target variable for correlation analysis
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show top correlations with target
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        print(f"\nTop 10 features correlated with {target_col}:")
        print(target_corr.head(11).iloc[1:])  # Exclude self-correlation


def plot_model_performance(y_true, y_pred, model_name="Model"):
    """
    Plot model performance metrics and residual analysis.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        model_name (str): Name of the model
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} Performance Analysis', fontsize=16)
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Pace (min/km)')
    axes[0, 0].set_ylabel('Predicted Pace (min/km)')
    axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.3f})')
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residuals Distribution')
    
    # 4. Error metrics bar chart
    metrics = ['MAE', 'RMSE']
    values = [mae, rmse]
    axes[1, 1].bar(metrics, values, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_ylabel('Error Value')
    axes[1, 1].set_title('Error Metrics')
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.01 * max(values), f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{model_name} Performance Summary:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


def plot_feature_importance(importance_df, top_n=15):
    """
    Plot feature importance from trained model.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with feature importance
        top_n (int): Number of top features to display
    """
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        print("Error: importance_df must have 'feature' and 'importance' columns")
        return
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Show most important at top
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("EDA visualization module loaded successfully")
    print("Use the plotting functions to visualize your data and model results")
