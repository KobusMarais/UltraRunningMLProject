"""
Model Evaluation Module

This module contains functions to evaluate machine learning model performance
using various metrics and validation techniques. It includes both standard
metrics and custom evaluation functions for ultramarathon pace prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_pace_metrics(y_true, y_pred):
    """
    Calculate comprehensive pace prediction metrics.
    
    Args:
        y_true (array): True pace values (min/km)
        y_pred (array): Predicted pace values (min/km)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Percentage-based metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Additional pace-specific metrics
    # Mean Absolute Percentage Error for pace
    # This is more interpretable for pace prediction
    mean_pace = np.mean(y_true)
    mae_percentage = (mae / mean_pace) * 100
    
    # Pace accuracy within thresholds
    errors = np.abs(y_true - y_pred)
    accuracy_30s = np.mean(errors <= 0.5) * 100  # Within 30 seconds per km
    accuracy_1min = np.mean(errors <= 1.0) * 100  # Within 1 minute per km
    accuracy_2min = np.mean(errors <= 2.0) * 100  # Within 2 minutes per km
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'MAE_percentage': mae_percentage,
        'Accuracy_30s': accuracy_30s,
        'Accuracy_1min': accuracy_1min,
        'Accuracy_2min': accuracy_2min
    }
    
    return metrics


def print_pace_metrics(y_true, y_pred, model_name="Model"):
    """
    Print pace prediction metrics in a formatted way.
    
    Args:
        y_true (array): True pace values
        y_pred (array): Predicted pace values
        model_name (str): Name of the model
    """
    metrics = calculate_pace_metrics(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics")
    print("=" * 40)
    print(f"MAE:              {metrics['MAE']:.4f} min/km")
    print(f"RMSE:             {metrics['RMSE']:.4f} min/km")
    print(f"R²:               {metrics['R²']:.4f}")
    print(f"MAPE:             {metrics['MAPE']:.2f}%")
    print(f"MAE (% of mean):  {metrics['MAE_percentage']:.2f}%")
    print(f"Accuracy (±30s):  {metrics['Accuracy_30s']:.1f}%")
    print(f"Accuracy (±1min): {metrics['Accuracy_1min']:.1f}%")
    print(f"Accuracy (±2min): {metrics['Accuracy_2min']:.1f}%")
    print("=" * 40)


def evaluate_by_distance(y_true, y_pred, distances, distance_bins=None):
    """
    Evaluate model performance by distance categories.
    
    Args:
        y_true (array): True pace values
        y_pred (array): Predicted pace values
        distances (array): Distance values for each prediction
        distance_bins (list): Custom distance bins
        
    Returns:
        pd.DataFrame: Performance metrics by distance category
    """
    if distance_bins is None:
        distance_bins = [0, 50, 100, 160, 200, np.inf]
    
    distance_labels = ['< 50km', '50-99km', '100-159km', '160-199km', '≥ 200km']
    
    results = []
    
    for i, (bin_start, bin_end) in enumerate(zip(distance_bins[:-1], distance_bins[1:])):
        mask = (distances >= bin_start) & (distances < bin_end)
        
        if np.sum(mask) > 0:
            y_true_bin = y_true[mask]
            y_pred_bin = y_pred[mask]
            
            metrics = calculate_pace_metrics(y_true_bin, y_pred_bin)
            metrics['distance_range'] = distance_labels[i]
            metrics['count'] = len(y_true_bin)
            
            results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_by_gender(y_true, y_pred, genders):
    """
    Evaluate model performance by gender.
    
    Args:
        y_true (array): True pace values
        y_pred (array): Predicted pace values
        genders (array): Gender values for each prediction
        
    Returns:
        pd.DataFrame: Performance metrics by gender
    """
    results = []
    
    for gender in np.unique(genders):
        mask = genders == gender
        
        if np.sum(mask) > 0:
            y_true_gender = y_true[mask]
            y_pred_gender = y_pred[mask]
            
            metrics = calculate_pace_metrics(y_true_gender, y_pred_gender)
            metrics['gender'] = gender
            metrics['count'] = len(y_true_gender)
            
            results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_by_experience(y_true, y_pred, num_races, experience_bins=None):
    """
    Evaluate model performance by athlete experience (number of races).
    
    Args:
        y_true (array): True pace values
        y_pred (array): Predicted pace values
        num_races (array): Number of races for each athlete
        experience_bins (list): Custom experience bins
        
    Returns:
        pd.DataFrame: Performance metrics by experience level
    """
    if experience_bins is None:
        experience_bins = [0, 5, 20, 50, 100, np.inf]
    
    experience_labels = ['0-4 races', '5-19 races', '20-49 races', '50-99 races', '100+ races']
    
    results = []
    
    for i, (bin_start, bin_end) in enumerate(zip(experience_bins[:-1], experience_bins[1:])):
        mask = (num_races >= bin_start) & (num_races < bin_end)
        
        if np.sum(mask) > 0:
            y_true_exp = y_true[mask]
            y_pred_exp = y_pred[mask]
            
            metrics = calculate_pace_metrics(y_true_exp, y_pred_exp)
            metrics['experience_range'] = experience_labels[i]
            metrics['count'] = len(y_true_exp)
            
            results.append(metrics)
    
    return pd.DataFrame(results)


def cross_validation_evaluation(model, X, y, cv_folds=5, scoring='neg_mean_absolute_error'):
    """
    Perform cross-validation evaluation of the model.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        cv_folds (int): Number of cross-validation folds
        scoring (str): Scoring metric
        
    Returns:
        dict: Cross-validation results
    """
    from sklearn.model_selection import cross_val_score
    
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
    
    results = {
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'cv_min': np.min(cv_scores),
        'cv_max': np.max(cv_scores)
    }
    
    print(f"Cross-Validation Results ({cv_folds} folds):")
    print(f"Mean CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    print(f"Range: {results['cv_min']:.4f} to {results['cv_max']:.4f}")
    
    return results


def compare_models(y_true, predictions_dict):
    """
    Compare performance of multiple models.
    
    Args:
        y_true (array): True values
        predictions_dict (dict): Dictionary of model predictions
                               {model_name: predictions}
        
    Returns:
        pd.DataFrame: Comparison of model performances
    """
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_pace_metrics(y_true, y_pred)
        metrics['model'] = model_name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    
    # Sort by MAE (best performing first)
    comparison_df = comparison_df.sort_values('MAE')
    
    print("Model Comparison Results:")
    print("=" * 60)
    print(comparison_df[['model', 'MAE', 'RMSE', 'R²', 'MAPE']].round(4).to_string(index=False))
    print("=" * 60)
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully")
    print("Use the evaluation functions to assess model performance")
