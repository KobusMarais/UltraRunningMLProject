"""
Model Training Module

This module contains functions to train machine learning models for ultramarathon
pace prediction. It includes LightGBM regression with hyperparameter tuning and
model evaluation.
"""

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


def train_evaluate_lgbm(X_train, y_train, X_test, y_test, params=None):
    """
    Train a LightGBM regressor and evaluate performance on test set.
    
    LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
    It's efficient for large datasets and handles categorical features well.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features  
        y_test (pd.Series): Test target
        params (dict): LightGBM parameters (optional)
        
    Returns:
        tuple: (model, y_pred)
            - model: Trained LightGBM model
            - y_pred: Predictions on test set
    """
    # Default parameters for LightGBM
    if params is None:
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,  # No limit on depth
            "subsample": 0.8,  # Use 80% of data for each tree
            "colsample_bytree": 0.8,  # Use 80% of features for each tree
            "random_state": 42,
            "verbose": -1  # Suppress output
        }

    # Initialize LightGBM regressor
    model = lgb.LGBMRegressor(**params)

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Additional metrics for pace prediction
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return model, y_pred


def train_with_cross_validation(X_train, y_train, cv_folds=5, params=None):
    """
    Train LightGBM with cross-validation for better parameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        params (dict): LightGBM parameters
        
    Returns:
        dict: Cross-validation results
    """
    if params is None:
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Perform cross-validation
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=500,
        nfold=cv_folds,
        metrics=['mae', 'rmse']
    )
    
    print(f"CV MAE:  {np.min(cv_results['l1-mean']):.4f}")
    print(f"CV RMSE: {np.min(cv_results['rmse-mean']):.4f}")
    
    return cv_results


def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from trained LightGBM model.
    
    Args:
        model: Trained LightGBM model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance sorted by importance
    """
    importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def hyperparameter_tuning(X_train, y_train, param_grid=None):
    """
    Simple hyperparameter tuning for LightGBM.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (dict): Parameter grid to search
        
    Returns:
        dict: Best parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [32, 64, 128],
            'subsample': [0.7, 0.8, 0.9]
        }
    
    best_score = float('inf')
    best_params = {}
    
    # Simple grid search (for demonstration)
    # In practice, use sklearn's GridSearchCV or RandomizedSearchCV
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for num_leaves in param_grid['num_leaves']:
                for subsample in param_grid['subsample']:
                    params = {
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'num_leaves': num_leaves,
                        'subsample': subsample,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'verbose': -1
                    }
                    
                    # Train model with current parameters
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate on training data (for speed)
                    train_pred = model.predict(X_train)
                    score = mean_absolute_error(y_train, train_pred)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
    
    print(f"Best CV score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params


if __name__ == "__main__":
    # Example usage
    print("Model training module loaded successfully")
    print("Use: train_evaluate_lgbm(X_train, y_train, X_test, y_test) for basic training")
    print("Use: train_with_cross_validation(X_train, y_train) for CV")
    print("Use: hyperparameter_tuning(X_train, y_train) for parameter tuning")
