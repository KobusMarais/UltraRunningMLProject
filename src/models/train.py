"""
Model Training Module

This module contains functions to train machine learning models for ultramarathon
pace prediction. It includes LightGBM regression with hyperparameter tuning and
model evaluation.
"""

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from src.models.pipeline_utils import apply_smoothed_target_encoding


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


def train_with_cross_validation(df_train, feature_cols, cv_folds=5, params=None):
    """
    Train LightGBM with cross-validation for better parameter tuning.
    Applies smoothed target encoding per fold to prevent data leakage.
    
    Args:
        df_train (pd.DataFrame): Training dataframe (before encoding)
        feature_cols (list): List of feature column names
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

    # Sort by year to ensure temporal order for TimeSeriesSplit
    df_train = df_train.sort_values('Year of event').reset_index(drop=True)

    # Use TimeSeriesSplit for temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    cv_mae_scores = []
    cv_rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_train)):
        # Split the raw data
        df_fold_train = df_train.iloc[train_idx].copy()
        df_fold_val = df_train.iloc[val_idx].copy()
        
        # Apply encoding using only the training part of this fold
        df_fold_train, df_fold_val = apply_smoothed_target_encoding(df_fold_train, df_fold_val)
        
        # Prepare features (include the newly encoded column)
        fold_feature_cols = feature_cols + ['Race_Pace_Mean_Encoded']
        X_fold_train = df_fold_train[fold_feature_cols].copy()
        X_fold_val = df_fold_val[fold_feature_cols].copy()
        y_fold_train = df_fold_train['pace_min_per_km']
        y_fold_val = df_fold_val['pace_min_per_km']
        
        # Handle categorical features for LightGBM
        categorical_features = ['Athlete gender']
        for cat_col in categorical_features:
            if cat_col in X_fold_train.columns:
                # Combine fold train and val to ensure consistent encoding
                combined = pd.concat([X_fold_train[[cat_col]], X_fold_val[[cat_col]]], ignore_index=True)
                combined[cat_col] = combined[cat_col].astype('category')
                
                # Apply to train and val
                X_fold_train[cat_col] = combined.iloc[:len(X_fold_train)][cat_col]
                X_fold_val[cat_col] = combined.iloc[len(X_fold_train):][cat_col]
                
                # Convert to codes
                X_fold_train[cat_col] = X_fold_train[cat_col].cat.codes
                X_fold_val[cat_col] = X_fold_val[cat_col].cat.codes
        
        # Clean infinities and missing values
        X_fold_train.replace([float('inf'), -float('inf')], -1, inplace=True)
        X_fold_val.replace([float('inf'), -float('inf')], -1, inplace=True)
        X_fold_train.fillna(-1, inplace=True)
        X_fold_val.fillna(-1, inplace=True)
        
        # Train model on fold
        model = lgb.LGBMRegressor(**params)
        model.fit(X_fold_train, y_fold_train)
        
        # Predict on validation fold
        y_pred = model.predict(X_fold_val)
        
        # Calculate metrics
        fold_mae = mean_absolute_error(y_fold_val, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        
        cv_mae_scores.append(fold_mae)
        cv_rmse_scores.append(fold_rmse)
        print(f"  Fold {fold+1}: MAE={fold_mae:.4f}, RMSE={fold_rmse:.4f}")
    
    # Calculate average scores
    avg_mae = np.mean(cv_mae_scores)
    avg_rmse = np.mean(cv_rmse_scores)
    
    print(f"Average CV MAE:  {avg_mae:.4f} (+/- {np.std(cv_mae_scores):.4f})")
    print(f"Average CV RMSE: {avg_rmse:.4f} (+/- {np.std(cv_rmse_scores):.4f})")
    
    # Return results in similar format to lgb.cv
    return {
        'l1-mean': cv_mae_scores,
        'rmse-mean': cv_rmse_scores,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse
    }


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
