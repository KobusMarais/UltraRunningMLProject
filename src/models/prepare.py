"""
Model Preparation Module

This module contains functions to prepare data for machine learning models.
It handles feature encoding, target preparation, and data cleaning for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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


def prepare_model_data(df_train, df_test, feature_cols, target_col='pace_min_per_km'):
    """
    Prepare train/test datasets for modeling.
    
    This function:
    - Selects relevant features
    - Applies one-hot encoding to categorical variables
    - Aligns columns between train and test sets
    - Handles missing values and infinities
    - Defines target variables
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        feature_cols (list): List of feature column names
        target_col (str): Name of target column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    # Apply race difficulty encoding
    df_train, df_test = apply_smoothed_target_encoding(df_train, df_test)
    
    # Select features for modeling
    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()

    # One-hot encode categorical variables (gender)
    X_train = pd.get_dummies(X_train, columns=['Athlete gender'], drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=['Athlete gender'], drop_first=True)

    # Align columns between train and test sets
    # This ensures both datasets have the same feature columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Define target variables
    y_train = df_train[target_col]
    y_test  = df_test[target_col]

    # Clean infinities and replace with NaN
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaNs (including former infinities) with -1
    # This is a common strategy for missing values in tree-based models
    X_train.fillna(-1, inplace=True)
    X_test.fillna(-1, inplace=True)

    print("Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, feature_cols=None):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        feature_cols (list): List of numerical columns to scale
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if feature_cols is None:
        # Assume all non-categorical columns are numerical
        feature_cols = [col for col in X_train.columns if col != 'Athlete gender_M']
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale numerical features
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_for_lightgbm(df_train, df_test, feature_cols, target_col='pace_min_per_km'):
    """
    Prepare data specifically for LightGBM model.
    
    LightGBM can handle categorical variables directly, so we don't need
    one-hot encoding for categorical features.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        feature_cols (list): List of feature column names
        target_col (str): Name of target column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, categorical_features)
    """
    
    # Apply race difficulty encoding
    df_train, df_test = apply_smoothed_target_encoding(df_train, df_test)
    
    # Select features
    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()
    
    # Define target
    y_train = df_train[target_col]
    y_test  = df_test[target_col]

    # Identify categorical features for LightGBM
    categorical_features = ['Athlete gender']
    
    # Clean infinities and missing values
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(-1, inplace=True)
    X_test.fillna(-1, inplace=True)

    print("LightGBM shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test, categorical_features


if __name__ == "__main__":
    # Example usage
    print("Model preparation module loaded successfully")
    print("Use: prepare_model_data(df_train, df_test, feature_cols) for general ML")
    print("Use: prepare_for_lightgbm(df_train, df_test, feature_cols) for LightGBM")
