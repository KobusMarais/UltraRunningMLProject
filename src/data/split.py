"""
Train/Test Split Module

This module contains functions to split the ultramarathon dataset into training
and testing sets while preventing data leakage. The main function creates a
time-based split where the test set contains future races that the model should predict.
"""

import polars as pl


def split_train_test(df):
    """
    Split dataset into train/test while preventing leakage from future races.
    
    This function creates a time-based split where:
    - Test set: Western States 2022 race
    - Train set: All prior races of athletes in test set + other athletes
    
    This prevents data leakage by ensuring no future information is available
    during training.
    
    Args:
        df (pl.DataFrame): Dataset with engineered features
        
    Returns:
        tuple: (df_train, df_test, feature_cols)
            - df_train: Training dataset
            - df_test: Test dataset  
            - feature_cols: List of feature column names
    """
    
    # Define the feature columns to use for modeling
    feature_cols = [
        'Year of event', 
        'Event number of finishers', 
        'Athlete gender', 
        'Event distance_numeric', 
        'cum_num_races', 
        'cum_avg_pace', 
        'cum_best_pace', 
        'cum_ws_finishes', 
        'cum_total_distance', 
        'cum_avg_distance', 
        'cum_shortest_distance', 
        'cum_longest_distance', 
        'recent_avg_distance', 
        'distance_gap_from_longest', 
        'athlete_age'
    ]

    # Define test set: Western States 2022
    df_test = df.filter(
        (pl.col("Year of event") == 2022) &
        (pl.col("Event name").str.contains("Western States", literal=False, strict=False))
    )

    # Identify runners in test set
    ws_mask = (
        (pl.col("Year of event") == 2022) & 
        pl.col("Event name").str.contains("Western States", literal=False, strict=False)
    )
    ws_2022_runners = df.filter(ws_mask).select("Athlete ID").unique().to_series().to_list()

    # Cutoff cumulative races per athlete
    # This tracks how many races each athlete had completed before the test race
    ws_cutoffs = df.filter(ws_mask).select(["Athlete ID", "cum_num_races"]).to_dict(as_series=False)
    ws_cutoffs_dict = dict(zip(ws_cutoffs["Athlete ID"], ws_cutoffs["cum_num_races"]))

    # Map cutoff to all data using a join
    cutoff_df = pl.DataFrame({
        "Athlete ID": list(ws_cutoffs_dict.keys()),
        "ws_cutoff": list(ws_cutoffs_dict.values())
    })
    
    df_with_cutoff = df.join(cutoff_df, on="Athlete ID", how="left")

    # Keep only races before test race OR athletes not in test set
    # This ensures no future data leaks into training
    mask = (
        (pl.col("ws_cutoff").is_not_null() & (pl.col("cum_num_races") < pl.col("ws_cutoff"))) | 
        pl.col("ws_cutoff").is_null()
    )
    df_train = df_with_cutoff.filter(mask).drop("ws_cutoff")

    print("Test set shape:", df_test.shape)
    print("Train set shape:", df_train.shape)

    return df_train, df_test, feature_cols


def split_by_year(df, test_year=2022, target_event=None):
    """
    Alternative split function that splits by year.
    
    Args:
        df (pl.DataFrame): Dataset with engineered features
        test_year (int): Year to use for test set
        target_event (str): Optional event name to filter test set
        
    Returns:
        tuple: (df_train, df_test, feature_cols)
    """
    feature_cols = [
        'Year of event', 
        'Event number of finishers', 
        'Athlete gender', 
        'Event distance_numeric', 
        'cum_num_races', 
        'cum_avg_pace', 
        'cum_best_pace', 
        'cum_ws_finishes', 
        'cum_total_distance', 
        'cum_avg_distance', 
        'cum_shortest_distance', 
        'cum_longest_distance', 
        'recent_avg_distance', 
        'distance_gap_from_longest', 
        'athlete_age'
    ]

    if target_event:
        # Filter test set by specific event
        df_test = df.filter(
            (pl.col("Year of event") == test_year) &
            (pl.col("Event name").str.contains(target_event, literal=False, strict=False))
        )
    else:
        # Use all races from test year
        df_test = df.filter(pl.col("Year of event") == test_year)

    # Training set: all data before test year
    df_train = df.filter(pl.col("Year of event") < test_year)

    print(f"Test set shape ({test_year}):", df_test.shape)
    print(f"Train set shape (< {test_year}):", df_train.shape)

    return df_train, df_test, feature_cols


if __name__ == "__main__":
    # Example usage
    print("Train/test split module loaded successfully")
    print("Use: split_train_test(df) for Western States 2022 split")
