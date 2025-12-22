"""
Feature Engineering Module

This module contains functions to create features for ultramarathon performance prediction.
It generates cumulative statistics, rolling averages, and derived metrics that capture
athlete progression and race characteristics over time.
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Create cumulative and rolling features for ultramarathon dataset.
    
    This function generates features that capture:
    - Athlete progression over time (cumulative stats)
    - Race difficulty indicators
    - Recent performance trends
    - Athlete experience metrics
    
    Args:
        df (pd.DataFrame): Cleaned ultramarathon data with pace_min_per_km
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    
    # Sort data chronologically for cumulative calculations
    df = df.sort_values(by=["Athlete ID", "Year of event", "Event name"]).reset_index(drop=True)

    # Cumulative number of races per athlete
    df["cum_num_races"] = df.groupby("Athlete ID").cumcount()

    # Cumulative average pace (excluding current race)
    # This shows how the athlete's average performance has evolved
    df["cum_avg_pace"] = df.groupby("Athlete ID")["pace_min_per_km"].expanding().mean().shift(1).reset_index(level=0, drop=True)

    # Cumulative best pace (excluding current race)
    # Shows the athlete's personal best progression
    df["cum_best_pace"] = df.groupby("Athlete ID")["pace_min_per_km"].expanding().min().shift(1).reset_index(level=0, drop=True)

    # Cumulative distance statistics
    df["cum_total_distance"] = df.groupby("Athlete ID")["Event distance_numeric"].cumsum()
    df["cum_avg_distance"] = df["cum_total_distance"] / df["cum_num_races"]
    
    # Shortest and longest distances completed by athlete
    grp = df.groupby("Athlete ID")["Event distance_numeric"]
    df["cum_shortest_distance"] = grp.cummin()
    df["cum_longest_distance"] = grp.cummax()

    # Cumulative Western States finishes
    # Western States is a prestigious race, so experience matters
    df["cum_ws_finishes"] = (
        df["Event name"].eq("Western States")  # True for WS
          .groupby(df["Athlete ID"])           # per athlete
          .cumsum()                            # cumulative sum
          .shift(1)                            # exclude current race
          .fillna(0)                           # first race has 0
          .astype(int)
    )

    # Recent average distance (rolling 3 races)
    # Shows what distance the athlete has been training for recently
    df["recent_avg_distance"] = (
        df.groupby("Athlete ID")["Event distance_numeric"]
          .rolling(3, min_periods=1).mean()
          .reset_index(level=0, drop=True)
          .shift(1)
    )

    # Distance gap from longest
    # How challenging is this race compared to the athlete's longest?
    df["distance_gap_from_longest"] = df["Event distance_numeric"] - df["cum_longest_distance"]

    # Athlete age at time of race
    df["athlete_age"] = df["Year of event"] - df["Athlete year of birth"]

    return df


def create_race_difficulty_features(df):
    """
    Create race difficulty indicators based on historical performance.
    
    Args:
        df (pd.DataFrame): Dataset with pace_min_per_km
        
    Returns:
        pd.DataFrame: Dataset with race difficulty features
    """
    # Calculate average pace for each race (difficulty indicator)
    race_difficulty = df.groupby(['Event_name_clean', 'Event distance_numeric'])['pace_min_per_km'].mean().reset_index()
    race_difficulty.columns = ['Event_name_clean', 'Event distance_numeric', 'race_avg_pace']
    
    # Merge back to main dataset
    df = df.merge(race_difficulty, on=['Event_name_clean', 'Event distance_numeric'], how='left')
    
    return df


def create_consistency_features(df):
    """
    Create features that measure athlete consistency.
    
    Args:
        df (pd.DataFrame): Dataset with pace_min_per_km
        
    Returns:
        pd.DataFrame: Dataset with consistency features
    """
    # Calculate pace variance for each athlete (consistency measure)
    pace_stats = df.groupby("Athlete ID")["pace_min_per_km"].agg(['std', 'mean']).reset_index()
    pace_stats['consistency_ratio'] = pace_stats['std'] / pace_stats['mean']
    pace_stats.columns = ['Athlete ID', 'pace_std', 'pace_mean', 'consistency_ratio']
    
    # Merge back to main dataset
    df = df.merge(pace_stats, on=['Athlete ID'], how='left')
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully")
    print("Use: engineer_features(df) to create features")
    print("Use: create_race_difficulty_features(df) for race difficulty")
    print("Use: create_consistency_features(df) for consistency metrics")
