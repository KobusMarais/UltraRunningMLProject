"""
Data Cleaning Module

This module contains functions to clean and preprocess ultramarathon data using Polars.
It handles missing values, standardizes event names, converts data types, and removes
irrelevant records for machine learning purposes using vectorized operations.
"""

import polars as pl
import re
from pathlib import Path


def clean_event_name_expr():
    """
    Create a Polars expression for cleaning event names using vectorized operations.
    
    Returns:
        pl.Expr: Polars expression for cleaning event names
    """
    return (
        pl.col("Event name")
        .str.to_lowercase()
        .str.replace_all(r'\(?\\b(19|20)\d{2}\b\)?', '')  # Remove years
        .str.replace_all(r'\([a-z]{3}\)', '')  # Remove country codes
        .str.replace_all(r'\d+(ª|th|st|nd|rd)', '')  # Remove edition numbers
        .str.replace_all(r'[^a-z0-9\s]', ' ')  # Keep only alphanumeric and spaces
        .str.replace_all(r'\s+', ' ')  # Normalize whitespace
        .str.strip_chars()
    )


def convert_to_seconds_expr():
    """
    Refined Polars expression to convert duration strings to total seconds.
    Handles both 'Xd HH:MM:SS' and 'HH:MM:SS' formats efficiently.
    """
    # Pattern explanation:
    # (?:(\d+)d\s+)? -> Optional non-capturing group for "days" (Group 1)
    # (\d+):(\d+):(\d+) -> Capturing groups for Hours, Minutes, Seconds (Groups 2, 3, 4)
    pattern = r"(?:(\d+)d\s+)?(\d+):(\d+):(\d+)"
    
    # Extract all groups into a struct in one pass
    groups = pl.col("Athlete performance").str.extract_groups(pattern)
    
    return (
        pl.duration(
            # fill_null("0") handles cases where the 'Xd' part is missing
            days=groups.struct.field("1").fill_null("0").cast(pl.Int32),
            hours=groups.struct.field("2").cast(pl.Int32),
            minutes=groups.struct.field("3").cast(pl.Int32),
            seconds=groups.struct.field("4").cast(pl.Int32),
        )
        .dt.total_seconds()  # Native conversion to float seconds
        .cast(pl.Int64)       # Optional: Cast back to integer
        .alias("performance_seconds")
    )


def extract_distance_expr():
    """
    Create a Polars expression for extracting numeric distance from text.
    
    Returns:
        pl.Expr: Polars expression for distance extraction
    """
    return (
        pl.col("Event distance/length")
        .str.to_lowercase()
        .str.replace_all(r'km|k', '')  # Remove km/k suffix
        .str.extract(r'(\d+(\.\d+)?)')  # Extract numeric value
        .cast(pl.Float64)
    )


def convert_miles_to_km_expr():
    """
    Create a Polars expression for converting miles to kilometers.
    
    Returns:
        pl.Expr: Polars expression for mile conversion
    """
    return (
        pl.when(pl.col("Event distance/length").str.contains(r'm|mi|mile|miles', literal=False))
        .then(pl.col("distance_numeric") * 1.60934)
        .otherwise(pl.col("distance_numeric"))
    )


def clean_data(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean ultramarathon dataset for machine learning using Polars.
    
    This function:
    - Drops columns with excessive missing values
    - Removes rows missing essential information
    - Cleans event names and converts distances to numeric
    - Removes irrelevant events and extreme outliers
    - Handles athlete data consistency
    - Uses vectorized operations for performance
    
    Args:
        lf (pl.LazyFrame): Lazy loaded ultramarathon data
        
    Returns:
        pl.LazyFrame: Cleaned dataset ready for feature engineering
    """
    
    # Remove irrelevant events that don't represent actual races
    events_to_remove = ["Stockholm Fotrally \\(SWE\\)", "Maratonmarschen Stockholm \\(SWE\\)"]
    
    # Apply cleaning pipeline using vectorized operations
    cleaned_lf = (
        lf
        # Drop weak-value columns with excessive missing data
        .drop(["Athlete club", "Athlete country"])
        
        # Clean and convert 'Event distance/length' to numeric km
        .with_columns([
            extract_distance_expr().alias("distance_numeric"),
            pl.col("Event distance/length").str.to_lowercase().alias("distance_str")
        ])
        
        # Convert miles to km
        .with_columns([
            convert_miles_to_km_expr().alias("Event distance_numeric")
        ])
        
        # Remove time-based or stage-based events (not distance races)
        .filter(
            ~pl.col("distance_str").str.contains(r'h|hr|hour|hours|d|day|days|min', literal=False) &
            ~pl.col("distance_str").str.contains(r'/|:|x', literal=False)
        )
        
        # Remove irrelevant events
        .filter(~pl.col("Event name").is_in(events_to_remove))
        
        # Clean event names
        .with_columns([
            clean_event_name_expr().alias("Event_name_clean")
        ])
        
        # Clean 'Athlete performance' and convert to seconds
        .with_columns([
            pl.col("Athlete performance").str.replace_all(r' h| km', '').alias("performance_clean")
        ])
        
        # Convert time to seconds
        .with_columns([
            convert_to_seconds_expr()
        ])
        
        # Compute pace (min/km) - handle division by zero
        .with_columns([
            pl.when(pl.col("Event distance_numeric") > 0)
            .then((pl.col("performance_seconds") / 60) / pl.col("Event distance_numeric"))
            .otherwise(None)
            .alias("pace_min_per_km")
        ])
        
        # Drop rows missing critical fields for performance prediction
        .drop_nulls([
            "Athlete gender", 
            "Athlete year of birth", 
            "Athlete age category", 
            "Athlete performance",
            "performance_seconds",
            "Event distance_numeric",
            "pace_min_per_km"
        ])
        
        # Remove extreme distances (>250 km) that are outliers
        .filter(pl.col("Event distance_numeric") <= 250)
        
        # Remove athletes with multiple genders (data inconsistency)
        .with_columns([
            pl.col("Athlete gender").n_unique().over("Athlete ID").alias("gender_count")
        ])
        .filter(pl.col("gender_count") == 1)
        .drop("gender_count")
    )
    
    return cleaned_lf

if __name__ == "__main__":
    # Example usage
    print("Data cleaning module loaded successfully")
    print("Use: clean_data(lf) to clean your dataset")
