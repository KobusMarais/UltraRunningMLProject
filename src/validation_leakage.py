"""
Data Leakage Validation Module

This module contains functions to validate that the ultramarathon pace prediction
pipeline maintains temporal integrity and prevents data leakage.
"""

import polars as pl
import pandas as pd
from pathlib import Path


def validate_temporal_integrity(df: pl.DataFrame) -> dict:
    """
    Validate that engineered features maintain temporal integrity.
    Checks that features for a given year don't include data from future years.

    Args:
        df (pl.DataFrame): DataFrame with engineered features

    Returns:
        dict: Validation results with any detected issues
    """
    issues = []

    # Convert to pandas for easier validation
    df_pd = df.to_pandas()

    # Group by athlete and check temporal integrity
    for athlete_id in df_pd['Athlete ID'].unique():
        athlete_data = df_pd[df_pd['Athlete ID'] == athlete_id].sort_values('Year of event')

        for idx, row in athlete_data.iterrows():
            current_year = row['Year of event']

            # Get all prior races (should match cum_num_races)
            prior_races = athlete_data[athlete_data['Year of event'] < current_year]

            if len(prior_races) > 0:
                # Check cum_num_races
                expected_races = len(prior_races)
                actual_races = row['cum_num_races']
                if actual_races != expected_races:
                    issues.append({
                        'type': 'cum_num_races_mismatch',
                        'athlete_id': athlete_id,
                        'year': current_year,
                        'expected': expected_races,
                        'actual': actual_races
                    })

                # Check cum_avg_pace
                if not pd.isna(row['cum_avg_pace']):
                    expected_avg = prior_races['pace_min_per_km'].mean()
                    actual_avg = row['cum_avg_pace']
                    if abs(actual_avg - expected_avg) > 1e-10:  # Small tolerance for floating point
                        issues.append({
                            'type': 'cum_avg_pace_mismatch',
                            'athlete_id': athlete_id,
                            'year': current_year,
                            'expected': expected_avg,
                            'actual': actual_avg
                        })

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_athletes_checked': len(df_pd['Athlete ID'].unique()),
        'total_records_checked': len(df_pd)
    }


def validate_train_test_split(df_train: pl.DataFrame, df_test: pl.DataFrame) -> dict:
    """
    Validate that train/test split prevents data leakage.

    Args:
        df_train (pl.DataFrame): Training dataset
        df_test (pl.DataFrame): Test dataset

    Returns:
        dict: Validation results
    """
    issues = []

    # Convert to pandas
    train_pd = df_train.to_pandas()
    test_pd = df_test.to_pandas()

    # Check test set is only Western States 2022
    test_events = test_pd['Event name'].unique()
    test_years = test_pd['Year of event'].unique()

    if not any('Western States' in str(event) for event in test_events):
        issues.append({'type': 'missing_western_states', 'message': 'Western States not found in test set'})

    if 2022 not in test_years:
        issues.append({'type': 'missing_2022', 'message': '2022 not found in test years'})

    if len(test_years) > 1:
        issues.append({'type': 'multiple_test_years', 'years': test_years.tolist()})

    # Check athletes in test set don't have future data in training
    test_athlete_ids = test_pd['Athlete ID'].unique()

    for athlete_id in test_athlete_ids:
        athlete_train = train_pd[train_pd['Athlete ID'] == athlete_id]

        if len(athlete_train) > 0:
            max_train_year = athlete_train['Year of event'].max()
            if max_train_year >= 2022:
                issues.append({
                    'type': 'future_data_in_training',
                    'athlete_id': athlete_id,
                    'max_train_year': max_train_year
                })

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'test_athletes': len(test_athlete_ids),
        'train_athletes': len(train_pd['Athlete ID'].unique())
    }


def run_leakage_validation(processed_data_path: str = "data/processed/final_features.parquet") -> dict:
    """
    Run complete data leakage validation on processed data.

    Args:
        processed_data_path (str): Path to processed features parquet file

    Returns:
        dict: Complete validation results
    """
    print("🔍 Running Data Leakage Validation")
    print("=" * 50)

    results = {
        'file_exists': False,
        'temporal_integrity': None,
        'split_validation': None,
        'overall_valid': False
    }

    # Check if file exists
    if not Path(processed_data_path).exists():
        print(f"❌ Processed data file not found: {processed_data_path}")
        return results

    results['file_exists'] = True
    print(f"✅ Found processed data: {processed_data_path}")

    try:
        # Load data
        df = pl.read_parquet(processed_data_path)
        print(f"✅ Loaded {len(df)} records from {len(df.columns)} columns")

        # Validate temporal integrity
        print("\n🔄 Validating temporal integrity of features...")
        temporal_results = validate_temporal_integrity(df)
        results['temporal_integrity'] = temporal_results

        if temporal_results['valid']:
            print(f"✅ Temporal integrity OK - checked {temporal_results['total_athletes_checked']} athletes")
        else:
            print(f"❌ Found {len(temporal_results['issues'])} temporal integrity issues")

        # Validate train/test split
        print("\n🔄 Validating train/test split...")
        from src.data.split import split_train_test
        df_train, df_test, _ = split_train_test(df)

        split_results = validate_train_test_split(df_train, df_test)
        results['split_validation'] = split_results

        if split_results['valid']:
            print(f"✅ Train/test split OK - {split_results['train_athletes']} train athletes, {split_results['test_athletes']} test athletes")
        else:
            print(f"❌ Found {len(split_results['issues'])} split validation issues")

        # Overall result
        results['overall_valid'] = temporal_results['valid'] and split_results['valid']

        print("\n" + "=" * 50)
        if results['overall_valid']:
            print("🎉 ALL VALIDATION CHECKS PASSED - No data leakage detected!")
        else:
            print("⚠️  VALIDATION ISSUES FOUND - Review details above")

        return results

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        results['error'] = str(e)
        return results


if __name__ == "__main__":
    # Run validation on default file
    results = run_leakage_validation()

    # Print detailed issues if any
    if not results.get('overall_valid', False):
        print("\n📋 Detailed Issues:")

        if results.get('temporal_integrity') and not results['temporal_integrity']['valid']:
            print("\nTemporal Integrity Issues:")
            for issue in results['temporal_integrity']['issues'][:5]:  # Show first 5
                print(f"  - {issue}")

        if results.get('split_validation') and not results['split_validation']['valid']:
            print("\nSplit Validation Issues:")
            for issue in results['split_validation']['issues'][:5]:  # Show first 5
                print(f"  - {issue}")
