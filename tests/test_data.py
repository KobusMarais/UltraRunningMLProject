"""
Tests for data processing modules and data leakage validation.

This module contains unit tests for the data processing pipeline with a focus
on preventing data leakage in the ultramarathon pace prediction model.
"""

import unittest
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

# Import the modules we want to test
from src.data.load import load_raw_data
from src.data.clean import clean_data
from src.data.sort_data import sort_chronologically
from src.features.build_features import engineer_features
from src.data.split import split_train_test


class TestDataLeakagePrevention(unittest.TestCase):
    """Test cases for data leakage prevention in the pipeline."""

    def setUp(self):
        """Set up test data for leakage validation."""
        # Create test data that simulates potential leakage scenarios
        # Include all columns that clean_data expects to drop
        self.test_data = {
            'Year of event': [2020, 2021, 2022, 2020, 2021, 2022, 2021, 2022],
            'Event dates': ['01.01.2020', '01.01.2021', '01.01.2022', '01.01.2020', '01.01.2021', '01.01.2022', '01.01.2021', '01.01.2022'],
            'Event name': ['Race A', 'Race A', 'Western States', 'Race B', 'Race B', 'Race C', 'Race D', 'Western States'],
            'Event distance/length': ['50km', '50km', '100km', '80km', '80km', '60km', '70km', '100km'],
            'Event number of finishers': [100, 120, 350, 80, 90, 60, 110, 320],
            'Athlete performance': ['3:30:00', '3:15:00', '8:00:00', '4:00:00', '3:45:00', '3:20:00', '3:50:00', '7:45:00'],
            'Athlete club': ['', '', '', '', '', '', '', ''],  # Will be dropped
            'Athlete country': ['', '', '', '', '', '', '', ''],  # Will be dropped
            'Athlete year of birth': [1985, 1985, 1985, 1990, 1990, 1990, 1980, 1980],
            'Athlete gender': ['M', 'M', 'M', 'F', 'F', 'F', 'M', 'M'],
            'Athlete age category': ['M35', 'M35', 'M37', 'F30', 'F31', 'F32', 'M41', 'M42'],
            'Athlete average speed': [10.0, 10.5, 12.5, 10.0, 10.2, 10.0, 9.8, 13.2],
            'Athlete ID': [1, 1, 1, 2, 2, 2, 3, 3]  # Athlete 1 and 3 in Western States 2022
        }

    def test_chronological_sorting(self):
        """Test that data is properly sorted chronologically."""
        df = pl.DataFrame(self.test_data)

        # Sort chronologically
        sorted_df = sort_chronologically(df.lazy()).collect()

        # Check that athlete data is sorted by year
        athlete_1_data = sorted_df.filter(pl.col('Athlete ID') == 1)
        years_athlete_1 = athlete_1_data['Year of event'].to_list()
        self.assertEqual(years_athlete_1, sorted(years_athlete_1), "Athlete data not chronologically sorted")

    def test_cumulative_features_exclude_current_race(self):
        """Test that cumulative features don't include the current race's data."""
        df = pl.DataFrame(self.test_data)

        # Clean and engineer features
        cleaned_lf = clean_data(df.lazy())
        sorted_lf = sort_chronologically(cleaned_lf)
        features_df = engineer_features(sorted_lf).collect()

        # Convert to pandas for easier testing
        features_pd = features_df.to_pandas()

        # For athlete 1's 2022 Western States race, cumulative features should exclude 2022 data
        ws_2022_athlete_1 = features_pd[
            (features_pd['Athlete ID'] == 1) &
            (features_pd['Year of event'] == 2022) &
            (features_pd['Event name'].str.contains('Western States'))
        ]

        if len(ws_2022_athlete_1) > 0:
            # cum_num_races should be 2 (2020 + 2021 races only)
            self.assertEqual(ws_2022_athlete_1['cum_num_races'].iloc[0], 2,
                           "cum_num_races includes current race - data leakage!")

            # cum_avg_pace should be average of 2020 and 2021 races only
            athlete_1_prior = features_pd[
                (features_pd['Athlete ID'] == 1) &
                (features_pd['Year of event'] < 2022)
            ]
            expected_avg_pace = athlete_1_prior['pace_min_per_km'].mean()
            actual_avg_pace = ws_2022_athlete_1['cum_avg_pace'].iloc[0]

            # Allow small floating point differences
            self.assertAlmostEqual(actual_avg_pace, expected_avg_pace, places=5,
                                 msg="cum_avg_pace includes current race - data leakage!")

    def test_train_test_split_prevents_leakage(self):
        """Test that train/test split prevents data leakage from future races."""
        df = pl.DataFrame(self.test_data)

        # Process data through pipeline
        cleaned_lf = clean_data(df.lazy())
        sorted_lf = sort_chronologically(cleaned_lf)
        features_df = engineer_features(sorted_lf).collect()

        # Split data
        df_train, df_test, feature_cols = split_train_test(features_df)

        # Test set should only contain Western States 2022
        test_events = df_test['Event name'].unique().to_list()
        test_years = df_test['Year of event'].unique().to_list()

        self.assertIn('Western States', test_events, "Western States not in test set")
        self.assertEqual(test_years, [2022], "Test set contains non-2022 data")

        # Check that athletes in test set have their pre-2022 data properly excluded
        test_athlete_ids = df_test['Athlete ID'].unique().to_list()

        for athlete_id in test_athlete_ids:
            # Get this athlete's training data
            athlete_train = df_train.filter(pl.col('Athlete ID') == athlete_id)

            if len(athlete_train) > 0:
                # Training data should only include races before the athlete's test race cutoff
                max_train_year = athlete_train['Year of event'].max()
                self.assertLess(max_train_year, 2022,
                              f"Athlete {athlete_id} has {max_train_year} data in training set - leakage!")

    def test_target_encoding_per_fold(self):
        """Test that target encoding is done properly per cross-validation fold."""
        # This is more complex to test directly, but we can verify the function exists
        # and that the CV pipeline uses it correctly
        from src.models.pipeline_utils import apply_smoothed_target_encoding

        # Create small test dataset
        fold_data = pd.DataFrame({
            'Event_name_clean': ['race_a', 'race_a', 'race_b', 'race_b'],
            'pace_min_per_km': [10.0, 9.5, 11.0, 10.5]
        })

        train_data = fold_data.iloc[:2].copy()
        val_data = fold_data.iloc[2:].copy()

        # Apply encoding
        train_encoded, val_encoded = apply_smoothed_target_encoding(train_data, val_data)

        # Validation set should get global mean for unseen categories
        global_mean = train_data['pace_min_per_km'].mean()
        val_encoded_pace = val_encoded['Race_Pace_Mean_Encoded'].iloc[0]

        # Should be close to global mean (allowing for smoothing factor)
        self.assertAlmostEqual(val_encoded_pace, global_mean, delta=2.0,
                             msg="Target encoding not properly handling unseen categories")


class TestPipelineIntegrity(unittest.TestCase):
    """Test cases for overall pipeline integrity."""

    def setUp(self):
        """Set up test data for integrity tests."""
        # Create test data that simulates potential leakage scenarios
        # Include all columns that clean_data expects
        self.test_data = {
            'Year of event': [2020, 2021, 2022, 2020, 2021, 2022, 2021, 2022],
            'Event dates': ['01.01.2020', '01.01.2021', '01.01.2022', '01.01.2020', '01.01.2021', '01.01.2022', '01.01.2021', '01.01.2022'],
            'Event name': ['Race A', 'Race A', 'Western States', 'Race B', 'Race B', 'Race C', 'Race D', 'Western States'],
            'Event distance/length': ['50km', '50km', '100km', '80km', '80km', '60km', '70km', '100km'],
            'Event number of finishers': [100, 120, 350, 80, 90, 60, 110, 320],
            'Athlete performance': ['3:30:00', '3:15:00', '8:00:00', '4:00:00', '3:45:00', '3:20:00', '3:50:00', '7:45:00'],
            'Athlete club': ['', '', '', '', '', '', '', ''],  # Will be dropped
            'Athlete country': ['', '', '', '', '', '', '', ''],  # Will be dropped
            'Athlete year of birth': [1985, 1985, 1985, 1990, 1990, 1990, 1980, 1980],
            'Athlete gender': ['M', 'M', 'M', 'F', 'F', 'F', 'M', 'M'],
            'Athlete age category': ['M35', 'M35', 'M37', 'F30', 'F31', 'F32', 'M41', 'M42'],
            'Athlete average speed': [10.0, 10.5, 12.5, 10.0, 10.2, 10.0, 9.8, 13.2],
            'Athlete ID': [1, 1, 1, 2, 2, 2, 3, 3]  # Athlete 1 and 3 in Western States 2022
        }

    def test_pipeline_produces_expected_columns(self):
        """Test that the full pipeline produces expected feature columns."""
        df = pl.DataFrame(self.test_data)

        # Run through pipeline
        cleaned_lf = clean_data(df.lazy())
        sorted_lf = sort_chronologically(cleaned_lf)
        features_df = engineer_features(sorted_lf).collect()

        # Check for critical columns
        expected_columns = [
            'pace_min_per_km', 'Event distance_numeric', 'Event_name_clean',
            'cum_num_races', 'cum_avg_pace', 'cum_best_pace',
            'cum_total_distance', 'athlete_age'
        ]

        for col in expected_columns:
            self.assertIn(col, features_df.columns, f"Missing expected column: {col}")

    def test_no_future_data_leakage_in_features(self):
        """Test that engineered features don't accidentally include future data."""
        df = pl.DataFrame(self.test_data)

        # Process data
        cleaned_lf = clean_data(df.lazy())
        sorted_lf = sort_chronologically(cleaned_lf)
        features_df = engineer_features(sorted_lf).collect()
        features_pd = features_df.to_pandas()

        # For each athlete, check that features for year Y don't include data from year > Y
        for athlete_id in features_pd['Athlete ID'].unique():
            athlete_data = features_pd[features_pd['Athlete ID'] == athlete_id].sort_values('Year of event')

            for idx, row in athlete_data.iterrows():
                current_year = row['Year of event']

                # Get all of this athlete's races up to but not including current year
                prior_races = athlete_data[athlete_data['Year of event'] < current_year]

                if len(prior_races) > 0:
                    # Check cum_num_races
                    expected_cum_races = len(prior_races)
                    actual_cum_races = row['cum_num_races']
                    self.assertEqual(actual_cum_races, expected_cum_races,
                                   f"cum_num_races leakage for athlete {athlete_id} in {current_year}")

                    # Check cum_avg_pace (should be mean of prior paces)
                    if not pd.isna(row['cum_avg_pace']):
                        expected_avg_pace = prior_races['pace_min_per_km'].mean()
                        actual_avg_pace = row['cum_avg_pace']
                        self.assertAlmostEqual(actual_avg_pace, expected_avg_pace, places=5,
                                             msg=f"cum_avg_pace leakage for athlete {athlete_id} in {current_year}")


if __name__ == '__main__':
    unittest.main()
