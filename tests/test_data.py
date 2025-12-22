"""
Tests for data processing modules.

This module contains unit tests for the data loading, cleaning, and feature
engineering functions to ensure they work correctly.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the modules we want to test
from src.data.load import load_raw_data, load_sample_data
from src.data.clean import clean_data, clean_event_name, convert_to_seconds
from src.data.features import engineer_features


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def test_load_raw_data(self):
        """Test that load_raw_data returns a DataFrame."""
        # This would require a test CSV file, but we can test the function signature
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
            mock_read_csv.return_value = mock_df
            
            result = load_raw_data("test_path.csv")
            
            self.assertIsInstance(result, pd.DataFrame)
            mock_read_csv.assert_called_once_with("test_path.csv", low_memory=False)
    
    def test_load_sample_data(self):
        """Test that load_sample_data limits rows correctly."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]})
            mock_read_csv.return_value = mock_df
            
            result = load_sample_data("test_path.csv", n_rows=3)
            
            self.assertEqual(len(result), 3)
            mock_read_csv.assert_called_once_with("test_path.csv", low_memory=False, nrows=3)


class TestDataCleaning(unittest.TestCase):
    """Test cases for data cleaning functions."""
    
    def test_clean_event_name(self):
        """Test event name cleaning function."""
        test_cases = [
            ("Western States 2022", "western states"),
            ("Ultra Trail (FRA)", "ultra trail"),
            ("100km Race 2021", "100km race"),
            ("Test Event", "test event"),
            (None, None),
            ("", "")
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = clean_event_name(input_name)
                self.assertEqual(result, expected)
    
    def test_convert_to_seconds(self):
        """Test time conversion function."""
        test_cases = [
            ("4:30:15", 4*3600 + 30*60 + 15),
            ("1:15:30", 1*3600 + 15*60 + 30),
            ("2d 6:45:20", 2*86400 + 6*3600 + 45*60 + 20)
        ]
        
        for time_str, expected in test_cases:
            with self.subTest(time_str=time_str):
                result = convert_to_seconds(time_str)
                self.assertEqual(result, expected)
    
    def test_clean_data_basic(self):
        """Test basic data cleaning functionality."""
        # Create a mock DataFrame with some test data
        test_data = {
            'Event name': ['Test Race 2022', 'Another Race 2021'],
            'Event distance/length': ['100km', '50 miles'],
            'Athlete gender': ['M', 'F'],
            'Athlete year of birth': [1985, 1990],
            'Athlete age category': ['M35', 'F25'],
            'Athlete performance': ['4:30:15', '5:15:30'],
            'Year of event': [2022, 2021],
            'Event number of finishers': [100, 50],
            'Athlete club': ['Test Club', 'Another Club'],
            'Athlete country': ['USA', 'CAN'],
            'Athlete ID': [1, 2]
        }
        
        df = pd.DataFrame(test_data)
        
        # Test that the function runs without error
        try:
            cleaned_df = clean_data(df)
            self.assertIsInstance(cleaned_df, pd.DataFrame)
            # Check that some expected columns exist
            self.assertIn('pace_min_per_km', cleaned_df.columns)
            self.assertIn('Event distance_numeric', cleaned_df.columns)
        except Exception as e:
            self.fail(f"clean_data raised an exception: {e}")


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering functions."""
    
    def test_engineer_features_basic(self):
        """Test basic feature engineering functionality."""
        # Create a mock DataFrame with minimal required data
        test_data = {
            'Athlete ID': [1, 1, 2, 2],
            'Year of event': [2020, 2021, 2020, 2021],
            'Event name': ['Race A', 'Race B', 'Race A', 'Race C'],
            'Event distance_numeric': [50, 100, 50, 80],
            'pace_min_per_km': [10.0, 9.5, 11.0, 10.5],
            'Athlete year of birth': [1980, 1980, 1985, 1985]
        }
        
        df = pd.DataFrame(test_data)
        
        # Test that the function runs without error
        try:
            features_df = engineer_features(df)
            self.assertIsInstance(features_df, pd.DataFrame)
            
            # Check that expected feature columns exist
            expected_features = [
                'cum_num_races', 'cum_avg_pace', 'cum_best_pace',
                'cum_total_distance', 'cum_avg_distance', 'athlete_age'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features_df.columns, f"Missing feature: {feature}")
                
        except Exception as e:
            self.fail(f"engineer_features raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
