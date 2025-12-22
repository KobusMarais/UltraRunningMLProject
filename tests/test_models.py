"""
Tests for model training and evaluation modules.

This module contains unit tests for the model training and evaluation functions.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the modules we want to test
from src.models.train import train_evaluate_lgbm
from src.evaluation.metrics import calculate_pace_metrics, print_pace_metrics


class TestModelTraining(unittest.TestCase):
    """Test cases for model training functions."""
    
    def test_calculate_pace_metrics(self):
        """Test pace metrics calculation."""
        # Create simple test data
        y_true = np.array([10.0, 9.5, 11.0, 10.5])
        y_pred = np.array([10.2, 9.3, 11.1, 10.4])
        
        metrics = calculate_pace_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE', 'MAE_percentage', 
                          'Accuracy_30s', 'Accuracy_1min', 'Accuracy_2min']
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that MAE is reasonable
        self.assertGreater(metrics['MAE'], 0)
        self.assertLess(metrics['MAE'], 1)  # Should be small for this test data
        
        # Check that R² is between -1 and 1
        self.assertGreaterEqual(metrics['R²'], -1)
        self.assertLessEqual(metrics['R²'], 1)
    
    def test_print_pace_metrics(self):
        """Test pace metrics printing (just check it doesn't crash)."""
        y_true = np.array([10.0, 9.5, 11.0, 10.5])
        y_pred = np.array([10.2, 9.3, 11.1, 10.4])
        
        # This should not raise an exception
        try:
            print_pace_metrics(y_true, y_pred, "Test Model")
        except Exception as e:
            self.fail(f"print_pace_metrics raised an exception: {e}")
    
    def test_calculate_pace_metrics_perfect_prediction(self):
        """Test pace metrics with perfect predictions."""
        y_true = np.array([10.0, 9.5, 11.0, 10.5])
        y_pred = y_true.copy()  # Perfect predictions
        
        metrics = calculate_pace_metrics(y_true, y_pred)
        
        # For perfect predictions, MAE and RMSE should be very close to 0
        self.assertAlmostEqual(metrics['MAE'], 0, places=10)
        self.assertAlmostEqual(metrics['RMSE'], 0, places=10)
        
        # R² should be 1 for perfect predictions
        self.assertAlmostEqual(metrics['R²'], 1, places=10)
        
        # Accuracy should be 100% for all thresholds
        self.assertEqual(metrics['Accuracy_30s'], 100.0)
        self.assertEqual(metrics['Accuracy_1min'], 100.0)
        self.assertEqual(metrics['Accuracy_2min'], 100.0)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    @patch('lightgbm.LGBMRegressor')
    def test_train_evaluate_lgbm_mock(self, mock_lgbm):
        """Test train_evaluate_lgbm with mocked LightGBM."""
        # Create test data
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train = pd.Series([10, 11, 12])
        X_test = pd.DataFrame({'feature1': [4, 5], 'feature2': [7, 8]})
        y_test = pd.Series([13, 14])
        
        # Mock the model and its methods
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([12.8, 13.9])  # Close predictions
        mock_lgbm.return_value = mock_model
        
        # Test the function
        result_model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test)
        
        # Check that the model was fitted
        mock_model.fit.assert_called_once_with(X_train, y_train)
        
        # Check that predictions were made
        mock_model.predict.assert_called_once_with(X_test)
        
        # Check that we get reasonable results
        self.assertIsNotNone(result_model)
        self.assertEqual(len(y_pred), len(y_test))


if __name__ == '__main__':
    unittest.main()
