import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from engines.forecasting_engine import (
    prepare_time_series_data,
    linear_regression_forecast,
    nowcasting_forecast,
    arima_forecast,
    sarima_forecast,
    var_forecast,
    dynamic_factor_forecast,
    state_space_forecast,
    lstm_forecast
)

class TestForecastingEngine(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset for testing
        np.random.seed(42)
        n_obs = 50
        self.df = pd.DataFrame({
            'Sales': np.linspace(100, 200, n_obs) + np.random.normal(0, 5, n_obs),
            'Marketing': np.linspace(10, 20, n_obs) + np.random.normal(0, 2, n_obs),
            'Price': np.random.normal(50, 2, n_obs)
        })
        self.target_col = 'Sales'

    def test_prepare_time_series_data(self):
        # Test with no datetime index
        ts_data = prepare_time_series_data(self.df, self.target_col)
        self.assertIsInstance(ts_data.index, pd.DatetimeIndex)
        self.assertEqual(len(ts_data), len(self.df))
        self.assertIn(self.target_col, ts_data.columns)

        # Test with additional columns
        ts_data = prepare_time_series_data(self.df, self.target_col, additional_cols=['Marketing'])
        self.assertIn('Marketing', ts_data.columns)
        self.assertIn('Sales', ts_data.columns)

    def test_linear_regression_forecast(self):
        ts_data = prepare_time_series_data(self.df, self.target_col)
        periods = 5
        result = linear_regression_forecast(ts_data, self.target_col, periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertEqual(len(result['lower_bound']), periods)
        self.assertEqual(len(result['upper_bound']), periods)
        self.assertIn('model_info', result)
        self.assertIn('slope', result['model_info'])

    def test_nowcasting_forecast(self):
        ts_data = prepare_time_series_data(self.df, self.target_col)
        periods = 5
        result = nowcasting_forecast(ts_data, self.target_col, periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertIn('method', result['model_info'])
        self.assertEqual(result['model_info']['method'], 'Exponential Smoothing')

    def test_arima_forecast(self):
        try:
            import statsmodels
        except ImportError:
            self.skipTest("statsmodels not installed")
            
        ts_data = prepare_time_series_data(self.df, self.target_col)
        periods = 5
        result = arima_forecast(ts_data, self.target_col, periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertIn('order', result['model_info'])

    def test_sarima_forecast(self):
        try:
            import statsmodels
        except ImportError:
            self.skipTest("statsmodels not installed")
            
        ts_data = prepare_time_series_data(self.df, self.target_col)
        periods = 5
        seasonal_period = 12
        result = sarima_forecast(ts_data, self.target_col, periods, seasonal_period)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertEqual(result['model_info']['seasonal_period'], seasonal_period)

    def test_var_forecast(self):
        try:
            import statsmodels
        except ImportError:
            self.skipTest("statsmodels not installed")
            
        ts_data = prepare_time_series_data(self.df, self.target_col, additional_cols=['Marketing'])
        periods = 5
        result = var_forecast(ts_data, self.target_col, ['Marketing'], periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertIn('lag_order', result['model_info'])

    def test_dynamic_factor_forecast(self):
        # Needs at least 3 variables
        df_df = self.df.copy()
        df_df['Extra'] = np.random.normal(0, 1, len(df_df))
        ts_data = prepare_time_series_data(df_df, self.target_col, additional_cols=['Marketing', 'Price', 'Extra'])
        periods = 5
        result = dynamic_factor_forecast(ts_data, self.target_col, ['Marketing', 'Price', 'Extra'], periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertIn('n_factors', result['model_info'])

    def test_state_space_forecast(self):
        try:
            import statsmodels
        except ImportError:
            self.skipTest("statsmodels not installed")
            
        ts_data = prepare_time_series_data(self.df, self.target_col)
        periods = 5
        result = state_space_forecast(ts_data, self.target_col, periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertIn('components', result['model_info'])

    def test_lstm_forecast(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
            
        ts_data = prepare_time_series_data(self.df, self.target_col)
        periods = 5
        result = lstm_forecast(ts_data, self.target_col, periods)
        
        self.assertEqual(len(result['forecast']), periods)
        self.assertIn('method', result['model_info'])
        self.assertEqual(result['model_info']['method'], 'LSTM (PyTorch)')

    def test_prepare_time_series_data_insufficient(self):
        # Test with insufficient data
        df_short = self.df.iloc[:2]
        with self.assertRaises(ValueError):
            prepare_time_series_data(df_short, self.target_col)

    def test_prepare_time_series_data_existing_datetime(self):
        # Test with existing datetime index
        df_dt = self.df.copy()
        df_dt['date'] = pd.date_range('2023-01-01', periods=len(df_dt), freq='D')
        df_dt = df_dt.set_index('date')
        ts_data = prepare_time_series_data(df_dt, self.target_col)
        self.assertIsInstance(ts_data.index, pd.DatetimeIndex)
        self.assertEqual(ts_data.index[0], pd.Timestamp('2023-01-01'))

    def test_prepare_time_series_data_nans(self):
        # Test with NaNs
        df_nan = self.df.copy()
        df_nan.loc[0, self.target_col] = np.nan
        ts_data = prepare_time_series_data(df_nan, self.target_col)
        self.assertEqual(len(ts_data), len(df_nan) - 1)

    def test_prepare_time_series_data_large(self):
        # Test with large dataset (to check 10,000 limit)
        n_large = 11000
        df_large = pd.DataFrame({
            'Sales': np.random.normal(100, 10, n_large)
        })
        ts_data = prepare_time_series_data(df_large, self.target_col)
        # Should be truncated to 10,000
        self.assertEqual(len(ts_data.index), 10000)
        self.assertIsInstance(ts_data.index, pd.DatetimeIndex)

    def test_perform_forecasting(self):
        # Mock dashboard_config.current_data
        from core import dashboard_config
        old_data = dashboard_config.current_data
        dashboard_config.current_data = self.df
        
        try:
            from engines.forecasting_engine import perform_forecasting
            # Use a fast model
            fig, summary, metrics = perform_forecasting(
                target_var=self.target_col,
                additional_vars=[],
                model_type="Linear Regression",
                periods=5,
                seasonal_period=12,
                confidence_level=0.95,
                progress=lambda x, desc=None: None
            )
            
            self.assertIsNotNone(fig)
            self.assertIn(self.target_col, summary)
            self.assertIn("Detailed Forecast Values", metrics)
        finally:
            dashboard_config.current_data = old_data

if __name__ == '__main__':
    unittest.main()
