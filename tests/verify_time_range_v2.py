#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from core import dashboard_config
    from core.data_handler import convert_date_columns
    from engines.visualization_engine import create_visualization
    from utils.data_utils import add_time_range_columns
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_time_range_logic():
    print("\n--- Testing Time-Range Logic ---")
    
    # 1. Create mock data
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 1000, size=100),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], size=100)
    })
    
    print(f"Initial columns: {df.columns.tolist()}")
    
    # 2. Test convert_date_columns (which calls add_time_range_columns)
    df_processed = convert_date_columns(df)
    
    print(f"Processed columns: {df_processed.columns.tolist()}")
    
    expected_cols = ['Date_Month', 'Date_Quarter', 'Date_Year']
    for col in expected_cols:
        if col in df_processed.columns:
            print(f"✅ Found new column: {col}")
            print(f"   Sample value: {df_processed[col].iloc[0]}")
        else:
            print(f"❌ Missing column: {col}")
            return False

    # 3. Test visualization engine swapping
    # Set as global data
    dashboard_config.current_data = df_processed
    
    print("\n--- Testing Visualization Swapping ---")
    
    # Test case: Color var is a date
    print("Case 1: Using 'Date' as color_var in Scatter Plot")
    # create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)
    # Note: Scatter Plot doesn't swap x-axis, but should swap color_var if it's a date
    fig = create_visualization('Region', 'Sales', 'Date', 'Scatter Plot', 'Light', 'Raw Data')
    # If swap happened, the print in visualization_engine.py should have triggered.
    # We can also check the figure's data if possible, but the simplest is checking the logic.
    
    # Test case: x_axis is a date for Bar Chart
    print("\nCase 2: Using 'Date' as x_axis in Bar Chart")
    fig2 = create_visualization('Date', 'Sales', 'Region', 'Bar Chart', 'Light', 'Raw Data')
    
    # Test case: x_axis is a date for Histogram
    print("\nCase 3: Using 'Date' as x_axis in Histogram")
    fig3 = create_visualization('Date', None, 'Region', 'Histogram', 'Light', 'Raw Data')

    print("\n✅ Verification script finished")
    return True

if __name__ == "__main__":
    success = test_time_range_logic()
    sys.exit(0 if success else 1)
