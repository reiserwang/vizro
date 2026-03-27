#!/usr/bin/env python3
"""
Data Utilities Module
Provides helper functions for date/time range generation and data transformation.
"""

import pandas as pd
import numpy as np

def get_time_range_series(series, range_type='Month'):
    """
    Generate range-based string labels for a datetime series.
    
    Args:
        series: pandas Series of datetime objects
        range_type: 'Month', 'Quarter', or 'Year'
        
    Returns:
        pandas Series of strings representing the ranges
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            series = pd.to_datetime(series, errors='coerce')
        except Exception:
            return series

    if range_type == 'Month':
        return series.dt.strftime('%b-%Y') # e.g., Jan-2015
    elif range_type == 'Quarter':
        return series.dt.year.astype(str) + '-Q' + series.dt.quarter.astype(str) # e.g., 2015-Q1
    elif range_type == 'Year':
        return series.dt.year.astype(str)
    else:
        return series.astype(str)

def add_time_range_columns(df, col):
    """
    Add multiple range-based columns for a given datetime column.
    
    Args:
        df: pandas DataFrame
        col: Name of the datetime column
        
    Returns:
        DataFrame with new range columns
    """
    if col not in df.columns:
        return df
        
    # Ensure it's datetime
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            return df
            
    # Add ranges
    df[f"{col}_Month"] = get_time_range_series(df[col], 'Month')
    df[f"{col}_Quarter"] = get_time_range_series(df[col], 'Quarter')
    df[f"{col}_Year"] = get_time_range_series(df[col], 'Year')
    
    return df
