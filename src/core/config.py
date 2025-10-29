#!/usr/bin/env python3
"""
Dashboard Configuration Module
Centralized configuration management for the analytics dashboard.
"""

import pandas as pd
from typing import Optional, Dict, Any

# Global configuration parameters for causal analysis
CAUSAL_ANALYSIS_PARAMS = {
    'max_variables': 12,
    'max_samples': 1500,
    'max_iter': 100,
    'h_tol': 1e-8,
    'w_threshold': 0.3
}

class DashboardConfig:
    """
    Centralized configuration and state management for the dashboard.
    
    This class manages global state including:
    - Current dataset
    - Analysis results
    - User preferences
    - System settings
    """
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.causal_results: Optional[Dict[str, Any]] = None
        self.forecasting_results: Optional[Dict[str, Any]] = None
        self.visualization_settings: Dict[str, Any] = {}
        
    def reset(self):
        """Reset all stored data and results."""
        self.current_data = None
        self.causal_results = None
        self.forecasting_results = None
        self.visualization_settings = {}
        
    def has_data(self) -> bool:
        """Check if data is currently loaded."""
        return self.current_data is not None
        
    def get_numeric_columns(self) -> list:
        """Get list of numeric columns from current data."""
        if not self.has_data():
            return []
        return self.current_data.select_dtypes(include=['number']).columns.tolist()
        
    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the current dataset."""
        if not self.has_data():
            return {}
            
        return {
            'shape': self.current_data.shape,
            'columns': self.current_data.columns.tolist(),
            'numeric_columns': self.get_numeric_columns(),
            'missing_values': self.current_data.isnull().sum().to_dict(),
            'data_types': self.current_data.dtypes.to_dict()
        }

# Global configuration instance
dashboard_config = DashboardConfig()