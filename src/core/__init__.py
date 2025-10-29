"""
Core module containing fundamental components and configuration.
"""

from .config import DashboardConfig, CAUSAL_ANALYSIS_PARAMS
from .data_handler import load_data_from_file, convert_date_columns

__all__ = ['DashboardConfig', 'CAUSAL_ANALYSIS_PARAMS', 'load_data_from_file', 'convert_date_columns']