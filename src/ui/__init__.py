"""
User interface module containing dashboard components and settings management.
"""

from .dashboard import create_gradio_interface
from .settings_manager import SettingsManager

__all__ = ['create_gradio_interface', 'SettingsManager']
