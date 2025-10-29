#!/usr/bin/env python3
"""
Settings Manager Module
Handles loading and managing dashboard configuration
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class SettingsManager:
    """Manages dashboard settings from JSON configuration file"""
    
    def __init__(self, settings_file: str = "dashboard_settings.json"):
        self.settings_file = settings_file
        self.settings = {}
        self.load_settings()
    
    def load_settings(self) -> None:
        """Load settings from JSON file with fallback defaults"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
                print(f"✅ Settings loaded from {self.settings_file}")
            else:
                print(f"⚠️ Settings file {self.settings_file} not found, using defaults")
                self.settings = self._get_default_settings()
                self.save_settings()  # Create default settings file
        except Exception as e:
            print(f"❌ Error loading settings: {e}")
            print("   Using default settings")
            self.settings = self._get_default_settings()
    
    def save_settings(self) -> None:
        """Save current settings to JSON file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"✅ Settings saved to {self.settings_file}")
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get setting value using dot notation (e.g., 'server.port')
        
        Args:
            key_path: Dot-separated path to setting (e.g., 'server.port')
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        try:
            keys = key_path.split('.')
            value = self.settings
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
        except Exception:
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set setting value using dot notation
        
        Args:
            key_path: Dot-separated path to setting
            value: Value to set
        """
        try:
            keys = key_path.split('.')
            current = self.settings
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the final value
            current[keys[-1]] = value
            
        except Exception as e:
            print(f"❌ Error setting {key_path}: {e}")
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration for Gradio launch"""
        return {
            'server_name': self.get('server.host', '0.0.0.0'),
            'server_port': self.get('server.port', 7860),
            'share': self.get('server.share', False),
            'debug': self.get('server.debug', False),
            'show_error': self.get('server.show_error', True),
            'quiet': self.get('server.quiet', False),
            'inbrowser': self.get('server.inbrowser', True),
            'ssl_verify': self.get('server.ssl_verify', False),
            'favicon_path': self.get('server.favicon_path'),
            'auth': self.get('server.auth'),
            'auth_message': self.get('server.auth_message'),
            'max_threads': self.get('server.max_threads', 40),
            'show_tips': self.get('server.show_tips', False),
            'height': self.get('server.height', 500),
            'width': self.get('server.width', "100%"),
            'prevent_thread_lock': self.get('server.prevent_thread_lock', False)
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            'title': self.get('ui.title', '🔍 Dynamic Data Analysis Dashboard'),
            'description': self.get('ui.description', 'Advanced Analytics Platform'),
            'theme': self.get('ui.theme', 'default'),
            'css_file': self.get('ui.css_file'),
            'analytics_enabled': self.get('ui.analytics_enabled', False),
            'show_api': self.get('ui.show_api', True),
            'show_error': self.get('ui.show_error', True),
            'layout': self.get('ui.layout', {}),
            'colors': self.get('ui.colors', {})
        }
    
    def get_causal_config(self) -> Dict[str, Any]:
        """Get causal analysis configuration"""
        return {
            'max_variables': self.get('causal_analysis.max_variables', 12),
            'max_samples': self.get('causal_analysis.max_samples', 1500),
            'notears_params': self.get('causal_analysis.notears_params', {
                'max_iter': 100,
                'h_tol': 1e-8,
                'w_threshold': 0.3
            }),
            'significance_threshold': self.get('causal_analysis.significance_threshold', 0.05),
            'correlation_thresholds': self.get('causal_analysis.correlation_thresholds', {
                'strong': 0.7,
                'moderate': 0.3,
                'weak': 0.1
            }),
            'discretization': self.get('causal_analysis.discretization', {}),
            'network_layout': self.get('causal_analysis.network_layout', {}),
            'progress_steps': self.get('causal_analysis.progress_steps', [])
        }
    
    def get_forecasting_config(self) -> Dict[str, Any]:
        """Get forecasting configuration"""
        return {
            'default_model': self.get('forecasting.default_model', 'Linear Regression'),
            'default_periods': self.get('forecasting.default_periods', 12),
            'default_confidence_level': self.get('forecasting.default_confidence_level', 0.95),
            'default_seasonal_period': self.get('forecasting.default_seasonal_period', 12),
            'max_forecast_periods': self.get('forecasting.max_forecast_periods', 100),
            'min_data_points': self.get('forecasting.min_data_points', 3),
            'recommended_min_points': self.get('forecasting.recommended_min_points', 10),
            'models': self.get('forecasting.models', {}),
            'plot_settings': self.get('forecasting.plot_settings', {})
        }
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return {
            'default_theme': self.get('visualization.default_theme', 'Light'),
            'chart_height': self.get('visualization.chart_height', 600),
            'enhanced_chart_height': self.get('visualization.enhanced_chart_height', 700),
            'default_chart_type': self.get('visualization.default_chart_type', 'Enhanced Scatter Plot'),
            'color_palette': self.get('visualization.color_palette', []),
            'show_correlation_threshold': self.get('visualization.show_correlation_threshold', 0.3),
            'max_categories_in_legend': self.get('visualization.max_categories_in_legend', 10),
            'animation_duration': self.get('visualization.animation_duration', 500),
            'hover_mode': self.get('visualization.hover_mode', 'closest'),
            'margin': self.get('visualization.margin', {})
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.get(f'features.enable_{feature}', True)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data handling configuration"""
        return {
            'max_file_size_mb': self.get('data.max_file_size_mb', 100),
            'supported_formats': self.get('data.supported_formats', ['.csv', '.xlsx', '.xls']),
            'preview_rows': self.get('data.preview_rows', 20),
            'max_columns_display': self.get('data.max_columns_display', 50),
            'auto_detect_types': self.get('data.auto_detect_types', True),
            'encoding': self.get('data.encoding', 'utf-8'),
            'decimal_separator': self.get('data.decimal_separator', '.'),
            'thousands_separator': self.get('data.thousands_separator', ','),
            'date_formats': self.get('data.date_formats', [])
        }
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings if no config file exists"""
        return {
            "server": {
                "port": 7860,
                "host": "0.0.0.0",
                "share": False,
                "debug": False,
                "show_error": True,
                "quiet": False,
                "inbrowser": True,
                "ssl_verify": False
            },
            "ui": {
                "title": "🔍 Dynamic Data Analysis Dashboard",
                "description": "Advanced Analytics Platform",
                "theme": "default"
            },
            "causal_analysis": {
                "max_variables": 12,
                "max_samples": 1500,
                "notears_params": {
                    "max_iter": 100,
                    "h_tol": 1e-8,
                    "w_threshold": 0.3
                }
            },
            "forecasting": {
                "default_model": "Linear Regression",
                "default_periods": 12,
                "default_confidence_level": 0.95
            },
            "visualization": {
                "default_theme": "Light",
                "chart_height": 600,
                "default_chart_type": "Enhanced Scatter Plot"
            },
            "features": {
                "enable_vizro": True,
                "enable_forecasting": True,
                "enable_causal_analysis": True,
                "enable_intervention_analysis": True,
                "enable_pathway_analysis": True,
                "enable_data_insights": True,
                "enable_export": True
            }
        }
    
    def update_from_env(self) -> None:
        """Update settings from environment variables"""
        env_mappings = {
            'DASHBOARD_PORT': 'server.port',
            'DASHBOARD_HOST': 'server.host',
            'DASHBOARD_DEBUG': 'server.debug',
            'DASHBOARD_SHARE': 'server.share',
            'DASHBOARD_THEME': 'ui.theme',
            'MAX_VARIABLES': 'causal_analysis.max_variables',
            'MAX_SAMPLES': 'causal_analysis.max_samples',
            'DEFAULT_FORECAST_PERIODS': 'forecasting.default_periods'
        }
        
        for env_var, setting_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_var.endswith('_PORT') or env_var.startswith('MAX_') or 'PERIODS' in env_var:
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        continue
                elif env_var.endswith('_DEBUG') or env_var.endswith('_SHARE'):
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                
                self.set(setting_path, env_value)
                print(f"✅ Updated {setting_path} from environment: {env_value}")
    
    def validate_settings(self) -> bool:
        """Validate settings and return True if valid"""
        try:
            # Validate port range
            port = self.get('server.port', 7860)
            if not (1024 <= port <= 65535):
                print(f"⚠️ Invalid port {port}, using default 7860")
                self.set('server.port', 7860)
            
            # Validate max variables
            max_vars = self.get('causal_analysis.max_variables', 12)
            if max_vars < 3 or max_vars > 50:
                print(f"⚠️ Invalid max_variables {max_vars}, using default 12")
                self.set('causal_analysis.max_variables', 12)
            
            # Validate forecast periods
            periods = self.get('forecasting.default_periods', 12)
            if periods < 1 or periods > 100:
                print(f"⚠️ Invalid forecast periods {periods}, using default 12")
                self.set('forecasting.default_periods', 12)
            
            return True
            
        except Exception as e:
            print(f"❌ Settings validation error: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get dashboard information"""
        return {
            'version': self.get('help.version', '2.0.0'),
            'build_date': self.get('help.build_date', '2024-01-01'),
            'settings_file': self.settings_file,
            'features_enabled': {
                feature: self.is_feature_enabled(feature.replace('enable_', ''))
                for feature in self.get('features', {}).keys()
                if feature.startswith('enable_')
            }
        }

# Global settings instance
settings = SettingsManager()

# Convenience functions for common settings
def get_server_config():
    """Get server configuration for Gradio launch"""
    return settings.get_server_config()

def get_causal_params():
    """Get causal analysis parameters"""
    config = settings.get_causal_config()
    return config['notears_params']

def get_chart_height():
    """Get default chart height"""
    return settings.get('visualization.chart_height', 600)

def is_vizro_enabled():
    """Check if Vizro features are enabled"""
    return settings.is_feature_enabled('vizro')

def get_default_theme():
    """Get default visualization theme"""
    return settings.get('visualization.default_theme', 'Light')