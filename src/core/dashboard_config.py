#!/usr/bin/env python3
"""
Dashboard Configuration and Constants
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Detect if running inside PyInstaller native .app bundle
_IS_FROZEN = getattr(sys, 'frozen', False)

# Global variables for data storage
current_data = None
causal_results = None

# Vizro availability check
def check_vizro_availability():
    """Check if Vizro is available and import components"""
    try:
        import vizro
        print(f"📊 Vizro base package loaded (version: {getattr(vizro, '__version__', 'unknown')})")
        
        # Try to import additional Vizro components
        try:
            from vizro import Vizro
            from vizro.models import Dashboard, Page, Graph
            print("📊 Vizro models imported successfully")
        except ImportError as e:
            print(f"⚠️ Some Vizro models not available: {e}")
        
        try:
            from vizro.models import Card, Container
            from vizro.models.types import capture
            print("📊 Vizro advanced models imported successfully")
        except ImportError as e:
            print(f"⚠️ Some Vizro advanced models not available: {e}")
        
        try:
            from vizro.figures import kpi_card
            print("📊 Vizro figures imported successfully")
        except ImportError as e:
            print(f"⚠️ Some Vizro figures not available: {e}")
        
        try:
            from vizro.figures import waterfall_chart
            print("📊 Vizro waterfall_chart available")
        except ImportError:
            print("ℹ️ Vizro waterfall_chart not available in this version")
        
        try:
            import vizro.plotly.express as vpx
            print("📊 Vizro plotly express imported successfully")
        except ImportError as e:
            print(f"⚠️ Vizro plotly express not available: {e}")
        
        print("✅ Vizro integration enabled - Enhanced visualizations available!")
        return True
        
    except ImportError as e:
        print(f"⚠️ Vizro not available: {e}")
        print("   Using standard Plotly visualizations")
        return False

# Initialize Vizro availability
VIZRO_AVAILABLE = check_vizro_availability()

# Dashboard themes and styling
DASHBOARD_THEMES = {
    'light': 'plotly_white',
    'dark': 'plotly_dark'
}

# Chart type configurations
STANDARD_CHART_TYPES = [
    "Scatter Plot",
    "Line Chart", 
    "Bar Chart",
    "Histogram"
]

VIZRO_ENHANCED_CHART_TYPES = [
    "Enhanced Scatter Plot",
    "Statistical Box Plot",
    "Correlation Heatmap", 
    "Distribution Analysis",
    "Time Series Analysis",
    "Advanced Bar Chart"
]

# Forecasting model configurations
# LSTM is excluded from the native .app build (torch not bundled, ~276 MB savings).
# It remains available in web/api mode when torch is installed.
_FORECASTING_MODELS_CORE = [
    "Linear Regression",
    "ARIMA",
    "SARIMA",
    "VAR (Vector Autoregression)",
    "Dynamic Factor Model",
    "State-Space Model",
    "Nowcasting",
]

_FORECASTING_MODELS_OPTIONAL = [
    "LSTM (Deep Learning)",   # Requires torch (~276 MB) — web mode only
]

FORECASTING_MODELS = (
    _FORECASTING_MODELS_CORE
    if _IS_FROZEN
    else _FORECASTING_MODELS_CORE + _FORECASTING_MODELS_OPTIONAL
)

# Y-axis aggregation options
Y_AXIS_AGGREGATIONS = [
    "Raw Data",
    "Average",
    "Sum", 
    "Count"
]

# Causal analysis parameters
CAUSAL_ANALYSIS_PARAMS = {
    'max_iter': 100,
    'h_tol': 1e-8,
    'w_threshold': 0.3,
    'max_variables': 12,
    'max_samples': 1500
}

# File format support
SUPPORTED_FILE_FORMATS = {
    'csv': ['.csv'],
    'excel': ['.xlsx', '.xls'],
    'json': ['.json'],
    'parquet': ['.parquet']
}

# Progress tracking steps
CAUSAL_ANALYSIS_STEPS = [
    "🔍 Loading and validating data...",
    "📊 Analyzing data characteristics...", 
    "🎯 Selecting optimal variables...",
    "📈 Computing correlation matrix...",
    "🧠 Building causal structure (NOTEARS)...",
    "🔗 Identifying causal relationships...",
    "📊 Calculating statistical significance...",
    "🎨 Creating network visualization...",
    "📋 Generating results table...",
    "🔍 Computing edge statistics...",
    "📊 Finalizing analysis...",
    "✅ Analysis complete!"
]

FORECASTING_STEPS = [
    "📊 Preparing time series data...",
    "🤖 Fitting forecasting model...",
    "📈 Creating forecast visualization...",
    "📋 Generating forecast summary...",
    "✅ Forecast complete!"
]

INTERVENTION_STEPS = [
    "🔬 Preparing intervention analysis...",
    "🔍 Validating data quality...",
    "🏗️ Building causal structure...",
    "🧠 Creating Bayesian Network...",
    "🎯 Performing intervention...",
    "📊 Generating results...",
    "✅ Intervention analysis complete!"
]