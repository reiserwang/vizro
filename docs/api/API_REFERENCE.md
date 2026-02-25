# API Reference

## 📚 Overview

This document provides comprehensive API reference for the Advanced Analytics Dashboard components.

## 🏗️ Core Module

### DashboardConfig

Central configuration management for the dashboard application.

```python
from src.core.config import dashboard_config, DashboardConfig
```

#### Class: `DashboardConfig`

**Description**: Manages global state and configuration for the dashboard.

**Attributes**:
- `current_data: Optional[pd.DataFrame]` - Currently loaded dataset
- `causal_results: Optional[Dict[str, Any]]` - Stored causal analysis results
- `forecasting_results: Optional[Dict[str, Any]]` - Stored forecasting results
- `visualization_settings: Dict[str, Any]` - Visualization preferences

**Methods**:

##### `reset() -> None`
Reset all stored data and results.

```python
dashboard_config.reset()
```

##### `has_data() -> bool`
Check if data is currently loaded.

```python
if dashboard_config.has_data():
    print("Data is loaded")
```

##### `get_numeric_columns() -> List[str]`
Get list of numeric columns from current data.

```python
numeric_cols = dashboard_config.get_numeric_columns()
```

##### `get_data_info() -> Dict[str, Any]`
Get comprehensive information about the current dataset.

```python
info = dashboard_config.get_data_info()
print(f"Dataset shape: {info['shape']}")
```

### DataHandler

Handles data loading, validation, and preprocessing operations.

```python
from src.core.data_handler import DataHandler
```

#### Class: `DataHandler`

**Description**: Static methods for data operations.

**Methods**:

##### `load_data(file_content: bytes, filename: str) -> Tuple[str, Optional[pd.DataFrame]]`
Load data from uploaded file.

**Parameters**:
- `file_content`: Raw file content as bytes
- `filename`: Name of the uploaded file

**Returns**: Tuple of (status_message, dataframe)

```python
with open('data.csv', 'rb') as f:
    content = f.read()
status, df = DataHandler.load_data(content, 'data.csv')
```

##### `validate_data(df: pd.DataFrame) -> Tuple[bool, str]`
Validate the loaded dataset.

**Parameters**:
- `df`: DataFrame to validate

**Returns**: Tuple of (is_valid, message)

```python
is_valid, message = DataHandler.validate_data(df)
if not is_valid:
    print(f"Validation failed: {message}")
```

##### `preprocess_data(df: pd.DataFrame, handle_missing: str = 'drop', remove_outliers: bool = False) -> pd.DataFrame`
Preprocess the dataset for analysis.

**Parameters**:
- `df`: DataFrame to preprocess
- `handle_missing`: How to handle missing values ('drop', 'fill_mean', 'fill_median')
- `remove_outliers`: Whether to remove statistical outliers

**Returns**: Preprocessed DataFrame

```python
clean_df = DataHandler.preprocess_data(
    df, 
    handle_missing='fill_mean', 
    remove_outliers=True
)
```

## 🔬 Analysis Engines

### CausalAnalysisEngine

Performs causal discovery and intervention analysis.

```python
from src.engines.causal_engine import CausalAnalysisEngine
```

#### Functions

##### `perform_causal_analysis(hide_nonsignificant: bool, min_correlation: float, theme: str, show_all_relationships: bool = False) -> Tuple[go.Figure, str, str]`
Perform causal analysis on the loaded data.

**Parameters**:
- `hide_nonsignificant`: Filter non-significant relationships
- `min_correlation`: Minimum correlation threshold
- `theme`: Visualization theme ('Light' or 'Dark')
- `show_all_relationships`: Show all relationships regardless of significance

**Returns**: Tuple of (network_figure, results_table_html, summary_markdown)

```python
network_fig, table_html, summary = perform_causal_analysis(
    hide_nonsignificant=True,
    min_correlation=0.3,
    theme='Light',
    show_all_relationships=False
)
```

##### `perform_causal_intervention_analysis(target_var: str, intervention_var: str, intervention_value: float) -> Tuple[str, str]`
Perform causal intervention analysis (do-calculus).

**Parameters**:
- `target_var`: Target variable name
- `intervention_var`: Intervention variable name
- `intervention_value`: Value to set for intervention

**Returns**: Tuple of (results_html, status_message)

```python
results_html, status = perform_causal_intervention_analysis(
    target_var='sales',
    intervention_var='marketing_spend',
    intervention_value=1500.0
)
```

### ForecastingEngine

Provides time series forecasting capabilities.

```python
from src.engines.forecasting_engine import ForecastingEngine
```

#### Functions

##### `generate_forecast(target_column: str, periods: int, model_type: str = 'auto', confidence_level: float = 0.95) -> Tuple[go.Figure, str]`
Generate time series forecast.

**Parameters**:
- `target_column`: Column name to forecast
- `periods`: Number of periods to forecast
- `model_type`: Forecasting model ('auto', 'arima', 'exponential', 'linear')
- `confidence_level`: Confidence level for prediction intervals

**Returns**: Tuple of (forecast_figure, summary_text)

```python
forecast_fig, summary = generate_forecast(
    target_column='sales',
    periods=12,
    model_type='auto',
    confidence_level=0.95
)
```

### VisualizationEngine

Creates interactive data visualizations.

```python
from src.engines.visualization_engine import VisualizationEngine
```

#### Functions

##### `create_visualization(chart_type: str, x_column: str, y_column: str, color_column: str = None, theme: str = 'Light') -> go.Figure`
Create interactive data visualization.

**Parameters**:
- `chart_type`: Type of chart ('line', 'scatter', 'histogram', 'box', 'heatmap')
- `x_column`: X-axis column name
- `y_column`: Y-axis column name
- `color_column`: Optional color grouping column
- `theme`: Visualization theme

**Returns**: Plotly figure object

```python
fig = create_visualization(
    chart_type='scatter',
    x_column='marketing_spend',
    y_column='sales',
    color_column='region',
    theme='Light'
)
```

## 🎨 UI Components

### Dashboard

Main Gradio dashboard interface.

```python
from src.ui.dashboard import create_dashboard
```

#### Functions

##### `create_dashboard() -> gr.Blocks`
Create the main dashboard interface.

**Returns**: Gradio Blocks interface

```python
dashboard = create_dashboard()
dashboard.launch()
```

### SettingsManager

Manages dashboard settings and preferences.

```python
from src.ui.settings_manager import SettingsManager
```

#### Class: `SettingsManager`

**Methods**:

##### `load_settings() -> Dict[str, Any]`
Load settings from configuration file.

```python
settings = SettingsManager.load_settings()
```

##### `save_settings(settings: Dict[str, Any]) -> bool`
Save settings to configuration file.

```python
success = SettingsManager.save_settings(new_settings)
```

## 🛠️ Utilities

### DataGenerator

Generates sample datasets for testing and demonstration.

```python
from src.utils.data_generator import DataGenerator
```

#### Functions

##### `generate_business_data(n_samples: int = 1000) -> pd.DataFrame`
Generate realistic business analytics dataset.

**Parameters**:
- `n_samples`: Number of samples to generate

**Returns**: Generated DataFrame

```python
business_data = generate_business_data(n_samples=500)
```

##### `generate_time_series_data(periods: int = 365, frequency: str = 'D') -> pd.DataFrame`
Generate time series dataset with trends and seasonality.

**Parameters**:
- `periods`: Number of time periods
- `frequency`: Time frequency ('D', 'M', 'Y')

**Returns**: Generated time series DataFrame

```python
ts_data = generate_time_series_data(periods=730, frequency='D')
```

## 📊 Data Structures

### Analysis Results

#### Causal Analysis Results
```python
causal_results = {
    'structure_model': StructureModel,  # CausalNex structure model
    'results_df': pd.DataFrame,         # Relationships dataframe
    'edge_stats': List[Dict],           # Edge statistics
    'summary_stats': Dict[str, int]     # Summary statistics
}
```

#### Forecasting Results
```python
forecasting_results = {
    'forecast_values': np.ndarray,      # Predicted values
    'confidence_intervals': np.ndarray, # Confidence intervals
    'model_metrics': Dict[str, float],  # Performance metrics
    'model_type': str,                  # Selected model type
    'parameters': Dict[str, Any]        # Model parameters
}
```

## 🔧 Configuration Parameters

### Causal Analysis Parameters
```python
CAUSAL_ANALYSIS_PARAMS = {
    'max_variables': 12,        # Maximum variables for analysis
    'max_samples': 1500,        # Maximum samples for performance
    'max_iter': 100,            # NOTEARS maximum iterations
    'h_tol': 1e-8,             # NOTEARS tolerance
    'w_threshold': 0.3          # Edge weight threshold
}
```

### Dashboard Settings
```python
dashboard_settings = {
    'theme': 'Light',           # UI theme
    'auto_refresh': True,       # Auto-refresh results
    'export_format': 'PNG',     # Default export format
    'max_file_size': 100,       # Max upload size (MB)
    'cache_results': True       # Cache analysis results
}
```

## 🚨 Error Handling

### Common Exceptions

#### `DataValidationError`
Raised when data validation fails.

```python
try:
    DataHandler.validate_data(df)
except DataValidationError as e:
    print(f"Data validation failed: {e}")
```

#### `AnalysisError`
Raised when analysis operations fail.

```python
try:
    perform_causal_analysis(...)
except AnalysisError as e:
    print(f"Analysis failed: {e}")
```

### Error Response Format
```python
error_response = {
    'success': False,
    'error_type': 'DataValidationError',
    'message': 'Dataset must have at least 2 numeric columns',
    'suggestions': [
        'Check your data format',
        'Ensure numeric columns are present'
    ]
}
```

## 📈 Performance Considerations

### Memory Usage
- Large datasets are automatically sampled
- Results are cached to avoid recomputation
- Garbage collection is performed after analysis

### Computational Complexity
- Causal analysis: O(n³) for n variables
- Forecasting: O(n) for n time points
- Visualization: O(n) for n data points

### Optimization Tips
```python
# For large datasets
dashboard_config.enable_sampling = True
dashboard_config.max_samples = 1000

# For better performance
dashboard_config.cache_results = True
dashboard_config.parallel_processing = True
```

## 🔗 Integration Examples

### Custom Analysis Pipeline
```python
from src.core.data_handler import DataHandler
from src.engines.causal_engine import perform_causal_analysis

# Load data
status, df = DataHandler.load_data(file_content, filename)

# Preprocess
clean_df = DataHandler.preprocess_data(df, handle_missing='fill_mean')

# Analyze
network_fig, table_html, summary = perform_causal_analysis(
    hide_nonsignificant=True,
    min_correlation=0.3,
    theme='Light'
)
```

### Batch Processing
```python
import os
from pathlib import Path

# Process multiple files
data_dir = Path('data/')
results = {}

for file_path in data_dir.glob('*.csv'):
    with open(file_path, 'rb') as f:
        content = f.read()
    
    status, df = DataHandler.load_data(content, file_path.name)
    if df is not None:
        results[file_path.name] = perform_causal_analysis(
            hide_nonsignificant=True,
            min_correlation=0.3,
            theme='Light'
        )
```