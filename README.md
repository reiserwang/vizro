# Dynamic Data Analysis Dashboard with CausalNex

An advanced interactive dashboard for comprehensive data analysis, causal inference, and time series forecasting with professional Material Design styling.

## 🚀 Features

### Data Input & Visualization
*   **Multi-format Data Upload**: Support for CSV and Excel files via drag-and-drop or URL loading
*   **Dynamic Plotting**: Interactive scatter, line, and bar charts with automatic column detection
*   **Smart Filtering**: Time-based filtering with predefined ranges (Last 3/6 months, Last Year, YTD)
*   **Aggregation Options**: Raw data or averaged values for cleaner visualizations
*   **Responsive Design**: Mobile-friendly interface with modern Material Design styling

### Advanced Causal Analysis
*   **CausalNex Integration**: Automated causal structure discovery using NOTEARS algorithm
*   **Statistical Validation**: Comprehensive model quality evaluation with standard statistical checks
*   **Interactive Causal Graph**: Color-coded relationships based on statistical significance
*   **Enhanced Visualizations**: Edge thickness proportional to causal strength, hover details with metrics

### Model Quality Evaluation
*   **Network Structure Analysis**: Node/edge counts, density, DAG validation, sparsity metrics
*   **Statistical Significance Testing**: 
    - Pearson and Spearman correlation coefficients
    - P-value calculations for each causal relationship
    - R² scores for linear relationships
*   **Cross-Validation Performance**: 5-fold CV with predictive accuracy assessment
*   **Residual Analysis**: Normality tests and model assumption validation
*   **Model Complexity Metrics**: Parent node analysis and network complexity assessment

### Time Series Forecasting
*   **Multiple Models**: Linear Regression, ARIMA, and Nowcasting capabilities
*   **Interactive Controls**: Dynamic model selection and forecast period configuration
*   **Visual Predictions**: Clear separation of historical data and forecasted values

### Professional UI/UX
*   **Material Design Theme**: Modern color palette with light/dark mode support
*   **Smooth Animations**: Hover effects, transitions, and elevation shadows
*   **Enhanced Typography**: Professional font stack with proper hierarchy
*   **Accessibility**: High contrast ratios and keyboard navigation support

## 📋 Requirements

### Core Dependencies
- **Python 3.8+**
- **Dash & Plotly**: Interactive web framework and visualization
- **CausalNex**: McKinsey's causal inference library
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities and cross-validation
- **SciPy & Statsmodels**: Statistical testing and time series analysis

### Complete Package List
```
dash
plotly
causalnex
pandas
numpy
scikit-learn
scipy
statsmodels
networkx
matplotlib
openpyxl
```

## 🛠️ Setup

1.  **Create and activate virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Alternative installation** (if requirements.txt not available):
    ```bash
    pip install dash plotly causalnex pandas numpy scikit-learn scipy statsmodels networkx matplotlib openpyxl
    ```

## 🚀 Running the Dashboard

Start the interactive dashboard:

```bash
python dashboard.py
```

The dashboard will be available at `http://127.0.0.1:8050` in your web browser.

## 📊 Usage Guide

### 1. Data Loading
- **Upload Files**: Drag and drop CSV/Excel files or use the file selector
- **URL Loading**: Enter direct URLs to CSV/Excel files for remote data access
- **Automatic Processing**: Date columns are automatically detected and converted

### 2. Visualization
- **Chart Types**: Choose between scatter, line, and bar charts
- **Axis Selection**: Dynamic dropdowns populated based on your data columns
- **Color Coding**: Add categorical or temporal color dimensions
- **Time Filtering**: Apply date ranges when temporal data is detected
- **Aggregation**: Switch between raw data and averaged values

### 3. Causal Analysis
- **Automatic Discovery**: CausalNex automatically identifies causal relationships
- **Statistical Validation**: Each relationship is tested for statistical significance
- **Interactive Graph**: 
  - Green/orange edges: Statistically significant (p < 0.05)
  - Blue/red edges: Non-significant relationships
  - Edge thickness: Proportional to causal strength
  - Hover for details: View correlation coefficients, p-values, and R² scores

### 4. Model Quality Assessment
- **Comprehensive Metrics**: Automatic evaluation of model reliability
- **Performance Indicators**: Cross-validation scores and predictive accuracy
- **Statistical Tests**: Normality tests and assumption validation
- **Quality Report**: Detailed analysis with actionable insights

### 5. Time Series Forecasting
- **Model Selection**: Choose from Linear Regression, ARIMA, or Nowcasting
- **Flexible Periods**: Specify custom forecast horizons
- **Visual Results**: Clear distinction between historical and predicted data

## 🎨 Theme Customization

The dashboard features a modern Material Design theme with:
- **Light/Dark Mode**: Toggle between themes using the radio buttons
- **CSS Variables**: Easy customization through CSS custom properties
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Professional Styling**: Elevation shadows, smooth transitions, and modern typography

## 📈 Statistical Methods

### Causal Discovery
- **NOTEARS Algorithm**: Non-parametric causal structure learning
- **DAG Validation**: Ensures acyclic graph structure
- **Weight Interpretation**: Positive/negative causal effects with magnitude

### Quality Evaluation
- **Correlation Analysis**: Pearson and Spearman coefficients
- **Significance Testing**: P-value calculations with 0.05 threshold
- **Cross-Validation**: 5-fold CV for predictive performance
- **Residual Analysis**: D'Agostino-Pearson normality tests
- **Network Metrics**: Density, sparsity, and complexity measures

### Forecasting Models
- **Linear Regression**: Trend-based predictions with time features
- **ARIMA**: Autoregressive integrated moving average for time series
- **Nowcasting**: Real-time prediction using recent data patterns

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed in the correct environment
2. **Data Loading**: Check file format and encoding (UTF-8 recommended)
3. **Memory Issues**: Large datasets may require data sampling or chunking
4. **Causal Analysis**: Requires at least 2 numeric columns for meaningful results

### Performance Tips
- **Data Size**: Optimal performance with datasets under 10,000 rows
- **Column Selection**: Focus on relevant variables for causal analysis
- **Browser**: Use modern browsers (Chrome, Firefox, Safari) for best experience

## 🔮 Future Works

We are planning to enhance the forecasting capabilities by adding more advanced models from the `statsmodels` library. Here are some of the models we are considering:

*   **Exponential Smoothing (ETS)**: To better handle data with trend and seasonality.
*   **Vector Autoregression (VAR)**: For multivariate time series analysis.
*   **State-Space Models**: To model time series in terms of unobserved states.
*   **Dynamic Factor Models**: To explain a large number of time series with a small number of unobserved common factors.

We also plan to add more features to the causal analysis section, such as:
*   Allowing users to add or remove edges from the causal graph.
*   Implementing what-if analysis to simulate interventions.

## 📝 License

This project is open source and available under the MIT License.