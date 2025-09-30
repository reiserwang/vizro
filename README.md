# Advanced Causal Analysis & Forecasting Dashboard

A comprehensive interactive platform for data analysis, causal inference, and time series forecasting with realistic business scenarios, statistical validation, and professional Material Design styling. Features enhanced datasets with business incidents, advanced forecasting models, and educational statistical explanations.

## 🚀 Features

### Data Input & Visualization
*   **Multi-format Data Upload**: Support for CSV and Excel files via drag-and-drop or URL loading
*   **Dynamic Plotting**: Interactive scatter, line, and bar charts with automatic column detection
*   **Smart Filtering**: Time-based filtering with predefined ranges (Last 3/6 months, Last Year, YTD)
*   **Aggregation Options**: Raw data or averaged values for cleaner visualizations
*   **Responsive Design**: Mobile-friendly interface with modern Material Design styling

### Advanced Causal Analysis with Business Realism
*   **CausalNex Integration**: Automated causal structure discovery using NOTEARS algorithm
*   **Realistic Business Dataset**: Enhanced sales data with 21 types of business incidents affecting performance
*   **Statistical Validation**: Comprehensive model quality evaluation with standard statistical checks
*   **Interactive Filtering**: Hide non-significant relationships and set correlation thresholds
*   **Interactive Causal Graph**: Color-coded relationships based on statistical significance
*   **Enhanced Visualizations**: Edge thickness proportional to causal strength, hover details with metrics
*   **Business Incident Analysis**: Track how external events (pandemics, product recalls, partnerships) impact sales and revenue

### Comprehensive Model Quality Evaluation
*   **Network Structure Analysis**: Node/edge counts, density, DAG validation, sparsity metrics
*   **Statistical Significance Testing**: 
    - Pearson and Spearman correlation coefficients
    - P-value calculations for each causal relationship
    - R² scores for linear relationships
*   **Cross-Validation Performance**: 5-fold CV with predictive accuracy assessment
*   **Residual Analysis**: Normality tests and model assumption validation
*   **Model Complexity Metrics**: Parent node analysis and network complexity assessment
*   **Performance Optimization**: Multiprocessing support for faster analysis of large datasets
*   **Educational Explanations**: Cascadeable statistical concept explanations with examples

### Advanced Time Series Forecasting
*   **Seven Forecasting Models**: Linear Regression, ARIMA, SARIMA, VAR, Dynamic Factor, State-Space, Nowcasting
*   **Multivariate Analysis**: VAR models for analyzing relationships between multiple time series
*   **Structural Decomposition**: State-Space models for trend, seasonal, and irregular components
*   **Dimension Reduction**: Dynamic Factor models for high-dimensional datasets
*   **Interactive Model Selection**: Detailed descriptions and use cases for each model
*   **Enhanced Visualizations**: Confidence intervals, component analysis, and uncertainty bands
*   **Business Context**: Models designed for real-world business forecasting scenarios

### Enhanced Business Dataset
*   **Realistic Causal Relationships**: Clear business logic with proper statistical properties
*   **Business Incident System**: 21 types of external events affecting sales and revenue
*   **Temporal Complexity**: Incidents with time-decay effects and realistic durations
*   **Multi-dimensional Impact**: Events affecting different sale types (subscription vs. buyout) and regions
*   **Statistical Rigor**: Strong relationships (p < 0.001) and weak control variables (p > 0.05)
*   **Educational Value**: Clear examples of both significant and non-significant relationships

### Professional UI/UX & Educational Features
*   **Material Design Theme**: Modern color palette with light/dark mode support
*   **Cascadeable Explanations**: Multi-level collapsible content with structured markdown
*   **Interactive Learning**: Progressive disclosure of statistical concepts with practical examples
*   **Statistical Education**: Built-in explanations of p-values, correlations, R², and causal weights
*   **Quality Assessment Framework**: Clear guidelines for evaluating relationship reliability
*   **Left-Aligned Structure**: Clean, professional layout with consistent typography
*   **Smooth Animations**: Hover effects, transitions, and elevation shadows
*   **Enhanced Typography**: Professional font stack with proper hierarchy
*   **Accessibility**: High contrast ratios and keyboard navigation support

## 📋 Requirements

### Core Dependencies
- **Python 3.8+**
- **Dash & Plotly**: Interactive web framework and visualization
- **CausalNex**: McKinsey's causal inference library for causal structure discovery
- **Pandas & NumPy**: Data manipulation and numerical analysis
- **Scikit-learn**: Machine learning utilities, cross-validation, and preprocessing
- **SciPy & Statsmodels**: Statistical testing, time series analysis, and advanced forecasting models
- **NetworkX**: Graph analysis and visualization for causal networks
- **Multiprocessing**: Performance optimization for large dataset analysis

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

### 3. Advanced Causal Analysis
- **Automatic Discovery**: CausalNex automatically identifies causal relationships using NOTEARS algorithm
- **Interactive Filtering**: 
  - Hide non-significant relationships (p ≥ 0.05) with toggle control
  - Set minimum correlation thresholds (0.0 to 0.8) with slider
  - Dynamic graph and table updates based on filter settings
- **Statistical Validation**: Each relationship tested with multiple statistical measures
- **Enhanced Interactive Graph**: 
  - Green/orange edges: Statistically significant (p < 0.05)
  - Blue/red edges: Non-significant relationships
  - Edge thickness: Proportional to causal strength
  - Hover for details: View correlation coefficients, p-values, and R² scores
- **Educational Explanations**: Cascadeable statistical concept explanations with practical examples

### 4. Comprehensive Model Quality Assessment
- **Comprehensive Metrics**: Automatic evaluation of model reliability with performance optimization
- **Statistical Significance**: Built-in explanations of p-values, correlations, R², and causal weights
- **Performance Indicators**: Cross-validation scores and predictive accuracy with multiprocessing
- **Statistical Tests**: Normality tests and assumption validation
- **Quality Framework**: Clear guidelines for evaluating relationship reliability (High/Moderate/Low quality)
- **Educational Content**: Progressive disclosure of statistical concepts with real-world examples
- **Interactive Learning**: Collapsible explanations covering interpretation and decision-making

### 5. Advanced Time Series Forecasting
- **Multiple Models**: Linear Regression, ARIMA, SARIMA, VAR, Dynamic Factor, State-Space, Nowcasting
- **Multivariate Analysis**: VAR models for analyzing relationships between multiple time series
- **Structural Decomposition**: State-Space models for trend, seasonal, and irregular components
- **Dimension Reduction**: Dynamic Factor models for high-dimensional datasets
- **Interactive Controls**: Model selection with detailed descriptions and use cases
- **Flexible Periods**: Specify custom forecast horizons with confidence intervals
- **Visual Results**: Enhanced plots with model components and uncertainty bands

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

### Advanced Forecasting Models
- **Linear Regression**: Simple trend-based predictions with time features
- **ARIMA**: Autoregressive integrated moving average for univariate time series
- **SARIMA**: Seasonal ARIMA for data with seasonal patterns
- **VAR (Vector Autoregression)**: Multivariate model capturing cross-variable relationships
- **Dynamic Factor Model**: Dimension reduction for high-dimensional time series data
- **State-Space Model**: Structural decomposition with trend, seasonal, and irregular components
- **Nowcasting**: Real-time estimation using most recent data patterns

## 🏢 Enhanced Business Dataset

### Realistic Causal Relationships
The dashboard includes a comprehensive enhanced sales dataset with:
- **Clear Business Logic**: Marketing Budget → Lead Generation → Sales Volume → Revenue
- **Statistical Rigor**: Strong relationships (p < 0.001) and weak control variables (p > 0.05)
- **28 Variables**: Including business metrics, categorical variables, and control variables
- **10,000 Records**: Spanning 2015-2025 with realistic business scenarios

### Business Incident System
- **21 Incident Types**: Including pandemics, product recalls, partnerships, regional events
- **46.8% Impact Coverage**: Nearly half of records affected by business incidents
- **Time-Decay Effects**: Stronger impact at incident start, gradual weakening over duration
- **Multi-dimensional Impact**: Different effects on subscription vs. buyout sales and regions
- **Cascading Effects**: Incidents affect sales → revenue → customer satisfaction → training

### Key Incident Categories
- **Major Negative**: Pandemic (-35%), Product Recall (-25%), Economic Downturn (-20%)
- **Major Positive**: Regional Expansion (+25%), Enterprise Sales Push (+22%), Viral Marketing (+20%)
- **Sale-Type Specific**: Subscription churn crisis, buyout discount campaigns, platform upgrades
- **Regional Events**: Market expansion, disruption, investment, competition by geography

### Statistical Validation
- **Strong Correlations**: Incident Impact ↔ Sales Volume (r = +0.413, p < 0.001)
- **Revenue Impact**: 21.2% increase during positive incidents, 8.1% decrease during negative
- **Educational Value**: Clear examples of both significant and spurious relationships
- **Control Variables**: Employee ID, office temperature, random factors show no correlation

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed using `uv pip install` or `pip install`
2. **Data Loading**: Check file format and encoding (UTF-8 recommended)
3. **Memory Issues**: Large datasets automatically use smart sampling and multiprocessing optimization
4. **Causal Analysis**: Requires at least 2 numeric columns for meaningful results
5. **Performance**: Use the enhanced dataset (`aed_sales_data_enhanced.csv`) for optimal demonstration

### Performance Optimization
- **Automatic Sampling**: Datasets >5000 rows automatically sampled for structure learning
- **Multiprocessing**: Utilizes multiple CPU cores for statistical computations
- **Variable Selection**: Automatically limits to top 15 most correlated variables for performance
- **Memory Management**: Efficient data handling prevents browser freezing
- **Browser Compatibility**: Optimized for Chrome, Firefox, Safari, and Edge

### Getting Started Quickly
1. **Use Enhanced Dataset**: Load `aed_sales_data_enhanced.csv` for immediate demonstration
2. **Try Filtering**: Use significance filter and correlation threshold controls
3. **Explore Incidents**: Filter by incident types to see business impact patterns
4. **Test Forecasting**: Try different models with the time series data
5. **Learn Statistics**: Expand the statistical explanation sections for educational content

## 🎯 Key Features Summary

### ✅ **Implemented Advanced Features**
- **7 Forecasting Models**: Linear, ARIMA, SARIMA, VAR, Dynamic Factor, State-Space, Nowcasting
- **Interactive Causal Filtering**: Hide non-significant edges and set correlation thresholds
- **Business Incident Analysis**: 21 types of external events affecting sales and revenue
- **Statistical Education**: Cascadeable explanations with practical examples and quality frameworks
- **Performance Optimization**: Multiprocessing support for large dataset analysis
- **Professional UI/UX**: Material Design with responsive layout and accessibility features

### 🔮 **Future Enhancement Opportunities**
- **Interactive Graph Editing**: Manual relationship editing with domain knowledge integration
- **"What-if" Analysis**: Intervention simulation for business scenario planning
- **Predictive Modeling**: DAGRegressor and DAGClassifier integration with causal structure
- **Advanced Discretization**: Multiple strategies for continuous variable handling
- **Latent Variable Detection**: Identification and modeling of unobserved factors
- **Real-time Data Integration**: Live data feeds and streaming analysis capabilities
- **Export Capabilities**: Report generation and model deployment features

## 🏆 **Use Cases & Applications**

### **Business Analytics**
- **Strategic Planning**: Identify key business drivers and causal relationships
- **Risk Assessment**: Analyze impact of external events on business performance
- **Marketing ROI**: Quantify marketing effectiveness and lead conversion patterns
- **Performance Optimization**: Understand training impact and operational efficiency

### **Research & Education**
- **Statistical Learning**: Interactive platform for understanding causal inference concepts
- **Hypothesis Testing**: Validate theoretical relationships with real data
- **Methodology Teaching**: Demonstrate proper statistical analysis techniques
- **Publication Preparation**: Generate publication-ready causal analysis results

### **Data Science & Analytics**
- **Exploratory Analysis**: Discover hidden patterns and relationships in complex datasets
- **Model Validation**: Cross-validate findings across multiple statistical approaches
- **Feature Engineering**: Identify important variables for predictive modeling
- **Time Series Analysis**: Advanced forecasting with multiple model comparison

## 🚀 **Getting Started**

1. **Clone the repository** and install dependencies
2. **Run the dashboard**: `python dashboard.py`
3. **Load enhanced dataset**: Use `aed_sales_data_enhanced.csv` for immediate demonstration
4. **Explore features**: Try causal filtering, incident analysis, and advanced forecasting
5. **Learn statistics**: Expand explanation sections for educational content

## 🤝 **Contributing**

We welcome contributions! Areas for enhancement include:
- Additional forecasting models and statistical tests
- Enhanced visualization capabilities and interactive features
- Performance optimizations and scalability improvements
- Educational content and documentation improvements
- Real-world dataset examples and use case studies

## 📚 **Documentation**

- **`CAUSAL_ANALYSIS_GUIDE.md`**: Comprehensive guide to causal analysis features
- **`FORECASTING_MODELS_GUIDE.md`**: Detailed forecasting model documentation
- **`PERFORMANCE_IMPROVEMENTS.md`**: Technical details on optimization features
- **`DATASET_IMPROVEMENTS.md`**: Enhanced dataset structure and relationships
- **`INCIDENT_ENHANCEMENTS.md`**: Business incident system documentation

## 🎓 **Educational Value**

This dashboard serves as a comprehensive educational platform for:
- **Causal Inference**: Understanding causation vs. correlation with real examples
- **Statistical Analysis**: Learning p-values, correlations, and significance testing
- **Business Analytics**: Analyzing real-world business scenarios and external impacts
- **Time Series Forecasting**: Comparing multiple forecasting approaches
- **Data Science Methodology**: Best practices for exploratory data analysis

## 📝 **License**

This project is open source and available under the MIT License.

---

**Built with ❤️ for the data science and business analytics community**

*Transform your data analysis workflow with advanced causal inference, realistic business scenarios, and professional-grade statistical validation.*