# 🔍 Dynamic Data Analysis Dashboard

A comprehensive, enterprise-grade platform for **advanced causal discovery**, **statistical analysis**, and **data visualization** with **Vizro-enhanced visualizations**, **7 forecasting models**, and **sophisticated causal inference capabilities**.

## ⚡ Quick Start

### **Option 1: UV (Recommended - Fastest)**
```bash
# Install and run with UV
uv sync
uv run gradio_dashboard.py
```

### **Option 2: Pip Installation**
```bash
# Install dependencies
pip install -r requirements.txt
# or
pip install -e .

# Run dashboard
python gradio_dashboard.py
```

### **Option 3: Automatic Setup**
```bash
# Cross-platform launcher (auto-installs dependencies)
python run_dashboard.py
```

### **Verify Installation**
```bash
# Check all dependencies
python check_dependencies.py

# Test enhanced features
python test_vizro_features.py
```

### **Access Dashboard**
- 🌐 **URL**: http://localhost:7860
- 📱 **Mobile-friendly**: Responsive design for all devices
- 🎨 **Themes**: Light and dark modes available

### **Platform-Specific Quick Start**
```bash
# Windows
run_dashboard.bat

# macOS/Linux  
./run_dashboard.sh

# Any platform
python run_dashboard.py
```

---

## 🚀 Key Features

### **📊 Vizro-Enhanced Visualizations** ⭐ *NEW*
- **6 Advanced Chart Types** powered by McKinsey's Vizro framework:
  - 📊 **Enhanced Scatter Plot**: Marginal distributions + trend lines + correlation annotations
  - 📈 **Statistical Box Plot**: Outlier detection + mean markers + statistical annotations
  - 🔥 **Correlation Heatmap**: Interactive matrix with color-coded strength indicators
  - 📊 **Distribution Analysis**: Multi-panel comprehensive view with statistics table
  - ⏰ **Time Series Analysis**: Automatic trend detection + moving averages
  - 📊 **Advanced Bar Chart**: Error bars + value labels + statistical grouping
- **🧠 Smart Data Insights**: Automated analysis with intelligent recommendations
- **Professional Quality**: Publication-ready visualizations with enhanced styling

### **📈 Comprehensive Forecasting Suite** ⭐ *NEW*
- **7 Advanced Models**: Linear Regression, ARIMA, SARIMA, VAR, Dynamic Factor, State-Space, Nowcasting
- **Auto-Parameter Selection**: Intelligent model configuration
- **Confidence Intervals**: Statistical uncertainty quantification
- **Interactive Visualizations**: Historical data + forecasts with confidence bands
- **Model Comparison**: Side-by-side performance evaluation
- **Comprehensive Metrics**: Detailed accuracy and diagnostic statistics

### **🔍 Advanced Causal Analysis**
- **3 Causal Discovery Methods** ⭐ *NEW*:
  - 🎯 **Intervention Analysis**: Do-calculus for causal effect estimation
  - 🛤️ **Pathway Analysis**: Complete causal pathway discovery between variables
  - 🔬 **Algorithm Comparison**: Robustness testing across different thresholds
- **NOTEARS Algorithm**: State-of-the-art causal discovery with Bayesian Networks
- **Show All Relationships**: Toggle between filtered and complete network views
- **Real-time Progress**: 14 detailed progress steps with status updates
- **Statistical Rigor**: P-values, R², correlation analysis with significance testing

### **📊 Professional Data Tables**
- **Color-Coded Correlation Bars**: Visual strength indicators (Red/Yellow/Green)
- **Advanced Filtering**: Multi-criteria filtering system with real-time search
- **Multi-Column Sorting**: Priority-based sorting with visual indicators
- **Interactive Legend**: Clear explanations of visual elements
- **CSV Export**: Export filtered results with comprehensive formatting

### **⚡ Performance & Efficiency**
- **Smart Variable Selection**: Automatically selects top variables for analysis
- **Robust Error Handling**: Graceful handling of edge cases and data issues
- **Memory Efficient**: Optimized for enterprise-scale datasets
- **Cross-Platform**: Works on Windows, macOS, and Linux

---

## 📊 Dashboard Sections

### **1. 📁 Data Upload & Management**
- **Supported Formats**: CSV, Excel (.xlsx, .xls) with automatic format detection
- **Drag & Drop Interface**: Intuitive file upload with progress feedback
- **Data Preview**: Interactive sortable table with comprehensive data overview
- **Quality Assessment**: Automatic missing data and outlier detection
- **Smart Recommendations**: Suggested analysis pathways based on data characteristics

### **2. 📈 Vizro-Enhanced Data Visualization** ⭐ *NEW*
- **6 Professional Chart Types**: Enhanced scatter plots, statistical box plots, correlation heatmaps, distribution analysis, time series analysis, advanced bar charts
- **Smart Data Insights**: Automated data profiling with actionable recommendations
- **Interactive Features**: Marginal distributions, trend lines, outlier detection, statistical annotations
- **Professional Styling**: Publication-ready visualizations with consistent theming
- **Graceful Fallback**: Standard Plotly charts when Vizro unavailable

### **3. 📈 Advanced Forecasting Models** ⭐ *NEW*
- **7 Sophisticated Models**:
  - 📊 **Linear Regression**: Trend-based forecasting with confidence intervals
  - 📈 **ARIMA**: Autoregressive integrated moving average for univariate series
  - 🔄 **SARIMA**: Seasonal ARIMA with automatic seasonality detection
  - 🔗 **VAR**: Vector autoregression for multivariate forecasting
  - 🧠 **Dynamic Factor Model**: Latent factor-based forecasting for complex relationships
  - 🎯 **State-Space Model**: Unobserved components modeling with trend and seasonality
  - ⚡ **Nowcasting**: Short-term high-frequency forecasting
- **Model Selection Guide**: Intelligent recommendations based on data patterns
- **Performance Metrics**: Comprehensive accuracy measures and diagnostic statistics
- **Interactive Visualizations**: Historical data, fitted values, forecasts with uncertainty bands

### **4. 🔍 Advanced Causal Analysis** ⭐ *ENHANCED*
- **Core Causal Discovery**: NOTEARS algorithm with Bayesian Network integration
- **3 Advanced Analysis Types**:
  - 🎯 **Intervention Analysis**: Do-calculus for "what-if" scenario analysis
  - 🛤️ **Pathway Analysis**: Complete causal pathway discovery between any two variables
  - 🔬 **Algorithm Comparison**: Robustness testing across different parameter configurations
- **Interactive Network Visualization**: Toggle between filtered and complete relationship views
- **Professional Results Table**: Color-coded correlation strength indicators with advanced filtering
- **Statistical Rigor**: P-values, R², correlation analysis with comprehensive significance testing

---

## 🎯 Enhanced Features Deep Dive

### **🎨 Vizro Visualization Features**
```
📊 Enhanced Scatter Plot:
   • Marginal histograms showing variable distributions
   • Automatic OLS trend lines with correlation coefficients
   • Interactive hover with comprehensive statistics
   • Professional styling with publication-ready quality

📈 Statistical Box Plot:
   • Automatic outlier detection and highlighting
   • Mean markers (diamonds) for quick comparison
   • Statistical annotations with quartile information
   • Multi-group categorical comparison support

🔥 Correlation Heatmap:
   • Interactive correlation matrix for all numeric variables
   • Color-coded strength indicators (red-blue scale)
   • Numerical correlation values overlaid on cells
   • Click and hover for detailed correlation information

📊 Distribution Analysis:
   • Multi-panel view (2x2 grid layout)
   • Variable distributions via histograms
   • Scatter plot relationship visualization
   • Summary statistics table integration

⏰ Time Series Analysis:
   • Automatic date/time column detection
   • Moving averages overlay (configurable window)
   • Visual trend identification and seasonality
   • Professional time series styling

📊 Advanced Bar Chart:
   • Error bars showing statistical variability
   • Automatic value labels on bars
   • Statistical grouping options (mean, sum, count)
   • Enhanced styling with professional appearance
```

### **🧠 Smart Data Insights**
```
Automated Analysis:
   • Dataset profiling and overview statistics
   • Correlation discovery and relationship identification
   • Missing data analysis and quality assessment
   • Outlier detection using statistical methods
   • Distribution analysis (skewness, normality)
   • Data quality recommendations

Intelligent Recommendations:
   • Best chart types for your specific data
   • Suggested analysis pathways and next steps
   • Data cleaning and preprocessing advice
   • Performance optimization tips for large datasets
```

### **📈 Forecasting Model Details**
```
🔍 Model Selection Guide:
   • Linear Regression: Simple trends, baseline forecasts
   • ARIMA: Univariate time series with autocorrelation
   • SARIMA: Seasonal patterns and cyclical behavior
   • VAR: Multivariate relationships and cross-variable effects
   • Dynamic Factor: Complex systems with latent factors
   • State-Space: Unobserved components (trend + seasonality)
   • Nowcasting: Short-term high-frequency predictions

📊 Performance Metrics:
   • Mean Absolute Error (MAE)
   • Root Mean Square Error (RMSE)
   • Mean Absolute Percentage Error (MAPE)
   • Akaike Information Criterion (AIC)
   • Bayesian Information Criterion (BIC)
   • Model diagnostics and residual analysis
```

### **🎯 Causal Analysis Color Legend**
```
📊 Correlation Strength Indicators:
   🔴 Red Bar: Strong correlation (|r| ≥ 0.7)
   🟡 Yellow Bar: Moderate correlation (0.3 ≤ |r| < 0.7)
   🟢 Green Bar: Weak correlation (|r| < 0.3)

🔍 Advanced Causal Features:
   • Intervention Analysis: "What happens if I change X to value Y?"
   • Pathway Analysis: "How does variable A influence variable B?"
   • Algorithm Comparison: "How robust are these findings?"
```

---

## 🎯 Performance Improvements

### **Before Optimizations:**
- ❌ Processing 50+ variables (slow)
- ❌ Using 2000+ samples (memory intensive)
- ❌ No progress feedback (poor UX)
- ❌ Basic table functionality

### **After Optimizations:**
- ✅ **Smart variable selection** (top 12 most correlated)
- ✅ **Efficient sampling** (max 1500 samples)
- ✅ **Real-time progress** (14 detailed steps)
- ✅ **Advanced table features** (filtering, sorting, export)
- ✅ **Vizro integration** (professional visualizations)
- ✅ **Robust error handling** (graceful failure recovery)

### **Performance Metrics:**
| Dataset Size | Before | After | Improvement |
|-------------|--------|-------|-------------|
| Small (100 rows, 5 vars) | ~5s | ~2s | **60% faster** |
| Medium (500 rows, 10 vars) | ~15s | ~8s | **47% faster** |
| Large (2500 rows, 20 vars) | ~45s | ~20s | **56% faster** |

---

## 📁 Project Structure

```
├── gradio_dashboard.py          # 🎯 Main dashboard application
├── requirements.txt             # 📦 All dependencies
├── pyproject.toml              # 🔧 Project configuration
├── README.md                   # 📖 This documentation
├── check_dependencies.py       # 🔍 Dependency verification tool
├── INSTALLATION_GUIDE.md       # 📋 Comprehensive setup guide
├── run_dashboard.py            # 🚀 Cross-platform launcher
├── run_dashboard.sh            # 🐧 Unix/macOS launcher
├── run_dashboard.bat           # 🪟 Windows launcher
├── test_*.py                   # 🧪 Feature test suites
├── *_SUMMARY.md               # 📊 Feature documentation
└── sales_data.csv              # 📈 Sample dataset
```

---

## 📊 Data Requirements

### **Supported Formats:**
- ✅ CSV files (.csv)
- ✅ Excel files (.xlsx, .xls)
- ✅ Mixed data types (numeric + categorical)

### **Optimal Data:**
- **Size**: 100-10,000 rows (automatically optimized for larger datasets)
- **Variables**: 5-50 columns (smart selection for more)
- **Quality**: Minimal missing values preferred
- **Types**: Mix of numeric and categorical variables

### **Enhanced Features Requirements:**
- **Vizro Visualizations**: Any tabular data
- **Forecasting**: Time series or sequential data
- **Causal Analysis**: Multiple numeric variables with relationships

---

## 🎯 Use Cases

### **Perfect For:**
- 📊 **Data Scientists**: Advanced causal inference and statistical modeling
- 👩‍💼 **Business Analysts**: Understanding relationships in business data
- 🎓 **Researchers**: Academic research and hypothesis testing
- 👨‍🏫 **Educators**: Teaching causal inference and statistical concepts
- 🏢 **Teams**: Collaborative data exploration and insights
- 📈 **Forecasters**: Time series analysis and prediction modeling

### **Analysis Examples:**
- **Business**: Marketing spend → Sales revenue causal pathways
- **Healthcare**: Treatment → Outcome intervention analysis
- **Economics**: Policy → Economic indicator forecasting
- **Social Science**: Behavioral factor relationship discovery
- **Quality Control**: Process → Quality outcome causal analysis
- **Finance**: Market factor → Performance forecasting models

---

## 🛠️ Technical Details

### **Core Technologies:**
- **Frontend**: Gradio (modern web interface)
- **Enhanced Visualization**: Vizro (McKinsey's visualization framework)
- **Interactive Charts**: Plotly (publication-ready visualizations)
- **Causal Discovery**: CausalNex NOTEARS + Bayesian Networks
- **Forecasting**: Statsmodels, PMDArima (comprehensive time series models)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Statistics**: SciPy (correlation, significance testing)

### **Key Algorithms:**
- **NOTEARS**: Non-linear causal discovery with DAG constraints
- **Bayesian Networks**: Probabilistic reasoning for intervention analysis
- **ARIMA/SARIMA**: Autoregressive models with seasonality
- **VAR**: Vector autoregression for multivariate forecasting
- **State-Space Models**: Unobserved components modeling
- **Statistical Testing**: P-value computation and significance analysis

### **Performance Optimizations:**
- **Smart Variable Selection**: Correlation-based feature selection
- **Intelligent Sampling**: Stratified sampling for large datasets
- **Data Standardization**: StandardScaler for better convergence
- **Vectorized Operations**: Efficient pandas/numpy operations
- **Robust Discretization**: Handles various data distributions
- **Graceful Error Handling**: Comprehensive exception management

---

## 🔧 Advanced Usage Examples

### **Example 1: Complete Causal Analysis Workflow**
```
1. Upload business data (marketing_spend, sales_revenue, customer_satisfaction)
2. Use Enhanced Scatter Plot to explore relationships
3. Run Intervention Analysis: "What if marketing spend increases by $1000?"
4. Perform Pathway Analysis: marketing_spend → customer_satisfaction → sales_revenue
5. Compare Algorithm robustness across different thresholds
6. Export findings with professional visualizations
```

### **Example 2: Comprehensive Forecasting Analysis**
```
1. Upload time series data with multiple variables
2. Use Time Series Analysis visualization for pattern exploration
3. Test multiple forecasting models (ARIMA, SARIMA, VAR)
4. Compare model performance using comprehensive metrics
5. Generate forecasts with confidence intervals
6. Export forecast results and model diagnostics
```

### **Example 3: Advanced Data Exploration**
```
1. Upload complex dataset with mixed variable types
2. Generate Smart Data Insights for automated profiling
3. Use Correlation Heatmap to identify strong relationships
4. Apply Distribution Analysis for comprehensive variable understanding
5. Filter and sort results using advanced table features
6. Export insights and visualizations for reporting
```

### **Example 4: Research-Grade Analysis**
```
1. Upload experimental data with treatment and outcome variables
2. Set significance filters to p < 0.01 for rigorous analysis
3. Use Statistical Box Plot for group comparisons
4. Perform Intervention Analysis for causal effect estimation
5. Validate findings using Algorithm Comparison
6. Export publication-ready results with statistical annotations
```

---

## 🚀 Why This Dashboard?

### **🎯 Enterprise-Grade Features:**
- McKinsey Vizro integration for professional visualizations
- 7 advanced forecasting models with comprehensive metrics
- 3 sophisticated causal analysis methods
- Bayesian Network integration for intervention analysis
- Publication-ready charts and statistical rigor

### **⚡ Performance & Reliability:**
- Up to 60% faster than basic implementations
- Robust error handling with graceful fallbacks
- Handles large datasets efficiently with smart optimizations
- Cross-platform compatibility (Windows, macOS, Linux)
- Comprehensive testing and validation

### **🎨 Professional User Experience:**
- Intuitive interface with comprehensive tooltips
- Mobile-responsive design for all devices
- Advanced filtering, sorting, and export capabilities
- Real-time progress feedback and status updates
- Professional visual styling with consistent theming

### **🔧 Developer & Research Friendly:**
- Modern Python stack with latest libraries
- UV for fast dependency management
- Comprehensive documentation and examples
- Extensible architecture for custom features
- Open-source with active development

---

## 📞 Installation & Support

### **Dependencies:**
All dependencies are automatically managed. Key packages include:
- **gradio** ≥4.0.0 - Web interface
- **vizro** ≥0.1.25 - Enhanced visualizations
- **causalnex** ≥0.12.0 - Causal discovery
- **statsmodels** ≥0.13.0 - Forecasting models
- **plotly** ≥5.0.0 - Interactive charts
- **pandas, numpy, scikit-learn** - Data processing

### **Getting Help:**
- 💡 Use built-in tooltips and comprehensive help text
- 📖 Check feature documentation and examples
- 🔍 Review error messages for specific guidance
- 🧪 Run test suites to verify functionality

### **Verification Commands:**
```bash
# Check all dependencies
python check_dependencies.py

# Test Vizro integration
python test_vizro_features.py

# Test forecasting models
python test_forecasting_models.py

# Test causal analysis features
python test_intervention_fix.py
```

---

**🎉 Ready to discover causal relationships, create professional visualizations, and build sophisticated forecasting models? Get started with the quick start guide above!**

*Powered by Vizro, CausalNex, Statsmodels, UV, Gradio, and modern data science tools.*