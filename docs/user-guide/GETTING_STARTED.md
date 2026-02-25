# Getting Started Guide

## 🚀 Quick Start

Welcome to the Advanced Analytics Dashboard! This guide will help you get up and running quickly.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

### Required Dependencies
The application uses UV for dependency management. All required packages are automatically installed.

## 🛠️ Installation

### Option 1: Using UV (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd advanced-analytics-dashboard

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the application
uv run python main.py
```

### Option 2: Using Pip
```bash
# Clone the repository
git clone <repository-url>
cd advanced-analytics-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 🌐 Accessing the Dashboard

1. **Start the Application**: Run `uv run python main.py`
2. **Open Browser**: Navigate to `http://localhost:7860`
3. **Dashboard Interface**: The web interface will load automatically

## 📊 Your First Analysis

### Step 1: Upload Data
1. Click the **"Upload Data"** section
2. Choose a CSV, Excel, or JSON file
3. Wait for data validation and preview

### Step 2: Explore Your Data
1. Review the **data summary** that appears
2. Check the **column types** and **missing values**
3. Note the **numeric columns** available for analysis

### Step 3: Run Causal Analysis
1. Go to the **"Causal Analysis"** tab
2. Adjust settings if needed:
   - **Hide Non-significant**: Filter weak relationships
   - **Minimum Correlation**: Set correlation threshold
   - **Theme**: Choose light or dark theme
3. Click **"Run Causal Analysis"**
4. Explore the **network visualization** and **results table**

### Step 4: Try Forecasting
1. Switch to the **"Forecasting"** tab
2. Select a **target variable** to forecast
3. Choose **forecast periods** (e.g., 12 months)
4. Select **forecasting model** or use "Auto"
5. Click **"Generate Forecast"**
6. Analyze the **forecast plot** and **metrics**

### Step 5: Create Visualizations
1. Go to the **"Data Visualization"** tab
2. Choose **chart type** (line, scatter, histogram, etc.)
3. Select **variables** for X and Y axes
4. Customize **colors** and **themes**
5. Generate and explore your **interactive chart**

## 📁 Sample Data

The dashboard includes sample datasets for testing:

### Business Analytics Dataset
- **Marketing spend**, **sales volume**, **revenue**
- **Customer satisfaction**, **market competition**
- Perfect for causal analysis and forecasting

### Time Series Dataset
- **Daily sales data** with seasonal patterns
- **Multiple product categories**
- Ideal for forecasting experiments

### Economic Indicators
- **GDP**, **inflation**, **unemployment** data
- **Regional and temporal variations**
- Great for complex causal relationships

## 🎯 Key Features Overview

### Causal Analysis
- **Automatic Discovery**: Finds causal relationships in your data
- **Intervention Analysis**: Predict effects of changes
- **Network Visualization**: See relationships as interactive graphs
- **Statistical Validation**: P-values and significance testing

### Forecasting
- **Multiple Models**: ARIMA, Exponential Smoothing, Linear Regression
- **Automatic Selection**: Finds the best model for your data
- **Confidence Intervals**: Uncertainty quantification
- **Performance Metrics**: MAE, RMSE, MAPE evaluation

### Visualization
- **Interactive Charts**: Zoom, pan, hover for details
- **Multiple Types**: Line, scatter, histogram, heatmap, box plots
- **Customization**: Colors, themes, labels, titles
- **Export Options**: PNG, SVG, HTML formats

## ⚙️ Configuration

### Dashboard Settings
Access settings through the **Settings** tab:
- **Theme**: Light or dark mode
- **Performance**: Adjust for your system
- **Export**: Default export formats
- **Analysis**: Default analysis parameters

### Data Preprocessing
Configure data handling:
- **Missing Values**: Drop, fill with mean/median
- **Outliers**: Remove statistical outliers
- **Sampling**: Handle large datasets efficiently

## 🔧 Troubleshooting

### Common Issues

#### "No Data Loaded" Error
- **Solution**: Upload a valid CSV, Excel, or JSON file
- **Check**: File format and data structure

#### "Insufficient Numeric Columns" Error
- **Solution**: Ensure at least 2 numeric columns in your data
- **Check**: Column data types and values

#### Analysis Takes Too Long
- **Solution**: Use data sampling for large datasets
- **Check**: Dataset size and complexity

#### Visualization Not Showing
- **Solution**: Check variable selection and data types
- **Check**: Browser compatibility and JavaScript enabled

### Performance Tips

#### For Large Datasets
- Enable **automatic sampling** in settings
- Use **fewer variables** for causal analysis
- Consider **data preprocessing** to remove unnecessary columns

#### For Better Results
- Ensure **good data quality** (minimal missing values)
- Use **meaningful variable names**
- Have **sufficient data points** (100+ rows recommended)

## 📚 Next Steps

### Learn More
- **User Manual**: Detailed feature documentation
- **API Reference**: Technical implementation details
- **Examples**: Sample analyses and use cases

### Advanced Features
- **Custom Settings**: Fine-tune analysis parameters
- **Export Options**: Save results and reports
- **Integration**: Use with other tools and workflows

### Get Help
- **Documentation**: Comprehensive guides and references
- **Examples**: Sample datasets and analyses
- **Support**: Community forums and issue tracking

## 🎉 Success!

You're now ready to explore your data with the Advanced Analytics Dashboard. Start with the sample data to familiarize yourself with the features, then upload your own datasets for real insights!

### Quick Tips for Success
1. **Start Simple**: Begin with small, clean datasets
2. **Explore First**: Use visualizations to understand your data
3. **Iterate**: Try different settings and approaches
4. **Export Results**: Save your findings for later use
5. **Read Documentation**: Detailed guides for advanced features