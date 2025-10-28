# 📦 Installation Guide

## 🚀 **Quick Start**

### **Option 1: Using UV (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd dynamic-data-analysis-dashboard

# Install with UV (fastest)
uv sync

# Run the dashboard
uv run gradio_dashboard.py
```

### **Option 2: Using Pip**
```bash
# Clone the repository
git clone <repository-url>
cd dynamic-data-analysis-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python gradio_dashboard.py
```

### **Option 3: Using Pip with pyproject.toml**
```bash
# Install as editable package
pip install -e .

# Run the dashboard
dashboard
```

## 📋 **Dependencies Overview**

### **Core Framework**
- **Gradio** (≥4.0.0) - Web interface and interactive components
- **Pandas** (≥1.5.0) - Data manipulation and analysis
- **NumPy** (≥1.21.0) - Numerical computing foundation

### **Visualization Stack**
- **Plotly** (≥5.0.0) - Interactive plotting library
- **Vizro** (≥0.1.25) - McKinsey's advanced visualization framework
- **Matplotlib** (≥3.5.0) - Additional plotting capabilities
- **Seaborn** (≥0.11.0) - Statistical data visualization

### **Machine Learning & Statistics**
- **Scikit-learn** (≥1.0.0) - Machine learning algorithms
- **SciPy** (≥1.7.0) - Scientific computing functions
- **Statsmodels** (≥0.13.0) - Statistical modeling and time series
- **Patsy** (≥0.5.0) - Statistical modeling formulas

### **Causal Analysis**
- **CausalNex** (≥0.12.0) - Causal discovery algorithms (NOTEARS)
- **NetworkX** (≥2.6.0) - Graph analysis and network algorithms

### **Forecasting Models**
- **PMDArima** (≥2.0.0) - Auto-ARIMA and seasonal forecasting
- **Statsmodels** (≥0.13.0) - ARIMA, SARIMA, VAR, State-Space models

### **File Handling**
- **OpenPyXL** (≥3.0.0) - Excel file reading/writing
- **XLRD** (≥2.0.0) - Legacy Excel file support

### **Utilities**
- **TQDM** (≥4.60.0) - Progress bars and tracking

## 🔧 **Development Dependencies**

For development and testing:
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

**Development tools include:**
- **Pytest** (≥7.0.0) - Testing framework
- **Black** (≥22.0.0) - Code formatting
- **Flake8** (≥4.0.0) - Code linting
- **MyPy** (≥0.950) - Type checking

## 🐍 **Python Version Requirements**

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10 or 3.11
- **Tested on**: Python 3.8, 3.9, 3.10, 3.11

## 🚨 **Troubleshooting**

### **Common Installation Issues**

#### **1. Vizro Installation Problems**
```bash
# If Vizro fails to install
pip install --upgrade pip setuptools wheel
pip install vizro

# Alternative: Install without Vizro (dashboard will use fallback visualizations)
pip install -r requirements.txt --ignore-installed vizro
```

#### **2. CausalNex Dependencies**
```bash
# CausalNex may require additional system dependencies
# On Ubuntu/Debian:
sudo apt-get install build-essential

# On macOS:
xcode-select --install
```

#### **3. Statsmodels/PMDArima Issues**
```bash
# If forecasting libraries fail
pip install --upgrade numpy scipy
pip install statsmodels pmdarima
```

#### **4. Memory Issues with Large Datasets**
```bash
# For large dataset support, consider installing:
pip install dask  # Parallel computing
pip install modin[ray]  # Faster pandas operations
```

### **Platform-Specific Notes**

#### **Windows**
- Use `pip install` instead of `uv` if UV is not available
- Some packages may require Visual Studio Build Tools
- Consider using Anaconda for easier dependency management

#### **macOS**
- Xcode Command Line Tools may be required
- Use Homebrew for system-level dependencies if needed

#### **Linux**
- Build tools (`gcc`, `g++`) may be required
- Some distributions need additional development headers

## 🎯 **Feature-Specific Dependencies**

### **For Enhanced Visualizations**
```bash
pip install vizro>=0.1.25
```

### **For Advanced Forecasting**
```bash
pip install statsmodels>=0.13.0 pmdarima>=2.0.0
```

### **For Causal Analysis**
```bash
pip install causalnex>=0.12.0 networkx>=2.6.0
```

## 📊 **Verification**

After installation, verify everything works:
```bash
# Run the test suite
python test_vizro_features.py
python test_forecasting_models.py
python test_intervention_fix.py

# Start the dashboard
python gradio_dashboard.py
```

## 🔄 **Updating Dependencies**

### **With UV**
```bash
uv sync --upgrade
```

### **With Pip**
```bash
pip install --upgrade -r requirements.txt
```

## 💡 **Optional Enhancements**

### **For Better Performance**
```bash
pip install numba  # JIT compilation for numerical functions
pip install bottleneck  # Fast NumPy array functions
```

### **For Additional File Formats**
```bash
pip install pyarrow  # Parquet file support
pip install h5py  # HDF5 file support
pip install sqlalchemy  # Database connectivity
```

### **For Advanced Analytics**
```bash
pip install xgboost  # Gradient boosting
pip install lightgbm  # Microsoft's gradient boosting
pip install catboost  # Yandex's gradient boosting
```

---

**🎉 Once installed, you'll have access to all the advanced features including Vizro-enhanced visualizations, comprehensive forecasting models, and sophisticated causal analysis capabilities!**