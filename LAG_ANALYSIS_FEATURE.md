# ⏱️ Lag Analysis Feature - Time-Delayed Causal Effects

## 🎯 Overview

The Lag Analysis feature enables discovery and quantification of time-delayed causal relationships between variables. This addresses a critical limitation in standard causal analysis: the assumption that causes and effects occur simultaneously.

## 🚀 Key Features

### **1. Cross-Correlation Analysis**
- Measures linear relationships at different time lags (0 to max_lags)
- Identifies optimal lag period with strongest correlation
- Visual bar chart showing correlation strength at each lag

### **2. Granger Causality Testing**
- Statistical test for predictive causality
- Tests if past values of predictor help forecast target
- P-values for each lag period (significant if p < 0.05)
- Requires `statsmodels` package (optional)

### **3. Lagged Scatter Plots**
- Visual comparison of relationships at lags 0, 1, 2, 3
- Correlation coefficients for each lag
- Helps identify non-linear lagged patterns

## 📊 Implementation Details

### **Function Signature**:
```python
def perform_lag_analysis(target_var, predictor_var, max_lags, progress=gr.Progress()):
    """
    Perform comprehensive lag analysis
    
    Parameters:
    - target_var: Variable to be predicted/explained
    - predictor_var: Variable that may have lagged effect
    - max_lags: Maximum number of time periods to test
    
    Returns:
    - fig_cross_corr: Cross-correlation bar chart
    - fig_scatter: 2x2 grid of lagged scatter plots
    - results_html: Detailed analysis results
    - status: Success/error message
    """
```

### **Cross-Correlation Calculation**:
```python
for lag in range(0, max_lags + 1):
    if lag == 0:
        # Contemporaneous correlation
        corr = df[target_var].corr(df[predictor_var])
    else:
        # Lagged correlation
        shifted = df[predictor_var].shift(lag)
        corr = df[target_var].corr(shifted)
```

### **Granger Causality Test**:
```python
from statsmodels.tsa.stattools import grangercausalitytests

# Test null hypothesis: predictor does NOT Granger-cause target
granger_results = grangercausalitytests(
    data[[target_var, predictor_var]], 
    maxlag=max_lags, 
    verbose=False
)

# Extract F-test p-values
for lag in range(1, max_lags + 1):
    p_value = granger_results[lag][0]['ssr_ftest'][1]
    significant = p_value < 0.05
```

## 🎨 User Interface

### **Input Controls**:
1. **Target Variable**: Dropdown with numeric columns
2. **Predictor Variable**: Dropdown with numeric columns
3. **Maximum Lags**: Slider (1-20, default 10)
4. **Analyze Button**: Triggers analysis

### **Output Displays**:
1. **Cross-Correlation Plot**: Bar chart showing correlation at each lag
2. **Lagged Scatter Plots**: 2x2 grid comparing lags 0-3
3. **Detailed Results**: HTML report with:
   - Optimal lag identification
   - Granger causality test results
   - Interpretation guide
   - Key findings summary

## 📈 Use Cases

### **1. Economic Analysis**
```
Question: How long does it take for GDP changes to affect consumption?
Setup:
- Target: Consumption
- Predictor: GDP
- Max Lags: 12 (months)

Result: Optimal lag = 3 months, correlation = 0.75
Interpretation: GDP changes affect consumption after 3-month delay
```

### **2. Marketing Attribution**
```
Question: When do advertising campaigns impact sales?
Setup:
- Target: Sales
- Predictor: Ad_Spend
- Max Lags: 8 (weeks)

Result: Optimal lag = 2 weeks, Granger p-value = 0.003
Interpretation: Ads significantly predict sales 2 weeks later
```

### **3. Policy Impact Assessment**
```
Question: How quickly do policy changes affect behavior?
Setup:
- Target: Compliance_Rate
- Predictor: Policy_Stringency
- Max Lags: 6 (months)

Result: Optimal lag = 4 months, correlation = 0.62
Interpretation: Policy effects emerge after 4-month adjustment period
```

### **4. Supply Chain Dynamics**
```
Question: What's the lead time between orders and inventory?
Setup:
- Target: Inventory_Level
- Predictor: Order_Volume
- Max Lags: 10 (days)

Result: Optimal lag = 5 days, correlation = 0.88
Interpretation: Orders affect inventory with 5-day delivery lag
```

## 🔍 Interpretation Guide

### **Cross-Correlation Values**:
- **|r| > 0.7**: Strong lagged relationship
- **0.3 < |r| < 0.7**: Moderate lagged relationship
- **|r| < 0.3**: Weak lagged relationship

### **Granger Causality P-values**:
- **p < 0.01**: Very strong evidence of predictive causality
- **0.01 < p < 0.05**: Significant predictive causality
- **p > 0.05**: No significant predictive causality

### **Optimal Lag**:
- **Lag = 0**: Contemporaneous (same-time) relationship
- **Lag > 0**: Delayed effect (predictor leads target)
- **Multiple significant lags**: Complex dynamic relationship

## ⚙️ Technical Requirements

### **Minimum Data Requirements**:
```python
min_data_points = max_lags + 10
```
- Example: max_lags=10 requires at least 20 data points
- More data = more reliable results
- Recommended: 50+ data points for robust analysis

### **Dependencies**:
- **Required**: pandas, numpy, plotly, scipy
- **Optional**: statsmodels (for Granger causality)

### **Installation**:
```bash
# Install statsmodels for full functionality
pip install statsmodels
```

## 🛡️ Error Handling

### **Insufficient Data**:
```
❌ Insufficient data for lag analysis

Problem: Need at least 20 data points for 10 lags
Available: 15 data points

Solutions:
• Reduce maximum lags
• Use more data
• Remove missing values
```

### **Missing Variables**:
```
❌ Variables not found

Selected variables not found in data

Solutions:
• Check variable names
• Ensure data is loaded
• Select numeric variables
```

### **Statsmodels Not Available**:
```
⚠️ Granger Causality Test Unavailable

Install statsmodels for Granger causality testing:
pip install statsmodels
```

## 📊 Output Visualizations

### **1. Cross-Correlation Bar Chart**:
- **X-axis**: Lag period (0 to max_lags)
- **Y-axis**: Correlation coefficient (-1 to +1)
- **Color**: Gradient from red (negative) to blue (positive)
- **Annotations**: Correlation values on bars
- **Reference line**: Horizontal line at y=0

### **2. Lagged Scatter Plots (2x2 Grid)**:
- **Top-left**: Lag 0 (contemporaneous)
- **Top-right**: Lag 1
- **Bottom-left**: Lag 2
- **Bottom-right**: Lag 3
- **Annotations**: Correlation coefficient in each panel

### **3. Results HTML Report**:
- **Analysis Setup**: Variables and parameters
- **Optimal Lag**: Best lag period and correlation
- **Granger Test Table**: P-values for each lag
- **Interpretation Guide**: How to read results
- **Key Findings**: Summary of main insights

## 🔬 Statistical Background

### **Cross-Correlation**:
Measures linear association between X(t-k) and Y(t):
```
r(k) = Corr(X(t-k), Y(t))
```
where k is the lag period.

### **Granger Causality**:
Tests if X Granger-causes Y by comparing:
- **Restricted model**: Y(t) = f(Y(t-1), ..., Y(t-k))
- **Unrestricted model**: Y(t) = f(Y(t-1), ..., Y(t-k), X(t-1), ..., X(t-k))

If unrestricted model significantly improves prediction, X Granger-causes Y.

### **F-Test**:
```
F = [(RSS_restricted - RSS_unrestricted) / k] / [RSS_unrestricted / (n - 2k - 1)]
```
where:
- RSS = Residual Sum of Squares
- k = number of lags
- n = number of observations

## 🚀 Performance Optimization

### **Computational Complexity**:
- **Cross-Correlation**: O(n × k) where n = data points, k = max lags
- **Granger Test**: O(n × k²) due to regression fitting

### **Memory Usage**:
- **Cross-Correlation**: O(n) - minimal memory
- **Granger Test**: O(n × k) - stores lagged data

### **Optimization Tips**:
1. **Limit max_lags**: Use domain knowledge to set reasonable maximum
2. **Sample large datasets**: For n > 10,000, consider sampling
3. **Cache results**: Store computed lags if running multiple analyses

## 🎯 Best Practices

### **1. Choose Appropriate Max Lags**:
- **Daily data**: 7-30 lags (1-4 weeks)
- **Weekly data**: 4-12 lags (1-3 months)
- **Monthly data**: 6-24 lags (6 months - 2 years)
- **Quarterly data**: 4-8 lags (1-2 years)

### **2. Interpret with Caution**:
- Correlation ≠ causation (even with lags)
- Granger causality = predictive causality (not true causality)
- Consider confounding variables
- Use domain knowledge to validate findings

### **3. Combine with Other Methods**:
- Use with standard causal analysis
- Validate with intervention analysis
- Cross-check with domain expertise
- Consider alternative explanations

## 📚 References

### **Granger Causality**:
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods"
- Geweke, J. (1982). "Measurement of Linear Dependence and Feedback between Multiple Time Series"

### **Cross-Correlation**:
- Box, G. E. P., & Jenkins, G. M. (1976). "Time Series Analysis: Forecasting and Control"
- Chatfield, C. (2003). "The Analysis of Time Series: An Introduction"

## 🎉 Impact

This feature transforms the dashboard from analyzing only contemporaneous relationships to understanding the full temporal dynamics of causal systems, enabling:

1. **Better Predictions**: Identify lead indicators
2. **Policy Planning**: Understand implementation timelines
3. **Resource Allocation**: Time interventions optimally
4. **Scientific Discovery**: Reveal hidden temporal patterns

**Result**: Users can now answer "when" questions in addition to "what" and "how" questions about causal relationships.