# 🔬 Enhanced Lag Analysis - Triple Method Approach

## 🎯 Overview

The lag analysis feature now combines **three complementary methods** for comprehensive time-delayed causal discovery:

1. **Cross-Correlation Functions** - Identify timing and strength
2. **Granger Causality Tests** - Confirm predictive power
3. **VAR (Vector Autoregression) Models** - Quantify dynamic effects

## 🚀 Why Combine Three Methods?

Each method provides unique insights:

| Method | What It Reveals | Strength | Limitation |
|--------|----------------|----------|------------|
| **Cross-Correlation** | Timing and direction of relationships | Fast, intuitive, visual | Only linear relationships |
| **Granger Causality** | Statistical significance of prediction | Rigorous hypothesis testing | Doesn't quantify effect size |
| **VAR Models** | Dynamic interactions and impulse responses | Captures bidirectional effects | Requires more data |

**Together**: Complete picture of lagged causal relationships!

## 📊 Method 1: Cross-Correlation Functions

### **Purpose**
Measures how variables relate at different time delays.

### **Implementation**
```python
for lag in range(0, max_lags + 1):
    if lag == 0:
        corr = df[target].corr(df[predictor])
    else:
        shifted = df[predictor].shift(lag)
        corr = df[target].corr(shifted)
```

### **Output**
- Bar chart showing correlation at each lag
- Optimal lag identification
- Correlation strength assessment

### **Interpretation**
- **|r| > 0.7**: Strong lagged relationship
- **0.3 < |r| < 0.7**: Moderate relationship
- **|r| < 0.3**: Weak relationship

## 🔍 Method 2: Granger Causality Tests

### **Purpose**
Tests if past values of predictor significantly help forecast target.

### **Null Hypothesis**
"Predictor does NOT Granger-cause target"

### **Implementation**
```python
from statsmodels.tsa.stattools import grangercausalitytests

granger_results = grangercausalitytests(
    data[[target, predictor]], 
    maxlag=max_lags, 
    verbose=False
)

# Extract F-test p-values
for lag in range(1, max_lags + 1):
    p_value = granger_results[lag][0]['ssr_ftest'][1]
```

### **Output**
- P-values for each lag
- Significance indicators (p < 0.05)
- Table of results

### **Interpretation**
- **p < 0.01**: Very strong evidence of predictive causality
- **0.01 < p < 0.05**: Significant predictive causality
- **p > 0.05**: No significant predictive causality

### **Statistical Test**
Compares two models:
1. **Restricted**: Y(t) = f(Y(t-1), ..., Y(t-k))
2. **Unrestricted**: Y(t) = f(Y(t-1), ..., Y(t-k), X(t-1), ..., X(t-k))

If unrestricted model significantly improves prediction → X Granger-causes Y

## 🔬 Method 3: VAR (Vector Autoregression) Models

### **Purpose**
Models dynamic interactions between multiple time series variables.

### **What is VAR?**
A system of equations where each variable is regressed on its own lags and lags of all other variables:

```
Y(t) = α₁Y(t-1) + ... + αₖY(t-k) + β₁X(t-1) + ... + βₖX(t-k) + ε₁(t)
X(t) = γ₁Y(t-1) + ... + γₖY(t-k) + δ₁X(t-1) + ... + δₖX(t-k) + ε₂(t)
```

### **Implementation**
```python
from statsmodels.tsa.api import VAR

# Fit VAR model
model = VAR(data[[target, predictor]])

# Select optimal lag using AIC
lag_order = model.select_order(maxlags=max_lags)
optimal_lag = lag_order.aic

# Fit with optimal lag
var_fitted = model.fit(optimal_lag)

# Get impulse response functions
irf = var_fitted.irf(periods=max_lags)
```

### **Output**
1. **Optimal Lag Order**: Selected by AIC criterion
2. **Model Fit Statistics**: AIC, BIC, FPE, HQIC
3. **Impulse Response Functions**: How shocks propagate
4. **Cumulative Effects**: Total impact over time

### **Key Metrics**

#### **AIC (Akaike Information Criterion)**
- Balances model fit and complexity
- Lower is better
- Formula: AIC = 2k - 2ln(L)

#### **BIC (Bayesian Information Criterion)**
- Similar to AIC but penalizes complexity more
- Lower is better
- Preferred for larger datasets

#### **FPE (Final Prediction Error)**
- Estimates out-of-sample prediction error
- Lower is better

#### **Impulse Response Function (IRF)**
- Shows response of target to a one-unit shock in predictor
- Traces effect over time
- Reveals dynamic propagation

#### **Cumulative Effect**
- Sum of impulse responses over all periods
- Total long-run impact
- Accounts for persistence

### **Advantages of VAR**
1. **Bidirectional**: Captures feedback effects
2. **Dynamic**: Models time-varying relationships
3. **Multivariate**: Can include many variables
4. **Forecasting**: Enables joint prediction
5. **Policy Analysis**: Simulates interventions

## 🎯 Combined Analysis Workflow

### **Step 1: Cross-Correlation (Exploratory)**
```
Purpose: Quick identification of potential lags
Output: "Strongest correlation at lag 3"
Decision: Focus detailed analysis on promising lags
```

### **Step 2: Granger Causality (Confirmation)**
```
Purpose: Statistical validation of predictive power
Output: "Lag 3 is significant (p = 0.002)"
Decision: Confirm that relationship is not spurious
```

### **Step 3: VAR Model (Quantification)**
```
Purpose: Measure effect size and dynamics
Output: "Cumulative effect = 0.45 over 10 periods"
Decision: Quantify practical significance
```

## 📊 Example Analysis

### **Scenario: Marketing Spend → Sales**

#### **Cross-Correlation Results**
```
Lag 0: r = 0.15 (weak)
Lag 1: r = 0.32 (moderate)
Lag 2: r = 0.58 (strong) ← Optimal
Lag 3: r = 0.41 (moderate)
```
**Finding**: Strongest effect at 2-week lag

#### **Granger Causality Results**
```
Lag 1: p = 0.082 (not significant)
Lag 2: p = 0.003 (significant) ✓
Lag 3: p = 0.021 (significant) ✓
```
**Finding**: Marketing spend significantly predicts sales at lags 2-3

#### **VAR Model Results**
```
Optimal Lag: 2 (by AIC)
AIC: -145.32
Cumulative Effect: 0.67
Impulse Response at lag 2: 0.42
```
**Finding**: $1 increase in marketing → $0.67 increase in sales over 10 weeks

### **Combined Interpretation**
1. **Timing**: Marketing affects sales after 2-week delay
2. **Significance**: Effect is statistically significant (p < 0.01)
3. **Magnitude**: Each dollar spent generates $0.67 in cumulative sales
4. **Recommendation**: Plan campaigns 2 weeks before target sales period

## 🔧 Technical Implementation

### **Data Requirements**
```python
min_data_points = max_lags + 10  # Minimum for reliable results
recommended_points = 50+          # For robust VAR estimation
```

### **Computational Complexity**
- **Cross-Correlation**: O(n × k) - Very fast
- **Granger Test**: O(n × k²) - Moderate
- **VAR Model**: O(n × k³) - Most intensive

### **Memory Usage**
- **Cross-Correlation**: O(n) - Minimal
- **Granger Test**: O(n × k) - Moderate
- **VAR Model**: O(n × k²) - Higher for large k

## 🎨 Visualization Outputs

### **1. Cross-Correlation Bar Chart**
- X-axis: Lag periods
- Y-axis: Correlation coefficient
- Color: Gradient (red = negative, blue = positive)
- Annotations: Correlation values

### **2. Lagged Scatter Plots (2×2 Grid)**
- Lags 0, 1, 2, 3
- Correlation coefficient in each panel
- Visual pattern identification

### **3. Results Table**
- Granger test p-values
- Significance indicators
- Color-coded by significance

### **4. VAR Model Statistics**
- Model fit criteria table
- Impulse response summary
- Cumulative effect display

## 🚀 Best Practices

### **1. Start with Cross-Correlation**
- Quick exploratory analysis
- Identify promising lags
- Guide detailed analysis

### **2. Validate with Granger Tests**
- Confirm statistical significance
- Avoid spurious correlations
- Test multiple lags

### **3. Quantify with VAR**
- Measure effect sizes
- Understand dynamics
- Enable forecasting

### **4. Interpret Holistically**
- Don't rely on single method
- Look for consistency across methods
- Use domain knowledge

### **5. Check Assumptions**
- **Stationarity**: VAR requires stationary data
- **Linearity**: All methods assume linear relationships
- **No confounders**: Consider omitted variables

## 📚 Statistical Background

### **Cross-Correlation**
```
ρ(k) = Cov(X(t-k), Y(t)) / (σₓ × σᵧ)
```

### **Granger Causality F-Test**
```
F = [(RSS_restricted - RSS_unrestricted) / k] / [RSS_unrestricted / (n - 2k - 1)]
```

### **VAR Model**
```
Y(t) = A₁Y(t-1) + ... + AₚY(t-p) + ε(t)
```
where Y(t) is a vector of variables

### **Impulse Response**
```
IRF(h) = ∂Y(t+h) / ∂ε(t)
```
Effect of shock at time t on Y at time t+h

## 🎯 When to Use Each Method

### **Use Cross-Correlation When:**
- Quick exploratory analysis needed
- Visual intuition desired
- Computational resources limited
- Initial hypothesis generation

### **Use Granger Tests When:**
- Statistical validation required
- Publication/reporting standards
- Hypothesis testing framework
- Significance levels needed

### **Use VAR Models When:**
- Effect size quantification needed
- Forecasting required
- Bidirectional effects suspected
- Policy simulation desired
- Multiple variables involved

## ✅ Quality Assurance

### **Validation Checks**
1. **Data Quality**: No missing values, outliers handled
2. **Stationarity**: Check with ADF test for VAR
3. **Lag Selection**: Compare AIC, BIC, HQIC
4. **Residual Diagnostics**: Check for autocorrelation
5. **Stability**: VAR roots inside unit circle

### **Robustness Tests**
1. **Subsample Analysis**: Split data and retest
2. **Alternative Lags**: Test sensitivity to lag choice
3. **Variable Transformations**: Try logs, differences
4. **Bootstrap**: Confidence intervals for IRFs

## 🎉 Impact

This triple-method approach provides:

1. **Comprehensive Understanding**: Multiple perspectives on same relationship
2. **Robust Conclusions**: Cross-validation across methods
3. **Actionable Insights**: From timing to magnitude
4. **Scientific Rigor**: Statistical validation
5. **Practical Utility**: Forecasting and simulation

**Result**: Users can confidently identify, validate, and quantify time-delayed causal effects in their data!