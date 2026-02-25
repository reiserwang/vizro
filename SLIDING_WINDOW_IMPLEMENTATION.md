# 📊 Sliding Window Implementation Guide

## Overview

The dashboard implements sliding window techniques in two main areas:
1. **Moving Averages** for time series smoothing
2. **Rolling Correlations** for dynamic relationship analysis

## 1. Moving Average (Time Series Smoothing)

### **Location**: Time Series Analysis Visualization

### **Implementation**:
```python
# Add moving average if enough data points
if len(df) > 10:
    # Dynamic window size: 10% of data points, minimum 3
    window = max(3, len(df) // 10)
    
    # Calculate rolling mean using pandas
    df['moving_avg'] = df[y_axis].rolling(window=window).mean()
    
    # Add to plot as dashed line
    fig.add_scatter(
        x=df[x_axis], 
        y=df['moving_avg'],
        mode='lines', 
        name=f'Moving Average ({window})',
        line=dict(dash='dash')
    )
```

### **How It Works**:

1. **Window Size Calculation**:
   ```python
   window = max(3, len(df) // 10)
   ```
   - Takes 10% of total data points
   - Minimum window size of 3 to ensure smoothing
   - Example: 100 data points → window size of 10

2. **Rolling Mean Calculation**:
   ```python
   df['moving_avg'] = df[y_axis].rolling(window=window).mean()
   ```
   - Uses pandas `.rolling()` method
   - Computes mean over sliding window
   - First (window-1) values will be NaN

3. **Visual Representation**:
   - Overlaid on original time series
   - Dashed line style for distinction
   - Shows smoothed trend without noise

### **Example**:
```
Data: [10, 12, 15, 11, 13, 16, 14, 17, 19, 18]
Window: 3

Moving Average:
- Position 0-1: NaN (insufficient data)
- Position 2: (10+12+15)/3 = 12.33
- Position 3: (12+15+11)/3 = 12.67
- Position 4: (15+11+13)/3 = 13.00
- ...and so on
```

### **Benefits**:
- ✅ **Noise Reduction**: Smooths out short-term fluctuations
- ✅ **Trend Identification**: Makes long-term patterns visible
- ✅ **Adaptive**: Window size scales with data size
- ✅ **Visual Clarity**: Easy to compare with original data

## 2. Rolling Correlation (Dynamic Relationship Analysis)

### **Location**: Correlation Heatmap Visualization

### **Implementation**:
```python
def create_correlation_heatmap(df, template, window_size=0):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        if window_size > 0:
            # Rolling correlation with specified window
            corr_matrix = df[numeric_cols].rolling(window=window_size).corr().iloc[-len(numeric_cols):]
            title = f'Rolling Correlation Heatmap (Window Size: {window_size})'
        else:
            # Standard correlation (entire dataset)
            corr_matrix = df[numeric_cols].corr()
            title = 'Correlation Heatmap'
        
        # Create heatmap visualization
        fig = px.imshow(
            corr_matrix,
            title=title,
            template=template,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        
        return fig
```

### **How It Works**:

1. **Rolling Window Correlation**:
   ```python
   df[numeric_cols].rolling(window=window_size).corr()
   ```
   - Computes correlation for each sliding window
   - Returns multi-level index DataFrame
   - Each window produces a correlation matrix

2. **Extract Last Window**:
   ```python
   .iloc[-len(numeric_cols):]
   ```
   - Takes the most recent correlation matrix
   - Represents current relationships
   - Useful for time-varying correlations

3. **Window Size Control**:
   - `window_size = 0`: Standard correlation (entire dataset)
   - `window_size > 0`: Rolling correlation with specified window

### **Example**:
```
Data (3 variables, 10 time points):
   A    B    C
0  10   20   30
1  12   22   32
2  15   25   35
...

Window Size: 5

Rolling Correlation:
- Computes correlation for rows 0-4
- Then rows 1-5
- Then rows 2-6
- ...
- Finally rows 5-9 (last window shown)
```

### **Use Cases**:
- **Time-Varying Relationships**: See how correlations change over time
- **Regime Detection**: Identify periods of strong/weak relationships
- **Dynamic Analysis**: Understand evolving data patterns
- **Stability Assessment**: Check if relationships are stable or changing

## 3. Technical Details

### **Pandas Rolling Method**:
```python
DataFrame.rolling(window, min_periods=None, center=False)
```

**Parameters**:
- `window`: Size of the moving window (number of observations)
- `min_periods`: Minimum observations required (default: window size)
- `center`: If True, window is centered (default: False, right-aligned)

**Methods Available**:
- `.mean()`: Moving average
- `.std()`: Moving standard deviation
- `.var()`: Moving variance
- `.sum()`: Moving sum
- `.corr()`: Rolling correlation
- `.cov()`: Rolling covariance
- `.min()`, `.max()`: Moving min/max

### **Window Alignment**:

**Right-aligned (default)**:
```
Data:     [1, 2, 3, 4, 5]
Window=3: [-, -, 2, 3, 4]  (mean of [1,2,3], [2,3,4], [3,4,5])
```

**Center-aligned**:
```
Data:     [1, 2, 3, 4, 5]
Window=3: [-, 2, 3, 4, -]  (mean of [1,2,3], [2,3,4], [3,4,5])
```

## 4. Performance Considerations

### **Computational Complexity**:
- **Moving Average**: O(n) - very efficient
- **Rolling Correlation**: O(n × m²) where m = number of variables

### **Memory Usage**:
- **Moving Average**: O(n) - stores one additional column
- **Rolling Correlation**: O(n × m²) - stores correlation matrix for each window

### **Optimization Tips**:
1. **Limit Window Size**: Smaller windows = faster computation
2. **Reduce Variables**: Fewer variables = faster rolling correlation
3. **Sample Data**: For large datasets, sample before rolling operations
4. **Cache Results**: Store computed rolling statistics if reused

## 5. Advanced Patterns

### **Exponential Moving Average (EMA)**:
```python
# Not currently implemented, but could be added:
df['ema'] = df[y_axis].ewm(span=window, adjust=False).mean()
```

**Benefits**:
- More weight to recent observations
- Smoother than simple moving average
- Better for trend following

### **Weighted Moving Average**:
```python
# Custom weights for different importance
weights = np.array([0.1, 0.2, 0.3, 0.4])
df['wma'] = df[y_axis].rolling(window=4).apply(lambda x: np.dot(x, weights))
```

### **Rolling Statistics Dashboard**:
```python
# Multiple rolling statistics at once
window = 20
df['rolling_mean'] = df[y_axis].rolling(window).mean()
df['rolling_std'] = df[y_axis].rolling(window).std()
df['rolling_min'] = df[y_axis].rolling(window).min()
df['rolling_max'] = df[y_axis].rolling(window).max()

# Bollinger Bands
df['upper_band'] = df['rolling_mean'] + 2 * df['rolling_std']
df['lower_band'] = df['rolling_mean'] - 2 * df['rolling_std']
```

## 6. Best Practices

### **Choosing Window Size**:
1. **Too Small**: Doesn't smooth enough, follows noise
2. **Too Large**: Over-smooths, misses important changes
3. **Rule of Thumb**: 5-10% of data length
4. **Domain Knowledge**: Use meaningful periods (e.g., 7 days, 30 days)

### **Handling Edge Effects**:
```python
# Option 1: Drop NaN values
df_clean = df.dropna()

# Option 2: Use min_periods
df['moving_avg'] = df[y_axis].rolling(window=10, min_periods=3).mean()

# Option 3: Forward fill
df['moving_avg'] = df[y_axis].rolling(window=10).mean().fillna(method='bfill')
```

### **Visualization Tips**:
- Use different line styles (solid vs dashed)
- Add transparency to distinguish layers
- Include window size in legend
- Show confidence intervals with rolling std

## 7. Current Implementation Summary

| Feature | Implementation | Window Size | Location |
|---------|---------------|-------------|----------|
| **Moving Average** | `rolling().mean()` | Dynamic (10% of data, min 3) | Time Series Analysis |
| **Rolling Correlation** | `rolling().corr()` | User-configurable | Correlation Heatmap |

### **Activation**:
- **Moving Average**: Automatic when data > 10 points
- **Rolling Correlation**: Manual via window size parameter

### **Future Enhancements**:
- [ ] Exponential moving average option
- [ ] Multiple window sizes comparison
- [ ] Bollinger Bands for volatility
- [ ] Rolling regression for trend analysis
- [ ] Seasonal decomposition with rolling windows

## 8. Example Use Cases

### **Sales Trend Analysis**:
```python
# Smooth daily sales to see weekly trends
window = 7  # 7-day moving average
df['weekly_avg'] = df['daily_sales'].rolling(window=7).mean()
```

### **Stock Price Analysis**:
```python
# 50-day and 200-day moving averages
df['ma_50'] = df['close'].rolling(window=50).mean()
df['ma_200'] = df['close'].rolling(window=200).mean()
```

### **Sensor Data Smoothing**:
```python
# Remove noise from sensor readings
window = 10
df['smooth_temp'] = df['temperature'].rolling(window=10).mean()
```

### **Dynamic Correlation Monitoring**:
```python
# Monitor changing relationships over time
window = 30
rolling_corr = df[['var1', 'var2']].rolling(window=30).corr()
```

---

The sliding window implementation in this dashboard provides powerful tools for time series analysis and dynamic relationship monitoring, with automatic parameter selection for ease of use and flexibility for advanced users.