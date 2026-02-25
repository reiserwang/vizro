# Enhanced Sales Data - Causal Analysis Guide

## Overview
The `aed_sales_data_enhanced.csv` file contains engineered causal relationships that will demonstrate significant statistical relationships in the CausalNex dashboard.

## How to Use

1. **Start the Dashboard**:
   ```bash
   python dashboard.py
   ```

2. **Load the Enhanced Dataset**:
   - Upload the file `aed_sales_data_enhanced.csv` using the drag-and-drop interface
   - Or enter the file path in the URL field if serving locally

3. **Expected Significant Causal Relationships**:

### Strong Relationships (p < 0.001)
- **Marketing_Budget → Sales_Volume** (r ≈ 0.997)
  - Very strong positive relationship
  - Green edge in the causal graph
  - Demonstrates how marketing investment drives sales

- **Economic_Index → Sales_Volume** (r ≈ 0.490)
  - Moderate positive relationship
  - Shows economic conditions affect sales performance

### Moderate Relationships (p < 0.05)
- **Sales_Volume → Revenue** (r ≈ 0.275)
  - Positive relationship (affected by price variation)
  - Shows volume impact on total revenue

- **Training_Hours → Sales_Volume** (r ≈ 0.064)
  - Weak but significant positive relationship
  - Demonstrates training effectiveness

### Subscription-Only Relationships
For records where `SaleType_Numeric = 0` (subscriptions):
- **Customer_Satisfaction → Repeat_Sales**
- **Sales_Volume → Customer_Satisfaction**

## Key Variables Explained

### **Primary Causal Chain Variables:**
- `Economic_Index`: External economic conditions (100-115 range)
- `Marketing_Budget`: Monthly marketing investment ($27K-$93K)
- `Lead_Generation`: Number of qualified leads (50-300 leads)
- `Sales_Volume`: Units sold (345-1000 units)
- `Revenue`: Total sales revenue (Price × Volume)
- `Customer_Satisfaction`: Service quality rating (1-10 scale)
- `Training_Hours`: Salesperson development (10-80 hours)
- `Market_Competition`: Competitive pressure (1-5 scale)
- `Profit_Margin`: Profitability percentage (0.1-0.5 range)

### **Supporting Business Variables:**
- `Salesperson_Skill`: Individual skill level (1-5 scale)
- `Product_Tier`: Product quality level (1=Basic, 2=Standard, 3=Premium)
- `Price`: Individual transaction price
- `Region`: Geographic sales region (North, South, East, West)
- `Product_Line`: Product category (Basic, Standard, Premium)

### **Incident Impact Variables:**
- `Incident_Impact`: Numerical impact of business incidents (-0.35 to +0.25)
- `Incident_Type`: Type of business incident (21 different types)
- `Incident_Description`: Human-readable incident description
- `Incident_Impact_Category`: Categorical impact level (Negative, Neutral, Positive)

### **Control Variables (Should Show No Causality):**
- `Employee_ID`: Random employee identifier (1000-9999)
- `Office_Temperature`: Workplace temperature (68-78°F)
- `Random_Factor`: Pure statistical noise

### Categorical Variables (for Color Coding):
- `Salesperson`: Alice, Bob, Charlie, David, Eve
- `SaleType`: Subscription, Buyout  
- `Discontinued`: Active, Discontinued, N/A
- `Performance_Category`: Low, Medium, High (based on Sales_Volume)
- `Marketing_Category`: Low, Medium, High, Very High (based on Marketing_Budget)
- `Revenue_Category`: Low, Medium, High (based on Revenue)
- `Season`: Spring, Summer, Fall, Winter
- `Year_Period`: 2015-2017, 2018-2020, 2021-2023, 2024+

### Encoded Variables (for Causal Analysis):
- `Salesperson_Numeric`: 1=Alice, 2=Bob, 3=Charlie, 4=David, 5=Eve
- `SaleType_Numeric`: 0=Subscription, 1=Buyout
- `Discontinued_Numeric`: 0=False, 1=True, 0.5=N/A

## Dashboard Features to Explore

### 1. Interactive Visualizations
- **Color Options**: Use categorical variables like Salesperson, SaleType, Season, Performance_Category
- **Chart Types**: Scatter, Line, and Bar charts with multiple variables
- **Time Filtering**: Filter by date ranges to see temporal patterns
- **Aggregation**: Switch between raw data and averaged values

### 2. Interactive Causal Graph Visualization
- **Significance Filtering**: Toggle to hide non-significant edges (p ≥ 0.05)
- **Correlation Threshold**: Slider to set minimum correlation strength (0.0 to 0.8)
- **Dynamic Updates**: Graph and table filter automatically together
- **Color Coding**: 
  - **Green/Orange edges**: Statistically significant (p < 0.05)
  - **Blue/Red edges**: Non-significant relationships
- **Edge thickness**: Proportional to causal weight
- **Hover details**: View correlation coefficients and p-values
- **Focus Mode**: Hide weak relationships to emphasize strong causal connections

### 3. Model Quality Evaluation
- **Network Statistics**: Nodes, edges, density, DAG validation
- **Statistical Tests**: Pearson/Spearman correlations, p-values
- **Cross-Validation**: 5-fold CV performance metrics
- **Residual Analysis**: Normality tests for model assumptions

### 3. Enhanced Data Table with Statistical Explanations
- **Comprehensive Metrics**: Causal weights, correlation coefficients, p-values, R² scores
- **Built-in Explanations**: Detailed descriptions of what each metric means
- **Color-coded Significance**: Visual indicators for statistical significance levels
- **Interactive Legend**: Quick reference for interpreting results
- **Sortable Columns**: Easy analysis of relationship strength
- **Quality Assessment**: Clear guidelines for evaluating relationship reliability

## Expected Results

When you load this dataset, you should see:

### **Strong Significant Relationships (p < 0.001, Green/Orange Edges):**
1. **Lead_Generation → Sales_Volume** (r ≈ 0.87) - Very strong conversion relationship
2. **Economic_Index → Marketing_Budget** (r ≈ 0.50) - Economic conditions drive investment
3. **Training_Hours → Sales_Volume** (r ≈ 0.47) - Training improves performance
4. **Product_Tier → Customer_Satisfaction** (r ≈ 0.40) - Premium products satisfy more
5. **Marketing_Budget → Lead_Generation** (r ≈ 0.39) - Marketing generates leads
6. **Salesperson_Skill → Customer_Satisfaction** (r ≈ 0.29) - Skill improves service
7. **Sales_Volume → Revenue** (r ≈ 0.23) - Volume drives revenue
8. **Market_Competition → Profit_Margin** (r ≈ -0.20) - Competition reduces margins

### **Weak/Non-Significant Relationships (p > 0.05, Blue/Red Edges):**
1. **Employee_ID → Sales_Volume** (r ≈ 0.005, p ≈ 0.59) - ID numbers don't predict sales
2. **Office_Temperature → Revenue** (r ≈ 0.015, p ≈ 0.14) - Temperature unrelated to business
3. **Random_Factor → Profit_Margin** (r ≈ 0.000, p ≈ 0.99) - Pure noise shows no correlation
4. **Random_Factor → Lead_Generation** (r ≈ -0.005, p ≈ 0.59) - Noise unrelated to leads

### **Business Incident Analysis:**
- **46.8% of records** affected by randomized business incidents
- **21 different incident types** including pandemics, product recalls, partnerships
- **Time-decay effects** with stronger impact at incident start, weaker over time
- **Sale-type specific incidents** affecting subscriptions vs. buyouts differently
- **Regional incidents** impacting specific geographic areas
- **Cascading effects** where incidents affect sales → revenue → satisfaction

### **Major Incidents Include:**
- **Pandemic Impact** (-35%): Affected 676 records, avg impact -20.7%
- **Product Recall** (-25%): Affected 217 records, avg impact -17.0%
- **Partnership Deal** (+18%): Affected 524 records, avg impact +10.7%
- **Viral Marketing** (+20%): Affected 230 records, avg impact +13.2%
- **Economic Downturn** (-20%): Affected 301 records, avg impact -13.1%

### **Educational Value:**
- **Clear distinction** between meaningful and spurious relationships
- **Proper p-value distribution** showing statistical rigor
- **Realistic business context** with logical causal chains and external shocks
- **Control variables** demonstrating what non-causality looks like
- **Time series patterns** showing how external events impact business metrics
- **Incident correlation analysis** with strong statistical significance (p < 0.001)

## Advanced Forecasting Models

The enhanced dataset works excellently with all forecasting models:

### **VAR (Vector Autoregression)**
- Analyzes relationships between Marketing_Budget, Sales_Volume, Economic_Index
- Shows how marketing affects sales and economic conditions influence both
- Provides system-wide forecasts considering variable interdependencies

### **Dynamic Factor Model**
- Extracts common factors from Marketing_Budget, Sales_Volume, Economic_Index, Training_Hours
- Identifies underlying business drivers affecting multiple metrics
- Excellent for high-dimensional business data analysis

### **State-Space Model**
- Decomposes Sales_Volume into trend, seasonal, and irregular components
- Shows structural changes in sales patterns over time
- Provides component-wise forecasts and uncertainty analysis

### **Model Selection Guide**
- **Linear/ARIMA**: Use for single variable analysis (Sales_Volume trends)
- **SARIMA**: Use when seasonal patterns are evident in the data
- **VAR**: Use to analyze Marketing_Budget → Sales_Volume → Revenue relationships
- **Dynamic Factor**: Use to identify common drivers across all business metrics
- **State-Space**: Use for structural analysis of sales performance components

## Troubleshooting

If relationships don't appear significant:
1. Ensure you're using the enhanced dataset (`aed_sales_data_enhanced.csv`)
2. Check that all required packages are installed
3. Verify the causal analysis section loads without errors
4. Look for green/orange edges in the network graph (significant relationships)

The enhanced dataset is specifically designed to showcase the dashboard's statistical validation capabilities with real business-meaningful causal relationships.