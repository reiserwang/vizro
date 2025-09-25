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

### Engineered Causal Variables:
- `Marketing_Budget`: Monthly marketing spend (primary driver)
- `Sales_Volume`: Units sold (main outcome variable)
- `Revenue`: Price × Sales_Volume (deterministic relationship)
- `Customer_Satisfaction`: 1-10 scale (subscription only)
- `Repeat_Sales`: Follow-up sales (subscription only)
- `Profit_Margin`: 0.05-0.45 range (affected by competition)
- `Training_Hours`: Salesperson training (10-60 hours)
- `Market_Competition`: 1-5 scale (affects margins)
- `Economic_Index`: Economic indicator (affects overall performance)
- `Lead_Generation`: Number of leads (marketing-driven)

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

### 2. Causal Graph Visualization
- **Green/Orange edges**: Statistically significant (p < 0.05)
- **Blue/Red edges**: Non-significant relationships
- **Edge thickness**: Proportional to causal weight
- **Hover details**: View correlation coefficients and p-values

### 2. Model Quality Evaluation
- **Network Statistics**: Nodes, edges, density, DAG validation
- **Statistical Tests**: Pearson/Spearman correlations, p-values
- **Cross-Validation**: 5-fold CV performance metrics
- **Residual Analysis**: Normality tests for model assumptions

### 3. Enhanced Data Table
- **Statistical Metrics**: Correlation coefficients, p-values, R² scores
- **Significance Indicators**: Color-coded significant relationships
- **Sortable Columns**: Easy analysis of relationship strength

## Expected Results

When you load this dataset, you should see:

1. **Strong Marketing → Sales relationship** with very high significance
2. **Economic conditions affecting sales** with moderate significance  
3. **Training impact on performance** with weak but detectable significance
4. **Competition affecting profit margins** 
5. **Clear causal network structure** with meaningful business relationships

## Troubleshooting

If relationships don't appear significant:
1. Ensure you're using the enhanced dataset (`aed_sales_data_enhanced.csv`)
2. Check that all required packages are installed
3. Verify the causal analysis section loads without errors
4. Look for green/orange edges in the network graph (significant relationships)

The enhanced dataset is specifically designed to showcase the dashboard's statistical validation capabilities with real business-meaningful causal relationships.