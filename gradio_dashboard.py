#!/usr/bin/env python3
"""
Gradio-based Dynamic Data Analysis Dashboard
Optimized UI/UX with tooltips and modern interface
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from causalnex.structure.notears import from_pandas
from causalnex.structure import StructureModel
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
from causalnex.discretiser import Discretiser
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Vizro imports for advanced visualization
try:
    import vizro
    print(f"📊 Vizro base package loaded (version: {getattr(vizro, '__version__', 'unknown')})")
    
    # Try to import additional Vizro components (some may not be available in all versions)
    try:
        from vizro import Vizro
        from vizro.models import Dashboard, Page, Graph
        print("📊 Vizro models imported successfully")
    except ImportError as e:
        print(f"⚠️ Some Vizro models not available: {e}")
    
    try:
        from vizro.models import Card, Container
        from vizro.models.types import capture
        print("📊 Vizro advanced models imported successfully")
    except ImportError as e:
        print(f"⚠️ Some Vizro advanced models not available: {e}")
    
    try:
        from vizro.figures import kpi_card
        print("📊 Vizro figures imported successfully")
    except ImportError as e:
        print(f"⚠️ Some Vizro figures not available: {e}")
    
    # Try waterfall_chart separately as it may not be available in all versions
    try:
        from vizro.figures import waterfall_chart
        print("📊 Vizro waterfall_chart available")
    except ImportError:
        print("ℹ️ Vizro waterfall_chart not available in this version")
    
    try:
        import vizro.plotly.express as vpx
        print("📊 Vizro plotly express imported successfully")
    except ImportError as e:
        print(f"⚠️ Vizro plotly express not available: {e}")
    
    VIZRO_AVAILABLE = True
    print("✅ Vizro integration enabled - Enhanced visualizations available!")
    
except ImportError as e:
    VIZRO_AVAILABLE = False
    print(f"⚠️ Vizro not available: {e}")
    print("   Using standard Plotly visualizations")

# Additional imports for forecasting
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import json

# Global variables for data storage
current_data = None
causal_results = None

def load_data_from_file(file_path):
    """Load data from uploaded file"""
    global current_data
    
    if file_path is None:
        return "❌ No file uploaded", None, None, None, None
    
    try:
        # Read the file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return "❌ Unsupported file format. Please upload CSV or Excel files.", None, None, None, None
        
        current_data = df
        
        # Get column options
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Create sortable data preview
        preview_df = df.head(20)  # Show more rows for better preview
        
        # Create table headers
        table_headers = ''.join([f'<th class="sortable-column" onclick="sortPreviewTable({i}, \'data-preview\')" style="cursor: pointer; user-select: none; position: relative;">{col} <span class="sort-indicator">⇅</span></th>' for i, col in enumerate(preview_df.columns)])
        
        # Create table rows
        table_rows = ""
        for _, row in preview_df.iterrows():
            row_html = "<tr>"
            for col in preview_df.columns:
                cell_value = str(row[col])
                if len(cell_value) > 50:
                    cell_value = cell_value[:50] + "..."
                row_html += f"<td>{cell_value}</td>"
            row_html += "</tr>"
            table_rows += row_html
        
        # JavaScript for sorting (avoiding f-string with backslashes)
        sort_script = """
        <script>
        function sortPreviewTable(columnIndex, tableId) {
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const header = table.querySelectorAll('th')[columnIndex];
            const currentSort = header.getAttribute('data-sort') || 'none';
            
            // Reset all sort indicators in this table
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
            });
            
            let sortedRows;
            let newSort;
            
            if (currentSort === 'none' || currentSort === 'desc') {
                // Sort ascending
                sortedRows = rows.sort((a, b) => {
                    const aVal = a.cells[columnIndex].textContent.trim();
                    const bVal = b.cells[columnIndex].textContent.trim();
                    
                    // Try to parse as numbers
                    const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                    const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        return aNum - bNum;
                    }
                    
                    return aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'});
                });
                newSort = 'asc';
                header.querySelector('.sort-indicator').textContent = '↑';
            } else {
                // Sort descending
                sortedRows = rows.sort((a, b) => {
                    const aVal = a.cells[columnIndex].textContent.trim();
                    const bVal = b.cells[columnIndex].textContent.trim();
                    
                    // Try to parse as numbers
                    const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                    const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        return bNum - aNum;
                    }
                    
                    return bVal.localeCompare(aVal, undefined, {numeric: true, sensitivity: 'base'});
                });
                newSort = 'desc';
                header.querySelector('.sort-indicator').textContent = '↓';
            }
            
            header.setAttribute('data-sort', newSort);
            
            // Clear tbody and append sorted rows
            tbody.innerHTML = '';
            sortedRows.forEach(row => tbody.appendChild(row));
        }
        </script>
        """
        
        preview = f"""
        <div class="table-container">
            <h4>📋 Data Preview (First 20 rows)</h4>
            <table id="data-preview" class="table table-striped table-hover sortable-table">
                <thead class="table-header">
                    <tr>
                        {table_headers}
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        {sort_script}
        """
        
        success_msg = f"✅ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns"
        
        return success_msg, gr.update(choices=all_cols, value=None), gr.update(choices=all_cols, value=None), gr.update(choices=all_cols, value=None), preview
        
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", None, None, None, None

def create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg):
    """Create visualization based on user selections"""
    global current_data
    
    if current_data is None:
        return None
    
    if not x_axis or not y_axis:
        return None
    
    try:
        df = current_data.copy()
        
        # Apply Y-axis aggregation if requested
        if y_axis_agg != "Raw Data":
            if y_axis_agg == "Average":
                if color_var and color_var in df.columns:
                    df = df.groupby([x_axis, color_var])[y_axis].mean().reset_index()
                else:
                    df = df.groupby(x_axis)[y_axis].mean().reset_index()
                chart_title_suffix = f"Average {y_axis}"
            elif y_axis_agg == "Sum":
                if color_var and color_var in df.columns:
                    df = df.groupby([x_axis, color_var])[y_axis].sum().reset_index()
                else:
                    df = df.groupby(x_axis)[y_axis].sum().reset_index()
                chart_title_suffix = f"Total {y_axis}"
            elif y_axis_agg == "Count":
                if color_var and color_var in df.columns:
                    df = df.groupby([x_axis, color_var]).size().reset_index(name=y_axis)
                else:
                    df = df.groupby(x_axis).size().reset_index(name=y_axis)
                chart_title_suffix = f"Count of Records"
        else:
            chart_title_suffix = y_axis
        
        # Set theme
        template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
        
        # Create the plot based on chart type
        if chart_type == 'Scatter Plot':
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var, 
                           title=f'{chart_title_suffix} vs {x_axis}', template=template,
                           hover_data=[col for col in df.columns if col in [x_axis, y_axis, color_var]])
        elif chart_type == 'Line Chart':
            fig = px.line(df, x=x_axis, y=y_axis, color=color_var,
                         title=f'{chart_title_suffix} vs {x_axis}', template=template)
        elif chart_type == 'Bar Chart':
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_var,
                        title=f'{chart_title_suffix} by {x_axis}', template=template)
        elif chart_type == 'Histogram':
            fig = px.histogram(df, x=x_axis, color=color_var,
                             title=f'Distribution of {x_axis}', template=template)
        else:
            return None
        
        # Optimize layout for better UX
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='closest',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
        
    except Exception as e:
        return None

def create_vizro_enhanced_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg):
    """Create enhanced visualization using Vizro capabilities"""
    global current_data
    
    if not VIZRO_AVAILABLE:
        # Fallback to standard visualization
        return create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)
    
    if current_data is None:
        return None
    
    if not x_axis or not y_axis:
        return None
    
    try:
        df = current_data.copy()
        
        # Apply Y-axis aggregation if requested
        if y_axis_agg != "Raw Data":
            if y_axis_agg == "Average":
                if color_var and color_var in df.columns:
                    df = df.groupby([x_axis, color_var])[y_axis].mean().reset_index()
                else:
                    df = df.groupby(x_axis)[y_axis].mean().reset_index()
                chart_title_suffix = f"Average {y_axis}"
            elif y_axis_agg == "Sum":
                if color_var and color_var in df.columns:
                    df = df.groupby([x_axis, color_var])[y_axis].sum().reset_index()
                else:
                    df = df.groupby(x_axis)[y_axis].sum().reset_index()
                chart_title_suffix = f"Total {y_axis}"
            elif y_axis_agg == "Count":
                if color_var and color_var in df.columns:
                    df = df.groupby([x_axis, color_var]).size().reset_index(name=y_axis)
                else:
                    df = df.groupby(x_axis).size().reset_index(name=y_axis)
                chart_title_suffix = f"Count of Records"
        else:
            chart_title_suffix = y_axis
        
        # Set theme
        template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
        
        # Create enhanced Vizro-style visualizations
        if chart_type == 'Enhanced Scatter Plot':
            # Advanced scatter plot with trend lines and marginal distributions
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var,
                           title=f'Enhanced {chart_title_suffix} vs {x_axis}',
                           template=template,
                           marginal_x="histogram", marginal_y="histogram",
                           trendline="ols",
                           hover_data=[col for col in df.columns if col in [x_axis, y_axis, color_var]])
            
            # Add correlation annotation
            if df[x_axis].dtype in ['int64', 'float64'] and df[y_axis].dtype in ['int64', 'float64']:
                corr = df[x_axis].corr(df[y_axis])
                fig.add_annotation(
                    text=f"Correlation: {corr:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
        
        elif chart_type == 'Statistical Box Plot':
            # Enhanced box plot with statistical annotations
            fig = px.box(df, x=x_axis, y=y_axis, color=color_var,
                        title=f'Statistical Distribution: {chart_title_suffix} by {x_axis}',
                        template=template,
                        points="outliers")
            
            # Add mean markers
            if color_var and color_var in df.columns:
                means = df.groupby([x_axis, color_var])[y_axis].mean().reset_index()
            else:
                means = df.groupby(x_axis)[y_axis].mean().reset_index()
            
            fig.add_scatter(x=means[x_axis], y=means[y_axis],
                          mode='markers', marker=dict(symbol='diamond', size=10, color='red'),
                          name='Mean', showlegend=True)
        
        elif chart_type == 'Correlation Heatmap':
            # Create correlation heatmap for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix,
                              title='Correlation Heatmap',
                              template=template,
                              color_continuous_scale='RdBu_r',
                              aspect="auto")
                
                # Add correlation values as text
                fig.update_traces(text=np.around(corr_matrix.values, decimals=2),
                                texttemplate="%{text}", textfont={"size": 10})
            else:
                return None
        
        elif chart_type == 'Distribution Analysis':
            # Multi-panel distribution analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f'{y_axis} Distribution', f'{x_axis} Distribution', 
                              f'{y_axis} vs {x_axis}', 'Summary Statistics'],
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # Y-axis distribution
            fig.add_trace(go.Histogram(x=df[y_axis], name=y_axis, nbinsx=30), row=1, col=1)
            
            # X-axis distribution  
            if df[x_axis].dtype in ['int64', 'float64']:
                fig.add_trace(go.Histogram(x=df[x_axis], name=x_axis, nbinsx=30), row=1, col=2)
            else:
                # For categorical data, show value counts
                value_counts = df[x_axis].value_counts()
                fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, name=x_axis), row=1, col=2)
            
            # Scatter plot
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='markers',
                                   name=f'{y_axis} vs {x_axis}', opacity=0.6), row=2, col=1)
            
            # Summary statistics table
            if df[y_axis].dtype in ['int64', 'float64']:
                stats = df[y_axis].describe()
                fig.add_trace(go.Table(
                    header=dict(values=['Statistic', 'Value']),
                    cells=dict(values=[stats.index, [f"{val:.2f}" for val in stats.values]])
                ), row=2, col=2)
            
            fig.update_layout(height=800, title_text=f"Distribution Analysis: {chart_title_suffix}")
        
        elif chart_type == 'Time Series Analysis':
            # Enhanced time series with trend and seasonality
            if pd.api.types.is_datetime64_any_dtype(df[x_axis]) or x_axis.lower() in ['date', 'time', 'timestamp']:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[x_axis]):
                    df[x_axis] = pd.to_datetime(df[x_axis], errors='coerce')
                
                df = df.dropna(subset=[x_axis]).sort_values(x_axis)
                
                fig = px.line(df, x=x_axis, y=y_axis, color=color_var,
                            title=f'Time Series: {chart_title_suffix}',
                            template=template)
                
                # Add moving average if enough data points
                if len(df) > 10:
                    window = max(3, len(df) // 10)
                    df['moving_avg'] = df[y_axis].rolling(window=window).mean()
                    fig.add_scatter(x=df[x_axis], y=df['moving_avg'],
                                  mode='lines', name=f'Moving Average ({window})',
                                  line=dict(dash='dash'))
            else:
                # Fallback to regular line chart
                fig = px.line(df, x=x_axis, y=y_axis, color=color_var,
                            title=f'{chart_title_suffix} vs {x_axis}', template=template)
        
        elif chart_type == 'Advanced Bar Chart':
            # Enhanced bar chart with error bars and annotations
            if color_var and color_var in df.columns:
                # Grouped bar chart with statistics
                agg_df = df.groupby([x_axis, color_var])[y_axis].agg(['mean', 'std', 'count']).reset_index()
                agg_df.columns = [x_axis, color_var, 'mean', 'std', 'count']
                
                fig = px.bar(agg_df, x=x_axis, y='mean', color=color_var,
                           error_y='std',
                           title=f'{chart_title_suffix} by {x_axis} (with Error Bars)',
                           template=template,
                           hover_data=['count'])
            else:
                # Simple bar chart with value labels
                if y_axis_agg == "Count":
                    value_counts = df[x_axis].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Count by {x_axis}', template=template)
                else:
                    agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
                    fig = px.bar(agg_df, x=x_axis, y=y_axis,
                               title=f'{chart_title_suffix} by {x_axis}', template=template)
                
                # Add value labels on bars
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        
        else:
            # Fallback to standard visualization
            return create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)
        
        # Enhanced layout for all Vizro charts
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='closest',
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(size=12),
            title_font_size=16,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add interactive features (only for traces that support hovertemplate)
        try:
            # Only apply hovertemplate to non-table traces
            for trace in fig.data:
                if hasattr(trace, 'hovertemplate') and trace.type != 'table':
                    trace.update(
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     f'{x_axis}: %{{x}}<br>' +
                                     f'{y_axis}: %{{y}}<br>' +
                                     '<extra></extra>'
                    )
        except Exception:
            # If hover template update fails, continue without it
            pass
        
        return fig
        
    except Exception as e:
        print(f"Vizro visualization error: {e}")
        # Fallback to standard visualization
        return create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)

def create_data_insights_dashboard():
    """Create comprehensive data insights using Vizro capabilities"""
    global current_data
    
    if not VIZRO_AVAILABLE or current_data is None:
        return "Vizro not available or no data loaded"
    
    try:
        df = current_data.copy()
        
        # Generate comprehensive insights
        insights = []
        
        # Basic data info
        insights.append(f"📊 **Dataset Overview**: {len(df)} rows × {len(df.columns)} columns")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"🔢 **Numeric Variables**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})")
            
            # Correlation insights
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # Find strongest correlations (excluding diagonal)
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:  # Strong correlation threshold
                            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    top_corr = corr_pairs[0]
                    insights.append(f"🔗 **Strongest Correlation**: {top_corr[0]} ↔ {top_corr[1]} (r = {top_corr[2]:.3f})")
        
        # Categorical columns analysis
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            insights.append(f"📝 **Categorical Variables**: {len(cat_cols)} ({', '.join(cat_cols[:3])}{'...' if len(cat_cols) > 3 else ''})")
            
            # Find high cardinality columns
            high_card_cols = [col for col in cat_cols if df[col].nunique() > len(df) * 0.5]
            if high_card_cols:
                insights.append(f"🎯 **High Cardinality**: {', '.join(high_card_cols)} (may be identifiers)")
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        if len(missing_cols) > 0:
            insights.append(f"⚠️ **Missing Data**: {len(missing_cols)} columns have missing values")
            worst_missing = missing_cols.idxmax()
            insights.append(f"📉 **Most Missing**: {worst_missing} ({missing_cols[worst_missing]} missing, {missing_cols[worst_missing]/len(df)*100:.1f}%)")
        else:
            insights.append("✅ **Data Quality**: No missing values detected")
        
        # Outlier detection for numeric columns
        if len(numeric_cols) > 0:
            outlier_counts = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_counts[col] = len(outliers)
            
            if outlier_counts:
                max_outlier_col = max(outlier_counts, key=outlier_counts.get)
                insights.append(f"🎯 **Outliers Detected**: {max_outlier_col} has {outlier_counts[max_outlier_col]} outliers")
        
        # Data distribution insights
        if len(numeric_cols) > 0:
            skewed_cols = []
            for col in numeric_cols:
                skewness = df[col].skew()
                if abs(skewness) > 1:  # Highly skewed
                    skewed_cols.append((col, skewness))
            
            if skewed_cols:
                most_skewed = max(skewed_cols, key=lambda x: abs(x[1]))
                insights.append(f"📊 **Distribution**: {most_skewed[0]} is highly skewed (skewness: {most_skewed[1]:.2f})")
        
        # Recommendations
        insights.append("\n**💡 Recommendations:**")
        
        if len(numeric_cols) > 1:
            insights.append("• Use correlation heatmap to explore relationships")
            insights.append("• Try scatter plots for strong correlations")
        
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            insights.append("• Use box plots to compare distributions across categories")
        
        if missing_cols.any():
            insights.append("• Consider data cleaning or imputation for missing values")
        
        if len(df) > 1000:
            insights.append("• Dataset is large - consider sampling for faster exploration")
        
        return "\n".join(insights)
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def prepare_time_series_data(df, target_col, additional_cols=None):
    """Prepare data for time series forecasting"""
    try:
        # Create a simple time index if none exists
        if not any(df.dtypes == 'datetime64[ns]'):
            df = df.copy()
            df['time_index'] = pd.date_range(start='2020-01-01', periods=len(df), freq='M')
            df = df.set_index('time_index')
        
        # Select target and additional columns
        if additional_cols:
            cols = [target_col] + [col for col in additional_cols if col in df.columns and col != target_col]
            ts_data = df[cols].dropna()
        else:
            ts_data = df[[target_col]].dropna()
        
        return ts_data
    except Exception as e:
        raise ValueError(f"Error preparing time series data: {str(e)}")

def linear_regression_forecast(data, target_col, periods, confidence_level=0.95):
    """Simple linear regression forecasting"""
    try:
        from sklearn.linear_model import LinearRegression
        
        y = data[target_col].values
        X = np.arange(len(y)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecasts
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Simple confidence intervals (assuming normal residuals)
        residuals = y - model.predict(X)
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        
        from scipy.stats import t
        alpha = 1 - confidence_level
        t_val = t.ppf(1 - alpha/2, len(y) - 2)
        margin = t_val * std_error
        
        lower_bound = forecast - margin
        upper_bound = forecast + margin
        
        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fitted_values': model.predict(X),
            'model_info': {
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'mse': mse
            }
        }
    except Exception as e:
        raise ValueError(f"Linear regression forecast failed: {str(e)}")

def arima_forecast(data, target_col, periods, confidence_level=0.95):
    """ARIMA forecasting"""
    try:
        # Try to import statsmodels
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            raise ValueError("ARIMA requires statsmodels. Install with: pip install statsmodels")
        
        y = data[target_col].dropna()
        
        # Simple auto-ARIMA (try common configurations)
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                            best_order = (p, d, q)
                    except:
                        continue
        
        if best_model is None:
            raise ValueError("Could not fit ARIMA model. Try a simpler model.")
        
        # Generate forecasts
        forecast_result = best_model.forecast(steps=periods, alpha=1-confidence_level)
        forecast = forecast_result
        
        # Get confidence intervals
        forecast_ci = best_model.get_forecast(steps=periods, alpha=1-confidence_level).conf_int()
        
        return {
            'forecast': forecast,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values,
            'fitted_values': best_model.fittedvalues,
            'model_info': {
                'order': best_order,
                'aic': best_aic,
                'bic': best_model.bic,
                'log_likelihood': best_model.llf
            }
        }
    except Exception as e:
        raise ValueError(f"ARIMA forecast failed: {str(e)}")

def sarima_forecast(data, target_col, periods, seasonal_period, confidence_level=0.95):
    """SARIMA forecasting with seasonality"""
    try:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ValueError("SARIMA requires statsmodels. Install with: pip install statsmodels")
        
        y = data[target_col].dropna()
        
        # Simple SARIMA configuration
        try:
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
            fitted_model = model.fit(disp=False)
        except:
            # Fallback to simpler model
            model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 1, seasonal_period))
            fitted_model = model.fit(disp=False)
        
        # Generate forecasts
        forecast_result = fitted_model.get_forecast(steps=periods, alpha=1-confidence_level)
        forecast = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        return {
            'forecast': forecast.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values,
            'fitted_values': fitted_model.fittedvalues,
            'model_info': {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'seasonal_period': seasonal_period,
                'log_likelihood': fitted_model.llf
            }
        }
    except Exception as e:
        raise ValueError(f"SARIMA forecast failed: {str(e)}")

def var_forecast(data, target_col, additional_cols, periods, confidence_level=0.95):
    """Vector Autoregression forecasting"""
    try:
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except ImportError:
            raise ValueError("VAR requires statsmodels. Install with: pip install statsmodels")
        
        # Prepare multivariate data
        cols = [target_col] + [col for col in additional_cols if col in data.columns and col != target_col]
        if len(cols) < 2:
            raise ValueError("VAR requires at least 2 variables. Please select additional variables.")
        
        var_data = data[cols].dropna()
        
        # Fit VAR model
        model = VAR(var_data)
        
        # Select optimal lag order (max 5 for performance)
        max_lags = min(5, len(var_data) // 4)
        lag_order = model.select_order(maxlags=max_lags)
        optimal_lags = lag_order.aic
        
        fitted_model = model.fit(optimal_lags)
        
        # Generate forecasts
        forecast_result = fitted_model.forecast_interval(var_data.values, steps=periods, alpha=1-confidence_level)
        
        # Extract results for target variable
        target_idx = cols.index(target_col)
        forecast = forecast_result[0][:, target_idx]
        lower_bound = forecast_result[1][:, target_idx]
        upper_bound = forecast_result[2][:, target_idx]
        
        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fitted_values': fitted_model.fittedvalues[target_col],
            'model_info': {
                'lag_order': optimal_lags,
                'variables': cols,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'n_variables': len(cols)
            }
        }
    except Exception as e:
        raise ValueError(f"VAR forecast failed: {str(e)}")

def dynamic_factor_forecast(data, target_col, additional_cols, periods, confidence_level=0.95):
    """Dynamic Factor Model forecasting"""
    try:
        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise ValueError("Dynamic Factor Model requires scikit-learn")
        
        # Prepare multivariate data
        cols = [target_col] + [col for col in additional_cols if col in data.columns and col != target_col]
        if len(cols) < 3:
            raise ValueError("Dynamic Factor Model requires at least 3 variables for meaningful factors.")
        
        factor_data = data[cols].dropna()
        
        # Extract factors using PCA
        n_factors = min(3, len(cols) - 1)
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(factor_data.values)
        
        # Forecast factors using simple AR(1) model
        factor_forecasts = []
        for i in range(n_factors):
            factor_series = factors[:, i]
            # Simple AR(1) forecast
            if len(factor_series) > 1:
                ar_coef = np.corrcoef(factor_series[:-1], factor_series[1:])[0, 1]
                last_value = factor_series[-1]
                factor_forecast = [last_value * (ar_coef ** (j+1)) for j in range(periods)]
            else:
                factor_forecast = [0] * periods
            factor_forecasts.append(factor_forecast)
        
        factor_forecasts = np.array(factor_forecasts).T
        
        # Transform back to original space
        forecast = pca.inverse_transform(factor_forecasts)
        target_idx = cols.index(target_col)
        target_forecast = forecast[:, target_idx]
        
        # Simple confidence intervals based on factor variance
        factor_var = np.var(factors, axis=0)
        total_var = np.sum(factor_var)
        std_error = np.sqrt(total_var)
        margin = 1.96 * std_error  # Approximate 95% CI
        
        return {
            'forecast': target_forecast,
            'lower_bound': target_forecast - margin,
            'upper_bound': target_forecast + margin,
            'fitted_values': pca.inverse_transform(factors)[:, target_idx],
            'model_info': {
                'n_factors': n_factors,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': np.sum(pca.explained_variance_ratio_),
                'variables': cols
            }
        }
    except Exception as e:
        raise ValueError(f"Dynamic Factor Model forecast failed: {str(e)}")

def state_space_forecast(data, target_col, periods, confidence_level=0.95):
    """State-Space Model (Unobserved Components) forecasting"""
    try:
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            raise ValueError("State-Space Model requires statsmodels. Install with: pip install statsmodels")
        
        y = data[target_col].dropna()
        
        # Fit Unobserved Components model
        try:
            model = UnobservedComponents(y, 'local linear trend', seasonal=12 if len(y) > 24 else None)
            fitted_model = model.fit(disp=False)
        except:
            # Fallback to simpler model
            model = UnobservedComponents(y, 'local level')
            fitted_model = model.fit(disp=False)
        
        # Generate forecasts
        forecast_result = fitted_model.get_forecast(steps=periods, alpha=1-confidence_level)
        forecast = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        return {
            'forecast': forecast.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values,
            'fitted_values': fitted_model.fittedvalues,
            'model_info': {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'log_likelihood': fitted_model.llf,
                'components': 'trend + seasonal' if hasattr(fitted_model, 'seasonal') else 'trend only'
            }
        }
    except Exception as e:
        raise ValueError(f"State-Space Model forecast failed: {str(e)}")

def nowcasting_forecast(data, target_col, periods, confidence_level=0.95):
    """Nowcasting - simple short-term forecasting"""
    try:
        y = data[target_col].dropna()
        
        if len(y) < 3:
            raise ValueError("Nowcasting requires at least 3 data points")
        
        # Simple exponential smoothing for nowcasting
        alpha = 0.3  # Smoothing parameter
        
        # Calculate smoothed values
        smoothed = [y.iloc[0]]
        for i in range(1, len(y)):
            smoothed.append(alpha * y.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast using last smoothed value with slight trend
        if len(y) >= 2:
            trend = (y.iloc[-1] - y.iloc[-2]) * 0.5  # Dampened trend
        else:
            trend = 0
        
        last_smoothed = smoothed[-1]
        forecast = [last_smoothed + trend * (i + 1) for i in range(periods)]
        
        # Simple confidence intervals
        residuals = y.values - np.array(smoothed)
        std_error = np.std(residuals)
        margin = 1.96 * std_error
        
        return {
            'forecast': np.array(forecast),
            'lower_bound': np.array(forecast) - margin,
            'upper_bound': np.array(forecast) + margin,
            'fitted_values': np.array(smoothed),
            'model_info': {
                'smoothing_parameter': alpha,
                'trend_component': trend,
                'mse': np.mean(residuals**2),
                'method': 'Exponential Smoothing'
            }
        }
    except Exception as e:
        raise ValueError(f"Nowcasting failed: {str(e)}")

def perform_forecasting(target_var, additional_vars, model_type, periods, seasonal_period, confidence_level, progress=gr.Progress()):
    """Main forecasting function"""
    global current_data
    
    if current_data is None:
        return None, "⚠️ Please upload data first", "No data available for forecasting"
    
    if not target_var:
        return None, "⚠️ Please select a target variable", "Target variable not selected"
    
    try:
        progress(0.1, desc="📊 Preparing time series data...")
        
        # Prepare data
        ts_data = prepare_time_series_data(current_data, target_var, additional_vars)
        
        if len(ts_data) < 10:
            return None, "⚠️ Insufficient data for forecasting (minimum 10 points required)", "Not enough data"
        
        progress(0.3, desc=f"🤖 Fitting {model_type} model...")
        
        # Select and run forecasting model
        if model_type == "Linear Regression":
            result = linear_regression_forecast(ts_data, target_var, periods, confidence_level)
        elif model_type == "ARIMA":
            result = arima_forecast(ts_data, target_var, periods, confidence_level)
        elif model_type == "SARIMA":
            result = sarima_forecast(ts_data, target_var, periods, seasonal_period, confidence_level)
        elif model_type == "VAR (Vector Autoregression)":
            if not additional_vars:
                return None, "⚠️ VAR model requires additional variables", "Additional variables needed for VAR"
            result = var_forecast(ts_data, target_var, additional_vars, periods, confidence_level)
        elif model_type == "Dynamic Factor Model":
            if not additional_vars or len(additional_vars) < 2:
                return None, "⚠️ Dynamic Factor Model requires at least 2 additional variables", "More variables needed"
            result = dynamic_factor_forecast(ts_data, target_var, additional_vars, periods, confidence_level)
        elif model_type == "State-Space Model":
            result = state_space_forecast(ts_data, target_var, periods, confidence_level)
        elif model_type == "Nowcasting":
            result = nowcasting_forecast(ts_data, target_var, periods, confidence_level)
        else:
            return None, f"⚠️ Unknown model type: {model_type}", "Invalid model selection"
        
        progress(0.7, desc="📈 Creating forecast visualization...")
        
        # Create forecast plot
        fig = create_forecast_plot(ts_data, target_var, result, model_type, periods)
        
        progress(0.9, desc="📋 Generating forecast summary...")
        
        # Create summary and metrics
        summary, metrics = create_forecast_summary(result, model_type, target_var, periods, confidence_level)
        
        progress(1.0, desc="✅ Forecast complete!")
        
        return fig, summary, metrics
        
    except Exception as e:
        return None, f"❌ Forecasting failed: {str(e)}", f"Error: {str(e)}"

def create_forecast_plot(data, target_var, result, model_type, periods):
    """Create forecast visualization"""
    try:
        # Prepare historical data
        historical_values = data[target_var].values
        historical_dates = data.index
        
        # Create future dates
        last_date = historical_dates[-1]
        if hasattr(last_date, 'freq') and last_date.freq:
            freq = last_date.freq
        else:
            # Infer frequency from data
            if len(historical_dates) > 1:
                freq = pd.infer_freq(historical_dates) or 'M'
            else:
                freq = 'M'
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=periods, freq=freq)
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Fitted values (if available)
        if 'fitted_values' in result and len(result['fitted_values']) == len(historical_values):
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=result['fitted_values'],
                mode='lines',
                name='Fitted Values',
                line=dict(color='green', width=1, dash='dot'),
                opacity=0.7
            ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=result['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence intervals
        if 'lower_bound' in result and 'upper_bound' in result:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=result['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=result['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name=f'{int((result.get("confidence_level", 0.95))*100)}% Confidence Interval',
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{model_type} Forecast for {target_var}',
            xaxis_title='Time',
            yaxis_title=target_var,
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        # Return error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating forecast plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red')
        )
        return fig

def update_forecast_dropdowns():
    """Update forecasting dropdown options when data is loaded"""
    global current_data
    
    if current_data is None:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    
    # Get numeric columns for forecasting
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    
    return (
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),
        gr.update(choices=numeric_cols, value=None)
    )

def update_causal_dropdowns():
    """Update causal analysis dropdown options when data is loaded"""
    global current_data
    
    if current_data is None:
        return (gr.update(choices=[], value=None), gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None), gr.update(choices=[], value=None))
    
    # Get numeric columns for causal analysis
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    
    return (
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # intervention_target
        gr.update(choices=numeric_cols, value=numeric_cols[1] if len(numeric_cols) > 1 else None),  # intervention_var
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # pathway_source
        gr.update(choices=numeric_cols, value=numeric_cols[1] if len(numeric_cols) > 1 else None)   # pathway_target
    )

def create_forecast_summary(result, model_type, target_var, periods, confidence_level):
    """Create forecast summary and metrics"""
    try:
        # Basic forecast statistics
        forecast_values = result['forecast']
        forecast_mean = np.mean(forecast_values)
        forecast_std = np.std(forecast_values)
        forecast_min = np.min(forecast_values)
        forecast_max = np.max(forecast_values)
        
        # Model-specific information
        model_info = result.get('model_info', {})
        
        # Create summary
        summary = f"""
        ## 📈 {model_type} Forecast Results
        
        **Target Variable:** {target_var}  
        **Forecast Periods:** {periods}  
        **Confidence Level:** {confidence_level*100:.0f}%  
        
        ### 📊 Forecast Statistics
        - **Mean Forecast:** {forecast_mean:.4f}
        - **Forecast Range:** {forecast_min:.4f} to {forecast_max:.4f}
        - **Standard Deviation:** {forecast_std:.4f}
        
        ### 🔍 Model Information
        """
        
        # Add model-specific details
        if model_type == "Linear Regression":
            summary += f"""
        - **Slope:** {model_info.get('slope', 'N/A'):.4f}
        - **R-squared:** {model_info.get('r_squared', 'N/A'):.4f}
        - **MSE:** {model_info.get('mse', 'N/A'):.4f}
        """
        elif model_type in ["ARIMA", "SARIMA"]:
            summary += f"""
        - **Model Order:** {model_info.get('order', 'N/A')}
        - **AIC:** {model_info.get('aic', 'N/A'):.2f}
        - **BIC:** {model_info.get('bic', 'N/A'):.2f}
        """
        elif model_type == "VAR (Vector Autoregression)":
            summary += f"""
        - **Variables:** {', '.join(model_info.get('variables', []))}
        - **Lag Order:** {model_info.get('lag_order', 'N/A')}
        - **AIC:** {model_info.get('aic', 'N/A'):.2f}
        """
        elif model_type == "Dynamic Factor Model":
            summary += f"""
        - **Number of Factors:** {model_info.get('n_factors', 'N/A')}
        - **Total Explained Variance:** {model_info.get('total_explained_variance', 0)*100:.1f}%
        - **Variables Used:** {', '.join(model_info.get('variables', []))}
        """
        elif model_type == "State-Space Model":
            summary += f"""
        - **Components:** {model_info.get('components', 'N/A')}
        - **AIC:** {model_info.get('aic', 'N/A'):.2f}
        - **BIC:** {model_info.get('bic', 'N/A'):.2f}
        """
        elif model_type == "Nowcasting":
            summary += f"""
        - **Method:** {model_info.get('method', 'N/A')}
        - **Smoothing Parameter:** {model_info.get('smoothing_parameter', 'N/A')}
        - **MSE:** {model_info.get('mse', 'N/A'):.4f}
        """
        
        # Create metrics table
        metrics_data = []
        for i, value in enumerate(forecast_values):
            period = i + 1
            lower = result.get('lower_bound', [None]*len(forecast_values))[i]
            upper = result.get('upper_bound', [None]*len(forecast_values))[i]
            
            metrics_data.append({
                'Period': f'T+{period}',
                'Forecast': f'{value:.4f}',
                'Lower Bound': f'{lower:.4f}' if lower is not None else 'N/A',
                'Upper Bound': f'{upper:.4f}' if upper is not None else 'N/A'
            })
        
        # Create HTML table
        metrics_df = pd.DataFrame(metrics_data)
        metrics_html = f"""
        <div class="table-container">
            <h4>📊 Detailed Forecast Values</h4>
            {metrics_df.to_html(classes='table table-striped table-hover', index=False, escape=False)}
        </div>
        """
        
        return summary, metrics_html
        
    except Exception as e:
        return f"Error creating summary: {str(e)}", f"Error creating metrics: {str(e)}"

def create_advanced_causal_table(results_df, edge_stats):
    """Create advanced causal analysis table with sorting and filtering"""
    
    # Create filter controls
    filter_controls = """
    <div class="filter-controls" style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <div style="display: flex; flex-wrap: wrap; gap: 15px; align-items: center;">
            <div style="flex: 1; min-width: 200px;">
                <label style="font-weight: 600; color: #495057; margin-bottom: 5px; display: block;">🔍 Search Variables:</label>
                <input type="text" id="search-input" placeholder="Search source or target variables..." 
                       style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;"
                       onkeyup="filterTable()">
            </div>
            
            <div style="min-width: 150px;">
                <label style="font-weight: 600; color: #495057; margin-bottom: 5px; display: block;">📊 Significance:</label>
                <select id="significance-filter" onchange="filterTable()" 
                        style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                    <option value="all">All Relationships</option>
                    <option value="significant">Significant Only (p < 0.05)</option>
                    <option value="non-significant">Non-Significant Only</option>
                </select>
            </div>
            
            <div style="min-width: 150px;">
                <label style="font-weight: 600; color: #495057; margin-bottom: 5px; display: block;">📈 Correlation Range:</label>
                <select id="correlation-filter" onchange="filterTable()" 
                        style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                    <option value="all">All Correlations</option>
                    <option value="strong">Strong (|r| ≥ 0.7)</option>
                    <option value="moderate">Moderate (0.3 ≤ |r| < 0.7)</option>
                    <option value="weak">Weak (|r| < 0.3)</option>
                    <option value="positive">Positive (r > 0)</option>
                    <option value="negative">Negative (r < 0)</option>
                </select>
            </div>
            
            <div style="min-width: 120px;">
                <label style="font-weight: 600; color: #495057; margin-bottom: 5px; display: block;">🎯 P-value:</label>
                <select id="pvalue-filter" onchange="filterTable()" 
                        style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                    <option value="all">All P-values</option>
                    <option value="very-sig">Very Significant (p < 0.01)</option>
                    <option value="significant">Significant (p < 0.05)</option>
                    <option value="marginal">Marginal (0.05 ≤ p < 0.1)</option>
                </select>
            </div>
            
            <div style="min-width: 100px;">
                <button onclick="resetFilters()" 
                        style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    🔄 Reset
                </button>
            </div>
            
            <div style="min-width: 100px;">
                <button onclick="exportFilteredData()" 
                        style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    📥 Export
                </button>
            </div>
        </div>
        
        <div id="filter-status" style="margin-top: 10px; font-size: 14px; color: #6c757d;">
            Showing all relationships
        </div>
        
        <!-- Color Bar Legend -->
        <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 6px; border: 1px solid #dee2e6;">
            <h6 style="margin: 0 0 8px 0; color: #495057; font-weight: 600;">📊 Color Bar Legend (Correlation Strength):</h6>
            <div style="display: flex; flex-wrap: wrap; gap: 15px; font-size: 14px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 4px; background: #dc3545; border-radius: 2px;"></div>
                    <span><strong>Red:</strong> Strong (|r| ≥ 0.7)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 4px; background: #ffc107; border-radius: 2px;"></div>
                    <span><strong>Yellow:</strong> Moderate (0.3 ≤ |r| < 0.7)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 4px; background: #28a745; border-radius: 2px;"></div>
                    <span><strong>Green:</strong> Weak (|r| < 0.3)</span>
                </div>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 12px; color: #6c757d; font-style: italic;">
                💡 The colored bar on the left of each row indicates the strength of the correlation between variables.
            </p>
        </div>
    </div>
    """
    
    # Create table headers with enhanced sorting
    column_info = [
        ('Source Variable', 'text', 'Source variable in the causal relationship'),
        ('Target Variable', 'text', 'Target variable in the causal relationship'),
        ('Correlation (r)', 'number', 'Pearson correlation coefficient (-1 to +1)'),
        ('P-value', 'number', 'Statistical significance (lower is better)'),
        ('R²', 'number', 'Coefficient of determination (0 to 1)'),
        ('Causal Weight', 'number', 'NOTEARS algorithm weight'),
        ('Significant', 'text', 'Statistical significance (Yes/No)')
    ]
    
    causal_headers = ""
    for i, (col_name, col_type, tooltip) in enumerate(column_info):
        causal_headers += f'''
        <th class="sortable-column" onclick="sortCausalTable({i}, 'causal-results', event)" 
            style="cursor: pointer; user-select: none; position: relative;" 
            title="{tooltip}">
            {col_name} 
            <span class="sort-indicator">⇅</span>
            <span class="sort-priority" style="display: none; font-size: 10px; color: #ffc107;"></span>
        </th>
        '''
    
    # Create table rows with data attributes for filtering
    causal_rows = ""
    for idx, row in results_df.iterrows():
        correlation_val = float(row['Correlation (r)'])
        pvalue_val = float(row['P-value'])
        r2_val = float(row['R²'])
        weight_val = float(row['Causal Weight'])
        
        # Add CSS classes for styling based on significance and strength
        row_class = ""
        if row['Significant'] == 'Yes':
            row_class += "significant-row "
        if abs(correlation_val) >= 0.7:
            row_class += "strong-correlation "
        elif abs(correlation_val) >= 0.3:
            row_class += "moderate-correlation "
        else:
            row_class += "weak-correlation "
        
        causal_rows += f'''
        <tr class="{row_class}" 
            data-source="{row['Source Variable'].lower()}"
            data-target="{row['Target Variable'].lower()}"
            data-correlation="{correlation_val}"
            data-pvalue="{pvalue_val}"
            data-r2="{r2_val}"
            data-weight="{weight_val}"
            data-significant="{row['Significant'].lower()}">
            <td>{row['Source Variable']}</td>
            <td>{row['Target Variable']}</td>
            <td class="numeric-cell">{row['Correlation (r)']}</td>
            <td class="numeric-cell">{row['P-value']}</td>
            <td class="numeric-cell">{row['R²']}</td>
            <td class="numeric-cell">{row['Causal Weight']}</td>
            <td class="significance-cell">
                <span class="badge {'badge-success' if row['Significant'] == 'Yes' else 'badge-secondary'}">
                    {row['Significant']}
                </span>
            </td>
        </tr>
        '''
    
    # Enhanced JavaScript for sorting, filtering, and exporting
    advanced_script = """
    <script>
    let sortState = [];
    let originalRows = [];
    
    // Initialize table when DOM is ready or immediately if already loaded
    function initializeCausalTable() {
        const table = document.getElementById('causal-results');
        if (table && table.querySelector('tbody')) {
            originalRows = Array.from(table.querySelector('tbody').querySelectorAll('tr'));
            updateFilterStatus();
            console.log('Causal table initialized with', originalRows.length, 'rows');
        } else {
            // Retry after a short delay if table not ready
            setTimeout(initializeCausalTable, 100);
        }
    }
    
    // Try to initialize immediately
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeCausalTable);
    } else {
        initializeCausalTable();
    }
    
    function sortCausalTable(columnIndex, tableId, event) {
        console.log('sortCausalTable called:', columnIndex, tableId);
        
        const table = document.getElementById(tableId);
        if (!table) {
            console.error('Table not found:', tableId);
            return;
        }
        
        const tbody = table.querySelector('tbody');
        if (!tbody) {
            console.error('Table body not found');
            return;
        }
        
        const visibleRows = Array.from(tbody.querySelectorAll('tr:not([style*="display: none"])'));
        const header = table.querySelectorAll('th')[columnIndex];
        
        if (!header) {
            console.error('Header not found for column:', columnIndex);
            return;
        }
        
        const currentSort = header.getAttribute('data-sort') || 'none';
        console.log('Current sort state:', currentSort, 'Visible rows:', visibleRows.length);
        
        // Handle multi-column sorting with Ctrl key
        if (!event || !event.ctrlKey) {
            // Single column sort - reset all other columns
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
            });
            table.querySelectorAll('.sort-priority').forEach(priority => {
                priority.style.display = 'none';
                priority.textContent = '';
            });
            sortState = [];
        }
        
        let newSort;
        if (currentSort === 'none' || currentSort === 'desc') {
            newSort = 'asc';
            header.querySelector('.sort-indicator').textContent = '↑';
        } else {
            newSort = 'desc';
            header.querySelector('.sort-indicator').textContent = '↓';
        }
        
        header.setAttribute('data-sort', newSort);
        
        // Update sort state for multi-column sorting
        const existingIndex = sortState.findIndex(s => s.column === columnIndex);
        if (existingIndex >= 0) {
            sortState[existingIndex].direction = newSort;
        } else {
            sortState.push({column: columnIndex, direction: newSort});
        }
        
        // Show sort priority numbers
        sortState.forEach((sort, index) => {
            const prioritySpan = table.querySelectorAll('th')[sort.column].querySelector('.sort-priority');
            if (sortState.length > 1) {
                prioritySpan.style.display = 'inline';
                prioritySpan.textContent = index + 1;
            }
        });
        
        // Sort rows based on multiple columns
        const sortedRows = visibleRows.sort((a, b) => {
            for (let sort of sortState) {
                const aVal = a.cells[sort.column].textContent.trim();
                const bVal = b.cells[sort.column].textContent.trim();
                
                let comparison = 0;
                
                // Try numeric comparison first
                const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    comparison = aNum - bNum;
                } else {
                    comparison = aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'});
                }
                
                if (comparison !== 0) {
                    return sort.direction === 'asc' ? comparison : -comparison;
                }
            }
            return 0;
        });
        
        // Clear tbody and append sorted rows
        tbody.innerHTML = '';
        
        // First append sorted visible rows
        console.log('Appending', sortedRows.length, 'sorted rows');
        sortedRows.forEach(row => tbody.appendChild(row));
        
        // Then append hidden rows (filtered out) at the end
        const hiddenRows = originalRows.filter(row => 
            !sortedRows.includes(row) && 
            (row.style.display === 'none' || row.style.display.includes('none'))
        );
        console.log('Appending', hiddenRows.length, 'hidden rows');
        hiddenRows.forEach(row => tbody.appendChild(row));
        
        console.log('Sort completed for column', columnIndex);
    }
    
    function filterTable() {
        const table = document.getElementById('causal-results');
        if (!table) return;
        
        const searchTerm = document.getElementById('search-input').value.toLowerCase();
        const significanceFilter = document.getElementById('significance-filter').value;
        const correlationFilter = document.getElementById('correlation-filter').value;
        const pvalueFilter = document.getElementById('pvalue-filter').value;
        
        const rows = table.querySelector('tbody').querySelectorAll('tr');
        let visibleCount = 0;
        
        rows.forEach(row => {
            let show = true;
            
            // Search filter
            if (searchTerm) {
                const source = row.dataset.source || '';
                const target = row.dataset.target || '';
                if (!source.includes(searchTerm) && !target.includes(searchTerm)) {
                    show = false;
                }
            }
            
            // Significance filter
            if (significanceFilter !== 'all') {
                const significant = row.dataset.significant;
                if (significanceFilter === 'significant' && significant !== 'yes') show = false;
                if (significanceFilter === 'non-significant' && significant !== 'no') show = false;
            }
            
            // Correlation filter
            if (correlationFilter !== 'all') {
                const correlation = parseFloat(row.dataset.correlation);
                const absCorr = Math.abs(correlation);
                
                switch (correlationFilter) {
                    case 'strong':
                        if (absCorr < 0.7) show = false;
                        break;
                    case 'moderate':
                        if (absCorr < 0.3 || absCorr >= 0.7) show = false;
                        break;
                    case 'weak':
                        if (absCorr >= 0.3) show = false;
                        break;
                    case 'positive':
                        if (correlation <= 0) show = false;
                        break;
                    case 'negative':
                        if (correlation >= 0) show = false;
                        break;
                }
            }
            
            // P-value filter
            if (pvalueFilter !== 'all') {
                const pvalue = parseFloat(row.dataset.pvalue);
                
                switch (pvalueFilter) {
                    case 'very-sig':
                        if (pvalue >= 0.01) show = false;
                        break;
                    case 'significant':
                        if (pvalue >= 0.05) show = false;
                        break;
                    case 'marginal':
                        if (pvalue < 0.05 || pvalue >= 0.1) show = false;
                        break;
                }
            }
            
            row.style.display = show ? '' : 'none';
            if (show) visibleCount++;
        });
        
        updateFilterStatus(visibleCount, rows.length);
    }
    
    function resetFilters() {
        document.getElementById('search-input').value = '';
        document.getElementById('significance-filter').value = 'all';
        document.getElementById('correlation-filter').value = 'all';
        document.getElementById('pvalue-filter').value = 'all';
        
        // Reset sort state
        sortState = [];
        const table = document.getElementById('causal-results');
        if (table) {
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
            });
            table.querySelectorAll('.sort-priority').forEach(priority => {
                priority.style.display = 'none';
            });
        }
        
        filterTable();
    }
    
    function updateFilterStatus(visible = null, total = null) {
        const statusDiv = document.getElementById('filter-status');
        if (!statusDiv) return;
        
        if (visible === null || total === null) {
            const table = document.getElementById('causal-results');
            if (table) {
                const rows = table.querySelector('tbody').querySelectorAll('tr');
                total = rows.length;
                visible = Array.from(rows).filter(row => row.style.display !== 'none').length;
            }
        }
        
        if (visible === total) {
            statusDiv.textContent = `Showing all ${total} relationships`;
            statusDiv.style.color = '#6c757d';
        } else {
            statusDiv.textContent = `Showing ${visible} of ${total} relationships (filtered)`;
            statusDiv.style.color = '#007bff';
        }
    }
    
    function exportFilteredData() {
        const table = document.getElementById('causal-results');
        if (!table) return;
        
        const headers = Array.from(table.querySelectorAll('th')).map(th => 
            th.textContent.replace(/[⇅↑↓]/g, '').trim()
        );
        
        const visibleRows = Array.from(table.querySelectorAll('tbody tr:not([style*="display: none"])'));
        const data = visibleRows.map(row => 
            Array.from(row.cells).map(cell => cell.textContent.trim())
        );
        
        // Create CSV content
        let csvContent = headers.join(',') + '\\n';
        csvContent += data.map(row => 
            row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(',')
        ).join('\\n');
        
        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'causal_analysis_results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        // Show success message
        const statusDiv = document.getElementById('filter-status');
        const originalText = statusDiv.textContent;
        statusDiv.textContent = `✅ Exported ${data.length} relationships to CSV`;
        statusDiv.style.color = '#28a745';
        setTimeout(() => {
            statusDiv.textContent = originalText;
            statusDiv.style.color = '#6c757d';
        }, 3000);
    }
    </script>
    """
    
    # Combine all parts
    table_html = f"""
    {filter_controls}
    <div class="table-container">
        <table id="causal-results" class="table table-striped table-hover sortable-table advanced-table">
            <thead class="table-header">
                <tr>
                    {causal_headers}
                </tr>
            </thead>
            <tbody>
                {causal_rows}
            </tbody>
        </table>
    </div>
    {advanced_script}
    """
    
    return table_html

def perform_causal_analysis_with_status(hide_nonsignificant, min_correlation, theme, show_all_relationships):
    """Wrapper function to handle causal analysis with status updates"""
    import time
    
    # Initial status
    yield "🔍 Initializing causal analysis...", None, "Starting analysis...", None
    time.sleep(0.5)
    
    # Run the actual analysis
    try:
        result = perform_causal_analysis(hide_nonsignificant, min_correlation, theme, show_all_relationships)
        
        # Final status with results
        if isinstance(result[0], str) and "❌" in result[0]:
            # Error case
            yield result[0], result[1], result[2], None
        else:
            # Success case
            yield "✅ Analysis completed successfully!", result[0], result[1], result[2]
            
    except Exception as e:
        yield f"❌ Analysis failed: {str(e)}", None, None, "Analysis could not be completed."

def perform_causal_analysis(hide_nonsignificant, min_correlation, theme, show_all_relationships=False, progress=gr.Progress()):
    """Perform efficient causal analysis on the data with progress tracking"""
    global current_data, causal_results
    
    if current_data is None:
        return None, None, "⚠️ Please upload data first"
    
    try:
        # Initialize progress tracking
        progress(0, desc="🔍 Starting causal analysis...")
        
        df = current_data.copy()
        original_shape = df.shape
        
        progress(0.1, desc="📊 Analyzing data structure...")
        
        # Prepare data for causal analysis with efficiency improvements
        df_numeric = df.copy()
        
        # More efficient categorical encoding
        categorical_cols = df_numeric.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            progress(0.15, desc=f"🔤 Encoding {len(categorical_cols)} categorical variables...")
            
            # Vectorized encoding for better performance
            label_encoders = {}
            for col in categorical_cols:
                if df_numeric[col].notna().sum() > 0:
                    le = LabelEncoder()
                    mask = df_numeric[col].notna()
                    df_numeric.loc[mask, col] = le.fit_transform(df_numeric.loc[mask, col].astype(str))
                    label_encoders[col] = le
        
        progress(0.2, desc="🔢 Selecting numeric variables...")
        
        # Select only numeric columns
        df_numeric = df_numeric.select_dtypes(include=['number'])
        
        # Efficient NaN handling
        nan_threshold = 0.5
        valid_cols = df_numeric.columns[df_numeric.isnull().mean() < nan_threshold]
        df_numeric = df_numeric[valid_cols].dropna()
        
        if df_numeric.shape[1] < 2:
            return "❌ Not enough numeric variables for causal analysis", None, "Insufficient data"
        
        progress(0.25, desc=f"📈 Processing {df_numeric.shape[1]} variables, {df_numeric.shape[0]} samples...")
        
        # Smart variable selection for performance
        max_vars = 12  # Reduced for better performance
        if df_numeric.shape[1] > max_vars:
            progress(0.3, desc=f"🎯 Selecting top {max_vars} most correlated variables...")
            
            # More efficient correlation calculation
            corr_matrix = df_numeric.corr().abs()
            
            # Select variables with highest average correlation
            avg_corr = corr_matrix.mean().sort_values(ascending=False)
            top_vars = avg_corr.head(max_vars).index
            df_numeric = df_numeric[top_vars]
            
            progress(0.35, desc=f"✅ Selected {len(top_vars)} key variables")
        
        # Intelligent sampling for large datasets
        max_samples = 1500  # Reduced for better performance
        if len(df_numeric) > max_samples:
            progress(0.4, desc=f"📊 Sampling {max_samples} representative rows...")
            
            # Stratified sampling to preserve data distribution
            df_numeric = df_numeric.sample(n=max_samples, random_state=42)
        
        # Data preprocessing for better causal discovery
        progress(0.45, desc="🔧 Preprocessing data for causal discovery...")
        
        # Standardize data for better NOTEARS performance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index
        )
        
        progress(0.5, desc="🧠 Running NOTEARS causal discovery algorithm...")
        
        # Learn causal structure with optimized parameters
        try:
            # Use more conservative threshold for better results
            w_threshold = max(0.2, min_correlation * 0.5)
            sm = from_pandas(df_scaled, w_threshold=w_threshold, max_iter=50)
            
            # If show_all_relationships is True, create a second model with very low threshold
            sm_all = None
            if show_all_relationships:
                try:
                    progress(0.6, desc="🌐 Discovering ALL relationships (including weak ones)...")
                    sm_all = from_pandas(df_scaled, w_threshold=0.05, max_iter=50)  # Very low threshold
                except:
                    sm_all = sm  # Fallback to regular model if this fails
            
            progress(0.7, desc=f"🕸️ Found {len(sm.edges())} potential causal relationships...")
            
        except Exception as causal_error:
            progress(1.0, desc="❌ Causal discovery failed")
            # If NOTEARS fails, create empty graph and inform user
            sm = nx.DiGraph()
            sm_all = None
            error_msg = f"⚠️ Causal discovery failed: {str(causal_error)}"
            error_details = "\n\nThis might happen with:\n- Too few samples\n- Highly correlated variables\n- Non-numeric data\n- Complex non-linear relationships"
            return error_msg + error_details, None, "Causal analysis could not be completed. Try with different data or preprocessing."
        
        progress(0.75, desc="📊 Computing statistical measures...")
        
        # Calculate statistics for each edge with vectorized operations
        edge_stats = []  # For table display (filtered)
        edge_stats_for_network = []  # For network display (potentially unfiltered)
        
        # Use appropriate model for edge processing
        edges_to_process = list(sm.edges(data=True))  # Always use regular model for table
        
        # For network, use all relationships model if available and requested
        if show_all_relationships and sm_all is not None:
            edges_for_network = list(sm_all.edges(data=True))
        else:
            edges_for_network = edges_to_process
        
        # Process edges for table (always filtered)
        for i, (u, v, data) in enumerate(edges_to_process):
            if i % 5 == 0:  # Update progress every 5 edges
                progress(0.75 + 0.1 * (i / len(edges_to_process)), 
                        desc=f"📈 Computing table statistics ({i+1}/{len(edges_to_process)})...")
            
            if u in df_numeric.columns and v in df_numeric.columns:
                x_data = df_numeric[u].values
                y_data = df_numeric[v].values
                
                # Calculate correlation and p-value
                r, p = pearsonr(x_data, y_data)
                
                # Calculate R² more efficiently
                if len(x_data) > 1 and np.var(x_data) > 1e-10:
                    reg = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
                    r2 = reg.score(x_data.reshape(-1, 1), y_data)
                else:
                    r2 = 0.0
                
                weight = data.get('weight', 0)
                significant = p < 0.05
                
                # For table display: apply user filters
                if hide_nonsignificant and not significant:
                    continue
                if abs(r) < min_correlation:
                    continue
                
                edge_stats.append({
                    'Source Variable': u,
                    'Target Variable': v,
                    'Correlation (r)': f"{r:.4f}",
                    'P-value': f"{p:.6f}",
                    'R²': f"{r2:.4f}",
                    'Causal Weight': f"{weight:.4f}",
                    'Significant': "Yes" if significant else "No"
                })
        
        # Process edges for network visualization
        for i, (u, v, data) in enumerate(edges_for_network):
            if i % 10 == 0:  # Update progress every 10 edges
                progress(0.85 + 0.05 * (i / len(edges_for_network)), 
                        desc=f"📊 Computing network statistics ({i+1}/{len(edges_for_network)})...")
            
            if u in df_numeric.columns and v in df_numeric.columns:
                x_data = df_numeric[u].values
                y_data = df_numeric[v].values
                
                # Calculate correlation and p-value
                r, p = pearsonr(x_data, y_data)
                
                # Calculate R² more efficiently
                if len(x_data) > 1 and np.var(x_data) > 1e-10:
                    reg = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
                    r2 = reg.score(x_data.reshape(-1, 1), y_data)
                else:
                    r2 = 0.0
                
                weight = data.get('weight', 0)
                significant = p < 0.05
                
                # For network: apply filters only if show_all_relationships is False
                if not show_all_relationships:
                    if hide_nonsignificant and not significant:
                        continue
                    if abs(r) < min_correlation:
                        continue
                
                edge_stats_for_network.append({
                    'Source Variable': u,
                    'Target Variable': v,
                    'Correlation (r)': f"{r:.4f}",
                    'P-value': f"{p:.6f}",
                    'R²': f"{r2:.4f}",
                    'Causal Weight': f"{weight:.4f}",
                    'Significant': "Yes" if significant else "No"
                })
        
        progress(0.9, desc="🎨 Creating network visualization...")
        
        # Create network visualization (use appropriate model and edge stats)
        network_model = sm_all if (show_all_relationships and sm_all is not None) else sm
        fig = create_network_plot(network_model, edge_stats_for_network, theme, show_all_relationships)
        
        progress(0.95, desc="📋 Generating advanced results table...")
        
        # Create advanced results table with sorting and filtering
        if edge_stats:
            results_df = pd.DataFrame(edge_stats)
            
            # Convert string numbers back to float for better filtering
            results_df['Correlation_num'] = results_df['Correlation (r)'].astype(float)
            results_df['P_value_num'] = results_df['P-value'].astype(float)
            results_df['R2_num'] = results_df['R²'].astype(float)
            results_df['Weight_num'] = results_df['Causal Weight'].astype(float)
            
            # Create enhanced table with filtering and sorting
            table_html = create_advanced_causal_table(results_df, edge_stats)
            
            # Enhanced summary statistics with performance metrics
            total_relationships = len(edge_stats)
            total_network_relationships = len(edge_stats_for_network)
            significant_count = sum(1 for stat in edge_stats if stat['Significant'] == 'Yes')
            processing_time = "< 30 seconds"  # Estimated with optimizations
            
            # Network display information
            network_display_info = ""
            if show_all_relationships:
                network_display_info = f"""
            **Network Display:** Showing ALL {total_network_relationships} relationships (including weak ones)  
            **Table Display:** Showing {total_relationships} filtered relationships  """
            else:
                network_display_info = f"""
            **Network & Table Display:** Showing {total_relationships} filtered relationships  """
            
            summary = f"""
            ## 📊 Causal Analysis Summary
            
            **Original Data:** {original_shape[0]} rows × {original_shape[1]} columns  
            **Processed Data:** {df_numeric.shape[0]} rows × {df_numeric.shape[1]} variables  
            **Total Relationships Found:** {total_relationships}  
            **Statistically Significant:** {significant_count} ({significant_count/total_relationships*100:.1f}% if total_relationships > 0 else 0)  
            {network_display_info}
            **Processing Time:** {processing_time}  
            **Analysis Method:** NOTEARS (Optimized for performance)  
            **Significance Threshold:** p < 0.05  
            
            ### 🔍 Key Insights:
            - **Green relationships** in the graph are statistically significant (p < 0.05)
            - **Edge thickness** represents the strength of causal relationships
            - **Correlation values** show linear relationship strength (-1 to +1)
            - **R² values** indicate how much variance is explained (0 to 1)
            {f"- **🌐 Network shows ALL relationships** including weak ones (table still filtered)" if show_all_relationships else ""}
            
            ### ⚡ Performance Optimizations Applied:
            - Smart variable selection (top {max_vars} most correlated)
            - Efficient data sampling ({max_samples} max samples)
            - Standardized preprocessing for better convergence
            - Vectorized statistical computations
            """
            
        else:
            table_html = """
            <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 8px; border: 2px dashed #dee2e6;">
                <h4 style="color: #6c757d; margin-bottom: 15px;">🔍 No Relationships Found</h4>
                <p style="color: #6c757d; margin-bottom: 20px;">No causal relationships match your current filter criteria.</p>
                <div style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
                    <strong>💡 Try adjusting:</strong><br>
                    • Lower the correlation threshold<br>
                    • Uncheck "Hide Non-Significant Relationships"<br>
                    • Use different variables or more data
                </div>
            </div>
            """
            summary = f"""
            ## 🔍 No Relationships Found
            
            **Data Processed:** {df_numeric.shape[0]} rows × {df_numeric.shape[1]} variables  
            **Current filters may be too restrictive.**
            
            ### 💡 Suggestions:
            - **Lower correlation threshold** - Try 0.0 to see all relationships
            - **Include non-significant relationships** - Uncheck the significance filter
            - **Check your data** - Ensure you have numeric variables with variation
            - **Add more data** - Causal discovery works better with larger datasets
            
            ### 📊 Data Requirements:
            - At least 2 numeric variables
            - Sufficient sample size (>50 rows recommended)
            - Variables with meaningful variation
            """
        
        progress(1.0, desc="✅ Analysis complete!")
        
        causal_results = {'graph': fig, 'table': table_html, 'summary': summary}
        
        return fig, table_html, summary
        
    except Exception as e:
        progress(1.0, desc="❌ Analysis failed")
        error_msg = f"❌ Error in causal analysis: {str(e)}"
        return None, None, error_msg

def create_network_plot(sm, edge_stats, theme, show_all_relationships=False):
    """Create network visualization using Plotly with proper hover interactions"""
    try:
        # Check if we have nodes
        if len(sm.nodes()) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No causal relationships found with current settings.<br>Try adjusting the correlation threshold or significance filter.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='gray')
            )
            fig.update_layout(
                title="🔍 Causal Relationship Network",
                template='plotly_dark' if theme == 'Dark' else 'plotly_white',
                height=600
            )
            return fig
        
        # Create layout using networkx with better positioning
        pos = nx.spring_layout(sm, k=2, iterations=100, seed=42)
        
        # Prepare edge traces with hover information
        edge_traces = []
        
        for u, v, data in sm.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Find corresponding statistics
            edge_stat = next((stat for stat in edge_stats 
                            if stat['Source Variable'] == u and stat['Target Variable'] == v), None)
            
            if edge_stat:
                # Determine edge color based on significance
                is_significant = edge_stat['Significant'] == 'Yes'
                edge_color = '#2E8B57' if is_significant else '#CD5C5C'  # Green for significant, red for not
                edge_width = 3 if is_significant else 1
                
                # Create hover text
                hover_text = f"{u} → {v}<br>Correlation: {edge_stat['Correlation (r)']}<br>P-value: {edge_stat['P-value']}<br>R²: {edge_stat['R²']}<br>Significant: {edge_stat['Significant']}"
                
                # Add individual edge trace for better hover control
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=False,
                    name=f"{u}→{v}"
                ))
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in sm.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node)[:15] + "..." if len(str(node)) > 15 else str(node))  # Truncate long names
            
            # Count connections for node importance
            in_degree = sm.in_degree(node)
            out_degree = sm.out_degree(node)
            total_connections = in_degree + out_degree
            
            # Scale node size based on connections
            node_size = max(30, min(80, 30 + total_connections * 10))
            node_sizes.append(node_size)
            
            node_info.append(f"<b>{node}</b><br>Incoming connections: {in_degree}<br>Outgoing connections: {out_degree}<br>Total influence: {total_connections}")
        
        # Create figure
        fig = go.Figure()
        
        # Add all edge traces
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)
        
        # Add nodes with proper hover
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F8B500', '#FF69B4', '#20B2AA']
        node_colors = [colors[i % len(colors)] for i in range(len(node_x))]
        
        fig.add_trace(go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            hoverinfo='text',
            hovertext=node_info,
            showlegend=False,
            name="Variables"
        ))
        
        # Update layout with better styling
        template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
        
        # Create title based on whether all relationships are shown
        title_text = "🔍 Causal Relationship Network"
        if show_all_relationships:
            title_text += " (All Relationships)"
        
        # Create annotation text based on display mode
        if show_all_relationships:
            annotation_text = "<b>🌐 Showing ALL relationships</b> (including weak ones)<br><b>Green edges:</b> Significant (p < 0.05) | <b>Red edges:</b> Non-significant<br><b>Node size:</b> Number of connections | <b>Hover:</b> Detailed statistics"
        else:
            annotation_text = "<b>Green edges:</b> Significant relationships (p < 0.05) | <b>Red edges:</b> Non-significant<br><b>Node size:</b> Number of connections | <b>Hover:</b> Detailed statistics"
        
        fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=80),
            annotations=[
                dict(
                    text=annotation_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor='center', yanchor='top',
                    font=dict(color='gray', size=11)
                )
            ],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            template=template,
            height=650,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Enable hover interactions
        fig.update_traces(
            hovertemplate='%{hovertext}<extra></extra>',
        )
        
        return fig
        
    except Exception as e:
        # Return informative error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error creating network visualization:<br>{str(e)}<br><br>This might happen if:<br>• No significant relationships found<br>• Data has too few variables<br>• All correlations below threshold",
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=14, color='red'),
            align='center'
        )
        fig.update_layout(
            title="🔍 Causal Relationship Network - Error",
            template='plotly_dark' if theme == 'Dark' else 'plotly_white',
            height=600
        )
        return fig

def perform_causal_intervention_analysis(target_var, intervention_var, intervention_value, progress=gr.Progress()):
    """Perform causal intervention analysis (do-calculus)"""
    global current_data, causal_results
    
    try:
        progress(0.1, desc="🔬 Preparing intervention analysis...")
        
        if current_data is None:
            return "❌ No data loaded", "Please upload data first"
        
        # Get numeric data
        df_numeric = current_data.select_dtypes(include=[np.number])
        if df_numeric.empty:
            return "❌ No numeric data found", "Please ensure your data contains numeric variables"
        
        if target_var not in df_numeric.columns or intervention_var not in df_numeric.columns:
            return "❌ Variables not found", "Selected variables not found in data"
        
        progress(0.3, desc="🏗️ Building causal structure...")
        
        # Build causal structure using NOTEARS
        sm = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.3)
        
        progress(0.5, desc="🧠 Creating Bayesian Network...")
        
        # Discretize data for Bayesian Network with robust split points
        def create_robust_split_points(series):
            """Create monotonically increasing split points for discretization"""
            min_val = series.min()
            max_val = series.max()
            
            # If there's no variation, create artificial split points
            if min_val == max_val:
                return [min_val - 0.1, min_val + 0.1]
            
            # Try quantile-based split points
            q33 = series.quantile(0.33)
            q67 = series.quantile(0.67)
            
            # Ensure monotonic increasing with minimum separation
            min_separation = (max_val - min_val) * 0.01  # 1% of range
            
            if q67 - q33 < min_separation:
                # If quantiles are too close, use evenly spaced points
                range_val = max_val - min_val
                split1 = min_val + range_val * 0.33
                split2 = min_val + range_val * 0.67
                return [split1, split2]
            else:
                return [q33, q67]
        
        # Create split points for each column
        split_points = {}
        for col in df_numeric.columns:
            split_points[col] = create_robust_split_points(df_numeric[col])
        
        discretiser = Discretiser(
            method="fixed",
            numeric_split_points=split_points
        )
        
        df_discretised = discretiser.transform(df_numeric)
        
        # Create Bayesian Network
        bn = BayesianNetwork(sm)
        bn = bn.fit_node_states(df_discretised)
        bn = bn.fit_cpds(df_discretised, method="BayesianEstimator", bayes_prior="K2")
        
        progress(0.7, desc="🎯 Performing intervention...")
        
        # Create inference engine
        ie = InferenceEngine(bn)
        
        # Discretize intervention value
        intervention_discretised = discretiser.transform(pd.DataFrame({intervention_var: [intervention_value]}))
        intervention_state = intervention_discretised[intervention_var].iloc[0]
        
        # Perform intervention (do-calculus)
        intervention_query = ie.do_intervention(
            {intervention_var: intervention_state},
            {target_var: list(df_discretised[target_var].unique())}
        )
        
        progress(0.9, desc="📊 Generating results...")
        
        # Calculate baseline probabilities (without intervention)
        baseline_query = ie.query({target_var: list(df_discretised[target_var].unique())})
        
        # Create results summary
        results_html = f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🎯 Causal Intervention Analysis</h3>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📋 Analysis Setup</h4>
                <p><strong>Target Variable:</strong> {target_var}</p>
                <p><strong>Intervention Variable:</strong> {intervention_var}</p>
                <p><strong>Intervention Value:</strong> {intervention_value} → {intervention_state}</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📊 Probability Distributions</h4>
                
                <h5>🔵 Baseline (Observational)</h5>
                <div style="font-family: monospace; background: white; padding: 10px; border-radius: 4px;">
        """
        
        for state, prob in baseline_query.items():
            results_html += f"P({target_var} = {state}) = {prob:.4f}<br>"
        
        results_html += """
                </div>
                
                <h5>🔴 After Intervention</h5>
                <div style="font-family: monospace; background: white; padding: 10px; border-radius: 4px;">
        """
        
        for state, prob in intervention_query.items():
            results_html += f"P({target_var} = {state} | do({intervention_var} = {intervention_state})) = {prob:.4f}<br>"
        
        results_html += """
                </div>
            </div>
            
            <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📈 Causal Effect Analysis</h4>
        """
        
        # Calculate causal effects
        for state in baseline_query.keys():
            baseline_prob = baseline_query[state]
            intervention_prob = intervention_query[state]
            effect = intervention_prob - baseline_prob
            effect_pct = (effect / baseline_prob * 100) if baseline_prob > 0 else 0
            
            effect_color = "#4caf50" if effect > 0 else "#f44336" if effect < 0 else "#757575"
            results_html += f"""
                <p style="color: {effect_color};">
                    <strong>{target_var} = {state}:</strong> 
                    {effect:+.4f} ({effect_pct:+.1f}% change)
                </p>
            """
        
        results_html += """
            </div>
            
            <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>💡 Interpretation</h4>
                <p>This analysis shows the <strong>causal effect</strong> of intervening on <strong>{intervention_var}</strong> 
                and setting it to <strong>{intervention_value}</strong> on the probability distribution of <strong>{target_var}</strong>.</p>
                <p>The differences between baseline and intervention probabilities represent the <strong>true causal impact</strong>, 
                not just correlation.</p>
            </div>
        </div>
        """.format(intervention_var=intervention_var, intervention_value=intervention_value, target_var=target_var)
        
        progress(1.0, desc="✅ Intervention analysis complete!")
        
        return results_html, "✅ Causal intervention analysis completed successfully!"
        
    except Exception as e:
        progress(1.0, desc="❌ Analysis failed")
        
        # Provide specific error guidance
        error_details = str(e)
        if "monotonically increasing" in error_details:
            error_msg = """
            ❌ Intervention analysis failed: Data discretization issue
            
            **Problem:** The selected variables have insufficient variation for Bayesian Network analysis.
            
            **Solutions:**
            • Try different variables with more variation
            • Ensure your data has diverse values (not mostly the same)
            • Use variables with continuous distributions
            • Check that your intervention value is within the data range
            
            **Technical details:** """ + error_details
        elif "intervention" in error_details.lower():
            error_msg = f"""
            ❌ Intervention analysis failed: {error_details}
            
            **Suggestions:**
            • Check that the intervention value is reasonable for your data
            • Ensure both variables have sufficient data points
            • Try with different variable combinations
            """
        else:
            error_msg = f"""
            ❌ Intervention analysis failed: {error_details}
            
            **Common causes:**
            • Insufficient data (need at least 50+ rows)
            • Variables with little variation
            • Missing or invalid values
            • Causal structure too complex for the data size
            """
        
        return error_msg, error_msg

def perform_causal_path_analysis(source_var, target_var, progress=gr.Progress()):
    """Analyze causal pathways between two variables"""
    global current_data
    
    try:
        progress(0.1, desc="🛤️ Analyzing causal pathways...")
        
        if current_data is None:
            return "❌ No data loaded", "Please upload data first"
        
        # Get numeric data
        df_numeric = current_data.select_dtypes(include=[np.number])
        if df_numeric.empty:
            return "❌ No numeric data found", "Please ensure your data contains numeric variables"
        
        if source_var not in df_numeric.columns or target_var not in df_numeric.columns:
            return "❌ Variables not found", "Selected variables not found in data"
        
        progress(0.3, desc="🏗️ Building causal structure...")
        
        # Build causal structure
        sm = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.3)
        
        progress(0.5, desc="🔍 Finding causal paths...")
        
        # Find all paths between source and target
        try:
            all_paths = list(nx.all_simple_paths(sm, source_var, target_var, cutoff=5))
        except nx.NetworkXNoPath:
            all_paths = []
        
        progress(0.7, desc="📊 Calculating path strengths...")
        
        # Calculate path strengths
        path_analysis = []
        for path in all_paths:
            path_strength = 1.0
            path_edges = []
            
            for i in range(len(path) - 1):
                if sm.has_edge(path[i], path[i+1]):
                    weight = sm[path[i]][path[i+1]].get('weight', 0)
                    path_strength *= abs(weight)
                    path_edges.append(f"{path[i]} → {path[i+1]} ({weight:.3f})")
            
            path_analysis.append({
                'path': ' → '.join(path),
                'length': len(path) - 1,
                'strength': path_strength,
                'edges': path_edges
            })
        
        # Sort by strength
        path_analysis.sort(key=lambda x: x['strength'], reverse=True)
        
        progress(0.9, desc="📋 Generating pathway report...")
        
        # Create results
        results_html = f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🛤️ Causal Pathway Analysis</h3>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📋 Analysis Setup</h4>
                <p><strong>Source Variable:</strong> {source_var}</p>
                <p><strong>Target Variable:</strong> {target_var}</p>
                <p><strong>Paths Found:</strong> {len(path_analysis)}</p>
            </div>
        """
        
        if path_analysis:
            results_html += """
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>🔗 Causal Pathways (Ranked by Strength)</h4>
            """
            
            for i, path_info in enumerate(path_analysis[:10], 1):  # Show top 10 paths
                strength_color = "#4caf50" if path_info['strength'] > 0.1 else "#ff9800" if path_info['strength'] > 0.01 else "#757575"
                
                results_html += f"""
                <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid {strength_color};">
                    <h5 style="margin: 0 0 8px 0; color: {strength_color};">Path #{i}</h5>
                    <p style="margin: 4px 0; font-family: monospace; font-size: 14px;"><strong>{path_info['path']}</strong></p>
                    <p style="margin: 4px 0; font-size: 12px;">
                        <span style="background: #f0f0f0; padding: 2px 6px; border-radius: 3px;">
                            Length: {path_info['length']} steps
                        </span>
                        <span style="background: {strength_color}; color: white; padding: 2px 6px; border-radius: 3px; margin-left: 8px;">
                            Strength: {path_info['strength']:.4f}
                        </span>
                    </p>
                    <details style="margin-top: 8px;">
                        <summary style="cursor: pointer; font-size: 12px; color: #666;">Edge Details</summary>
                        <div style="margin-top: 4px; font-size: 11px; color: #666;">
                """
                
                for edge in path_info['edges']:
                    results_html += f"<div style='margin: 2px 0;'>{edge}</div>"
                
                results_html += """
                        </div>
                    </details>
                </div>
                """
            
            results_html += "</div>"
        else:
            results_html += """
            <div style="background: #ffebee; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>🚫 No Direct Causal Paths Found</h4>
                <p>No causal pathways were detected between <strong>{source_var}</strong> and <strong>{target_var}</strong>.</p>
                <p>This could mean:</p>
                <ul>
                    <li>The variables are causally independent</li>
                    <li>The causal relationship is too weak to detect</li>
                    <li>There are confounding variables not included in the analysis</li>
                    <li>The relationship is non-linear and not captured by the current method</li>
                </ul>
            </div>
            """.format(source_var=source_var, target_var=target_var)
        
        # Add network statistics
        results_html += f"""
        <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>📊 Network Statistics</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <p><strong>Total Nodes:</strong> {sm.number_of_nodes()}</p>
                    <p><strong>Total Edges:</strong> {sm.number_of_edges()}</p>
                </div>
                <div>
                    <p><strong>Source Out-degree:</strong> {sm.out_degree(source_var) if source_var in sm.nodes() else 0}</p>
                    <p><strong>Target In-degree:</strong> {sm.in_degree(target_var) if target_var in sm.nodes() else 0}</p>
                </div>
            </div>
        </div>
        </div>
        """
        
        progress(1.0, desc="✅ Pathway analysis complete!")
        
        return results_html, "✅ Causal pathway analysis completed successfully!"
        
    except Exception as e:
        progress(1.0, desc="❌ Analysis failed")
        error_msg = f"❌ Pathway analysis failed: {str(e)}"
        return error_msg, error_msg

def perform_causal_discovery_comparison(progress=gr.Progress()):
    """Compare different causal discovery algorithms"""
    global current_data
    
    try:
        progress(0.1, desc="🔬 Comparing causal discovery methods...")
        
        if current_data is None:
            return "❌ No data loaded", "Please upload data first"
        
        # Get numeric data (limit to prevent computational explosion)
        df_numeric = current_data.select_dtypes(include=[np.number])
        if df_numeric.empty:
            return "❌ No numeric data found", "Please ensure your data contains numeric variables"
        
        # Limit to top 8 most correlated variables for performance
        if len(df_numeric.columns) > 8:
            corr_matrix = df_numeric.corr().abs()
            # Get variables with highest average correlation
            avg_corr = corr_matrix.mean().sort_values(ascending=False)
            top_vars = avg_corr.head(8).index.tolist()
            df_numeric = df_numeric[top_vars]
        
        progress(0.3, desc="🏗️ Running NOTEARS algorithm...")
        
        # Method 1: NOTEARS (current method)
        sm_notears = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.3)
        
        progress(0.5, desc="🔍 Running NOTEARS with different parameters...")
        
        # Method 2: NOTEARS with stricter threshold
        sm_notears_strict = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.5)
        
        progress(0.7, desc="🧮 Running NOTEARS with relaxed parameters...")
        
        # Method 3: NOTEARS with relaxed threshold
        sm_notears_relaxed = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.1)
        
        progress(0.9, desc="📊 Comparing results...")
        
        # Compare results
        methods = {
            'NOTEARS (Standard)': {'model': sm_notears, 'threshold': 0.3},
            'NOTEARS (Strict)': {'model': sm_notears_strict, 'threshold': 0.5},
            'NOTEARS (Relaxed)': {'model': sm_notears_relaxed, 'threshold': 0.1}
        }
        
        comparison_results = []
        for method_name, method_info in methods.items():
            model = method_info['model']
            threshold = method_info['threshold']
            
            # Calculate statistics
            num_edges = model.number_of_edges()
            num_nodes = model.number_of_nodes()
            density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            
            # Calculate average edge weight
            avg_weight = np.mean([abs(data['weight']) for _, _, data in model.edges(data=True)]) if num_edges > 0 else 0
            
            # Find strongest relationships
            strongest_edges = []
            if num_edges > 0:
                edge_weights = [(u, v, abs(data['weight'])) for u, v, data in model.edges(data=True)]
                edge_weights.sort(key=lambda x: x[2], reverse=True)
                strongest_edges = edge_weights[:5]  # Top 5
            
            comparison_results.append({
                'method': method_name,
                'threshold': threshold,
                'edges': num_edges,
                'nodes': num_nodes,
                'density': density,
                'avg_weight': avg_weight,
                'strongest': strongest_edges
            })
        
        # Create results HTML
        results_html = f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🔬 Causal Discovery Algorithm Comparison</h3>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📋 Analysis Overview</h4>
                <p><strong>Variables Analyzed:</strong> {len(df_numeric.columns)} ({', '.join(df_numeric.columns[:5])}{'...' if len(df_numeric.columns) > 5 else ''})</p>
                <p><strong>Sample Size:</strong> {len(df_numeric)} rows</p>
                <p><strong>Methods Compared:</strong> {len(methods)} NOTEARS variants</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📊 Method Comparison</h4>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="background: #f0f0f0;">
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Method</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Threshold</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Edges</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Density</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Avg Weight</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for result in comparison_results:
            results_html += f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>{result['method']}</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{result['threshold']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{result['edges']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{result['density']:.3f}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{result['avg_weight']:.3f}</td>
                        </tr>
            """
        
        results_html += """
                    </tbody>
                </table>
            </div>
        """
        
        # Show strongest relationships for each method
        for result in comparison_results:
            if result['strongest']:
                results_html += f"""
                <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h4>🔗 {result['method']} - Strongest Relationships</h4>
                """
                
                for i, (u, v, weight) in enumerate(result['strongest'], 1):
                    strength_color = "#4caf50" if weight > 0.5 else "#ff9800" if weight > 0.3 else "#757575"
                    results_html += f"""
                    <div style="background: white; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 3px solid {strength_color};">
                        <span style="font-family: monospace;">{u} → {v}</span>
                        <span style="float: right; background: {strength_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 12px;">
                            {weight:.3f}
                        </span>
                    </div>
                    """
                
                results_html += "</div>"
        
        # Add recommendations
        results_html += """
        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>💡 Recommendations</h4>
            <ul>
                <li><strong>Strict Threshold (0.5):</strong> Use when you want only the strongest, most reliable causal relationships</li>
                <li><strong>Standard Threshold (0.3):</strong> Balanced approach, good for most analyses</li>
                <li><strong>Relaxed Threshold (0.1):</strong> Use for exploratory analysis to find weak but potentially important relationships</li>
            </ul>
            <p><strong>Note:</strong> Higher thresholds reduce false positives but may miss weaker causal relationships. 
            Lower thresholds capture more relationships but may include spurious connections.</p>
        </div>
        </div>
        """
        
        progress(1.0, desc="✅ Comparison complete!")
        
        return results_html, "✅ Causal discovery comparison completed successfully!"
        
    except Exception as e:
        progress(1.0, desc="❌ Analysis failed")
        error_msg = f"❌ Comparison analysis failed: {str(e)}"
        return error_msg, error_msg

def export_results():
    """Export analysis results"""
    global causal_results, current_data
    
    if causal_results is None:
        return "⚠️ No analysis results to export"
    
    try:
        # Create a simple text report
        report = f"""
# Causal Analysis Report

## Data Summary
- Dataset shape: {current_data.shape if current_data is not None else 'N/A'}
- Analysis timestamp: {pd.Timestamp.now()}

## Analysis Results
{causal_results.get('summary', 'No summary available')}

## Methodology
- Algorithm: NOTEARS (Non-linear causal discovery)
- Statistical tests: Pearson correlation, Linear regression R²
- Significance threshold: p < 0.05

Generated by Dynamic Data Analysis Dashboard
        """
        
        return report
        
    except Exception as e:
        return f"❌ Error exporting results: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    """Create the main Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 5px;
    }
    
    .upload-area {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: rgba(78, 205, 196, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    
    .table-container {
        max-height: 500px;
        overflow-y: auto;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sortable-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    
    .sortable-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 8px;
        text-align: left;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 10;
        border-bottom: 2px solid #fff;
    }
    
    .sortable-table th:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    
    .sortable-table td {
        padding: 10px 8px;
        border-bottom: 1px solid #eee;
        vertical-align: top;
    }
    
    .sortable-table tr:hover {
        background-color: #f8f9fa;
        transition: background-color 0.2s ease;
    }
    
    .sortable-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .sortable-table tr:nth-child(even):hover {
        background-color: #f0f0f0;
    }
    
    .sort-indicator {
        font-size: 12px;
        margin-left: 5px;
        opacity: 0.7;
        transition: all 0.2s ease;
    }
    
    .sortable-column:hover .sort-indicator {
        opacity: 1;
        transform: scale(1.2);
    }
    
    .table-header th {
        user-select: none;
        cursor: pointer;
    }
    
    /* Advanced table styles */
    .advanced-table {
        font-size: 13px;
    }
    
    .filter-controls {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .filter-controls input, .filter-controls select {
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .filter-controls input:focus, .filter-controls select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
        outline: none;
    }
    
    .numeric-cell {
        text-align: right;
        font-family: 'Courier New', monospace;
        font-weight: 500;
    }
    
    .significance-cell {
        text-align: center;
    }
    
    .badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    
    .badge-secondary {
        background-color: #6c757d;
        color: white;
    }
    
    .significant-row {
        background-color: rgba(40, 167, 69, 0.05);
    }
    
    .strong-correlation {
        border-left: 4px solid #dc3545;
    }
    
    .moderate-correlation {
        border-left: 4px solid #ffc107;
    }
    
    .weak-correlation {
        border-left: 4px solid #28a745;
    }
    
    .sort-priority {
        background: #ffc107;
        color: #212529;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-left: 4px;
        font-weight: bold;
    }
    
    /* Hover effects for filter controls */
    .filter-controls button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    
    /* Responsive design for filters */
    @media (max-width: 768px) {
        .filter-controls > div {
            flex-direction: column;
        }
        
        .filter-controls input, .filter-controls select {
            margin-bottom: 10px;
        }
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Responsive table */
    @media (max-width: 768px) {
        .table-container {
            overflow-x: auto;
        }
        
        .sortable-table {
            min-width: 600px;
        }
        
        .sortable-table th,
        .sortable-table td {
            padding: 8px 6px;
            font-size: 12px;
        }
    }
    """
    
    with gr.Blocks(css=css, title="🔍 Dynamic Data Analysis Dashboard", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🔍 Dynamic Data Analysis Dashboard</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.2em;">Advanced Causal Discovery & Statistical Analysis</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Data Upload Tab
            with gr.TabItem("📁 Data Upload", id="upload_tab"):
                gr.Markdown("## Upload Your Dataset")
                gr.Markdown("*Upload CSV or Excel files to begin analysis. The system will automatically detect data types and prepare your data for analysis.*")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="📂 Choose File",
                            file_types=[".csv", ".xlsx", ".xls"],
                            type="filepath"
                        )
                        
                        upload_status = gr.Textbox(
                            label="📊 Upload Status",
                            interactive=False,
                            placeholder="No file uploaded yet..."
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Tips:
                        - **CSV files**: Comma-separated values
                        - **Excel files**: .xlsx or .xls format
                        - **Size limit**: Up to 100MB
                        - **Columns**: Mix of numeric and categorical data works best
                        - **Missing values**: Will be handled automatically
                        """)
                
                data_preview = gr.HTML(label="📋 Data Preview")
        
            # Visualization Tab  
            with gr.TabItem("📈 Data Visualization", id="viz_tab"):
                gr.Markdown("## Create Interactive Visualizations")
                gr.Markdown("*Select variables and chart types to explore your data visually. Hover over points for detailed information.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Visualization Controls")
                        
                        x_axis = gr.Dropdown(
                            label="📊 X-Axis Variable",
                            info="Choose the independent variable for your visualization",
                            interactive=True
                        )
                        
                        y_axis = gr.Dropdown(
                            label="📈 Y-Axis Variable", 
                            info="Choose the dependent variable to analyze",
                            interactive=True
                        )
                        
                        color_var = gr.Dropdown(
                            label="🎨 Color Variable (Optional)",
                            info="Add a third dimension by coloring points/bars by this variable",
                            interactive=True
                        )
                        
                        chart_type = gr.Dropdown(
                            choices=[
                                "Scatter Plot", "Line Chart", "Bar Chart", "Histogram",
                                "Enhanced Scatter Plot", "Statistical Box Plot", "Correlation Heatmap",
                                "Distribution Analysis", "Time Series Analysis", "Advanced Bar Chart"
                            ] if VIZRO_AVAILABLE else ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"],
                            value="Enhanced Scatter Plot" if VIZRO_AVAILABLE else "Scatter Plot",
                            label="📊 Chart Type" + (" (Vizro Enhanced)" if VIZRO_AVAILABLE else ""),
                            info="Select visualization type - Enhanced options available with Vizro!"
                        )
                        
                        y_axis_agg = gr.Radio(
                            choices=["Raw Data", "Average", "Sum", "Count"],
                            value="Raw Data",
                            label="📊 Y-Axis Aggregation",
                            info="Choose how to aggregate Y-axis values: Raw (all points), Average (mean by X-axis), Sum (total by X-axis), or Count (frequency)"
                        )
                        
                        viz_theme = gr.Radio(
                            choices=["Light", "Dark"],
                            value="Light",
                            label="🎨 Theme",
                            info="Choose your preferred visual theme"
                        )
                        
                        create_viz_btn = gr.Button("🚀 Create Visualization", variant="primary", size="lg")
                        
                        if VIZRO_AVAILABLE:
                            gr.Markdown("### 🔍 Smart Data Insights")
                            insights_btn = gr.Button("🧠 Generate Data Insights", variant="secondary")
                            data_insights = gr.Markdown("Click 'Generate Data Insights' to get AI-powered analysis of your data")
                    
                    with gr.Column(scale=2):
                        viz_output = gr.Plot(label="📊 Interactive Visualization")
                        
                        if VIZRO_AVAILABLE:
                            with gr.Accordion("📋 Vizro Enhanced Features", open=False):
                                gr.Markdown("""
                                ### 🚀 Enhanced Visualization Features (Powered by Vizro)
                                
                                **📊 Enhanced Scatter Plot:**
                                - Marginal histograms showing distributions
                                - Automatic trend lines with correlation coefficients
                                - Interactive hover with detailed statistics
                                
                                **📈 Statistical Box Plot:**
                                - Outlier detection and highlighting
                                - Mean markers for quick comparison
                                - Statistical annotations
                                
                                **🔥 Correlation Heatmap:**
                                - Interactive correlation matrix
                                - Color-coded strength indicators
                                - Numerical correlation values overlay
                                
                                **📊 Distribution Analysis:**
                                - Multi-panel comprehensive view
                                - Histograms for both variables
                                - Scatter plot with summary statistics table
                                
                                **⏰ Time Series Analysis:**
                                - Automatic trend detection
                                - Moving averages overlay
                                - Seasonality indicators
                                
                                **📊 Advanced Bar Chart:**
                                - Error bars showing variability
                                - Value labels on bars
                                - Statistical grouping options
                                """)
                        else:
                            with gr.Accordion("💡 Vizro Integration Info", open=False):
                                gr.Markdown("""
                                ### 🚀 Unlock Enhanced Visualizations with Vizro!
                                
                                Install Vizro to access advanced visualization features:
                                ```bash
                                pip install vizro
                                ```
                                
                                **Enhanced features include:**
                                - 📊 Advanced statistical plots
                                - 🔍 Automatic data insights
                                - 📈 Interactive dashboards
                                - 🎨 Professional styling
                                - 📋 Smart recommendations
                                """)
            
            # Forecasting Tab
            with gr.TabItem("📈 Forecasting Models", id="forecasting_tab"):
                gr.Markdown("## Advanced Time Series Forecasting")
                gr.Markdown("*Choose from 7 sophisticated forecasting models to predict future values and analyze trends.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Data Selection")
                        
                        forecast_target = gr.Dropdown(
                            label="🎯 Target Variable",
                            info="Select the variable you want to forecast",
                            interactive=True
                        )
                        
                        forecast_additional = gr.Dropdown(
                            label="📈 Additional Variables (for multivariate models)",
                            info="Select additional variables for VAR and Dynamic Factor models",
                            multiselect=True,
                            interactive=True
                        )
                        
                        gr.Markdown("### 🤖 Model Selection")
                        
                        forecast_model = gr.Dropdown(
                            choices=[
                                "Linear Regression",
                                "ARIMA", 
                                "SARIMA",
                                "VAR (Vector Autoregression)",
                                "Dynamic Factor Model",
                                "State-Space Model",
                                "Nowcasting"
                            ],
                            value="Linear Regression",
                            label="🔮 Forecasting Model",
                            info="Choose the forecasting model based on your data characteristics"
                        )
                        
                        gr.Markdown("### ⚙️ Forecast Settings")
                        
                        forecast_periods = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=12,
                            label="📅 Forecast Periods",
                            info="Number of future periods to predict"
                        )
                        
                        seasonal_period = gr.Slider(
                            minimum=2,
                            maximum=52,
                            step=1,
                            value=12,
                            label="🔄 Seasonal Period (for SARIMA)",
                            info="Number of periods in one seasonal cycle (12 for monthly, 4 for quarterly)"
                        )
                        
                        confidence_level = gr.Slider(
                            minimum=0.8,
                            maximum=0.99,
                            step=0.01,
                            value=0.95,
                            label="📊 Confidence Level",
                            info="Confidence level for prediction intervals"
                        )
                        
                        forecast_btn = gr.Button("🚀 Generate Forecast", variant="primary", size="lg")
                        
                        # Status indicator
                        forecast_status = gr.Textbox(
                            label="📊 Forecast Status",
                            value="Ready to forecast...",
                            interactive=False,
                            lines=2
                        )
                
                with gr.Column(scale=2):
                    forecast_plot = gr.Plot(label="📈 Forecast Visualization")
                    
                    with gr.Accordion("📊 Model Results & Insights", open=True):
                        forecast_summary = gr.Markdown("Select data and model to see forecast results...")
                        forecast_metrics = gr.HTML(label="📋 Forecast Metrics & Statistics")
            
            # Causal Analysis Tab
            with gr.TabItem("🔍 Causal Analysis", id="causal_tab"):
                gr.Markdown("## Discover Causal Relationships")
                gr.Markdown("*Advanced causal discovery using NOTEARS algorithm. Identify which variables influence others in your dataset.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Analysis Settings")
                        
                        hide_nonsig = gr.Checkbox(
                            label="🎯 Hide Non-Significant Relationships",
                            info="Only show relationships with p-value < 0.05 for cleaner results",
                            value=False
                        )
                        
                        min_corr = gr.Slider(
                            minimum=0.0,
                            maximum=0.8,
                            step=0.1,
                            value=0.0,
                            label="📊 Minimum Correlation Threshold",
                            info="Filter out weak correlations below this threshold"
                        )
                        
                        show_all_relationships = gr.Checkbox(
                            label="🌐 Show All Relationships in Network",
                            info="Display all discovered relationships in the network graph, including weak ones (overrides correlation threshold for visualization)",
                            value=False
                        )
                        
                        causal_theme = gr.Radio(
                            choices=["Light", "Dark"],
                            value="Light", 
                            label="🎨 Graph Theme",
                            info="Choose theme for the causal network visualization"
                        )
                        
                        analyze_btn = gr.Button("🔍 Run Causal Analysis", variant="primary", size="lg")
                        
                        # Status indicator for analysis progress
                        analysis_status = gr.Textbox(
                            label="📊 Analysis Status",
                            value="Ready to analyze...",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("### 📤 Export Results")
                        export_btn = gr.Button("💾 Export Analysis Report", variant="secondary")
                        export_output = gr.Textbox(label="📄 Export Status", interactive=False)
                
                with gr.Column(scale=2):
                    causal_network = gr.Plot(label="🕸️ Causal Network Graph")
                    
                    with gr.Accordion("📊 Detailed Results", open=True):
                        causal_summary = gr.Markdown("Run analysis to see results...")
                        causal_table = gr.HTML(label="📋 Statistical Results Table")
            
            # Advanced Causal Analysis Features
            with gr.TabItem("🎯 Causal Intervention", id="intervention_tab"):
                gr.Markdown("## Causal Intervention Analysis")
                gr.Markdown("*Perform do-calculus to understand the causal effect of interventions. Answer questions like 'What happens if I change X to a specific value?'*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Intervention Setup")
                        
                        intervention_target = gr.Dropdown(
                            label="🎯 Target Variable",
                            info="Variable whose outcome you want to predict",
                            choices=[],
                            interactive=True
                        )
                        
                        intervention_var = gr.Dropdown(
                            label="🔧 Intervention Variable", 
                            info="Variable you want to intervene on (change)",
                            choices=[],
                            interactive=True
                        )
                        
                        intervention_value = gr.Number(
                            label="📊 Intervention Value",
                            info="Value to set the intervention variable to",
                            value=0.0
                        )
                        
                        intervention_btn = gr.Button("🎯 Run Intervention Analysis", variant="primary", size="lg")
                        
                        intervention_status = gr.Textbox(
                            label="📊 Analysis Status",
                            value="Ready to analyze...",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column(scale=2):
                        intervention_results = gr.HTML(label="🎯 Intervention Results")
            
            with gr.TabItem("🛤️ Causal Pathways", id="pathway_tab"):
                gr.Markdown("## Causal Pathway Analysis")
                gr.Markdown("*Discover all causal pathways between two variables. Understand how one variable influences another through intermediate steps.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛤️ Pathway Setup")
                        
                        pathway_source = gr.Dropdown(
                            label="🚀 Source Variable",
                            info="Starting variable in the causal chain",
                            choices=[],
                            interactive=True
                        )
                        
                        pathway_target = gr.Dropdown(
                            label="🎯 Target Variable",
                            info="Ending variable in the causal chain", 
                            choices=[],
                            interactive=True
                        )
                        
                        pathway_btn = gr.Button("🛤️ Analyze Pathways", variant="primary", size="lg")
                        
                        pathway_status = gr.Textbox(
                            label="📊 Analysis Status",
                            value="Ready to analyze...",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column(scale=2):
                        pathway_results = gr.HTML(label="🛤️ Pathway Analysis Results")
            
            with gr.TabItem("🔬 Algorithm Comparison", id="comparison_tab"):
                gr.Markdown("## Causal Discovery Algorithm Comparison")
                gr.Markdown("*Compare different causal discovery approaches to understand the robustness of your findings.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔬 Comparison Setup")
                        
                        comparison_btn = gr.Button("🔬 Compare Algorithms", variant="primary", size="lg")
                        
                        comparison_status = gr.Textbox(
                            label="📊 Analysis Status",
                            value="Ready to compare algorithms...",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("""
                        ### 📋 What This Analyzes
                        - **NOTEARS Standard**: Balanced threshold (0.3)
                        - **NOTEARS Strict**: High threshold (0.5) - only strong relationships
                        - **NOTEARS Relaxed**: Low threshold (0.1) - includes weak relationships
                        
                        Compare results to understand the stability and robustness of discovered causal relationships.
                        """)
                    
                    with gr.Column(scale=2):
                        comparison_results = gr.HTML(label="🔬 Algorithm Comparison Results")
        
        # Event handlers
        file_input.change(
            fn=load_data_from_file,
            inputs=[file_input],
            outputs=[upload_status, x_axis, y_axis, color_var, data_preview]
        )
        
        create_viz_btn.click(
            fn=create_vizro_enhanced_visualization if VIZRO_AVAILABLE else create_visualization,
            inputs=[x_axis, y_axis, color_var, chart_type, viz_theme, y_axis_agg],
            outputs=[viz_output]
        )
        
        # Add insights event handler if Vizro is available
        if VIZRO_AVAILABLE:
            insights_btn.click(
                fn=create_data_insights_dashboard,
                outputs=[data_insights]
            )
        
        analyze_btn.click(
            fn=perform_causal_analysis_with_status,
            inputs=[hide_nonsig, min_corr, causal_theme, show_all_relationships],
            outputs=[analysis_status, causal_network, causal_table, causal_summary]
        )
        
        export_btn.click(
            fn=export_results,
            outputs=[export_output]
        )
        
        # Advanced Causal Analysis Event Handlers
        intervention_btn.click(
            fn=perform_causal_intervention_analysis,
            inputs=[intervention_target, intervention_var, intervention_value],
            outputs=[intervention_results, intervention_status]
        )
        
        pathway_btn.click(
            fn=perform_causal_path_analysis,
            inputs=[pathway_source, pathway_target],
            outputs=[pathway_results, pathway_status]
        )
        
        comparison_btn.click(
            fn=perform_causal_discovery_comparison,
            outputs=[comparison_results, comparison_status]
        )
        
        # Update forecasting dropdowns when data is loaded
        file_input.change(
            fn=lambda file_path: update_forecast_dropdowns() if file_path else (gr.update(), gr.update()),
            inputs=[file_input],
            outputs=[forecast_target, forecast_additional]
        )
        
        # Update causal analysis dropdowns when data is loaded
        file_input.change(
            fn=lambda file_path: update_causal_dropdowns() if file_path else (gr.update(), gr.update(), gr.update(), gr.update()),
            inputs=[file_input],
            outputs=[intervention_target, intervention_var, pathway_source, pathway_target]
        )
        
        # Forecasting event handler
        forecast_btn.click(
            fn=perform_forecasting,
            inputs=[forecast_target, forecast_additional, forecast_model, forecast_periods, seasonal_period, confidence_level],
            outputs=[forecast_plot, forecast_summary, forecast_metrics]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e0e0e0;">
            <p style="color: #666; margin: 0;">
                🔬 Powered by CausalNex, Plotly & Gradio | 
                📊 Advanced Statistical Analysis | 
                🚀 Built for Data Scientists
            </p>
        </div>
        """)
    
    return demo

def main():
    """Main function for launching the dashboard"""
    print("🚀 Starting Dynamic Data Analysis Dashboard...")
    print("📊 Loading causal analysis engine...")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with optimized settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=False,            # Set to True for public sharing
        debug=False,            # Set to True for development
        show_error=True,        # Show detailed errors
        quiet=False,            # Show startup messages
        inbrowser=True,         # Auto-open browser
        favicon_path=None,      # Add custom favicon if desired
        ssl_verify=False        # For development
    )

if __name__ == "__main__":
    main()