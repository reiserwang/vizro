#!/usr/bin/env python3
"""
Visualization Engine Module
Handles both standard and Vizro-enhanced visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core import dashboard_config
from core.dashboard_config import VIZRO_AVAILABLE
from core.data_handler import convert_date_columns

def create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg):
    """Create visualization based on user selections"""
    
    if dashboard_config.current_data is None:
        return None
    
    if not x_axis or not y_axis:
        return None
    
    try:
        df = dashboard_config.current_data.copy()
        
        # Handle date columns - convert string dates to datetime
        if x_axis in df.columns:
            if x_axis.lower() in ['date', 'time', 'timestamp'] or 'date' in x_axis.lower():
                if not pd.api.types.is_datetime64_any_dtype(df[x_axis]):
                    try:
                        df[x_axis] = pd.to_datetime(df[x_axis], errors='coerce')
                        print(f"✅ Converted {x_axis} to datetime for standard visualization")
                    except Exception as e:
                        print(f"⚠️ Could not convert {x_axis} to datetime: {e}")
        
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

def create_vizro_enhanced_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg, correlation_window=0):
    """Create enhanced visualization using Vizro capabilities"""
    
    if not VIZRO_AVAILABLE:
        # Fallback to standard visualization
        return create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)
    
    if dashboard_config.current_data is None:
        return None
    
    if not x_axis or not y_axis:
        return None
    
    try:
        df = dashboard_config.current_data.copy()
        
        # Convert date columns
        df = convert_date_columns(df)
        
        # For scatter plots and other charts, if x_axis is a date column, ensure it's properly handled
        if x_axis in df.columns:
            if x_axis.lower() in ['date', 'time', 'timestamp'] or 'date' in x_axis.lower():
                if not pd.api.types.is_datetime64_any_dtype(df[x_axis]):
                    try:
                        df[x_axis] = pd.to_datetime(df[x_axis], errors='coerce')
                        print(f"✅ Converted x-axis {x_axis} to datetime for visualization")
                    except Exception as e:
                        print(f"⚠️ Could not convert x-axis {x_axis} to datetime: {e}")
        
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
            return create_enhanced_scatter_plot(df, x_axis, y_axis, color_var, chart_title_suffix, template)
        elif chart_type == 'Statistical Box Plot':
            return create_statistical_box_plot(df, x_axis, y_axis, color_var, chart_title_suffix, template)
        elif chart_type == 'Correlation Heatmap':
            return create_correlation_heatmap(df, template, window_size=correlation_window)
        elif chart_type == 'Distribution Analysis':
            return create_distribution_analysis(df, x_axis, y_axis, chart_title_suffix, template)
        elif chart_type == 'Time Series Analysis':
            return create_time_series_analysis(df, x_axis, y_axis, color_var, chart_title_suffix, template)
        elif chart_type == 'Advanced Bar Chart':
            return create_advanced_bar_chart(df, x_axis, y_axis, color_var, chart_title_suffix, template, y_axis_agg)
        else:
            # Fallback to standard visualization
            return create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)
        
    except Exception as e:
        print(f"Vizro visualization error: {e}")
        # Fallback to standard visualization
        return create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg)

def create_enhanced_scatter_plot(df, x_axis, y_axis, color_var, chart_title_suffix, template):
    """Create enhanced scatter plot with datetime handling"""
    # Check if x_axis is datetime - adjust visualization accordingly
    if pd.api.types.is_datetime64_any_dtype(df[x_axis]):
        # For datetime x-axis, create time series scatter plot without marginal histograms
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var,
                       title=f'Enhanced {chart_title_suffix} vs {x_axis}',
                       template=template,
                       hover_data=[col for col in df.columns if col in [x_axis, y_axis, color_var]])
        
        # Add trend line for datetime series if y is numeric
        if df[y_axis].dtype in ['int64', 'float64']:
            # Convert datetime to numeric for trend calculation
            df_numeric = df.copy()
            df_numeric['x_numeric'] = pd.to_numeric(df_numeric[x_axis])
            
            # Calculate trend line
            X = df_numeric['x_numeric'].values.reshape(-1, 1)
            y = df_numeric[y_axis].values
            
            # Remove NaN values
            mask = ~(np.isnan(X.flatten()) | np.isnan(y))
            if mask.sum() > 1:  # Need at least 2 points
                X_clean = X[mask]
                y_clean = y[mask]
                
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                trend_y = model.predict(X)
                
                fig.add_scatter(x=df[x_axis], y=trend_y,
                              mode='lines', name='Trend Line',
                              line=dict(dash='dash', color='red'))
    else:
        # Advanced scatter plot with trend lines and marginal distributions for numeric x-axis
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var,
                       title=f'Enhanced {chart_title_suffix} vs {x_axis}',
                       template=template,
                       marginal_x="histogram", marginal_y="histogram",
                       trendline="ols",
                       hover_data=[col for col in df.columns if col in [x_axis, y_axis, color_var]])
    
    # Add correlation annotation (only for numeric variables)
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
    
    return apply_enhanced_layout(fig)

def create_statistical_box_plot(df, x_axis, y_axis, color_var, chart_title_suffix, template):
    """Create enhanced box plot with statistical annotations"""
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
    
    return apply_enhanced_layout(fig)

def create_correlation_heatmap(df, template, window_size=0):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        if window_size > 0:
            # When using rolling correlation, a multi-level index is returned.
            # We select the correlation matrix for the last window.
            corr_matrix = df[numeric_cols].rolling(window=window_size).corr().iloc[-len(numeric_cols):]
            title = f'Rolling Correlation Heatmap (Window Size: {window_size})'
        else:
            corr_matrix = df[numeric_cols].corr()
            title = 'Correlation Heatmap'

        fig = px.imshow(corr_matrix,
                        title=title,
                        template=template,
                        color_continuous_scale='RdBu_r',
                        aspect="auto")

        fig.update_traces(text=np.around(corr_matrix.values, decimals=2),
                          texttemplate="%{text}", textfont={"size": 10})

        return apply_enhanced_layout(fig)
    else:
        return None

def create_distribution_analysis(df, x_axis, y_axis, chart_title_suffix, template):
    """Create multi-panel distribution analysis"""
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
    elif pd.api.types.is_datetime64_any_dtype(df[x_axis]):
        # For datetime data, show time series histogram
        fig.add_trace(go.Histogram(x=df[x_axis], name=x_axis, nbinsx=20), row=1, col=2)
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
    return fig

def create_time_series_analysis(df, x_axis, y_axis, color_var, chart_title_suffix, template):
    """Create enhanced time series analysis"""
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
    
    return apply_enhanced_layout(fig)

def create_advanced_bar_chart(df, x_axis, y_axis, color_var, chart_title_suffix, template, y_axis_agg):
    """Create enhanced bar chart with error bars and annotations"""
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
    
    return apply_enhanced_layout(fig)

def apply_enhanced_layout(fig):
    """Apply enhanced layout for all Vizro charts"""
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
                                 'X: %{x}<br>' +
                                 'Y: %{y}<br>' +
                                 '<extra></extra>'
                )
    except Exception:
        # If hover template update fails, continue without it
        pass
    
    return fig

def create_data_insights_dashboard():
    """Create comprehensive data insights using Vizro capabilities"""
    
    if not VIZRO_AVAILABLE or dashboard_config.current_data is None:
        return "Vizro not available or no data loaded"
    
    try:
        df = dashboard_config.current_data.copy()
        
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
        
        # Add recommendations
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