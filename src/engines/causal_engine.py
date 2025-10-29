#!/usr/bin/env python3
"""
Causal Analysis Engine Module
Handles causal discovery, intervention analysis, and pathway analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from causalnex.structure.notears import from_pandas
from causalnex.structure import StructureModel
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
from causalnex.discretiser import Discretiser
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import gradio as gr
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core import dashboard_config
from core.config import CAUSAL_ANALYSIS_PARAMS

def perform_causal_analysis_with_status(hide_nonsignificant, min_correlation, theme, show_all_relationships):
    """Wrapper function to handle causal analysis with status updates"""
    import time
    
    try:
        # Yield initial status
        yield "🔍 Starting causal analysis...", None, None, "Initializing analysis..."
        time.sleep(0.1)
        
        # Run the actual analysis
        network_fig, table_html, summary = perform_causal_analysis(hide_nonsignificant, min_correlation, theme, show_all_relationships)
        
        # Yield final result
        yield summary, network_fig, table_html, "✅ Analysis complete!"
            
    except Exception as e:
        yield f"❌ Analysis failed: {str(e)}", None, None, "Analysis could not be completed."

def perform_causal_analysis(hide_nonsignificant, min_correlation, theme, show_all_relationships=False, progress=gr.Progress()):
    """Perform efficient causal analysis on the data with progress tracking"""
    
    try:
        progress(0.05, desc="🔍 Loading and validating data...")
        
        if dashboard_config.current_data is None:
            return None, None, "❌ No data loaded. Please upload a dataset first."
        
        # Get numeric data only
        df_numeric = dashboard_config.current_data.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            return None, None, "❌ No numeric columns found. Please ensure your data contains numeric variables for causal analysis."
        
        progress(0.1, desc="📊 Analyzing data characteristics...")
        
        # Smart variable selection for performance
        max_vars = CAUSAL_ANALYSIS_PARAMS['max_variables']
        if len(df_numeric.columns) > max_vars:
            # Calculate correlation with all other variables to select most connected ones
            corr_matrix = df_numeric.corr().abs()
            # Sum of correlations for each variable (excluding self-correlation)
            corr_sums = corr_matrix.sum() - 1  # Subtract 1 to exclude self-correlation
            top_vars = corr_sums.nlargest(max_vars).index.tolist()
            df_numeric = df_numeric[top_vars]
            print(f"📊 Selected top {max_vars} most correlated variables for analysis")
        
        progress(0.15, desc="🎯 Selecting optimal variables...")
        
        # Smart sampling for large datasets
        max_samples = CAUSAL_ANALYSIS_PARAMS['max_samples']
        if len(df_numeric) > max_samples:
            df_numeric = df_numeric.sample(n=max_samples, random_state=42)
            print(f"📊 Sampled {max_samples} rows for efficient analysis")
        
        progress(0.2, desc="📈 Computing correlation matrix...")
        
        # Remove columns with no variation
        df_numeric = df_numeric.loc[:, df_numeric.std() > 1e-10]
        
        if df_numeric.empty:
            return None, None, "❌ No variables with sufficient variation found. Please check your data quality."
        
        # Standardize the data for better NOTEARS performance
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index
        )
        
        progress(0.3, desc="🧠 Building causal structure (NOTEARS)...")
        
        # Build causal structure using NOTEARS with optimized parameters
        sm = from_pandas(
            df_scaled, 
            max_iter=CAUSAL_ANALYSIS_PARAMS['max_iter'],
            h_tol=CAUSAL_ANALYSIS_PARAMS['h_tol'],
            w_threshold=CAUSAL_ANALYSIS_PARAMS['w_threshold']
        )
        
        progress(0.5, desc="🔗 Identifying causal relationships...")
        
        # Get edges and calculate statistics
        edges = sm.edges()
        
        if not edges:
            return None, None, "❌ No causal relationships found. Try adjusting the minimum correlation threshold or check data quality."
        
        progress(0.6, desc="📊 Calculating statistical significance...")
        
        # Calculate edge statistics
        edge_stats = []
        for source, target in edges:
            if source in df_numeric.columns and target in df_numeric.columns:
                # Calculate correlation and p-value
                corr, p_value = pearsonr(df_numeric[source], df_numeric[target])
                
                # Calculate R-squared
                r_squared = corr ** 2
                
                edge_stats.append({
                    'source': source,
                    'target': target,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'p_value': p_value,
                    'r_squared': r_squared,
                    'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
                    'strength': 'Strong' if abs(corr) >= 0.7 else 'Moderate' if abs(corr) >= 0.3 else 'Weak'
                })
        
        progress(0.7, desc="🎨 Creating network visualization...")
        
        # Create results DataFrame
        results_df = pd.DataFrame(edge_stats)
        
        if results_df.empty:
            return None, None, "❌ No valid relationships found after statistical analysis."
        
        # Filter results based on user preferences
        if hide_nonsignificant:
            results_df = results_df[results_df['p_value'] < 0.05]
        
        if min_correlation > 0:
            results_df = results_df[results_df['abs_correlation'] >= min_correlation]
        
        if results_df.empty:
            return None, None, f"❌ No relationships found meeting criteria (p < 0.05, |r| >= {min_correlation}). Try lowering the minimum correlation threshold."
        
        progress(0.8, desc="📋 Generating results table...")
        
        # Sort by absolute correlation (strongest first)
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        
        # Create network plot
        network_fig = create_network_plot(sm, edge_stats, theme, show_all_relationships)
        
        progress(0.9, desc="🔍 Computing edge statistics...")
        
        # Create advanced results table
        table_html = create_advanced_causal_table(results_df, edge_stats)
        
        progress(0.95, desc="📊 Finalizing analysis...")
        
        # Create summary
        total_relationships = len(results_df)
        significant_relationships = len(results_df[results_df['p_value'] < 0.05])
        strong_relationships = len(results_df[results_df['abs_correlation'] >= 0.7])
        
        summary = f"""
        ## 🔍 Causal Analysis Results
        
        **📊 Dataset:** {len(df_numeric)} rows × {len(df_numeric.columns)} variables  
        **🔗 Total Relationships:** {total_relationships}  
        **✅ Significant (p < 0.05):** {significant_relationships}  
        **💪 Strong (|r| ≥ 0.7):** {strong_relationships}  
        
        ### 🏆 Top Relationships:
        """
        
        # Add top 3 relationships
        for i, row in results_df.head(3).iterrows():
            direction = "→" if row['correlation'] > 0 else "⟷"
            summary += f"- **{row['source']} {direction} {row['target']}**: r = {row['correlation']:.3f} (p = {row['p_value']:.3f})\n"
        
        # Store results globally
        dashboard_config.causal_results = {
            'structure_model': sm,
            'results_df': results_df,
            'edge_stats': edge_stats,
            'summary_stats': {
                'total_relationships': total_relationships,
                'significant_relationships': significant_relationships,
                'strong_relationships': strong_relationships
            }
        }
        
        progress(1.0, desc="✅ Analysis complete!")
        
        return network_fig, table_html, summary
        
    except Exception as e:
        error_msg = f"❌ Causal analysis failed: {str(e)}\n\n**Common solutions:**\n• Ensure data has numeric variables\n• Check for sufficient data variation\n• Try lowering correlation threshold\n• Verify data quality (no excessive missing values)"
        return None, None, error_msg

def create_network_plot(sm, edge_stats, theme, show_all_relationships=False):
    """Create network visualization using Plotly with proper hover interactions"""
    try:
        # Convert to NetworkX graph
        G = nx.DiGraph()
        
        # Add edges with weights
        edge_dict = {}
        for stat in edge_stats:
            source = stat['source']
            target = stat['target']
            correlation = stat['correlation']
            p_value = stat['p_value']
            
            # Filter based on significance if not showing all
            if not show_all_relationships and p_value >= 0.05:
                continue
                
            G.add_edge(source, target, weight=abs(correlation), correlation=correlation, p_value=p_value)
            edge_dict[(source, target)] = stat
        
        if len(G.edges()) == 0:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No significant relationships found.<br>Try lowering the significance threshold or minimum correlation.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='gray')
            )
            fig.update_layout(
                title="Causal Network (No Relationships Found)",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=600
            )
            return fig
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Calculate node statistics
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            node_info.append(f"Variable: {node}<br>Incoming: {in_degree}<br>Outgoing: {out_degree}")
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=node_info,
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Variables'
        )
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            source, target = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge statistics
            correlation = G[source][target]['correlation']
            p_value = G[source][target]['p_value']
            edge_info.append(f"{source} → {target}<br>Correlation: {correlation:.3f}<br>P-value: {p_value:.3f}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            name='Relationships'
        )
        
        # Create arrows for directed edges
        arrow_traces = []
        for edge in G.edges():
            source, target = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Calculate arrow position (80% along the edge)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)
            
            # Calculate arrow direction
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize direction
                dx /= length
                dy /= length
                
                # Create arrow
                arrow_trace = go.Scatter(
                    x=[arrow_x],
                    y=[arrow_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='red',
                        angle=np.degrees(np.arctan2(dy, dx))
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                arrow_traces.append(arrow_trace)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace] + arrow_traces)
        
        # Update layout
        template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
        
        fig.update_layout(
            title=f"Causal Network ({len(G.edges())} relationships)",
            title_font_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Arrows show causal direction. Hover over nodes and edges for details.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=template,
            height=600
        )
        
        return fig
        
    except Exception as e:
        # Return error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating network plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red')
        )
        return fig

def create_ultra_robust_split_points(series):
    """Create guaranteed monotonically increasing split points with extensive validation"""
    # Remove NaN and infinite values
    clean_series = series.dropna()
    clean_series = clean_series[np.isfinite(clean_series)]
    
    if len(clean_series) == 0:
        print(f"⚠️ No valid data points for split points, using default [0.0, 1.0]")
        return [0.0, 1.0]
    
    # Convert to float and get basic stats
    try:
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        mean_val = float(clean_series.mean())
        std_val = float(clean_series.std())
    except (ValueError, TypeError) as e:
        print(f"⚠️ Error computing statistics: {e}, using default split points")
        return [0.0, 1.0]
    
    # Check for invalid values
    if not all(np.isfinite([min_val, max_val, mean_val, std_val])):
        print(f"⚠️ Invalid statistics detected, using default split points")
        return [0.0, 1.0]
    
    # If there's no variation, create artificial split points
    range_val = max_val - min_val
    if range_val <= 1e-12:  # Essentially no variation
        print(f"⚠️ No variation detected (range={range_val:.2e}), creating artificial splits")
        if abs(min_val) < 1e-6:  # Value is near zero
            return [-1.0, 1.0]
        else:
            margin = max(abs(min_val) * 0.1, 1.0)
            return [min_val - margin, min_val + margin]
    
    # Calculate minimum separation (at least 1% of range or 1e-6)
    min_separation = max(range_val * 0.01, 1e-6)
    
    # Strategy 1: Try quantile-based split points
    try:
        unique_values = np.unique(clean_series)
        if len(unique_values) >= 3:
            q33 = float(np.percentile(unique_values, 33.33))
            q67 = float(np.percentile(unique_values, 66.67))
            
            if q67 - q33 >= min_separation:
                splits = [q33, q67]
                print(f"✅ Using quantile splits: {splits}")
                return splits
    except Exception as e:
        print(f"⚠️ Quantile method failed: {e}")
    
    # Strategy 2: Try mean ± 0.5 * std
    try:
        if std_val > min_separation / 2:
            split1 = mean_val - 0.5 * std_val
            split2 = mean_val + 0.5 * std_val
            
            # Ensure splits are within data range
            split1 = max(split1, min_val + min_separation)
            split2 = min(split2, max_val - min_separation)
            
            if split2 - split1 >= min_separation:
                splits = [float(split1), float(split2)]
                print(f"✅ Using mean±std splits: {splits}")
                return splits
    except Exception as e:
        print(f"⚠️ Mean±std method failed: {e}")
    
    # Strategy 3: Simple range-based splits
    try:
        split1 = min_val + range_val * 0.4
        split2 = min_val + range_val * 0.6
        
        # Ensure minimum separation
        if split2 - split1 < min_separation:
            mid_point = (min_val + max_val) / 2
            split1 = mid_point - min_separation / 2
            split2 = mid_point + min_separation / 2
        
        splits = [float(split1), float(split2)]
        print(f"✅ Using range-based splits: {splits}")
        return splits
        
    except Exception as e:
        print(f"⚠️ Range-based method failed: {e}")
    
    # Final fallback: Use data range with padding
    try:
        padding = max(range_val * 0.1, min_separation)
        split1 = min_val + padding
        split2 = max_val - padding
        
        if split2 <= split1:
            # If still not enough separation, use midpoint approach
            mid_point = (min_val + max_val) / 2
            split1 = mid_point - min_separation / 2
            split2 = mid_point + min_separation / 2
        
        splits = [float(split1), float(split2)]
        
        # Final validation - ensure strictly increasing
        if splits[1] <= splits[0]:
            splits[1] = splits[0] + min_separation
        
        print(f"✅ Using fallback splits: {splits}")
        return splits
        
    except Exception as e:
        print(f"❌ All methods failed: {e}, using emergency fallback")
        return [0.0, 1.0]

def perform_causal_intervention_analysis(target_var, intervention_var, intervention_value, progress=gr.Progress()):
    """Perform causal intervention analysis (do-calculus)"""
    
    try:
        progress(0.1, desc="🔬 Preparing intervention analysis...")
        
        if dashboard_config.current_data is None:
            return "❌ No data loaded", "Please upload data first"
        
        # Get numeric data
        df_numeric = dashboard_config.current_data.select_dtypes(include=[np.number])
        if df_numeric.empty:
            return "❌ No numeric data found", "Please ensure your data contains numeric variables"
        
        if target_var not in df_numeric.columns or intervention_var not in df_numeric.columns:
            return "❌ Variables not found", "Selected variables not found in data"
        
        progress(0.4, desc="🔍 Validating data quality...")
        
        # Check for sufficient variation in key variables
        target_variation = df_numeric[target_var].std()
        intervention_variation = df_numeric[intervention_var].std()
        
        if target_variation < 1e-10:
            return f"""
            ❌ Intervention analysis failed: Insufficient variation in target variable
            
            **Problem:** The target variable '{target_var}' has no variation (std = {target_variation:.2e})
            
            **Solutions:**
            • Choose a different target variable with more diverse values
            • Check if the data contains only constant values
            • Ensure the variable represents a meaningful outcome
            """, "Target variable has insufficient variation"
        
        if intervention_variation < 1e-10:
            return f"""
            ❌ Intervention analysis failed: Insufficient variation in intervention variable
            
            **Problem:** The intervention variable '{intervention_var}' has no variation (std = {intervention_variation:.2e})
            
            **Solutions:**
            • Choose a different intervention variable with more diverse values
            • Check if the data contains only constant values
            • Ensure the variable can be meaningfully changed
            """, "Intervention variable has insufficient variation"
        
        progress(0.3, desc="🏗️ Building causal structure...")
        
        # Build causal structure using NOTEARS
        sm = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.3)
        
        progress(0.5, desc="🧠 Creating Bayesian Network...")
        
        # Create split points for each column with extensive validation
        split_points = {}
        failed_columns = []
        
        for col in df_numeric.columns:
            try:
                # Get column data and check quality
                col_data = df_numeric[col]
                unique_count = col_data.nunique()
                
                print(f"📊 Processing {col}: {len(col_data)} values, {unique_count} unique")
                
                if unique_count < 2:
                    print(f"⚠️ Column {col} has insufficient variation ({unique_count} unique values)")
                    failed_columns.append(col)
                    continue
                
                splits = create_ultra_robust_split_points(col_data)
                
                # Validate splits
                if len(splits) != 2:
                    raise ValueError(f"Expected 2 split points, got {len(splits)}")
                
                if splits[1] <= splits[0]:
                    raise ValueError(f"Split points not monotonic: {splits}")
                
                if not all(np.isfinite(splits)):
                    raise ValueError(f"Split points contain invalid values: {splits}")
                
                split_points[col] = splits
                print(f"✅ Created valid split points for {col}: {splits}")
                
            except Exception as e:
                print(f"❌ Failed to create split points for {col}: {e}")
                failed_columns.append(col)
        
        # Remove failed columns from analysis
        if failed_columns:
            print(f"⚠️ Removing {len(failed_columns)} columns with discretization issues: {failed_columns}")
            df_numeric = df_numeric.drop(columns=failed_columns)
            
            # Check if we still have enough variables
            if len(df_numeric.columns) < 2:
                return f"""
                ❌ Intervention analysis failed: Insufficient valid variables
                
                **Problem:** After removing columns with discretization issues, only {len(df_numeric.columns)} variables remain.
                
                **Failed columns:** {', '.join(failed_columns)}
                
                **Solutions:**
                • Use a dataset with more diverse numeric variables
                • Check data quality (remove constant or near-constant columns)
                • Ensure variables have at least 10+ unique values
                • Try with different variable combinations
                """, "Insufficient variables for analysis"
            
            # Check if target/intervention variables are still available
            if target_var not in df_numeric.columns:
                return f"""
                ❌ Intervention analysis failed: Target variable removed
                
                **Problem:** Target variable '{target_var}' was removed due to discretization issues.
                
                **Solutions:**
                • Choose a different target variable with more variation
                • Check if '{target_var}' has sufficient unique values
                • Ensure the variable represents a meaningful outcome
                """, "Target variable unavailable"
            
            if intervention_var not in df_numeric.columns:
                return f"""
                ❌ Intervention analysis failed: Intervention variable removed
                
                **Problem:** Intervention variable '{intervention_var}' was removed due to discretization issues.
                
                **Solutions:**
                • Choose a different intervention variable with more variation
                • Check if '{intervention_var}' has sufficient unique values
                • Ensure the variable can be meaningfully changed
                """, "Intervention variable unavailable"
        
        # Final validation and cleanup of all split points before creating discretizer
        print(f"🔍 Final validation of split points for {len(split_points)} variables...")
        cleaned_split_points = {}
        
        for col, splits in split_points.items():
            # Round to avoid floating point precision issues
            rounded_splits = [round(float(s), 10) for s in splits]
            
            # Ensure strict monotonic increasing with minimum separation
            if len(rounded_splits) != 2:
                print(f"❌ Invalid number of splits for {col}: {len(rounded_splits)}")
                continue
                
            if not all(np.isfinite(rounded_splits)):
                print(f"❌ Non-finite splits for {col}: {rounded_splits}")
                continue
                
            # Ensure monotonic with minimum separation
            min_separation = 1e-8
            if rounded_splits[1] <= rounded_splits[0]:
                print(f"⚠️ Non-monotonic splits for {col}: {rounded_splits}, fixing...")
                rounded_splits[1] = rounded_splits[0] + min_separation
            elif rounded_splits[1] - rounded_splits[0] < min_separation:
                print(f"⚠️ Insufficient separation for {col}: {rounded_splits}, fixing...")
                rounded_splits[1] = rounded_splits[0] + min_separation
            
            cleaned_split_points[col] = rounded_splits
            print(f"✅ Validated splits for {col}: {rounded_splits} (diff: {rounded_splits[1] - rounded_splits[0]:.2e})")
        
        # Update split_points with cleaned version
        split_points = cleaned_split_points
        
        # Skip CausalNex discretizer entirely - use manual discretization only
        print(f"🏗️ Using manual discretization (bypassing CausalNex discretizer issues)...")
        
        # Manual discretization using pandas cut - this always works
        df_discretised = df_numeric.copy()
        discretization_info = {}
        
        for col in df_numeric.columns:
            try:
                # Calculate quantile thresholds
                q33 = df_numeric[col].quantile(0.33)
                q67 = df_numeric[col].quantile(0.67)
                
                # Store thresholds for intervention value discretization
                discretization_info[col] = {'q33': q33, 'q67': q67}
                
                # Apply discretization
                df_discretised[col] = pd.cut(
                    df_numeric[col], 
                    bins=[-np.inf, q33, q67, np.inf], 
                    labels=['low', 'medium', 'high']
                )
                
                print(f"✅ Discretized {col}: low ≤ {q33:.3f}, medium ≤ {q67:.3f}, high > {q67:.3f}")
                
            except Exception as e:
                return f"""
                ❌ Intervention analysis failed: Manual discretization error
                
                **Problem:** Could not discretize variable '{col}': {str(e)}
                
                **Solutions:**
                • Check that '{col}' contains valid numeric values
                • Ensure the variable has sufficient variation
                • Remove any infinite or extremely large values
                """, f"Manual discretization error for {col}: {str(e)}"
        
        print(f"✅ Manual discretization completed successfully for {len(df_numeric.columns)} variables")
        
        # Apply discretization with error handling
        try:
            print(f"🔄 Attempting to discretize data with shape: {df_numeric.shape}")
            print(f"📊 Data types: {df_numeric.dtypes.to_dict()}")
            print(f"📊 Data sample:\n{df_numeric.head()}")
            
            # Try to fit and transform in separate steps for better error handling
            discretiser = discretiser.fit(df_numeric)
            df_discretised = discretiser.transform(df_numeric)
            print(f"✅ Successfully discretized data: {df_discretised.shape}")
            
        except Exception as e:
            print(f"❌ Discretization failed with error: {str(e)}")
            
            # Try alternative approach: manual discretization
            try:
                print("🔄 Trying manual discretization as fallback...")
                df_discretised = df_numeric.copy()
                
                for col in df_numeric.columns:
                    # Simple quantile-based discretization
                    q33 = df_numeric[col].quantile(0.33)
                    q67 = df_numeric[col].quantile(0.67)
                    
                    df_discretised[col] = pd.cut(
                        df_numeric[col], 
                        bins=[-np.inf, q33, q67, np.inf], 
                        labels=['low', 'medium', 'high']
                    )
                
                print(f"✅ Manual discretization successful: {df_discretised.shape}")
                print(f"📊 Discretized sample:\n{df_discretised.head()}")
                
            except Exception as e2:
                return f"""
                ❌ Intervention analysis failed: Data transformation error
                
                **Problem:** Both automatic and manual discretization failed
                
                **Primary error:** {str(e)}
                **Fallback error:** {str(e2)}
                
                **Solutions:**
                • Check that your data contains valid numeric values
                • Remove any infinite or extremely large values
                • Ensure variables have reasonable ranges
                • Try with a smaller subset of variables
                • Consider using different variable combinations
                
                **Data info:**
                • Target variable range: {df_numeric[target_var].min():.3f} to {df_numeric[target_var].max():.3f}
                • Intervention variable range: {df_numeric[intervention_var].min():.3f} to {df_numeric[intervention_var].max():.3f}
                """, f"Data transformation error: {str(e)}"
        
        # Create Bayesian Network
        bn = BayesianNetwork(sm)
        bn = bn.fit_node_states(df_discretised)
        bn = bn.fit_cpds(df_discretised, method="BayesianEstimator", bayes_prior="K2")
        
        progress(0.7, desc="🎯 Performing intervention...")
        
        # Create inference engine
        ie = InferenceEngine(bn)
        
        # Validate and discretize intervention value
        intervention_min = df_numeric[intervention_var].min()
        intervention_max = df_numeric[intervention_var].max()
        
        # Check if intervention value is within reasonable range
        if intervention_value < intervention_min * 0.5 or intervention_value > intervention_max * 2.0:
            return f"""
            ❌ Intervention analysis failed: Intervention value out of range
            
            **Problem:** Intervention value {intervention_value} is outside reasonable range
            
            **Data range for {intervention_var}:**
            • Minimum: {intervention_min:.3f}
            • Maximum: {intervention_max:.3f}
            • Suggested range: {intervention_min:.3f} to {intervention_max:.3f}
            
            **Solutions:**
            • Use an intervention value within the data range
            • Try values between {intervention_min:.1f} and {intervention_max:.1f}
            • Consider the practical meaning of your intervention
            """, "Intervention value out of range"
        
        try:
            # Use the same manual discretization approach for intervention value
            q33 = df_numeric[intervention_var].quantile(0.33)
            q67 = df_numeric[intervention_var].quantile(0.67)
            
            if intervention_value <= q33:
                intervention_state = 'low'
            elif intervention_value <= q67:
                intervention_state = 'medium'
            else:
                intervention_state = 'high'
                
            print(f"✅ Intervention value {intervention_value} discretized to state: {intervention_state}")
            print(f"📊 Discretization thresholds: low ≤ {q33:.3f}, medium ≤ {q67:.3f}, high > {q67:.3f}")
            
        except Exception as e:
            return f"""
            ❌ Intervention analysis failed: Could not discretize intervention value
            
            **Problem:** Error discretizing intervention value {intervention_value}: {str(e)}
            
            **Solutions:**
            • Try an intervention value closer to the data mean: {df_numeric[intervention_var].mean():.3f}
            • Ensure the value is a valid number
            • Check that the value makes sense for your variable
            """, f"Intervention discretization error: {str(e)}"
        
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

def export_results():
    """Export analysis results"""
    
    if dashboard_config.causal_results is None:
        return "❌ No analysis results to export. Please run causal analysis first."
    
    try:
        # Create export data
        results_df = dashboard_config.causal_results['results_df']
        
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"causal_analysis_results_{timestamp}.csv"
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        
        # Create summary
        summary_stats = dashboard_config.causal_results['summary_stats']
        
        export_summary = f"""
        ✅ **Results exported successfully!**
        
        **📁 File:** {filename}  
        **📊 Total relationships:** {summary_stats['total_relationships']}  
        **✅ Significant relationships:** {summary_stats['significant_relationships']}  
        **💪 Strong relationships:** {summary_stats['strong_relationships']}  
        
        **📋 Exported columns:**
        - Source variable
        - Target variable  
        - Correlation coefficient
        - P-value
        - R-squared
        - Significance level
        - Relationship strength
        """
        
        return export_summary
        
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

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
        correlation_val = float(row['correlation'])
        pvalue_val = float(row['p_value'])
        r2_val = float(row['r_squared'])
        
        # Add CSS classes for styling based on significance and strength
        row_class = ""
        if row['significance'] == 'Significant':
            row_class += "significant-row "
        if abs(correlation_val) >= 0.7:
            row_class += "strong-correlation "
        elif abs(correlation_val) >= 0.3:
            row_class += "moderate-correlation "
        else:
            row_class += "weak-correlation "
        
        causal_rows += f'''
        <tr class="{row_class}" 
            data-source="{row['source'].lower()}"
            data-target="{row['target'].lower()}"
            data-correlation="{correlation_val}"
            data-pvalue="{pvalue_val}"
            data-r2="{r2_val}"
            data-significant="{row['significance'].lower()}">
            <td>{row['source']}</td>
            <td>{row['target']}</td>
            <td class="numeric-cell">{correlation_val:.4f}</td>
            <td class="numeric-cell">{pvalue_val:.4f}</td>
            <td class="numeric-cell">{r2_val:.4f}</td>
            <td class="significance-cell">
                <span class="badge {'badge-success' if row['significance'] == 'Significant' else 'badge-secondary'}">
                    {row['significance']}
                </span>
            </td>
        </tr>
        '''
    
    # Enhanced JavaScript for sorting, filtering, and exporting
    advanced_script = """
    <script>
    let sortState = [];
    let originalRows = [];
    
    function initializeCausalTable() {
        const table = document.getElementById('causal-results');
        if (table && table.querySelector('tbody')) {
            originalRows = Array.from(table.querySelector('tbody').querySelectorAll('tr'));
            updateFilterStatus();
        } else {
            setTimeout(initializeCausalTable, 100);
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeCausalTable);
    } else {
        initializeCausalTable();
    }
    
    function sortCausalTable(columnIndex, tableId, event) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        
        const visibleRows = Array.from(tbody.querySelectorAll('tr:not([style*="display: none"])'));
        const header = table.querySelectorAll('th')[columnIndex];
        
        if (!header) return;
        
        const currentSort = header.getAttribute('data-sort') || 'none';
        
        if (!event || !event.ctrlKey) {
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
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
        
        const existingIndex = sortState.findIndex(s => s.column === columnIndex);
        if (existingIndex >= 0) {
            sortState[existingIndex].direction = newSort;
        } else {
            sortState.push({column: columnIndex, direction: newSort});
        }
        
        const sortedRows = visibleRows.sort((a, b) => {
            for (let sort of sortState) {
                const aVal = a.cells[sort.column].textContent.trim();
                const bVal = b.cells[sort.column].textContent.trim();
                
                let comparison = 0;
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
        
        tbody.innerHTML = '';
        sortedRows.forEach(row => tbody.appendChild(row));
        
        const hiddenRows = originalRows.filter(row => 
            !sortedRows.includes(row) && 
            (row.style.display === 'none' || row.style.display.includes('none'))
        );
        hiddenRows.forEach(row => tbody.appendChild(row));
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
            
            if (searchTerm) {
                const source = row.dataset.source || '';
                const target = row.dataset.target || '';
                if (!source.includes(searchTerm) && !target.includes(searchTerm)) {
                    show = false;
                }
            }
            
            if (significanceFilter !== 'all') {
                const significant = row.dataset.significant;
                if (significanceFilter === 'significant' && significant !== 'significant') show = false;
                if (significanceFilter === 'non-significant' && significant !== 'not significant') show = false;
            }
            
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
        
        sortState = [];
        const table = document.getElementById('causal-results');
        if (table) {
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
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
        
        let csvContent = headers.join(',') + '\\n';
        csvContent += data.map(row => 
            row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(',')
        ).join('\\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'causal_analysis_results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
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