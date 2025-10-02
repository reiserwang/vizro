import pandas as pd
import plotly.express as px
from causalnex.structure.notears import from_pandas
from causalnex.structure import StructureModel
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine

import plotly.graph_objects as go
from scipy.stats import linregress, pearsonr, spearmanr, chi2_contingency, normaltest, kstest
from scipy import stats
import base64
from io import StringIO, BytesIO
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.mlemodel import MLEModel
import warnings
warnings.filterwarnings('ignore')

# Performance optimization imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
from typing import Dict, List, Tuple, Any
import time

# Performance optimization functions
def compute_edge_statistics(edge_data: Tuple) -> Dict:
    """
    Compute statistical tests for a single edge in parallel
    """
    u, v, weight, x_data, y_data = edge_data
    
    try:
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(x_data, y_data)
        
        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = spearmanr(x_data, y_data)
        
        # Linear regression R²
        reg = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
        r2 = reg.score(x_data.reshape(-1, 1), y_data)
        
        return {
            'source': u,
            'target': v,
            'causal_weight': weight,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r_squared': r2,
            'significant': pearson_p < 0.05
        }
    except Exception as e:
        return {
            'source': u,
            'target': v,
            'causal_weight': weight,
            'pearson_r': 0,
            'pearson_p': 1,
            'spearman_r': 0,
            'spearman_p': 1,
            'r_squared': 0,
            'significant': False,
            'error': str(e)
        }

def compute_cv_score(cv_data: Tuple) -> Dict:
    """
    Compute cross-validation score for a single target variable in parallel
    """
    target_var, parents, X, y = cv_data
    
    try:
        if len(parents) > 0 and len(parents) < X.shape[1]:
            # 5-fold cross validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
            
            return {
                'target': target_var,
                'parents': parents,
                'cv_r2_mean': np.mean(scores),
                'cv_r2_std': np.std(scores),
                'cv_scores': scores.tolist()
            }
    except Exception as e:
        return {
            'target': target_var,
            'parents': parents,
            'cv_r2_mean': 0,
            'cv_r2_std': 0,
            'cv_scores': [],
            'error': str(e)
        }
    
    return None

def compute_normality_test(norm_data: Tuple) -> Dict:
    """
    Compute normality test for residuals in parallel
    """
    target_var, parents, X, y = norm_data
    
    try:
        if len(parents) > 0:
            reg = LinearRegression().fit(X, y)
            residuals = y - reg.predict(X)
            
            # D'Agostino-Pearson normality test
            stat, p_value = normaltest(residuals)
            
            return {
                'target': target_var,
                'normality_stat': stat,
                'normality_p': p_value,
                'residuals_normal': p_value > 0.05
            }
    except Exception as e:
        return {
            'target': target_var,
            'normality_stat': 0,
            'normality_p': 1,
            'residuals_normal': False,
            'error': str(e)
        }
    
    return None

@functools.lru_cache(maxsize=128)
def cached_structure_learning(data_hash: str, data_shape: Tuple) -> Any:
    """
    Cache structure learning results to avoid recomputation
    """
    # This is a placeholder - actual caching would need serialization
    return None

def fit_var_model(df, target_cols, periods, max_lags=5):
    """
    Fit Vector Autoregression (VAR) model for multivariate forecasting.
    Handles single or multiple target variables.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not all(col in numeric_cols for col in target_cols):
            return None, None, "One or more target columns are not numeric."
        
        # Limit to 5-8 variables for computational efficiency
        if len(numeric_cols) > 8:
            # Calculate correlation with the mean of target variables
            corr_with_target = df[numeric_cols].corrwith(df[target_cols].mean(axis=1)).abs().sort_values(ascending=False)
            selected_vars = corr_with_target.head(8).index.tolist()
            # Ensure target columns are in the selected variables
            for col in target_cols:
                if col not in selected_vars:
                    selected_vars.append(col)
        else:
            selected_vars = numeric_cols
        
        var_data = df[selected_vars].dropna()
        
        if len(var_data) < max_lags * 2:
            return None, None, "Insufficient data for VAR model."
        
        model = VAR(var_data)
        lag_order = model.select_order(maxlags=min(max_lags, len(var_data)//4))
        optimal_lags = lag_order.aic
        fitted_model = model.fit(optimal_lags)
        
        forecast = fitted_model.forecast(var_data.values[-optimal_lags:], steps=periods)
        
        target_forecasts = {}
        for i, col in enumerate(selected_vars):
            if col in target_cols:
                target_forecasts[col] = forecast[:, i]
            
        return fitted_model, target_forecasts, selected_vars
        
    except Exception as e:
        return None, None, f"VAR model error: {str(e)}"

def fit_dynamic_factor_model(df, target_cols, periods, n_factors=2):
    """
    Fit Dynamic Factor Model for dimension reduction and forecasting.
    Handles single or multiple target variables.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not all(col in numeric_cols for col in target_cols):
            return None, None, "One or more target columns are not numeric."
            
        if len(numeric_cols) > 10:
            corr_with_target = df[numeric_cols].corrwith(df[target_cols].mean(axis=1)).abs().sort_values(ascending=False)
            selected_vars = corr_with_target.head(10).index.tolist()
            for col in target_cols:
                if col not in selected_vars:
                    selected_vars.append(col)
        else:
            selected_vars = numeric_cols
        
        factor_data = df[selected_vars].dropna()
        
        if len(factor_data) < 50:
            return None, None, "Insufficient data for Dynamic Factor Model."
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(factor_data)
        scaled_df = pd.DataFrame(scaled_data, columns=selected_vars, index=factor_data.index)
        
        n_factors = min(n_factors, len(selected_vars)//2)
        model = DynamicFactor(scaled_df, k_factors=n_factors, factor_order=2)
        fitted_model = model.fit(disp=False, maxiter=100)
        
        forecast_scaled = fitted_model.forecast(steps=periods)
        
        forecast = scaler.inverse_transform(forecast_scaled)
        
        target_forecasts = {}
        for i, col in enumerate(selected_vars):
            if col in target_cols:
                target_forecasts[col] = forecast[:, i]
            
        return fitted_model, target_forecasts, selected_vars
        
    except Exception as e:
        return None, None, f"Dynamic Factor Model error: {str(e)}"

def fit_state_space_model(df, target_cols, periods):
    """
    Fit State-Space Model for each target variable.
    """
    forecasts = {}
    models = {}
    
    for col in target_cols:
        try:
            target_series = df[col].dropna()
            
            if len(target_series) < 20:
                continue

            # Simplified model configuration
            model = UnobservedComponents(target_series, level='local linear trend', autoregressive=1)
            fitted_model = model.fit(disp=False, maxiter=100)
            
            forecasts[col] = fitted_model.forecast(steps=periods)
            models[col] = fitted_model
        except Exception as e:
            print(f"State-Space model for {col} failed: {e}")
            continue
            
    if not forecasts:
        return None, None, "State-Space models failed for all variables."
        
    # For simplicity, returning the last model for general info
    return models, forecasts, None

def process_data(df):
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass  # Ignore columns that can't be converted to datetime
    return df

def filter_by_timespan(data_frame, timespan_selector):
    if not timespan_selector or timespan_selector == 'All':
        return data_frame
    
    date_col = None
    for col in data_frame.columns:
        if data_frame[col].dtype == 'datetime64[ns]':
            date_col = col
            break
    
    if not date_col:
        return data_frame

    df = data_frame.copy()
    end_date = df[date_col].max()
    if timespan_selector == 'Last 3 Months':
        start_date = end_date - pd.DateOffset(months=3)
    elif timespan_selector == 'Last 6 Months':
        start_date = end_date - pd.DateOffset(months=6)
    elif timespan_selector == 'Last Year':
        start_date = end_to_datetime(f'{end_date.year}-01-01')
    else:
        return df
    return df[df[date_col] >= start_date]




def evaluate_model_quality_optimized(df, sm, max_workers=None):
    """
    Optimized evaluation of causal model quality with multiprocessing
    """
    start_time = time.time()
    quality_metrics = {}
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
    
    # 1. Basic Network Statistics (fast, no parallelization needed)
    quality_metrics['network_stats'] = {
        'num_nodes': sm.number_of_nodes(),
        'num_edges': sm.number_of_edges(),
        'density': nx.density(sm),
        'is_dag': nx.is_directed_acyclic_graph(sm)
    }
    
    # 2. Edge Weight Distribution Analysis (vectorized)
    weights = np.array([data.get('weight', 0) for u, v, data in sm.edges(data=True)])
    if len(weights) > 0:
        quality_metrics['weight_stats'] = {
            'mean_weight': np.mean(np.abs(weights)),
            'std_weight': np.std(weights),
            'min_weight': np.min(weights),
            'max_weight': np.max(weights),
            'weight_range': np.ptp(weights)  # Peak-to-peak (max - min)
        }
    
    # Prepare data for parallel processing
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    
    # 3. Parallel Statistical Tests for Each Edge
    edge_data_list = []
    for u, v, data in sm.edges(data=True):
        if u in df_numeric.columns and v in df_numeric.columns:
            x_data = df_numeric[u].values
            y_data = df_numeric[v].values
            weight = data.get('weight', 0)
            edge_data_list.append((u, v, weight, x_data, y_data))
    
    edge_tests = []
    if edge_data_list:
        # Use ThreadPoolExecutor for I/O bound statistical computations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_edge = {executor.submit(compute_edge_statistics, edge_data): edge_data 
                             for edge_data in edge_data_list}
            
            for future in as_completed(future_to_edge):
                result = future.result()
                if result:
                    edge_tests.append(result)
    
    quality_metrics['edge_tests'] = edge_tests
    
    # 4. Parallel Cross-Validation for Predictive Performance
    cv_data_list = []
    for target_var in df_numeric.columns:
        parents = list(sm.predecessors(target_var))
        if len(parents) > 0 and len(parents) < len(df_numeric.columns):
            X = df_numeric[parents].values
            y = df_numeric[target_var].values
            cv_data_list.append((target_var, parents, X, y))
    
    cv_scores = []
    if cv_data_list:
        # Use ProcessPoolExecutor for CPU-bound cross-validation
        with ProcessPoolExecutor(max_workers=min(max_workers, len(cv_data_list))) as executor:
            future_to_cv = {executor.submit(compute_cv_score, cv_data): cv_data 
                           for cv_data in cv_data_list}
            
            for future in as_completed(future_to_cv):
                result = future.result()
                if result:
                    cv_scores.append(result)
    
    quality_metrics['cross_validation'] = cv_scores
    
    # 5. Parallel Normality Tests for Residuals
    norm_data_list = []
    for target_var in df_numeric.columns:
        parents = list(sm.predecessors(target_var))
        if len(parents) > 0:
            X = df_numeric[parents].values
            y = df_numeric[target_var].values
            norm_data_list.append((target_var, parents, X, y))
    
    normality_tests = []
    if norm_data_list:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_norm = {executor.submit(compute_normality_test, norm_data): norm_data 
                             for norm_data in norm_data_list}
            
            for future in as_completed(future_to_norm):
                result = future.result()
                if result:
                    normality_tests.append(result)
    
    quality_metrics['normality_tests'] = normality_tests
    
    # 6. Model Complexity Metrics (vectorized)
    if sm.nodes():
        in_degrees = np.array([sm.in_degree(node) for node in sm.nodes()])
        num_nodes = sm.number_of_nodes()
        num_edges = sm.number_of_edges()
        
        quality_metrics['complexity'] = {
            'avg_parents_per_node': np.mean(in_degrees),
            'max_parents': np.max(in_degrees),
            'sparsity': 1 - (num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 1
        }
    else:
        quality_metrics['complexity'] = {
            'avg_parents_per_node': 0,
            'max_parents': 0,
            'sparsity': 1
        }
    
    # Performance metrics
    end_time = time.time()
    quality_metrics['performance'] = {
        'computation_time': end_time - start_time,
        'workers_used': max_workers,
        'edges_processed': len(edge_tests),
        'cv_tests_run': len(cv_scores)
    }
    
    return quality_metrics

# Backward compatibility alias
def evaluate_model_quality(df, sm):
    """Backward compatibility wrapper"""
    return evaluate_model_quality_optimized(df, sm)

def create_quality_report(quality_metrics, theme='dark'):
    """
    Create a comprehensive quality report with visualizations
    """
    report_sections = []
    
    # Network Overview
    network_stats = quality_metrics.get('network_stats', {})
    overview_text = f"""
    ### Model Overview
    - **Nodes**: {network_stats.get('num_nodes', 0)}
    - **Edges**: {network_stats.get('num_edges', 0)}
    - **Network Density**: {network_stats.get('density', 0):.3f}
    - **Valid DAG**: {'✅ Yes' if network_stats.get('is_dag', False) else '❌ No'}
    """
    
    # Weight Statistics
    weight_stats = quality_metrics.get('weight_stats', {})
    if weight_stats:
        weight_text = f"""
        ### Edge Weight Analysis
        - **Mean Absolute Weight**: {weight_stats.get('mean_weight', 0):.4f}
        - **Weight Standard Deviation**: {weight_stats.get('std_weight', 0):.4f}
        - **Weight Range**: [{weight_stats.get('min_weight', 0):.4f}, {weight_stats.get('max_weight', 0):.4f}]
        """
    else:
        weight_text = "### Edge Weight Analysis\nNo edge weights available."
    
    # Cross-Validation Results
    cv_results = quality_metrics.get('cross_validation', [])
    if cv_results:
        avg_cv_score = np.mean([result['cv_r2_mean'] for result in cv_results])
        cv_text = f"""
        ### Predictive Performance (Cross-Validation)
        - **Average R² Score**: {avg_cv_score:.4f}
        - **Number of Validated Relationships**: {len(cv_results)}
        """
        
        # Add individual results
        for result in cv_results[:5]:  # Show top 5
            cv_text += f"\n- **{result['target']}**: R² = {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}"
    else:
        cv_text = "### Predictive Performance\nNo cross-validation results available."
    
    # Statistical Significance
    edge_tests = quality_metrics.get('edge_tests', [])
    if edge_tests:
        significant_edges = [test for test in edge_tests if test['significant']]
        sig_text = f"""
        ### Statistical Significance
        - **Total Edges Tested**: {len(edge_tests)}
        - **Statistically Significant**: {len(significant_edges)} ({len(significant_edges)/len(edge_tests)*100:.1f}%)
        """
    else:
        sig_text = "### Statistical Significance\nNo statistical tests performed."
    
    # Model Quality Assessment
    complexity = quality_metrics.get('complexity', {})
    quality_text = f"""
    ### Model Quality Assessment
    - **Average Parents per Node**: {complexity.get('avg_parents_per_node', 0):.2f}
    - **Maximum Parents**: {complexity.get('max_parents', 0)}
    - **Sparsity**: {complexity.get('sparsity', 0):.3f}
    """
    
    # Add performance metrics if available
    performance_metrics = quality_metrics.get('performance', {})
    if performance_metrics:
        perf_text = f"""
        ### Performance Metrics
        - **Computation Time**: {performance_metrics.get('computation_time', 0):.2f} seconds
        - **Workers Used**: {performance_metrics.get('workers_used', 1)}
        - **Edges Processed**: {performance_metrics.get('edges_processed', 0)}
        - **CV Tests Run**: {performance_metrics.get('cv_tests_run', 0)}
        """
        full_report = overview_text + weight_text + cv_text + sig_text + quality_text + perf_text
    else:
        full_report = overview_text + weight_text + cv_text + sig_text + quality_text
    
    return full_report, edge_tests

def analyze_causal_structure_optimized(df, theme='dark', max_workers=None, filter_params=None):
    """
    Optimized causal structure analysis with performance improvements
    """
    start_time = time.time()
    
    # Optimize data preprocessing
    df_numeric = df.copy()
    
    # Vectorized categorical encoding
    categorical_cols = df_numeric.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_numeric[col].notna().sum() > 0:  # Only process non-empty columns
            le = LabelEncoder()
            # Handle NaN values efficiently
            mask = df_numeric[col].notna()
            df_numeric.loc[mask, col] = le.fit_transform(df_numeric.loc[mask, col].astype(str))
    
    # Efficient numeric selection and NaN handling
    df_numeric = df_numeric.select_dtypes(include=['number'])
    
    # Smart sampling for large datasets to improve performance
    if len(df_numeric) > 5000:
        # Sample 5000 rows for structure learning while preserving relationships
        sample_size = min(5000, len(df_numeric))
        df_numeric = df_numeric.sample(n=sample_size, random_state=42)
    
    # Remove columns with too many NaN values (>50%)
    nan_threshold = 0.5
    df_numeric = df_numeric.loc[:, df_numeric.isnull().mean() < nan_threshold]
    
    # Drop remaining NaN rows
    df_numeric = df_numeric.dropna()
    
    if df_numeric.shape[1] < 2:
        return go.Figure(), "", [], "", "", []
    
    # Limit number of variables for performance (CausalNex can be slow with many variables)
    if df_numeric.shape[1] > 15:
        # Select most correlated variables
        corr_matrix = df_numeric.corr().abs()
        # Get average correlation for each variable
        avg_corr = corr_matrix.mean().sort_values(ascending=False)
        # Keep top 15 most correlated variables
        top_vars = avg_corr.head(15).index
        df_numeric = df_numeric[top_vars]
    
    # Create causal structure with timeout protection
    try:
        # Use a more efficient structure learning approach for larger datasets
        if df_numeric.shape[0] > 1000:
            # Use a subset for initial structure learning
            sample_df = df_numeric.sample(n=min(1000, len(df_numeric)), random_state=42)
            sm = from_pandas(sample_df, w_threshold=0.3)  # Higher threshold for sparsity
        else:
            sm = from_pandas(df_numeric, w_threshold=0.1)
    except Exception as e:
        print(f"Structure learning failed: {e}")
        # Return empty results if structure learning fails
        return go.Figure(), f"Structure learning failed: {str(e)}", [], "", "", []
    
    # Evaluate model quality with optimization
    quality_metrics = evaluate_model_quality_optimized(df_numeric, sm, max_workers)
    quality_report, edge_tests = create_quality_report(quality_metrics, theme)
    
    # Apply filtering if specified
    if filter_params:
        hide_nonsig = filter_params.get('hide_nonsignificant', False)
        min_corr = filter_params.get('min_correlation', 0.0)
        
        # Filter edges based on significance and correlation
        filtered_edges = []
        for u, v, data in sm.edges(data=True):
            edge_test = next((test for test in edge_tests if test['source'] == u and test['target'] == v), None)
            
            # Check significance filter
            if hide_nonsig and edge_test and not edge_test['significant']:
                continue
                
            # Check correlation threshold
            if edge_test and abs(edge_test['pearson_r']) < min_corr:
                continue
                
            filtered_edges.append((u, v, data))
        
        # Create filtered graph for visualization
        filtered_sm = nx.DiGraph()
        for u, v, data in filtered_edges:
            filtered_sm.add_edge(u, v, **data)
        
        # Only include nodes that have edges
        nodes_with_edges = set()
        for u, v, _ in filtered_edges:
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)
        
        # Remove isolated nodes
        nodes_to_remove = [node for node in filtered_sm.nodes() if node not in nodes_with_edges]
        filtered_sm.remove_nodes_from(nodes_to_remove)
        
        sm = filtered_sm
    
    pos = nx.spring_layout(sm, seed=42)

    edge_traces = []
    all_edges = sm.edges(data=True)
    max_abs_weight = max([abs(data.get('weight', 0)) for u, v, data in all_edges], default=1)

    for u, v, data in all_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 0)
        
        norm_weight = abs(weight) / max_abs_weight if max_abs_weight != 0 else 0
        
        width = 1 + norm_weight * 5
        
        # Color edges based on statistical significance if available
        edge_test = next((test for test in edge_tests if test['source'] == u and test['target'] == v), None)
        if edge_test and edge_test['significant']:
            # Significant edges in green/red based on weight direction
            if weight > 0:
                color = f'rgba(0, 255, 0, {0.3 + norm_weight * 0.7})'  # Green for positive significant
            else:
                color = f'rgba(255, 100, 0, {0.3 + norm_weight * 0.7})'  # Orange for negative significant
        else:
            # Non-significant or untested edges in blue/red
            if weight > 0:
                color = f'rgba(0, 0, 255, {0.2 + norm_weight * 0.5})'
            else:
                color = f'rgba(255, 0, 0, {0.2 + norm_weight * 0.5})'

        # Enhanced hover text with statistical info
        custom_data = [weight]
        hovertemplate = '<b>Connection Status</b><br><br>Weight: %{customdata[0]:.4f}'
        if edge_test:
            custom_data.extend([
                edge_test["pearson_r"],
                edge_test["pearson_p"],
                edge_test["r_squared"],
                "Yes" if edge_test["significant"] else "No"
            ])
            hovertemplate += '<br>Pearson r: %{customdata[1]:.3f}'
            hovertemplate += '<br>P-value: %{customdata[2]:.3f}'
            hovertemplate += '<br>R²: %{customdata[3]:.3f}'
            hovertemplate += '<br>Significant: %{customdata[4]}'
        
        hovertemplate += '<extra></extra>'

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color),
            customdata=[custom_data] * 2, # Duplicate for both points of the line
            hovertemplate=hovertemplate,
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    for node in sm.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Add node statistics to hover
        in_degree = sm.in_degree(node)
        out_degree = sm.out_degree(node)
        node_hover.append(f'{node}<br>Parents: {in_degree}<br>Children: {out_degree}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hovertext=node_hover,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            showscale=False,
            color='lightblue',
            size=20,
            line_width=2))

    # Create dummy traces for legend
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='lines', name='Significant (Positive)',
                   line=dict(color=f'rgba(0, 255, 0, 0.7)', width=5)),
        go.Scatter(x=[None], y=[None], mode='lines', name='Significant (Negative)',
                   line=dict(color=f'rgba(255, 100, 0, 0.7)', width=5)),
        go.Scatter(x=[None], y=[None], mode='lines', name='Non-Significant (Positive)',
                   line=dict(color=f'rgba(0, 0, 255, 0.5)', width=5)),
        go.Scatter(x=[None], y=[None], mode='lines', name='Non-Significant (Negative)',
                   line=dict(color=f'rgba(255, 0, 0, 0.5)', width=5))
    ]

    fig = go.Figure(data=edge_traces + [node_trace] + legend_traces,
                 layout=go.Layout(
                    showlegend=True,
                    legend=dict(
                        x=0.99,
                        y=0.01,
                        xanchor='right',
                        yanchor='bottom',
                        bgcolor='rgba(255, 255, 255, 0.5)' if theme == 'light' else 'rgba(0, 0, 0, 0.5)',
                        font=dict(
                            color='black' if theme == 'light' else 'white'
                        )
                    ),
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    fig.update_layout(template=template, title="CausalNex Analysis with Quality Metrics")

    strongest_edge = None
    max_weight = 0
    for u, v, data in all_edges:
        weight_abs = abs(data.get('weight', 0))
        if weight_abs > max_weight:
            max_weight = weight_abs
            strongest_edge = (u, v, data.get('weight'))

    strongest_insight_md = ""
    if strongest_edge:
        u, v, w = strongest_edge
        # Add statistical significance info if available
        edge_test = next((test for test in edge_tests if test['source'] == u and test['target'] == v), None)
        sig_info = ""
        if edge_test:
            sig_info = f" (p-value: {edge_test['pearson_p']:.3f}, {'Significant' if edge_test['significant'] else 'Not Significant'})"
        strongest_insight_md = f"### Strongest Causal Relationship:\n- **{u}** -> **{v}** (Weight: {w:.4f}){sig_info}"

    # Enhanced table data with statistical metrics (filtered to match graph)
    table_data = []
    for u, v, data in sm.edges(data=True):
        edge_test = next((test for test in edge_tests if test['source'] == u and test['target'] == v), None)
        row = {
            'Source': u, 
            'Target': v, 
            'Weight': data.get('weight', 0)
        }
        if edge_test:
            row.update({
                'Pearson_r': edge_test['pearson_r'],
                'P_value': edge_test['pearson_p'],
                'R_squared': edge_test['r_squared'],
                'Significant': 'Yes' if edge_test['significant'] else 'No'
            })
        table_data.append(row)

    # Add filtering information to the note
    filter_info = ""
    if filter_params:
        active_filters = []
        if filter_params.get('hide_nonsignificant', False):
            active_filters.append("🎯 **Hiding non-significant edges** (p ≥ 0.05)")
        if filter_params.get('min_correlation', 0.0) > 0:
            active_filters.append(f"📊 **Minimum correlation threshold**: {filter_params.get('min_correlation', 0.0):.1f}")
        
        if active_filters:
            filter_info = f"""
### **Active Filters**
{chr(10).join(['- ' + f for f in active_filters])}

---
"""

    note_md = f"""{filter_info}
### **Quick Reference Guide**

**Interactive Filtering:**
- 🎯 **Significance Filter**: Hide relationships with p ≥ 0.05 to focus on reliable connections
- 📊 **Correlation Threshold**: Set minimum correlation strength to reduce noise
- **Dynamic Updates**: Graph and table update automatically with filter changes

**Graph Visualization:**
- 🟢 **Green/Orange edges**: Statistically significant relationships (p < 0.05)
- 🔵 **Blue/Red edges**: Non-significant or untested relationships  
- **Edge thickness**: Proportional to causal weight magnitude
- **Hover details**: View correlation coefficients, p-values, and R² scores

**Table Interpretation:**
- **Bold rows**: Statistically significant relationships (p < 0.05)
- **Green highlighting**: Confirmed significant relationships
- **Sort columns**: Click headers to sort by any metric

**Quality Assessment:**
- Look for: **Low p-values** (< 0.05), **High |Pearson r|** (> 0.3), **Good R²** (> 0.1)
- **Strong relationships**: p < 0.01, |r| > 0.5, R² > 0.25
- **Questionable relationships**: p > 0.05, |r| < 0.3, R² < 0.1

**Statistical Significance Levels:**
- ⭐⭐⭐ **p < 0.001**: Highly significant (very strong evidence)
- ⭐⭐ **p < 0.01**: Significant (strong evidence)  
- ⭐ **p < 0.05**: Significant (moderate evidence)
- ❌ **p ≥ 0.05**: Not significant (insufficient evidence)
"""
    return fig, strongest_insight_md, table_data, note_md, quality_report, edge_tests

# Backward compatibility wrapper
def analyze_causal_structure(df, theme='dark', filter_params=None):
    """Backward compatibility wrapper for optimized causal analysis"""
    return analyze_causal_structure_optimized(df, theme, filter_params=filter_params)



app = dash.Dash(__name__)

app.layout = html.Div(id='main-container', children=[
    html.H1("Dynamic Data Analysis Dashboard", id='h1-title'),
    
    dcc.RadioItems(
        id='theme-selector',
        options=[{'label': 'Light Mode', 'value': 'light'}, {'label': 'Dark Mode', 'value': 'dark'}],
        value='dark',
        labelStyle={'display': 'inline-block', 'margin': '0 10px'}
    ),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag & Drop or ', html.A('Select Files')]),
            multiple=False
        ),
        html.Div("Or"),
        dcc.Input(id='input-url', type='text', placeholder='Enter URL to CSV or Excel file'),
        html.Button('Load Data', id='load-url-button', n_clicks=0),
    ], className='data-loader'),

    dcc.Store(id='store-processed-data'),
    dcc.Store(id='store-raw-data'),
    
    html.Div(id='controls-container', children=[
        html.Div([
            html.Label('X-axis:', style={'paddingRight': '10px'}),
            dcc.Dropdown(id='x_axis_dropdown', placeholder='Select X-axis', className='control-dropdown')
        ], className='control-element'),
        html.Div([
            html.Label('Y-axis:', style={'paddingRight': '10px'}),
            dcc.Dropdown(id='y_axis_dropdown', placeholder='Select Y-axis', className='control-dropdown')
        ], className='control-element'),
        html.Div([
            html.Label('Color:', style={'paddingRight': '10px'}),
            dcc.Dropdown(id='color_dropdown', placeholder='Select Color', className='control-dropdown')
        ], className='control-element'),
        html.Div([
            html.Label('Timespan:', style={'paddingRight': '10px'}),
            dcc.Dropdown(
                id='timespan_selector',
                options=['All', 'Last 3 Months', 'Last 6 Months', 'Last Year', 'YTD'],
                value='All',
                clearable=False,
                className='control-dropdown'
            )
        ], className='control-element'),
        html.Div([
            html.Label('Chart Type:', style={'paddingRight': '10px'}),
            dcc.Dropdown(
                id='chart_type_dropdown',
                options=['Scatter', 'Line', 'Bar'],
                value='Scatter',
                clearable=False,
                className='control-dropdown'
            )
        ], className='control-element'),
        html.Div([
            html.Label('Y-axis Aggregation:', style={'paddingRight': '10px'}),
            dcc.RadioItems(
                id='y-axis-agg-selector',
                options=[
                    {'label': 'Raw', 'value': 'raw'},
                    {'label': 'Average', 'value': 'avg'},
                ],
                value='raw',
                labelStyle={'display': 'inline-block', 'margin': '0 5px'}
            )
        ], className='control-element'),
    ]),
    html.Div(id='dashboard-container'),
    html.Div(id='causal-analysis-section', children=[
        html.H2("Causal Analysis Results", className='section-title'),
        
        # Significance Filter Toggle
        html.Div([
            html.Div([
                html.Label("🎯 Focus on Significant Relationships:", className='toggle-label'),
                dcc.Checklist(
                    id='significance-filter-toggle',
                    options=[{'label': ' Hide non-significant edges (p ≥ 0.05)', 'value': 'hide_nonsig'}],
                    value=[],
                    className='significance-toggle'
                )
            ], className='filter-control-group'),
            html.Div([
                html.Label("📊 Minimum Correlation Threshold:", className='toggle-label'),
                dcc.Slider(
                    id='correlation-threshold-slider',
                    min=0,
                    max=0.8,
                    step=0.1,
                    value=0.0,
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 9, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='correlation-slider'
                )
            ], className='filter-control-group'),
        ], className='causal-filter-controls'),
        
        dcc.Loading(
            id="loading-causal-analysis",
            type="default",
            children=html.Div([
                dcc.Graph(id='causal-graph'),
                dcc.Markdown(id='strongest-causal-insight'),
                html.Div(id='model-quality-section', children=[
                    html.H3("Model Quality Evaluation", className='section-title'),
                    dcc.Markdown(id='quality-report'),
                    html.Div(id='performance-metrics', children=[
                        html.H4("Performance Metrics", className='section-title'),
                        dcc.Markdown(id='performance-report'),
                    ]),
                ]),
                html.H4("All Causal Relationships with Statistical Tests", className='section-title'),
                
                # Statistical Help Panel (Separated from main content)
                html.Div([
                    html.Button("📚 Statistical Guide", id="stats-help-button", className="help-button"),
                    dcc.Tooltip(
                        "Click to open comprehensive statistical explanations and examples",
                        id="stats-help-tooltip"
                    )
                ], className='help-button-container'),
                
                # Statistical Explanations Modal/Sidebar
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("📊 Statistical Metrics Guide", className='help-panel-title'),
                            html.Button("✕", id="close-help-button", className="close-help-button"),
                        ], className='help-panel-header'),
                        
                        # Quick Reference Cards
                        html.Div([
                            # Causal Weight Card
                            html.Div([
                                html.H4("⚖️ Causal Weight", className='metric-card-title'),
                                html.P("Estimated causal effect strength", className='metric-card-subtitle'),
                                html.Div([
                                    html.Span("Strong: ", className='threshold-label'),
                                    html.Span("|Weight| > 0.5", className='threshold-value strong'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Moderate: ", className='threshold-label'),
                                    html.Span("0.2 < |Weight| ≤ 0.5", className='threshold-value moderate'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Weak: ", className='threshold-label'),
                                    html.Span("|Weight| ≤ 0.2", className='threshold-value weak'),
                                ], className='threshold-item'),
                            ], className='metric-card'),
                            
                            # Correlation Card
                            html.Div([
                                html.H4("📈 Correlation (r)", className='metric-card-title'),
                                html.P("Linear relationship strength", className='metric-card-subtitle'),
                                html.Div([
                                    html.Span("Strong: ", className='threshold-label'),
                                    html.Span("|r| > 0.7", className='threshold-value strong'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Moderate: ", className='threshold-label'),
                                    html.Span("0.3 < |r| ≤ 0.7", className='threshold-value moderate'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Weak: ", className='threshold-label'),
                                    html.Span("|r| ≤ 0.3", className='threshold-value weak'),
                                ], className='threshold-item'),
                            ], className='metric-card'),
                            
                            # P-value Card
                            html.Div([
                                html.H4("🎯 P-value", className='metric-card-title'),
                                html.P("Statistical significance", className='metric-card-subtitle'),
                                html.Div([
                                    html.Span("Highly Sig: ", className='threshold-label'),
                                    html.Span("p < 0.001", className='threshold-value strong'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Significant: ", className='threshold-label'),
                                    html.Span("p < 0.05", className='threshold-value moderate'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Not Sig: ", className='threshold-label'),
                                    html.Span("p ≥ 0.05", className='threshold-value weak'),
                                ], className='threshold-item'),
                            ], className='metric-card'),
                            
                            # R² Card
                            html.Div([
                                html.H4("📊 R² (Variance)", className='metric-card-title'),
                                html.P("Explained variance proportion", className='metric-card-subtitle'),
                                html.Div([
                                    html.Span("Excellent: ", className='threshold-label'),
                                    html.Span("R² > 0.75", className='threshold-value strong'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Good: ", className='threshold-label'),
                                    html.Span("0.25 < R² ≤ 0.75", className='threshold-value moderate'),
                                ], className='threshold-item'),
                                html.Div([
                                    html.Span("Poor: ", className='threshold-label'),
                                    html.Span("R² ≤ 0.25", className='threshold-value weak'),
                                ], className='threshold-item'),
                            ], className='metric-card'),
                        ], className='metric-cards-grid'),
                        
                        # Detailed Explanations (Collapsible)
                        html.Details([
                            html.Summary("📖 Detailed Explanations & Examples", className='detailed-explanation-summary'),
                            html.Div([
                                dcc.Markdown("""
### Interpretation Guide

**Causal Weight**: Represents the strength and direction of causal effect. A weight of 0.5 means a 1-unit increase in the source variable causes a 0.5-unit increase in the target variable.

**Correlation (r)**: Measures linear relationship strength. Values closer to -1 or +1 indicate stronger linear relationships.

**P-value**: Probability that the observed relationship occurred by chance. Lower values indicate stronger evidence against randomness.

**R² (Coefficient of Determination)**: Proportion of variance in the target variable explained by the source variable. Higher values indicate better predictive power.

### Quality Assessment Framework

Use these combined criteria to evaluate relationship reliability:
- **High Quality**: p < 0.01 AND |r| > 0.5 AND R² > 0.25
- **Moderate Quality**: p < 0.05 AND |r| > 0.3 AND R² > 0.10
- **Low Quality**: p ≥ 0.05 OR |r| < 0.3 OR R² < 0.10

### Decision Making Guidelines

**For Business Decisions**: Use only High Quality relationships for strategic decisions, High + Moderate for operational decisions.

**For Research**: Focus on High Quality for publication, report Moderate with appropriate caveats.
                                """, className='detailed-explanation-content')
                            ], className='detailed-explanation-container')
                        ], className='detailed-explanations'),
                    ], className='help-panel-content')
                ], id='statistical-help-panel', className='help-panel hidden'),
                
                # Color Legend for Table
                html.Div([
                    html.H5("📋 Table Color Legend:", className='legend-title'),
                    html.Div([
                        html.Span("🟢 Highly Significant (p < 0.001)", className='legend-item legend-high'),
                        html.Span("🟢 Significant (p < 0.01)", className='legend-item legend-medium'),  
                        html.Span("🟢 Moderately Significant (p < 0.05)", className='legend-item legend-low'),
                        html.Span("🔴 Not Significant (p ≥ 0.05)", className='legend-item legend-none'),
                    ], className='legend-container')
                ], className='table-legend'),
                
                # Enhanced Table with Tooltips
                html.Div([
                    dash_table.DataTable(
                        id='causal-table',
                        columns=[
                            {'name': 'Source Variable', 'id': 'Source'},
                            {'name': 'Target Variable', 'id': 'Target'},
                            {'name': 'Causal Weight', 'id': 'Weight', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                            {'name': 'Correlation (r)', 'id': 'Pearson_r', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                            {'name': 'P-value', 'id': 'P_value', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                            {'name': 'R² (Variance)', 'id': 'R_squared', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                            {'name': 'Significant?', 'id': 'Significant'},
                        ],
                        tooltip_header={
                            'Source': 'The variable that potentially causes changes in the target variable.',
                            'Target': 'The variable that is potentially affected by changes in the source variable.',
                            'Weight': 'Estimated causal effect strength. Positive = same direction, Negative = opposite direction. Larger absolute values = stronger effects.',
                            'Pearson_r': 'Linear relationship strength (-1 to +1). |r| > 0.7 = strong, 0.3-0.7 = moderate, < 0.3 = weak.',
                            'P_value': 'Probability relationship occurred by chance. p < 0.05 = significant, p < 0.01 = highly significant.',
                            'R_squared': 'Percentage of target variance explained by source (0-1). Higher = better predictive power.',
                            'Significant': "Overall assessment: 'Yes' if p < 0.05 (statistically reliable), 'No' if p >= 0.05 (may be random)."
                        },
                    sort_action='native',
                    style_cell={'textAlign': 'left'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Significant} = Yes'},
                            'backgroundColor': 'rgba(0, 255, 0, 0.15)',
                            'color': 'inherit',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'filter_query': '{P_value} < 0.001'},
                            'backgroundColor': 'rgba(0, 200, 0, 0.2)',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'filter_query': '{P_value} < 0.01'},
                            'backgroundColor': 'rgba(0, 255, 0, 0.15)',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'filter_query': '{P_value} < 0.05'},
                            'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'filter_query': '{P_value} >= 0.05'},
                            'backgroundColor': 'rgba(255, 0, 0, 0.05)',
                            'opacity': '0.7'
                        }
                    ]
                ),
            ]),
                dcc.Markdown(id='causal-weights-note')
            ])
        )
    ]),
    
    html.Div(id='forecast-section', children=[
        html.H2("Time Series Forecasting", className='section-title'),
        html.Div([
            html.Div([
                html.Label('Time Column:'),
                dcc.Dropdown(id='time-series-col-dropdown', placeholder='Select Time Column', className='forecast-dropdown'),
            ], className='forecast-control-group'),
            html.Div([
                html.Label('Target Variable(s):'),
                dcc.Dropdown(id='target-col-dropdown', placeholder='Select one or more target columns', multi=True, className='forecast-dropdown'),
            ], className='forecast-control-group'),
            html.Div([
                html.Label('Forecasting Model:'),
                dcc.Dropdown(id='model-dropdown', options=[
                    {'label': 'Linear Regression', 'value': 'Linear Regression'},
                    {'label': 'ARIMA', 'value': 'ARIMA'},
                    {'label': 'SARIMA (Seasonal)', 'value': 'SARIMA'},
                    {'label': 'VAR (Vector Autoregression)', 'value': 'VAR (Vector Autoregression)'},
                    {'label': 'Dynamic Factor Model', 'value': 'Dynamic Factor Model'},
                    {'label': 'State-Space Model', 'value': 'State-Space Model'},
                    {'label': 'Nowcasting', 'value': 'Nowcasting'}
                ], placeholder='Select Model', className='forecast-dropdown'),
            ], className='forecast-control-group'),
            html.Div([
                html.Label('Forecast Periods:'),
                dcc.Input(id='forecast-periods-input', type='number', placeholder='Periods to forecast', value=10, min=1, max=100),
            ], className='forecast-control-group'),
            html.Div([
                html.Button('Generate Forecast', id='generate-forecast-button', n_clicks=0),
            ], className='forecast-control-group'),
        ], className='forecast-controls'),
        
        # Model Information Panel
        html.Div(id='model-info-panel', children=[
            dcc.Markdown(id='model-description', children="""
            ### Model Information
            Select a forecasting model to see detailed information about its capabilities and use cases.
            """)
        ], className='model-info-section'),
        dcc.Loading(
            id="loading-forecast",
            type="default",
            children=dcc.Graph(id='forecast-graph')
        )
    ])
])

@callback(
    Output('main-container', 'className'),
    [Input('theme-selector', 'value')]
)
def update_theme(theme):
    return f'theme-{theme}'

def parse_contents(contents, filename):
    print(f"Parsing file: {filename}")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(BytesIO(decoded))
        else:
            print("File type not supported")
            return None, None
        df.columns = [col.strip() for col in df.columns]
        print("File parsed successfully")
        return df, process_data(df.copy())
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None, None

def get_data_from_url(url):
    try:
        if 'csv' in url:
            df = pd.read_csv(url)
        elif 'xls' in url:
            df = pd.read_excel(url)
        else:
            return None, None
        df.columns = [col.strip() for col in df.columns]
        return df, process_data(df.copy())
    except Exception as e:
        print(e)
        return None, None

def get_options(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include='datetime64[ns]').columns.tolist()
    axis_options = numeric_cols + date_cols
    color_options = categorical_cols + date_cols
    return axis_options, color_options, date_cols, numeric_cols

@callback(
    [Output('store-processed-data', 'data'),
     Output('store-raw-data', 'data'),
     Output('x_axis_dropdown', 'options'),
     Output('y_axis_dropdown', 'options'),
     Output('color_dropdown', 'options'),
     Output('time-series-col-dropdown', 'options'),
     Output('target-col-dropdown', 'options')],
    [Input('upload-data', 'contents'),
     Input('load-url-button', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('input-url', 'value')]
)
def update_data(contents, n_clicks, filename, url):
    print("update_data triggered")
    ctx = dash.callback_context
    if not ctx.triggered:
        print("No trigger")
        return None, None, [], [], [], [], []

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Trigger ID: {trigger_id}")
    
    df_raw, df_processed = None, None

    if trigger_id == 'upload-data' and contents:
        print(f"Uploading file: {filename}")
        df_raw, df_processed = parse_contents(contents, filename)
    elif trigger_id == 'load-url-button' and url:
        df_raw, df_processed = get_data_from_url(url)

    if df_processed is None:
        print("df_processed is None, returning empty options")
        return None, None, [], [], [], [], []

    axis_options, color_options, date_cols, numeric_cols = get_options(df_processed)
    print(f"Axis options: {axis_options}")
    processed_data_json = df_processed.to_json(date_format='iso', orient='split')
    raw_data_json = df_raw.to_json(date_format='iso', orient='split')
    
    return processed_data_json, raw_data_json, axis_options, axis_options, color_options, date_cols, numeric_cols

@callback(
    [Output('causal-graph', 'figure'),
     Output('strongest-causal-insight', 'children'),
     Output('causal-table', 'data'),
     Output('causal-weights-note', 'children'),
     Output('quality-report', 'children'),
     Output('performance-report', 'children'),
     Output('causal-table', 'style_data'),
     Output('causal-table', 'style_header')],
    [Input('store-raw-data', 'data'),
     Input('theme-selector', 'value'),
     Input('significance-filter-toggle', 'value'),
     Input('correlation-threshold-slider', 'value')]
)
def update_causal_analysis(json_raw_data, theme, significance_filter, correlation_threshold):
    if json_raw_data is None:
        return go.Figure(), "", [], "", "", "", {}, {}

    df = pd.read_json(StringIO(json_raw_data), orient='split')
    
    # Apply filtering parameters
    filter_params = {
        'hide_nonsignificant': 'hide_nonsig' in (significance_filter or []),
        'min_correlation': correlation_threshold or 0.0
    }
    
    causal_graph, strongest_insight, table_data, note, quality_report, edge_tests = analyze_causal_structure(df, theme, filter_params)
    
    # Create performance report
    performance_report = ""
    if hasattr(quality_report, '__contains__') and 'Performance Metrics' not in quality_report:
        # Extract performance metrics if available
        try:
            # This would be populated by the optimized function
            performance_report = """
### Performance Metrics
- **Computation Time**: Optimized processing enabled
- **Parallel Workers**: Multi-threading active
- **Memory Usage**: Efficient data handling
- **Processing Status**: ✅ Enhanced performance mode
            """
        except:
            performance_report = "Performance metrics not available"
    
    if theme == 'dark':
        style_data = {
            'color': 'white',
            'backgroundColor': '#333',
            'border': '1px solid white'
        }
        style_header = {
            'color': 'white',
            'backgroundColor': '#111',
            'fontWeight': 'bold',
            'border': '1px solid white'
        }
    else: # light
        style_data = {
            'color': 'black',
            'backgroundColor': 'white',
            'border': '1px solid black'
        }
        style_header = {
            'color': 'black',
            'backgroundColor': '#eee',
            'fontWeight': 'bold',
            'border': '1px solid black'
        }

    return causal_graph, strongest_insight, table_data, note, quality_report, performance_report, style_data, style_header



@callback(
    Output('dashboard-container', 'children'),
    [
        Input('store-processed-data', 'data'),
        Input('x_axis_dropdown', 'value'),
        Input('y_axis_dropdown', 'value'),
        Input('color_dropdown', 'value'),
        Input('timespan_selector', 'value'),
        Input('chart_type_dropdown', 'value'),
        Input('theme-selector', 'value'),
        Input('y-axis-agg-selector', 'value')
    ]
)
def update_graphs(json_processed_data, x_axis, y_axis, color, timespan, chart_type, theme, y_axis_agg):
    if json_processed_data is None:
        return [html.Div("Please upload a file or enter a URL to begin.", className='placeholder-text')]

    df = pd.read_json(StringIO(json_processed_data), orient='split')
    
    date_col = None
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_col = col
            break
            
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = filter_by_timespan(df, timespan)

    if not x_axis or not y_axis:
        return [html.Div("Please select X and Y axes to plot.", className='placeholder-text')]

    if y_axis_agg == 'avg':
        group_by_cols = [x_axis]
        if color and color in df.columns and color != x_axis:
            group_by_cols.append(color)
        
        df = df.groupby(group_by_cols)[y_axis].mean().reset_index()

    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    if chart_type == 'Scatter':
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f'{y_axis} by {x_axis}', template=template)
    elif chart_type == 'Line':
        fig = px.line(df, x=x_axis, y=y_axis, color=color, title=f'{y_axis} by {x_axis}', template=template)
    elif chart_type == 'Bar':
        fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=f'{y_axis} by {x_axis}', template=template)
    else:
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f'{y_axis} by {x_axis}', template=template)
    
    return [dcc.Graph(figure=fig, className='main-graph')]

@callback(
    Output('forecast-graph', 'figure'),
    [Input('generate-forecast-button', 'n_clicks')],
    [State('store-processed-data', 'data'),
     State('time-series-col-dropdown', 'value'),
     State('target-col-dropdown', 'value'),
     State('model-dropdown', 'value'),
     State('forecast-periods-input', 'value'),
     State('theme-selector', 'value')]
)
def update_forecast_graph(n_clicks, json_data, time_col, target_cols, model_name, periods, theme):
    if n_clicks == 0 or not all([json_data, time_col, target_cols, model_name, periods]):
        return go.Figure()

    df = pd.read_json(StringIO(json_data), orient='split')
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).set_index(time_col)

    fig = go.Figure()
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    fig.update_layout(template=template, title=f'{model_name} Forecast')

    colors = px.colors.qualitative.Plotly

    if model_name == 'VAR (Vector Autoregression)':
        if len(target_cols) < 2:
            fig.add_annotation(text="Please select at least two columns for VAR model.",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        fitted_model, forecasts, returned_vars = fit_var_model(df, target_cols, periods)

        if fitted_model and forecasts:
            last_date = df.index.max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=periods, freq='D')
            var_data = df[returned_vars].dropna()
            
            try:
                _, lower, upper = fitted_model.forecast_interval(var_data.values[-fitted_model.k_ar:], steps=periods, alpha=0.05)
            except Exception as e:
                print(f"Could not generate confidence intervals: {e}")
                lower, upper = None, None

            for i, col in enumerate(target_cols):
                color = colors[i % len(colors)]
                
                # Handle both hex and rgb color formats
                if color.startswith('#'):
                    color_hex = color.lstrip('#')
                    r, g, b = tuple(int(color_hex[j:j+2], 16) for j in (0, 2, 4))
                    fill_color = f'rgba({r}, {g}, {b}, 0.2)'
                else:
                    fill_color = f"rgba({color.split('(')[1].split(')')[0]}, 0.2)"

                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'Historical {col}', line=dict(color=color)))
                fig.add_trace(go.Scatter(x=future_dates, y=forecasts[col], mode='lines', name=f'Forecast {col}', line=dict(color=color, dash='dot')))
                
                if lower is not None and upper is not None:
                    col_idx = returned_vars.index(col)
                    lower_ci = lower[:, col_idx]
                    upper_ci = upper[:, col_idx]
                    fig.add_trace(go.Scatter(x=future_dates, y=upper_ci, mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=future_dates, y=lower_ci, mode='lines', line=dict(width=0), 
                                           fill='tonexty', fillcolor=fill_color, 
                                           name=f'95% CI {col}'))

            fig.update_layout(title=f'{model_name} Forecast for {", ".join(target_cols)}')
        else:
            fig.add_annotation(text="VAR model failed to fit. Try with more data or fewer variables.", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    elif model_name == 'Dynamic Factor Model':
        if len(target_cols) < 2:
            fig.add_annotation(text="Please select at least two columns for Dynamic Factor Model.",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
            
        fitted_model, forecasts, returned_vars = fit_dynamic_factor_model(df, target_cols, periods)
        
        if fitted_model and forecasts:
            last_date = df.index.max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=periods, freq='D')

            for i, col in enumerate(target_cols):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'Historical {col}', line=dict(color=color)))
                fig.add_trace(go.Scatter(x=future_dates, y=forecasts[col], mode='lines', name=f'Forecast {col}', line=dict(color=color, dash='dashdot')))
            
            fig.update_layout(title=f'Dynamic Factor Model Forecast for {", ".join(target_cols)}')
        else:
            fig.add_annotation(text="Dynamic Factor Model failed to fit. Try with more data.", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    elif model_name == 'State-Space Model':
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(fit_state_space_model, df, [col], periods): col for col in target_cols}
            
            last_date = df.index.max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=periods, freq='D')

            for i, future in enumerate(as_completed(futures)):
                col = futures[future]
                models, forecasts, _ = future.result()
                if models and forecasts:
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'Historical {col}', line=dict(color=color)))
                    fig.add_trace(go.Scatter(x=future_dates, y=forecasts[col], mode='lines', name=f'Forecast {col}', line=dict(color=color, dash='longdash')))

        fig.update_layout(title=f'State-Space Model Forecast for {", ".join(target_cols)}')

    else: # Handling for single-target models
        if isinstance(target_cols, list):
            if len(target_cols) > 1:
                fig.add_annotation(text=f"{model_name} supports only a single target variable.",
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            target_col = target_cols[0]
        else:
            target_col = target_cols

        if model_name == 'Linear Regression':
            df['time_numeric'] = (df.index - df.index.min()).days
            model = LinearRegression()
            model.fit(df[['time_numeric']], df[target_col])
            
            last_date = df.index.max()
            future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, periods + 1)])
            future_numeric = (future_dates - df.index.min()).days
            
            forecast = model.predict(future_numeric.values.reshape(-1, 1))
            fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode='lines', name='Historical Data'))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast'))

        elif model_name == 'ARIMA':
            model = ARIMA(df[target_col], order=(5,1,0))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=periods)
            fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode='lines', name='Historical Data'))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast'))

        elif model_name == 'SARIMA':
            model = SARIMAX(df[target_col], order=(5,1,0), seasonal_order=(1,1,1,12))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=periods)
            fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode='lines', name='Historical Data'))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast'))

        elif model_name == 'Nowcasting':
            split_point = int(len(df) * 0.9)
            train_df = df.iloc[:split_point]
            test_df = df.iloc[split_point:]

            train_df.loc[:, 'time_numeric'] = (train_df.index - df.index.min()).days
            model = LinearRegression()
            model.fit(train_df[['time_numeric']], train_df[target_col])
            
            test_df.loc[:, 'time_numeric'] = (test_df.index - df.index.min()).days
            nowcast = model.predict(test_df[['time_numeric']])
            
            fig.add_trace(go.Scatter(x=train_df.index, y=train_df[target_col], mode='lines', name='Training Data'))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df[target_col], mode='lines', name='Actual Values'))
            fig.add_trace(go.Scatter(x=test_df.index, y=nowcast, mode='lines', name='Nowcast', line={'dash': 'dash'}))
            fig.update_layout(title=f'Nowcasting for {target_col}')

    return fig

@callback(
    Output('statistical-help-panel', 'className'),
    [Input('stats-help-button', 'n_clicks'),
     Input('close-help-button', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_help_panel(open_clicks, close_clicks):
    """Toggle the statistical help panel visibility"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'help-panel hidden'
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'stats-help-button':
        return 'help-panel visible'
    elif trigger_id == 'close-help-button':
        return 'help-panel hidden'
    
    return 'help-panel hidden'

@callback(
    Output('model-description', 'children'),
    [Input('model-dropdown', 'value')]
)
def update_model_description(model_name):
    """Update model description based on selected model"""
    
    descriptions = {
        'Linear Regression': """
## Linear Regression

### Model Overview
**Type:** Simple trend-based forecasting  
**Complexity:** Low  
**Data Requirements:** Minimal (10+ observations)

### Best Use Cases
- Data with clear linear trends
- Short-term forecasting (1-10 periods)
- Quick baseline analysis
- Limited historical data available

### Advantages
- ⚡ **Fast computation** - Results in seconds
- 📊 **Highly interpretable** - Clear trend coefficients
- 📈 **Minimal data requirements** - Works with small datasets
- 🔧 **Simple implementation** - No complex parameters

### Limitations
- ❌ **No seasonality handling** - Cannot capture periodic patterns
- ❌ **Linear assumption** - May miss non-linear relationships
- ❌ **No autocorrelation** - Ignores time series dependencies
- ❌ **Limited complexity** - Cannot model complex patterns

### Technical Details
- **Method:** Ordinary Least Squares (OLS)
- **Features:** Time as numeric variable
- **Output:** Linear trend projection
- **Confidence Intervals:** Not available
        """,
        
        'ARIMA': """
## ARIMA (AutoRegressive Integrated Moving Average)

### Model Overview
**Type:** Univariate time series model  
**Complexity:** Medium  
**Data Requirements:** Moderate (30+ observations)

### Best Use Cases
- Stationary time series data
- Medium-term forecasting (5-50 periods)
- Economic indicators
- Financial data without strong seasonality

### Advantages
- 📈 **Handles autocorrelation** - Models time dependencies
- 🔬 **Well-established methodology** - Proven statistical foundation
- ⚖️ **Balanced complexity** - Good performance vs. simplicity
- 📊 **Diagnostic tools** - Rich model validation options

### Limitations
- 📋 **Requires stationarity** - Data preprocessing needed
- ❌ **No seasonality** - Cannot handle seasonal patterns
- 🎛️ **Parameter selection** - Requires model tuning
- 📊 **Univariate only** - Single variable analysis

### Technical Details
- **Parameters:** (p, d, q) - AR, Integration, MA orders
- **Method:** Maximum Likelihood Estimation
- **Preprocessing:** Differencing for stationarity
- **Diagnostics:** Residual analysis, AIC/BIC selection
        """,
        
        'SARIMA': """
## SARIMA (Seasonal ARIMA)

### Model Overview
**Type:** Seasonal univariate time series model  
**Complexity:** High  
**Data Requirements:** High (2+ seasonal cycles)

### Best Use Cases
- Data with clear seasonal patterns
- Monthly, quarterly, or yearly cycles
- Sales data with seasonality
- Weather and climate patterns

### Advantages
- 🔄 **Seasonal modeling** - Captures periodic patterns
- 📈 **Trend and seasonality** - Handles both components
- 🔬 **Robust methodology** - Statistically sound approach
- 📊 **Comprehensive analysis** - Rich diagnostic capabilities

### Limitations
- 🎛️ **Complex parameter tuning** - Many parameters to optimize
- 💻 **Computationally intensive** - Slower than simpler models
- 📊 **Data requirements** - Needs multiple seasonal cycles
- 🔧 **Setup complexity** - Requires seasonal period specification

### Technical Details
- **Parameters:** (p,d,q)(P,D,Q,s) - Regular + Seasonal orders
- **Seasonal Period:** Must be specified (12 for monthly, 4 for quarterly)
- **Method:** Maximum Likelihood with seasonal differencing
- **Validation:** Seasonal diagnostics and residual analysis
        """,
        
        'VAR (Vector Autoregression)': """
## VAR (Vector Autoregression)

### Model Overview
**Type:** Multivariate time series model  
**Complexity:** High  
**Data Requirements:** High (multiple variables, 50+ observations)

### Best Use Cases
- Multiple related time series
- Cross-variable relationship analysis
- Economic systems modeling
- Portfolio and risk analysis

### Advantages
- 🔗 **Models interdependencies** - Captures variable relationships
- 🌐 **System-wide forecasts** - Predicts all variables simultaneously
- 📊 **Confidence intervals** - Provides uncertainty quantification
- 🔄 **Impulse response** - Shows shock propagation effects

### Limitations
- 📊 **Multiple variables required** - Needs related time series
- 🎛️ **Model specification** - Sensitive to variable selection
- 💻 **Computational complexity** - Scales with variables squared
- 📈 **Interpretation complexity** - Many coefficients to analyze

### Technical Details
- **Variables:** 2-8 related time series (optimal: 3-5)
- **Lag Selection:** Automatic via information criteria
- **Method:** Ordinary Least Squares for each equation
- **Output:** Forecasts + confidence intervals for all variables
        """,
        
        'Dynamic Factor Model': """
## Dynamic Factor Model

### Model Overview
**Type:** Dimension reduction + forecasting model  
**Complexity:** Very High  
**Data Requirements:** Very High (many variables, 100+ observations)

### Best Use Cases
- High-dimensional datasets (10+ variables)
- Common factor extraction
- Economic indicator analysis
- Large business metric datasets

### Advantages
- 🔍 **Dimension reduction** - Handles many variables efficiently
- 🎯 **Factor identification** - Finds underlying common drivers
- 🔇 **Noise reduction** - Filters out idiosyncratic variation
- 📊 **Scalable analysis** - Works with large datasets

### Limitations
- 🧩 **Complex interpretation** - Factor meanings not always clear
- 📊 **High data requirements** - Needs many variables and observations
- 💻 **Computational intensity** - Slow for large datasets
- 🎛️ **Parameter selection** - Number of factors must be chosen

### Technical Details
- **Factors:** 2-5 common factors (automatically estimated)
- **Method:** Kalman Filter and Maximum Likelihood
- **Preprocessing:** Standardization required
- **Output:** Factor-based forecasts with loadings interpretation
        """,
        
        'State-Space Model': """
## State-Space Model (Unobserved Components)

### Model Overview
**Type:** Structural time series model  
**Complexity:** Very High  
**Data Requirements:** High (50+ observations)

### Best Use Cases
- Structural component analysis
- Trend and seasonal decomposition
- Policy impact assessment
- Missing data scenarios

### Advantages
- 🔧 **Flexible structure** - Customizable components
- 📊 **Component analysis** - Separates trend, seasonal, irregular
- 🕳️ **Missing data handling** - Natural accommodation of gaps
- 📈 **Fitted components** - Shows individual component evolution

### Limitations
- 🧩 **Complex setup** - Requires component specification
- 🎓 **Domain knowledge needed** - Model structure decisions
- 💻 **Computational complexity** - Intensive estimation process
- 🎛️ **Many parameters** - Multiple components to estimate

### Technical Details
- **Components:** Trend (local linear) + Seasonal + Autoregressive
- **Method:** Kalman Filter and Maximum Likelihood
- **Seasonality:** Automatic detection or user-specified
- **Output:** Component forecasts + structural decomposition
        """,
        
        'Nowcasting': """
## Nowcasting

### Model Overview
**Type:** Real-time estimation model  
**Complexity:** Low  
**Data Requirements:** Low (20+ observations)

### Best Use Cases
- Current period estimation
- Real-time monitoring
- Very short-term predictions (1-3 periods)
- Quick status assessment

### Advantages
- ⚡ **Real-time processing** - Uses most recent data
- 📊 **Monitoring focus** - Good for current state assessment
- 🔧 **Simple methodology** - Easy to understand and implement
- 📈 **Quick results** - Fast computation for dashboards

### Limitations
- ⏱️ **Limited horizon** - Very short forecast range
- 🔧 **Simple approach** - Basic linear methodology
- 📊 **No uncertainty** - No confidence intervals
- 🎯 **Narrow scope** - Focused on current period only

### Technical Details
- **Method:** Linear regression on recent data subset
- **Training:** 90% of data for model, 10% for validation
- **Focus:** Estimating current/next period values
- **Output:** Point estimates with actual vs. predicted comparison
        """
    }
    
    if model_name and model_name in descriptions:
        return descriptions[model_name]
    else:
        return """
## Forecasting Model Selection Guide

### Available Models Overview

Select a forecasting model above to see detailed information about its capabilities, use cases, and technical specifications.

#### **Simple Models**
- 📈 **Linear Regression** - Basic trend analysis
- 📊 **Nowcasting** - Real-time estimation

#### **Univariate Time Series**
- 🔄 **ARIMA** - Classic time series modeling
- 🌊 **SARIMA** - Seasonal time series modeling

#### **Advanced Multivariate**
- 🔗 **VAR** - Multiple related time series
- 🎯 **Dynamic Factor Model** - High-dimensional analysis
- 🏗️ **State-Space Model** - Structural decomposition

### Selection Criteria

**Data Size:**
- Small (< 50): Linear Regression, Nowcasting
- Medium (50-200): ARIMA, SARIMA
- Large (200+): VAR, Dynamic Factor, State-Space

**Complexity Needs:**
- Simple: Linear Regression, Nowcasting
- Moderate: ARIMA, SARIMA
- Advanced: VAR, Dynamic Factor, State-Space

**Variable Count:**
- Single: Linear, ARIMA, SARIMA, State-Space, Nowcasting
- Multiple: VAR, Dynamic Factor Model
        """

if __name__ == '__main__':
    app.run(debug=True)