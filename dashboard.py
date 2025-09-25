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
import warnings
warnings.filterwarnings('ignore')

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




def evaluate_model_quality(df, sm):
    """
    Evaluate causal model quality with comprehensive statistical checks
    """
    quality_metrics = {}
    
    # 1. Basic Network Statistics
    quality_metrics['network_stats'] = {
        'num_nodes': sm.number_of_nodes(),
        'num_edges': sm.number_of_edges(),
        'density': nx.density(sm),
        'is_dag': nx.is_directed_acyclic_graph(sm)
    }
    
    # 2. Edge Weight Distribution Analysis
    weights = [data.get('weight', 0) for u, v, data in sm.edges(data=True)]
    if weights:
        quality_metrics['weight_stats'] = {
            'mean_weight': np.mean(np.abs(weights)),
            'std_weight': np.std(weights),
            'min_weight': np.min(weights),
            'max_weight': np.max(weights),
            'weight_range': np.max(weights) - np.min(weights)
        }
    
    # 3. Statistical Tests for Each Edge
    edge_tests = []
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    
    for u, v, data in sm.edges(data=True):
        if u in df_numeric.columns and v in df_numeric.columns:
            x_data = df_numeric[u].values
            y_data = df_numeric[v].values
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(x_data, y_data)
            
            # Spearman correlation (non-parametric)
            spearman_r, spearman_p = spearmanr(x_data, y_data)
            
            # Linear regression R²
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
            r2 = reg.score(x_data.reshape(-1, 1), y_data)
            
            edge_tests.append({
                'source': u,
                'target': v,
                'causal_weight': data.get('weight', 0),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'r_squared': r2,
                'significant': pearson_p < 0.05
            })
    
    quality_metrics['edge_tests'] = edge_tests
    
    # 4. Cross-Validation for Predictive Performance
    cv_scores = []
    for target_var in df_numeric.columns:
        try:
            # Get parents of target variable in the causal graph
            parents = list(sm.predecessors(target_var))
            if len(parents) > 0 and len(parents) < len(df_numeric.columns):
                X = df_numeric[parents].values
                y = df_numeric[target_var].values
                
                # 5-fold cross validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
                
                cv_scores.append({
                    'target': target_var,
                    'parents': parents,
                    'cv_r2_mean': np.mean(scores),
                    'cv_r2_std': np.std(scores),
                    'cv_scores': scores.tolist()
                })
        except:
            continue
    
    quality_metrics['cross_validation'] = cv_scores
    
    # 5. Normality Tests for Residuals
    normality_tests = []
    for target_var in df_numeric.columns:
        try:
            parents = list(sm.predecessors(target_var))
            if len(parents) > 0:
                X = df_numeric[parents].values
                y = df_numeric[target_var].values
                
                reg = LinearRegression().fit(X, y)
                residuals = y - reg.predict(X)
                
                # D'Agostino-Pearson normality test
                stat, p_value = normaltest(residuals)
                
                normality_tests.append({
                    'target': target_var,
                    'normality_stat': stat,
                    'normality_p': p_value,
                    'residuals_normal': p_value > 0.05
                })
        except:
            continue
    
    quality_metrics['normality_tests'] = normality_tests
    
    # 6. Model Complexity Metrics
    quality_metrics['complexity'] = {
        'avg_parents_per_node': np.mean([sm.in_degree(node) for node in sm.nodes()]),
        'max_parents': max([sm.in_degree(node) for node in sm.nodes()]) if sm.nodes() else 0,
        'sparsity': 1 - (sm.number_of_edges() / (sm.number_of_nodes() * (sm.number_of_nodes() - 1)))
    }
    
    return quality_metrics

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
    
    full_report = overview_text + weight_text + cv_text + sig_text + quality_text
    
    return full_report, edge_tests

def analyze_causal_structure(df, theme='dark'):
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            le = LabelEncoder()
            df_numeric[col] = le.fit_transform(df_numeric[col])
    df_numeric = df_numeric.select_dtypes(include=['number']).dropna()
    if df_numeric.shape[1] < 2:
        return go.Figure(), "", [], "", "", []
    
    # Create causal structure
    sm = from_pandas(df_numeric)
    
    # Evaluate model quality
    quality_metrics = evaluate_model_quality(df_numeric, sm)
    quality_report, edge_tests = create_quality_report(quality_metrics, theme)
    
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
        hover_text = f'Weight: {weight:.4f}'
        if edge_test:
            hover_text += f'<br>Pearson r: {edge_test["pearson_r"]:.3f}'
            hover_text += f'<br>P-value: {edge_test["pearson_p"]:.3f}'
            hover_text += f'<br>R²: {edge_test["r_squared"]:.3f}'
            hover_text += f'<br>Significant: {"Yes" if edge_test["significant"] else "No"}'

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=hover_text,
            mode='lines')
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

    fig = go.Figure(data=edge_traces + [node_trace],
                 layout=go.Layout(
                    showlegend=False,
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

    # Enhanced table data with statistical metrics
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

    note_md = """
---
**Enhanced Analysis Notes:**
- **Green/Orange edges**: Statistically significant relationships (p < 0.05)
- **Blue/Red edges**: Non-significant or untested relationships
- **Edge thickness**: Proportional to causal weight magnitude
- **Hover over edges**: View detailed statistical metrics
- **Quality metrics**: Comprehensive model evaluation included below
"""
    return fig, strongest_insight_md, table_data, note_md, quality_report, edge_tests



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
        dcc.Graph(id='causal-graph'),
        dcc.Markdown(id='strongest-causal-insight'),
        
        # Model Quality Evaluation Section
        html.Div(id='model-quality-section', children=[
            html.H3("Model Quality Evaluation", className='section-title'),
            dcc.Markdown(id='quality-report'),
        ]),
        
        html.H4("All Causal Relationships with Statistical Tests", className='section-title'),
        dash_table.DataTable(
            id='causal-table',
            columns=[
                {'name': 'Source', 'id': 'Source'},
                {'name': 'Target', 'id': 'Target'},
                {'name': 'Weight', 'id': 'Weight', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                {'name': 'Pearson r', 'id': 'Pearson_r', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'P-value', 'id': 'P_value', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'R²', 'id': 'R_squared', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'Significant', 'id': 'Significant'},
            ],
            sort_action='native',
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Significant} = Yes'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                    'color': 'inherit',
                },
                {
                    'if': {'filter_query': '{P_value} < 0.05'},
                    'fontWeight': 'bold'
                }
            ]
        ),
        dcc.Markdown(id='causal-weights-note')
    ]),
    
    html.Div(id='forecast-section', children=[
        html.H2("Time Series Forecasting", className='section-title'),
        html.Div([
            dcc.Dropdown(id='time-series-col-dropdown', placeholder='Select Time Column', className='forecast-dropdown'),
            dcc.Dropdown(id='target-col-dropdown', placeholder='Select Target Column', className='forecast-dropdown'),
            dcc.Dropdown(id='model-dropdown', options=['Linear Regression', 'ARIMA', 'Nowcasting'], placeholder='Select Model', className='forecast-dropdown'),
            dcc.Input(id='forecast-periods-input', type='number', placeholder='Periods to forecast', value=10),
            html.Button('Generate Forecast', id='generate-forecast-button', n_clicks=0),
        ], className='forecast-controls'),
        dcc.Graph(id='forecast-graph')
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
     Output('causal-table', 'style_data'),
     Output('causal-table', 'style_header')],
    [Input('store-raw-data', 'data'),
     Input('theme-selector', 'value')]
)
def update_causal_analysis(json_raw_data, theme):
    if json_raw_data is None:
        return go.Figure(), "", [], "", "", {}, {}

    df = pd.read_json(StringIO(json_raw_data), orient='split')
    causal_graph, strongest_insight, table_data, note, quality_report, edge_tests = analyze_causal_structure(df, theme)
    
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

    return causal_graph, strongest_insight, table_data, note, quality_report, style_data, style_header



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
def update_forecast_graph(n_clicks, json_data, time_col, target_col, model_name, periods, theme):
    if n_clicks == 0 or not all([json_data, time_col, target_col, model_name, periods]):
        return go.Figure()

    df = pd.read_json(StringIO(json_data), orient='split')
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).set_index(time_col)

    fig = go.Figure()
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    fig.update_layout(template=template, title=f'{model_name} Forecast for {target_col}')

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

if __name__ == '__main__':
    app.run(debug=True)