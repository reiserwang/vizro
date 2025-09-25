import pandas as pd
import plotly.express as px
from causalnex.structure.notears import from_pandas
import plotly.graph_objects as go
from scipy.stats import linregress
import base64
from io import StringIO, BytesIO
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

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

def analyze_causal_structure(df, theme='dark'):
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            le = LabelEncoder()
            df_numeric[col] = le.fit_transform(df_numeric[col])
    df_numeric = df_numeric.select_dtypes(include=['number']).dropna()
    if df_numeric.shape[1] < 2:
        return go.Figure(), "", [], ""
    sm = from_pandas(df_numeric)
    
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
        
        if weight > 0:
            color = f'rgba(0, 0, 255, {0.2 + norm_weight * 0.8})'
        else:
            color = f'rgba(255, 0, 0, {0.2 + norm_weight * 0.8})'

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f'Weight: {weight:.4f}',
            mode='lines')
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    for node in sm.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
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
    fig.update_layout(template=template, title="CausalNex Analysis")

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
        strongest_insight_md = f"""### Strongest Causal Relationship:\n- **{u}** -> **{v}** (Weight: {w:.4f})"""

    table_data = []
    for u, v, data in sm.edges(data=True):
        table_data.append({'Source': u, 'Target': v, 'Weight': data.get('weight', 0)})

    note_md = """
---
**Note on Weights:**
The weight of a causal relationship indicates the strength of the effect.
- A **positive weight** suggests that an increase in the source variable causes an increase in the target variable.
- A **negative weight** suggests that an increase in the source variable causes a decrease in the target variable.
- The **magnitude** of the weight indicates the strength of the relationship. Larger absolute values imply a stronger effect.
"""
    return fig, strongest_insight_md, table_data, note_md

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
        dcc.Input(id='input-url', type='text', placeholder='Enter URL to CSV or Excel file', style={'width': 'calc(100% - 110px)'}),
        html.Button('Load Data', id='load-url-button', n_clicks=0, style={'width': '100px', 'marginLeft': '10px'}),
    ], style={'width': '50%', 'margin': 'auto', 'textAlign': 'center'}),

    dcc.Store(id='store-processed-data'),
    dcc.Store(id='store-raw-data'),
    
    html.Div(id='controls-container', children=[
        html.Div([
            html.Label('X-axis:', style={'paddingRight': '10px'}),
            dcc.Dropdown(id='x_axis_dropdown', placeholder='Select X-axis', style={'width': '200px', 'display': 'inline-block'})
        ], style={'display': 'inline-block', 'margin': '10px'}),
        html.Div([
            html.Label('Y-axis:', style={'paddingRight': '10px'}),
            dcc.Dropdown(id='y_axis_dropdown', placeholder='Select Y-axis', style={'width': '200px', 'display': 'inline-block'})
        ], style={'display': 'inline-block', 'margin': '10px'}),
        html.Div([
            html.Label('Color:', style={'paddingRight': '10px'}),
            dcc.Dropdown(id='color_dropdown', placeholder='Select Color', style={'width': '200px', 'display': 'inline-block'})
        ], style={'display': 'inline-block', 'margin': '10px'}),
        html.Div([
            html.Label('Timespan:', style={'paddingRight': '10px'}),
            dcc.Dropdown(
                id='timespan_selector',
                options=['All', 'Last 3 Months', 'Last 6 Months', 'Last Year', 'YTD'],
                value='All',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'margin': '10px'}),
        html.Div([
            html.Label('Chart Type:', style={'paddingRight': '10px'}),
            dcc.Dropdown(
                id='chart_type_dropdown',
                options=['Scatter', 'Line', 'Bar'],
                value='Scatter',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'margin': '10px'}),
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
        ], style={'display': 'inline-block', 'margin': '10px'}),
    ], style={'textAlign': 'center', 'padding': '20px'}),
    html.Div(id='dashboard-container', style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '10px'}),
    html.Div(id='causal-analysis-section', children=[
        html.H2("Causal Analysis Results", style={'textAlign': 'center'}),
        dcc.Graph(id='causal-graph'),
        dcc.Markdown(id='strongest-causal-insight', style={'padding': '20px', 'textAlign': 'center'}),
        html.H4("All Causal Relationships", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='causal-table',
            columns=[
                {'name': 'Source', 'id': 'Source'},
                {'name': 'Target', 'id': 'Target'},
                {'name': 'Weight', 'id': 'Weight', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            ],
            sort_action='native',
            style_table={'overflowX': 'auto', 'margin': 'auto', 'width': '50%'},
            style_cell={'textAlign': 'left'},
        ),
        dcc.Markdown(id='causal-weights-note', style={'padding': '20px'})
    ], style={'padding': '10px'}),
    html.Div(id='forecast-section', children=[
        html.H2("Time Series Forecasting", style={'textAlign': 'center'}),
        html.Div([
            dcc.Dropdown(id='time-series-col-dropdown', placeholder='Select Time Column', style={'width': '200px', 'display': 'inline-block'}),
            dcc.Dropdown(id='target-col-dropdown', placeholder='Select Target Column', style={'width': '200px', 'display': 'inline-block'}),
            dcc.Dropdown(id='model-dropdown', options=['Linear Regression', 'ARIMA', 'Nowcasting'], placeholder='Select Model', style={'width': '200px', 'display': 'inline-block'}),
            dcc.Input(id='forecast-periods-input', type='number', placeholder='Periods to forecast', value=10, style={'width': '150px', 'display': 'inline-block'}),
            html.Button('Generate Forecast', id='generate-forecast-button', n_clicks=0, style={'display': 'inline-block'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px'}),
        dcc.Graph(id='forecast-graph')
    ])
])

@callback(
    [Output('main-container', 'style'),
     Output('h1-title', 'style'),
     Output('upload-data', 'style')],
    [Input('theme-selector', 'value')]
)
def update_theme(theme):
    if theme == 'dark':
        main_style = {'backgroundColor': '#111111', 'color': 'white', 'fontFamily': 'Arial, sans-serif', 'textAlign': 'center'}
        h1_style = {'textAlign': 'center', 'color': '#00CFD8'}
        upload_style = {
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px', 
            'textAlign': 'center', 'margin': '10px 0', 'backgroundColor': '#222222',
            'color': 'white'
        }
    else:
        main_style = {'backgroundColor': '#FFFFFF', 'color': 'black', 'fontFamily': 'Arial, sans-serif', 'textAlign': 'center'}
        h1_style = {'textAlign': 'center', 'color': '#007BFF'}
        upload_style = {
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px', 
            'textAlign': 'center', 'margin': '10px 0', 'backgroundColor': '#F0F0F0',
            'color': 'black'
        }
    return main_style, h1_style, upload_style

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(BytesIO(decoded))
        else:
            return None, None
        df.columns = [col.strip() for col in df.columns]
        return df, process_data(df.copy())
    except Exception as e:
        print(e)
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
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, None, [], [], [], [], []

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    df_raw, df_processed = None, None

    if trigger_id == 'upload-data' and contents:
        df_raw, df_processed = parse_contents(contents, filename)
    elif trigger_id == 'load-url-button' and url:
        df_raw, df_processed = get_data_from_url(url)

    if df_processed is None:
        return None, None, [], [], [], [], []

    axis_options, color_options, date_cols, numeric_cols = get_options(df_processed)
    processed_data_json = df_processed.to_json(date_format='iso', orient='split')
    raw_data_json = df_raw.to_json(date_format='iso', orient='split')
    
    return processed_data_json, raw_data_json, axis_options, axis_options, color_options, date_cols, numeric_cols

@callback(
    [Output('causal-graph', 'figure'),
     Output('strongest-causal-insight', 'children'),
     Output('causal-table', 'data'),
     Output('causal-weights-note', 'children'),
     Output('causal-table', 'style_data'),
     Output('causal-table', 'style_header')],
    [Input('store-raw-data', 'data'),
     Input('theme-selector', 'value')]
)
def update_causal_analysis(json_raw_data, theme):
    if json_raw_data is None:
        return go.Figure(), "", [], "", {}, {}

    df = pd.read_json(StringIO(json_raw_data), orient='split')
    causal_graph, strongest_insight, table_data, note = analyze_causal_structure(df, theme)
    
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

    return causal_graph, strongest_insight, table_data, note, style_data, style_header

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
        return [html.Div("Please upload a file or enter a URL to begin.", style={'textAlign': 'center', 'width': '100%', 'padding': '20px'})]

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
        return [html.Div("Please select X and Y axes to plot.", style={'textAlign': 'center', 'width': '100%', 'padding': '20px'})]

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
    
    return [dcc.Graph(figure=fig, style={'width': '100%'})]

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