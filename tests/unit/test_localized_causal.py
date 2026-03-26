import pytest
from fastapi.testclient import TestClient
from src.api.routes import app
import pandas as pd
import numpy as np
import io
import json

client = TestClient(app)

def test_localized_causal_analysis():
    # Setup data with a clear causal link A -> B
    np.random.seed(42)
    A = np.random.normal(100, 10, 100)
    B = A * 2.5 + np.random.normal(0, 2, 100)
    C = np.random.normal(50, 5, 100)
    
    df = pd.DataFrame({'A_cause': A, 'B_effect': B, 'C_random': C})
    f = io.BytesIO()
    df.to_csv(f, index=False)
    f.seek(0)
    
    # Upload
    upload_res = client.post('/api/v1/upload', files={'file': ('causal_test.csv', f, 'text/csv')})
    assert upload_res.status_code == 200
    
    # Request localized causal analysis targeting only A and B
    req_body = {
        'x_axis': 'A_cause',
        'y_axis': 'B_effect',
        'color_var': '',
        'chart_type': 'Enhanced Scatter Plot',
        'theme': 'Light',
        'y_axis_agg': 'Raw Data',
        'correlation_window': 0
    }
    
    res = client.post('/api/v1/visualize/causal', json=req_body)
    
    # Depending on randomness, 2 variables might fail NOTEARS cyclic constraints or yield a valid graph
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'success'
    
    if "Analysis Stopped" not in data['summary']:
        assert 'plot' in data
        assert 'table' in data
        assert 'summary' in data
        # Ensure it parsed correctly, C_random should not be in the results
        assert 'A_cause' in data['summary'] or 'B_effect' in data['summary']
        assert 'C_random' not in data['summary']
    else:
        assert "No causal" in data['summary'] or "Cyclic structure" in data['summary']

def test_localized_causal_categorical():
    # Setup data with a clear causal link Categorical -> Numeric -> Categorical
    np.random.seed(42)
    regions = np.random.choice(['North', 'South', 'East', 'West'], size=100)
    # The categorical region drives the numeric salary
    salary_map = {'North': 5000, 'South': 4000, 'East': 4500, 'West': 6000}
    salary = np.array([salary_map[r] for r in regions]) + np.random.normal(0, 500, 100)
    
    df = pd.DataFrame({'Region_Cat': regions, 'Salary_Num': salary})
    f = io.BytesIO()
    df.to_csv(f, index=False)
    f.seek(0)
    
    upload_res = client.post('/api/v1/upload', files={'file': ('cat_test.csv', f, 'text/csv')})
    assert upload_res.status_code == 200
    
    req_body = {
        'x_axis': 'Region_Cat',
        'y_axis': 'Salary_Num',
        'color_var': '',
        'chart_type': 'Enhanced Scatter Plot',
        'theme': 'Light',
        'y_axis_agg': 'Raw Data',
        'correlation_window': 0
    }
    
    res = client.post('/api/v1/visualize/causal', json=req_body)
    
    # Should automatically factorize Region_Cat
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'success'

    if "Analysis Stopped" in data['summary']:
        assert "No causal" in data['summary'] or "Cyclic structure" in data['summary']
    else:
        assert 'Region_Cat' in data['summary'] or 'Salary_Num' in data['summary']

def test_localized_causal_nan_infinity():
    # Setup data with NaNs and Infinite values to ensure the engine gracefully scrubs them
    np.random.seed(42)
    A = np.random.normal(100, 10, 100)
    B = A * 2.5 + np.random.normal(0, 2, 100)
    
    # Introduce NaNs and Infs
    A[10] = np.nan
    B[20] = np.inf
    B[21] = -np.inf
    
    df = pd.DataFrame({'A_cause': A, 'B_effect': B})
    f = io.BytesIO()
    df.to_csv(f, index=False)
    f.seek(0)
    
    upload_res = client.post('/api/v1/upload', files={'file': ('nan_inf.csv', f, 'text/csv')})
    assert upload_res.status_code == 200
    
    req_body = {
        'x_axis': 'A_cause',
        'y_axis': 'B_effect',
        'color_var': '',
        'chart_type': 'Enhanced Scatter Plot',
        'theme': 'Light',
        'y_axis_agg': 'Raw Data',
        'correlation_window': 0
    }
    
    res = client.post('/api/v1/visualize/causal', json=req_body)
    
    # Engine should replace NaNs/Infs with median and process normally
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'success'

    if "Analysis Stopped" in data['summary']:
        assert "No causal" in data['summary'] or "Cyclic structure" in data['summary']
    else:
        assert 'A_cause' in data['summary'] or 'B_effect' in data['summary']
