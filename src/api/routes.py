from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import sys
import os
import json

app = FastAPI(title="Advanced Analytics API", description="API for Time Series Forecasting and Causal Discovery")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_dir = sys._MEIPASS
        frontend_path = os.path.join(base_dir, 'src', 'ui', 'frontend')
    else:
        frontend_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'frontend')
        
    if os.path.exists(frontend_path):
        app.mount("/ui", StaticFiles(directory=frontend_path, html=True), name="frontend")
    else:
        print(f"Warning: Frontend directory not found at {frontend_path}")
except Exception as e:
    print(f"Warning: Could not mount frontend directory: {e}")
from core.data_handler import impute_missing_values, convert_date_columns, load_data_from_url as dh_load_url
from engines.forecasting_engine import perform_forecasting
from engines.visualization_engine import create_vizro_enhanced_visualization
from engines.causal_intervention import perform_causal_intervention_analysis

# In the original app, causal_engine is used but it yields. We can import the core function if we need to return directly.
try:
    from engines.causal_analysis import perform_causal_analysis
except ImportError:
    # Fallback to importing from causal_engine if it got merged
    pass

from pydantic import BaseModel
from typing import Optional, List, Any
import base64
import struct
import math

def _sanitize_floats(arr):
    return [None if math.isnan(x) or math.isinf(x) else x for x in arr]

def clean_plotly_bdata(obj):
    """
    Recursively walk the parsed Plotly JSON to unpack any base64 'bdata'
    arrays back into native float/int lists. This is required because
    Plotly.to_json() encodes numpy sequences in a way that standard
    Plotly.js cannot natively decode in the frontend, leading to sequential
    index values taking their place in bar charts.
    """
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            raw = base64.b64decode(obj['bdata'])
            dt = obj['dtype']
            if dt == 'f8':
                return _sanitize_floats(struct.unpack(f'<{len(raw)//8}d', raw))
            elif dt == 'f4':
                return _sanitize_floats(struct.unpack(f'<{len(raw)//4}f', raw))
            elif dt == 'i8':
                return list(struct.unpack(f'<{len(raw)//8}q', raw))
            elif dt == 'i4':
                return list(struct.unpack(f'<{len(raw)//4}i', raw))
        return {k: clean_plotly_bdata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_plotly_bdata(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

class VizRequest(BaseModel):
    x_axis: str
    y_axis: str
    color_var: Optional[str] = None
    chart_type: str = "Enhanced Scatter Plot"
    theme: str = "Light"
    y_axis_agg: str = "Raw Data"
    correlation_window: int = 0

class CausalRequest(BaseModel):
    hide_nonsignificant: bool = True
    min_correlation: float = 0.3
    causal_theme: str = "Light"
    show_all_relationships: bool = False

class InterventionRequest(BaseModel):
    target_var: str
    intervention_var: str
    intervention_value: float
@app.get("/")
def read_root():
    return {"message": "Welcome to Dynamic Data Analysis Dashboard API"}

@app.post("/api/v1/forecast")
async def forecast(
    target_var: str, 
    model_type: str = "Linear Regression", 
    periods: int = 12, 
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, JSON or Parquet.")

        df = impute_missing_values(df)
        df = convert_date_columns(df)
        
        # Patch dashboard config temporarily for downstream functions
        from core import dashboard_config
        dashboard_config.current_data = df
        
        # Call forecasting engine
        fig, summary, metrics = perform_forecasting(
            target_var=target_var,
            additional_vars=[],
            model_type=model_type,
            periods=periods,
            seasonal_period=12,
            confidence_level=0.95
        )

        if not fig:
            # If fig is None, perform_forecasting returns error message in summary
            raise HTTPException(status_code=400, detail=summary)
            
        import json
        return {
            "status": "success",
            "plot": clean_plotly_bdata(json.loads(fig.to_json())),
            "summary": summary,
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        df = impute_missing_values(df)
        df = convert_date_columns(df)
        
        from core import dashboard_config
        dashboard_config.current_data = df
        
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        return {
            "status": "success",
            "message": f"Loaded {len(df)} rows.",
            "columns": cols,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UrlRequest(BaseModel):
    url: str

@app.post("/api/v1/load-url")
def load_url(req: UrlRequest):
    try:
        msg, _, _, _, df = dh_load_url(req.url)
        if df is None:
            raise HTTPException(status_code=400, detail=msg)
            
        from core import dashboard_config
        dashboard_config.current_data = df
        
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        return {
            "status": "success",
            "message": msg,
            "columns": cols,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/visualize")
def visualize(req: VizRequest):
    try:
        from core import dashboard_config
        if dashboard_config.current_data is None:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a dataset first.")

        fig = create_vizro_enhanced_visualization(
            x_axis=req.x_axis,
            y_axis=req.y_axis,
            color_var=req.color_var,
            chart_type=req.chart_type,
            theme=req.theme,
            y_axis_agg=req.y_axis_agg,
            correlation_window=req.correlation_window
        )
        if fig is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not create '{req.chart_type}' chart. "
                       f"Check that '{req.x_axis}' and '{req.y_axis}' exist and have "
                       f"compatible types for this chart type."
            )

        import json
        return {
            "status": "success",
            "plot": clean_plotly_bdata(json.loads(fig.to_json()))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/visualize/causal")
def visualize_causal(req: VizRequest):
    try:
        from core import dashboard_config
        if dashboard_config.current_data is None:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a dataset first.")

        # Extract target variables from the visualization request
        target_vars = []
        for var in [req.x_axis, req.y_axis, req.color_var]:
            if var and var.lower() != "none" and var not in target_vars:
                target_vars.append(var)

        from engines.causal_analysis import perform_causal_analysis
        fig, table, summary = perform_causal_analysis(
            hide_nonsignificant=True,
            min_correlation=0.1,  # Lowered for sparse local variables
            theme=req.theme,
            show_all_relationships=True,
            progress=lambda x, desc=None: None,
            target_vars=target_vars
        )

        if not fig:
            raise HTTPException(status_code=400, detail=summary)

        import json
        return {
            "status": "success",
            "plot": clean_plotly_bdata(json.loads(fig.to_json())),
            "table": table,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/causal")
def causal_analysis(req: CausalRequest):
    try:
        from core import dashboard_config
        if dashboard_config.current_data is None:
            raise HTTPException(status_code=400, detail="No data available on the backend server. Please re-upload your dataset (session expired).")
            
        from engines.causal_analysis import perform_causal_analysis
        
        # progress argument is usually positional or kwarg, let's provide None if no UI
        fig, table, summary = perform_causal_analysis(
            req.hide_nonsignificant, 
            req.min_correlation, 
            req.causal_theme, 
            req.show_all_relationships,
            progress=lambda x, desc=None: None
        )
        if not fig:
            raise HTTPException(status_code=400, detail=summary)
            
        import json
        return {
            "status": "success",
            "plot": clean_plotly_bdata(json.loads(fig.to_json())),
            "table": table,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/intervention")
def intervention(req: InterventionRequest):
    try:
        def dummy_progress(*args, **kwargs): pass
        
        html_out, status = perform_causal_intervention_analysis(
            req.target_var,
            req.intervention_var,
            req.intervention_value,
            progress=dummy_progress
        )
        if "❌" in html_out or html_out.startswith("❌"):
            raise HTTPException(status_code=400, detail=html_out)
            
        return {
            "status": "success",
            "html": html_out,
            "message": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
