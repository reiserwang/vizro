from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import sys
import os
import json

app = FastAPI(title="Advanced Analytics API", description="API for Time Series Forecasting and Causal Discovery")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.data_handler import impute_missing_values, convert_date_columns
from engines.forecasting_engine import perform_forecasting

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
            
        return {
            "status": "success",
            "summary": summary,
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
