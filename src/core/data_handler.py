#!/usr/bin/env python3
"""
Data Loading and Processing Module
"""

import pandas as pd
import numpy as np
import gradio as gr
import io
import requests
import json
from core import dashboard_config
from core.dashboard_config import SUPPORTED_FILE_FORMATS

def load_data_from_file(file_obj):
    """Load data from uploaded file (path or object)"""
    
    if file_obj is None:
        return "❌ No file uploaded", None, None, None, None
    
    try:
        # Determine file path if it's a string, otherwise it's likely a file-like object or Gradio file wrapper
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

        df = None

        # CSV Handling with resilience
        if file_path.endswith('.csv'):
            # Try default reading
            try:
                df = pd.read_csv(file_obj)
            except UnicodeDecodeError:
                # Try common encodings
                encodings = ['latin1', 'iso-8859-1', 'cp1252']
                for enc in encodings:
                    try:
                        # Reset file pointer if possible
                        if hasattr(file_obj, 'seek'):
                            file_obj.seek(0)
                        df = pd.read_csv(file_obj, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    raise ValueError("Could not decode CSV file with standard encodings.")
            except pd.errors.ParserError:
                # Try python engine with sniffing
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, sep=None, engine='python')

        # Excel Handling
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_obj)

        # JSON Handling
        elif file_path.endswith('.json'):
            df = pd.read_json(file_obj)

        # Parquet Handling
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_obj)

        else:
            return "❌ Unsupported file format. Please upload CSV, Excel, JSON, or Parquet.", None, None, None, None
        
        # Clean data and convert dates
        df = impute_missing_values(df)
        df = convert_date_columns(df)
        
        if df is None:
             return "❌ Failed to load data.", None, None, None, None

        # Post-load processing
        dashboard_config.current_data = df
        
        # Get column options
        all_cols = df.columns.tolist()
        
        success_msg = f"✅ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns"
        
        return success_msg, gr.update(choices=all_cols, value=None), gr.update(choices=all_cols, value=None), gr.update(choices=all_cols, value=None), df
        
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", None, None, None, None

def load_data_from_url(url):
    """Load data from a URL"""
    if not url:
        return "❌ Please enter a URL", None, None, None, None

    # Security check: Prevent SSRF
    if not (url.startswith('http://') or url.startswith('https://')):
        return "❌ Invalid URL scheme. Only http:// and https:// are supported.", None, None, None, None
        
    # Additional SSRF protection - basic check for localhost/internal IPs
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        
        # Block common internal hostnames and IPs
        blocklist = [
            'localhost', '127.0.0.1', '0.0.0.0', '169.254.169.254',
            'host.docker.internal', 'metadata.google.internal',
            '10.', '172.16.', '172.17.', '172.18.', '172.19.', '172.2', '172.3',
            '192.168.', '::1', '[::1]'
        ]
        
        if any(hostname == blocked or hostname.startswith(blocked) for blocked in blocklist):
            return "❌ Access to internal network resources is blocked.", None, None, None, None
            
    except Exception as e:
        return f"❌ Invalid URL format: {str(e)}", None, None, None, None
        
    try:
        # Added timeout to prevent hanging requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Create a file-like object from content
        content = io.BytesIO(response.content)
        
        # Try to infer filename/extension from URL or headers
        if 'content-disposition' in response.headers:
            # Simple extraction (could be improved)
            fname = response.headers['content-disposition'].split('filename=')[-1].strip('"')
            content.name = fname
        else:
            content.name = url.split('/')[-1]
            if '?' in content.name:
                content.name = content.name.split('?')[0]

        # If no extension found, default to CSV if content looks like text, or try to peek
        if '.' not in content.name:
            # Fallback assumption
            content.name += '.csv'

        return load_data_from_file(content)
        
    except Exception as e:
         return f"❌ Error loading from URL: {str(e)}", None, None, None, None

def impute_missing_values(df):
    """Clean data by imputing missing values and handling infinite values"""
    # Replace inf with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Missing data imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric with median
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"✅ Imputed missing values in '{col}' with median")
            
    # Impute categorical with mode
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            print(f"✅ Imputed missing values in '{col}' with mode")
            
    return df

def convert_date_columns(dataframe):
    """Convert potential date columns to datetime"""
    for col in dataframe.columns:
        if col.lower() in ['date', 'time', 'timestamp'] or 'date' in col.lower():
            if not pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                try:
                    # Try to convert to datetime robustly
                    dataframe[col] = pd.to_datetime(dataframe[col], format='mixed', errors='coerce')
                    print(f"✅ Converted {col} to datetime")
                except Exception as e:
                    print(f"⚠️ Could not convert {col} to datetime: {e}")
    return dataframe

def validate_data_quality(df, target_var=None, intervention_var=None):
    """Validate data quality for analysis"""
    issues = []
    
    # Check for missing data
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if len(missing_cols) > 0:
        issues.append(f"Missing data in {len(missing_cols)} columns")
    
    # Check for constant variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = []
    for col in numeric_cols:
        if df[col].std() < 1e-10:
            constant_cols.append(col)
    
    if constant_cols:
        issues.append(f"Constant variables: {', '.join(constant_cols)}")
    
    # Check specific variables if provided
    if target_var and target_var in df.columns:
        target_variation = df[target_var].std()
        if target_variation < 1e-10:
            issues.append(f"Target variable '{target_var}' has no variation")
    
    if intervention_var and intervention_var in df.columns:
        intervention_variation = df[intervention_var].std()
        if intervention_variation < 1e-10:
            issues.append(f"Intervention variable '{intervention_var}' has no variation")
    
    return issues

def get_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {
        'shape': df.shape,
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'missing_data': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return summary

def update_forecast_dropdowns():
    """Update forecasting dropdown options when data is loaded"""
    
    if dashboard_config.current_data is None:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    
    # Get numeric columns for forecasting
    numeric_cols = dashboard_config.current_data.select_dtypes(include=[np.number]).columns.tolist()
    
    return (
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),
        gr.update(choices=numeric_cols, value=None)
    )

def update_causal_dropdowns():
    """Update causal analysis dropdown options when data is loaded"""
    
    if dashboard_config.current_data is None:
        return (gr.update(choices=[], value=None), gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None), gr.update(choices=[], value=None))
    
    # Get numeric columns for causal analysis
    numeric_cols = dashboard_config.current_data.select_dtypes(include=[np.number]).columns.tolist()
    
    return (
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # intervention_target
        gr.update(choices=numeric_cols, value=numeric_cols[1] if len(numeric_cols) > 1 else None),  # intervention_var
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # pathway_source
        gr.update(choices=numeric_cols, value=numeric_cols[1] if len(numeric_cols) > 1 else None)   # pathway_target
    )
