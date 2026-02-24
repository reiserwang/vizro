#!/usr/bin/env python3
"""
Data Loading and Processing Module
"""

import pandas as pd
import numpy as np
import gradio as gr
from . import dashboard_config
from .dashboard_config import SUPPORTED_FILE_FORMATS

def load_data_from_file(file_path):
    """Load data from uploaded file"""
    
    if file_path is None:
        return "❌ No file uploaded", None, None, None, None
    
    try:
        # Read the file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return "❌ Unsupported file format. Please upload CSV, Excel, JSON or Parquet files.", None, None, None, None
        
        # Clean data and convert dates
        df = impute_missing_values(df)
        df = convert_date_columns(df)
        
        dashboard_config.current_data = df
        
        # Get column options
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Create sortable data preview
        preview_df = df.head(20)  # Show more rows for better preview
        
        # Create table headers
        table_headers = ''.join([
            f'<th class="sortable-column" onclick="sortPreviewTable({i}, \'data-preview\')" '
            f'style="cursor: pointer; user-select: none; position: relative;">'
            f'{col} <span class="sort-indicator">⇅</span></th>' 
            for i, col in enumerate(preview_df.columns)
        ])
        
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
        
        # JavaScript for sorting
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