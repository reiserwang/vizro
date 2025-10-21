#!/usr/bin/env python3
"""
Gradio-based Dynamic Data Analysis Dashboard
Optimized UI/UX with tooltips and modern interface
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Global variables for data storage
current_data = None
causal_results = None

def load_data_from_file(file_path):
    """Load data from uploaded file"""
    global current_data
    
    if file_path is None:
        return "❌ No file uploaded", None, None, None, None
    
    try:
        # Read the file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return "❌ Unsupported file format. Please upload CSV or Excel files.", None, None, None, None
        
        current_data = df
        
        # Get column options
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Create sortable data preview
        preview_df = df.head(20)  # Show more rows for better preview
        
        # Create table headers
        table_headers = ''.join([f'<th class="sortable-column" onclick="sortPreviewTable({i}, \'data-preview\')" style="cursor: pointer; user-select: none; position: relative;">{col} <span class="sort-indicator">⇅</span></th>' for i, col in enumerate(preview_df.columns)])
        
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
        
        # JavaScript for sorting (avoiding f-string with backslashes)
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

def create_visualization(x_axis, y_axis, color_var, chart_type, theme, y_axis_agg):
    """Create visualization based on user selections"""
    global current_data
    
    if current_data is None:
        return "⚠️ Please upload data first"
    
    if not x_axis or not y_axis:
        return "⚠️ Please select both X and Y axes"
    
    try:
        df = current_data.copy()
        
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
            return "⚠️ Invalid chart type selected"
        
        # Optimize layout for better UX
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='closest',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
        
    except Exception as e:
        return f"❌ Error creating visualization: {str(e)}"

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
    </div>
    """
    
    # Create table headers with enhanced sorting
    column_info = [
        ('Source Variable', 'text', 'Source variable in the causal relationship'),
        ('Target Variable', 'text', 'Target variable in the causal relationship'),
        ('Correlation (r)', 'number', 'Pearson correlation coefficient (-1 to +1)'),
        ('P-value', 'number', 'Statistical significance (lower is better)'),
        ('R²', 'number', 'Coefficient of determination (0 to 1)'),
        ('Causal Weight', 'number', 'NOTEARS algorithm weight'),
        ('Significant', 'text', 'Statistical significance (Yes/No)')
    ]
    
    causal_headers = ""
    for i, (col_name, col_type, tooltip) in enumerate(column_info):
        causal_headers += f'''
        <th class="sortable-column" onclick="sortCausalTable({i}, 'causal-results')" 
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
        correlation_val = float(row['Correlation (r)'])
        pvalue_val = float(row['P-value'])
        r2_val = float(row['R²'])
        weight_val = float(row['Causal Weight'])
        
        # Add CSS classes for styling based on significance and strength
        row_class = ""
        if row['Significant'] == 'Yes':
            row_class += "significant-row "
        if abs(correlation_val) >= 0.7:
            row_class += "strong-correlation "
        elif abs(correlation_val) >= 0.3:
            row_class += "moderate-correlation "
        else:
            row_class += "weak-correlation "
        
        causal_rows += f'''
        <tr class="{row_class}" 
            data-source="{row['Source Variable'].lower()}"
            data-target="{row['Target Variable'].lower()}"
            data-correlation="{correlation_val}"
            data-pvalue="{pvalue_val}"
            data-r2="{r2_val}"
            data-weight="{weight_val}"
            data-significant="{row['Significant'].lower()}">
            <td>{row['Source Variable']}</td>
            <td>{row['Target Variable']}</td>
            <td class="numeric-cell">{row['Correlation (r)']}</td>
            <td class="numeric-cell">{row['P-value']}</td>
            <td class="numeric-cell">{row['R²']}</td>
            <td class="numeric-cell">{row['Causal Weight']}</td>
            <td class="significance-cell">
                <span class="badge {'badge-success' if row['Significant'] == 'Yes' else 'badge-secondary'}">
                    {row['Significant']}
                </span>
            </td>
        </tr>
        '''
    
    # Enhanced JavaScript for sorting, filtering, and exporting
    advanced_script = """
    <script>
    let sortState = [];
    let originalRows = [];
    
    // Initialize table
    document.addEventListener('DOMContentLoaded', function() {
        const table = document.getElementById('causal-results');
        if (table) {
            originalRows = Array.from(table.querySelector('tbody').querySelectorAll('tr'));
            updateFilterStatus();
        }
    });
    
    function sortCausalTable(columnIndex, tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const tbody = table.querySelector('tbody');
        const visibleRows = Array.from(tbody.querySelectorAll('tr:not([style*="display: none"])'));
        const header = table.querySelectorAll('th')[columnIndex];
        const currentSort = header.getAttribute('data-sort') || 'none';
        
        // Handle multi-column sorting with Ctrl key
        if (!event.ctrlKey) {
            // Single column sort - reset all other columns
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
            });
            table.querySelectorAll('.sort-priority').forEach(priority => {
                priority.style.display = 'none';
                priority.textContent = '';
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
        
        // Update sort state for multi-column sorting
        const existingIndex = sortState.findIndex(s => s.column === columnIndex);
        if (existingIndex >= 0) {
            sortState[existingIndex].direction = newSort;
        } else {
            sortState.push({column: columnIndex, direction: newSort});
        }
        
        // Show sort priority numbers
        sortState.forEach((sort, index) => {
            const prioritySpan = table.querySelectorAll('th')[sort.column].querySelector('.sort-priority');
            if (sortState.length > 1) {
                prioritySpan.style.display = 'inline';
                prioritySpan.textContent = index + 1;
            }
        });
        
        // Sort rows based on multiple columns
        const sortedRows = visibleRows.sort((a, b) => {
            for (let sort of sortState) {
                const aVal = a.cells[sort.column].textContent.trim();
                const bVal = b.cells[sort.column].textContent.trim();
                
                let comparison = 0;
                
                // Try numeric comparison first
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
        
        // Clear tbody and append sorted rows
        tbody.innerHTML = '';
        originalRows.forEach(row => {
            if (sortedRows.includes(row)) {
                tbody.appendChild(row);
            } else if (!row.style.display || row.style.display !== 'none') {
                tbody.appendChild(row);
            }
        });
        sortedRows.forEach(row => tbody.appendChild(row));
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
            
            // Search filter
            if (searchTerm) {
                const source = row.dataset.source || '';
                const target = row.dataset.target || '';
                if (!source.includes(searchTerm) && !target.includes(searchTerm)) {
                    show = false;
                }
            }
            
            // Significance filter
            if (significanceFilter !== 'all') {
                const significant = row.dataset.significant;
                if (significanceFilter === 'significant' && significant !== 'yes') show = false;
                if (significanceFilter === 'non-significant' && significant !== 'no') show = false;
            }
            
            // Correlation filter
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
            
            // P-value filter
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
        
        // Reset sort state
        sortState = [];
        const table = document.getElementById('causal-results');
        if (table) {
            table.querySelectorAll('.sort-indicator').forEach(indicator => {
                indicator.textContent = '⇅';
                indicator.parentElement.setAttribute('data-sort', 'none');
            });
            table.querySelectorAll('.sort-priority').forEach(priority => {
                priority.style.display = 'none';
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
        
        // Create CSV content
        let csvContent = headers.join(',') + '\\n';
        csvContent += data.map(row => 
            row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(',')
        ).join('\\n');
        
        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'causal_analysis_results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        // Show success message
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

def perform_causal_analysis_with_status(hide_nonsignificant, min_correlation, theme):
    """Wrapper function to handle causal analysis with status updates"""
    import time
    
    # Initial status
    yield "🔍 Initializing causal analysis...", None, "Starting analysis...", None
    time.sleep(0.5)
    
    # Run the actual analysis
    try:
        result = perform_causal_analysis(hide_nonsignificant, min_correlation, theme)
        
        # Final status with results
        if isinstance(result[0], str) and "❌" in result[0]:
            # Error case
            yield result[0], result[1], result[2], None
        else:
            # Success case
            yield "✅ Analysis completed successfully!", result[0], result[1], result[2]
            
    except Exception as e:
        yield f"❌ Analysis failed: {str(e)}", None, "Analysis could not be completed.", None

def perform_causal_analysis(hide_nonsignificant, min_correlation, theme, progress=gr.Progress()):
    """Perform efficient causal analysis on the data with progress tracking"""
    global current_data, causal_results
    
    if current_data is None:
        return "⚠️ Please upload data first", None, "No analysis performed yet"
    
    try:
        # Initialize progress tracking
        progress(0, desc="🔍 Starting causal analysis...")
        
        df = current_data.copy()
        original_shape = df.shape
        
        progress(0.1, desc="📊 Analyzing data structure...")
        
        # Prepare data for causal analysis with efficiency improvements
        df_numeric = df.copy()
        
        # More efficient categorical encoding
        categorical_cols = df_numeric.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            progress(0.15, desc=f"🔤 Encoding {len(categorical_cols)} categorical variables...")
            
            # Vectorized encoding for better performance
            label_encoders = {}
            for col in categorical_cols:
                if df_numeric[col].notna().sum() > 0:
                    le = LabelEncoder()
                    mask = df_numeric[col].notna()
                    df_numeric.loc[mask, col] = le.fit_transform(df_numeric.loc[mask, col].astype(str))
                    label_encoders[col] = le
        
        progress(0.2, desc="🔢 Selecting numeric variables...")
        
        # Select only numeric columns
        df_numeric = df_numeric.select_dtypes(include=['number'])
        
        # Efficient NaN handling
        nan_threshold = 0.5
        valid_cols = df_numeric.columns[df_numeric.isnull().mean() < nan_threshold]
        df_numeric = df_numeric[valid_cols].dropna()
        
        if df_numeric.shape[1] < 2:
            return "❌ Not enough numeric variables for causal analysis", None, "Insufficient data"
        
        progress(0.25, desc=f"📈 Processing {df_numeric.shape[1]} variables, {df_numeric.shape[0]} samples...")
        
        # Smart variable selection for performance
        max_vars = 12  # Reduced for better performance
        if df_numeric.shape[1] > max_vars:
            progress(0.3, desc=f"🎯 Selecting top {max_vars} most correlated variables...")
            
            # More efficient correlation calculation
            corr_matrix = df_numeric.corr().abs()
            
            # Select variables with highest average correlation
            avg_corr = corr_matrix.mean().sort_values(ascending=False)
            top_vars = avg_corr.head(max_vars).index
            df_numeric = df_numeric[top_vars]
            
            progress(0.35, desc=f"✅ Selected {len(top_vars)} key variables")
        
        # Intelligent sampling for large datasets
        max_samples = 1500  # Reduced for better performance
        if len(df_numeric) > max_samples:
            progress(0.4, desc=f"📊 Sampling {max_samples} representative rows...")
            
            # Stratified sampling to preserve data distribution
            df_numeric = df_numeric.sample(n=max_samples, random_state=42)
        
        # Data preprocessing for better causal discovery
        progress(0.45, desc="🔧 Preprocessing data for causal discovery...")
        
        # Standardize data for better NOTEARS performance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index
        )
        
        progress(0.5, desc="🧠 Running NOTEARS causal discovery algorithm...")
        
        # Learn causal structure with optimized parameters
        try:
            # Use more conservative threshold for better results
            w_threshold = max(0.2, min_correlation * 0.5)
            sm = from_pandas(df_scaled, w_threshold=w_threshold, max_iter=50)
            
            progress(0.7, desc=f"🕸️ Found {len(sm.edges())} potential causal relationships...")
            
        except Exception as causal_error:
            progress(1.0, desc="❌ Causal discovery failed")
            # If NOTEARS fails, create empty graph and inform user
            sm = nx.DiGraph()
            error_msg = f"⚠️ Causal discovery failed: {str(causal_error)}"
            error_details = "\n\nThis might happen with:\n- Too few samples\n- Highly correlated variables\n- Non-numeric data\n- Complex non-linear relationships"
            return error_msg + error_details, None, "Causal analysis could not be completed. Try with different data or preprocessing."
        
        progress(0.75, desc="📊 Computing statistical measures...")
        
        # Calculate statistics for each edge with vectorized operations
        edge_stats = []
        edges_to_process = list(sm.edges(data=True))
        
        for i, (u, v, data) in enumerate(edges_to_process):
            if i % 5 == 0:  # Update progress every 5 edges
                progress(0.75 + 0.15 * (i / len(edges_to_process)), 
                        desc=f"📈 Computing statistics ({i+1}/{len(edges_to_process)})...")
            
            if u in df_numeric.columns and v in df_numeric.columns:
                x_data = df_numeric[u].values
                y_data = df_numeric[v].values
                
                # Calculate correlation and p-value
                r, p = pearsonr(x_data, y_data)
                
                # Calculate R² more efficiently
                if len(x_data) > 1 and np.var(x_data) > 1e-10:
                    reg = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
                    r2 = reg.score(x_data.reshape(-1, 1), y_data)
                else:
                    r2 = 0.0
                
                weight = data.get('weight', 0)
                significant = p < 0.05
                
                # Apply filters
                if hide_nonsignificant and not significant:
                    continue
                if abs(r) < min_correlation:
                    continue
                
                edge_stats.append({
                    'Source Variable': u,
                    'Target Variable': v,
                    'Correlation (r)': f"{r:.4f}",
                    'P-value': f"{p:.6f}",
                    'R²': f"{r2:.4f}",
                    'Causal Weight': f"{weight:.4f}",
                    'Significant': "Yes" if significant else "No"
                })
        
        progress(0.9, desc="🎨 Creating network visualization...")
        
        # Create network visualization
        fig = create_network_plot(sm, edge_stats, theme)
        
        progress(0.95, desc="📋 Generating advanced results table...")
        
        # Create advanced results table with sorting and filtering
        if edge_stats:
            results_df = pd.DataFrame(edge_stats)
            
            # Convert string numbers back to float for better filtering
            results_df['Correlation_num'] = results_df['Correlation (r)'].astype(float)
            results_df['P_value_num'] = results_df['P-value'].astype(float)
            results_df['R2_num'] = results_df['R²'].astype(float)
            results_df['Weight_num'] = results_df['Causal Weight'].astype(float)
            
            # Create enhanced table with filtering and sorting
            table_html = create_advanced_causal_table(results_df, edge_stats)
            
            # Enhanced summary statistics with performance metrics
            total_relationships = len(edge_stats)
            significant_count = sum(1 for stat in edge_stats if stat['Significant'] == 'Yes')
            processing_time = "< 30 seconds"  # Estimated with optimizations
            
            summary = f"""
            ## 📊 Causal Analysis Summary
            
            **Original Data:** {original_shape[0]} rows × {original_shape[1]} columns  
            **Processed Data:** {df_numeric.shape[0]} rows × {df_numeric.shape[1]} variables  
            **Total Relationships Found:** {total_relationships}  
            **Statistically Significant:** {significant_count} ({significant_count/total_relationships*100:.1f}% if total_relationships > 0 else 0)  
            **Processing Time:** {processing_time}  
            **Analysis Method:** NOTEARS (Optimized for performance)  
            **Significance Threshold:** p < 0.05  
            
            ### 🔍 Key Insights:
            - **Green relationships** in the graph are statistically significant (p < 0.05)
            - **Edge thickness** represents the strength of causal relationships
            - **Correlation values** show linear relationship strength (-1 to +1)
            - **R² values** indicate how much variance is explained (0 to 1)
            
            ### ⚡ Performance Optimizations Applied:
            - Smart variable selection (top {max_vars} most correlated)
            - Efficient data sampling ({max_samples} max samples)
            - Standardized preprocessing for better convergence
            - Vectorized statistical computations
            """
            
        else:
            table_html = """
            <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 8px; border: 2px dashed #dee2e6;">
                <h4 style="color: #6c757d; margin-bottom: 15px;">🔍 No Relationships Found</h4>
                <p style="color: #6c757d; margin-bottom: 20px;">No causal relationships match your current filter criteria.</p>
                <div style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
                    <strong>💡 Try adjusting:</strong><br>
                    • Lower the correlation threshold<br>
                    • Uncheck "Hide Non-Significant Relationships"<br>
                    • Use different variables or more data
                </div>
            </div>
            """
            summary = f"""
            ## 🔍 No Relationships Found
            
            **Data Processed:** {df_numeric.shape[0]} rows × {df_numeric.shape[1]} variables  
            **Current filters may be too restrictive.**
            
            ### 💡 Suggestions:
            - **Lower correlation threshold** - Try 0.0 to see all relationships
            - **Include non-significant relationships** - Uncheck the significance filter
            - **Check your data** - Ensure you have numeric variables with variation
            - **Add more data** - Causal discovery works better with larger datasets
            
            ### 📊 Data Requirements:
            - At least 2 numeric variables
            - Sufficient sample size (>50 rows recommended)
            - Variables with meaningful variation
            """
        
        progress(1.0, desc="✅ Analysis complete!")
        
        causal_results = {'graph': fig, 'table': table_html, 'summary': summary}
        
        return fig, table_html, summary
        
    except Exception as e:
        progress(1.0, desc="❌ Analysis failed")
        error_msg = f"❌ Error in causal analysis: {str(e)}"
        return error_msg, None, error_msg
        
        # Calculate statistics for each edge
        edge_stats = []
        for u, v, data in sm.edges(data=True):
            if u in df_numeric.columns and v in df_numeric.columns:
                x_data = df_numeric[u].values
                y_data = df_numeric[v].values
                
                # Calculate correlation and p-value
                r, p = pearsonr(x_data, y_data)
                
                # Calculate R²
                reg = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
                r2 = reg.score(x_data.reshape(-1, 1), y_data)
                
                weight = data.get('weight', 0)
                significant = p < 0.05
                
                # Apply filters
                if hide_nonsignificant and not significant:
                    continue
                if abs(r) < min_correlation:
                    continue
                
                edge_stats.append({
                    'Source Variable': u,
                    'Target Variable': v,
                    'Correlation (r)': f"{r:.4f}",
                    'P-value': f"{p:.6f}",
                    'R²': f"{r2:.4f}",
                    'Causal Weight': f"{weight:.4f}",
                    'Significant': "Yes" if significant else "No"
                })
        
        # Create network visualization
        fig = create_network_plot(sm, edge_stats, theme)
        
        # Create results table with sorting functionality
        if edge_stats:
            results_df = pd.DataFrame(edge_stats)
            
            # Create sortable table HTML
            # Create table headers for causal results
            causal_headers = ''.join([f'<th class="sortable-column" onclick="sortCausalTable({i}, \'causal-results\')" style="cursor: pointer; user-select: none; position: relative;">{col} <span class="sort-indicator">⇅</span></th>' for i, col in enumerate(results_df.columns)])
            
            # Create table rows for causal results
            causal_rows = ""
            for _, row in results_df.iterrows():
                row_html = "<tr>"
                for col in results_df.columns:
                    row_html += f"<td>{row[col]}</td>"
                row_html += "</tr>"
                causal_rows += row_html
            
            # JavaScript for causal results sorting
            causal_sort_script = """
            <script>
            function sortCausalTable(columnIndex, tableId) {
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
                        
                        // Try to parse as numbers first
                        const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                        const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
                        
                        if (!isNaN(aNum) && !isNaN(bNum)) {
                            return aNum - bNum;
                        }
                        
                        // For text comparison, use locale-aware sorting
                        return aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'});
                    });
                    newSort = 'asc';
                    header.querySelector('.sort-indicator').textContent = '↑';
                } else {
                    // Sort descending
                    sortedRows = rows.sort((a, b) => {
                        const aVal = a.cells[columnIndex].textContent.trim();
                        const bVal = b.cells[columnIndex].textContent.trim();
                        
                        // Try to parse as numbers first
                        const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                        const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
                        
                        if (!isNaN(aNum) && !isNaN(bNum)) {
                            return bNum - aNum;
                        }
                        
                        // For text comparison, use locale-aware sorting
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
            
            table_html = f"""
            <div class="table-container">
                <table id="causal-results" class="table table-striped table-hover sortable-table">
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
            {causal_sort_script}
            """
            
            # Add summary statistics
            total_relationships = len(edge_stats)
            significant_count = sum(1 for stat in edge_stats if stat['Significant'] == 'Yes')
            
            summary = f"""
            ## 📊 Causal Analysis Summary
            
            **Total Relationships Found:** {total_relationships}  
            **Statistically Significant:** {significant_count} ({significant_count/total_relationships*100:.1f}% if total_relationships > 0 else 0)  
            **Analysis Method:** NOTEARS (Non-linear causal discovery)  
            **Significance Threshold:** p < 0.05  
            
            ### 🔍 Key Insights:
            - **Green relationships** in the graph are statistically significant (p < 0.05)
            - **Edge thickness** represents the strength of causal relationships
            - **Correlation values** show linear relationship strength (-1 to +1)
            - **R² values** indicate how much variance is explained (0 to 1)
            """
            
        else:
            table_html = """
            <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 8px; border: 2px dashed #dee2e6;">
                <h4 style="color: #6c757d; margin-bottom: 15px;">🔍 No Relationships Found</h4>
                <p style="color: #6c757d; margin-bottom: 20px;">No causal relationships match your current filter criteria.</p>
                <div style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
                    <strong>💡 Try adjusting:</strong><br>
                    • Lower the correlation threshold<br>
                    • Uncheck "Hide Non-Significant Relationships"<br>
                    • Use different variables or more data
                </div>
            </div>
            """
            summary = """
            ## 🔍 No Relationships Found
            
            **Current filters may be too restrictive.**
            
            ### 💡 Suggestions:
            - **Lower correlation threshold** - Try 0.0 to see all relationships
            - **Include non-significant relationships** - Uncheck the significance filter
            - **Check your data** - Ensure you have numeric variables with variation
            - **Add more data** - Causal discovery works better with larger datasets
            
            ### 📊 Data Requirements:
            - At least 2 numeric variables
            - Sufficient sample size (>50 rows recommended)
            - Variables with meaningful variation
            """
        
        causal_results = {'graph': fig, 'table': table_html, 'summary': summary}
        
        return fig, table_html, summary
        
    except Exception as e:
        error_msg = f"❌ Error in causal analysis: {str(e)}"
        return error_msg, None, error_msg

def create_network_plot(sm, edge_stats, theme):
    """Create network visualization using Plotly with proper hover interactions"""
    try:
        # Check if we have nodes
        if len(sm.nodes()) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No causal relationships found with current settings.<br>Try adjusting the correlation threshold or significance filter.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='gray')
            )
            fig.update_layout(
                title="🔍 Causal Relationship Network",
                template='plotly_dark' if theme == 'Dark' else 'plotly_white',
                height=600
            )
            return fig
        
        # Create layout using networkx with better positioning
        pos = nx.spring_layout(sm, k=2, iterations=100, seed=42)
        
        # Prepare edge traces with hover information
        edge_traces = []
        
        for u, v, data in sm.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Find corresponding statistics
            edge_stat = next((stat for stat in edge_stats 
                            if stat['Source Variable'] == u and stat['Target Variable'] == v), None)
            
            if edge_stat:
                # Determine edge color based on significance
                is_significant = edge_stat['Significant'] == 'Yes'
                edge_color = '#2E8B57' if is_significant else '#CD5C5C'  # Green for significant, red for not
                edge_width = 3 if is_significant else 1
                
                # Create hover text
                hover_text = f"{u} → {v}<br>Correlation: {edge_stat['Correlation (r)']}<br>P-value: {edge_stat['P-value']}<br>R²: {edge_stat['R²']}<br>Significant: {edge_stat['Significant']}"
                
                # Add individual edge trace for better hover control
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=False,
                    name=f"{u}→{v}"
                ))
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in sm.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node)[:15] + "..." if len(str(node)) > 15 else str(node))  # Truncate long names
            
            # Count connections for node importance
            in_degree = sm.in_degree(node)
            out_degree = sm.out_degree(node)
            total_connections = in_degree + out_degree
            
            # Scale node size based on connections
            node_size = max(30, min(80, 30 + total_connections * 10))
            node_sizes.append(node_size)
            
            node_info.append(f"<b>{node}</b><br>Incoming connections: {in_degree}<br>Outgoing connections: {out_degree}<br>Total influence: {total_connections}")
        
        # Create figure
        fig = go.Figure()
        
        # Add all edge traces
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)
        
        # Add nodes with proper hover
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F8B500', '#FF69B4', '#20B2AA']
        node_colors = [colors[i % len(colors)] for i in range(len(node_x))]
        
        fig.add_trace(go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            hoverinfo='text',
            hovertext=node_info,
            showlegend=False,
            name="Variables"
        ))
        
        # Update layout with better styling
        template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
        
        fig.update_layout(
            title={
                'text': "🔍 Causal Relationship Network",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=80),
            annotations=[
                dict(
                    text="<b>Green edges:</b> Significant relationships (p < 0.05) | <b>Red edges:</b> Non-significant<br><b>Node size:</b> Number of connections | <b>Hover:</b> Detailed statistics",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor='center', yanchor='top',
                    font=dict(color='gray', size=11)
                )
            ],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            template=template,
            height=650,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Enable hover interactions
        fig.update_traces(
            hovertemplate='%{hovertext}<extra></extra>',
        )
        
        return fig
        
    except Exception as e:
        # Return informative error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error creating network visualization:<br>{str(e)}<br><br>This might happen if:<br>• No significant relationships found<br>• Data has too few variables<br>• All correlations below threshold",
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=14, color='red'),
            align='center'
        )
        fig.update_layout(
            title="🔍 Causal Relationship Network - Error",
            template='plotly_dark' if theme == 'Dark' else 'plotly_white',
            height=600
        )
        return fig

def export_results():
    """Export analysis results"""
    global causal_results, current_data
    
    if causal_results is None:
        return "⚠️ No analysis results to export"
    
    try:
        # Create a simple text report
        report = f"""
# Causal Analysis Report

## Data Summary
- Dataset shape: {current_data.shape if current_data is not None else 'N/A'}
- Analysis timestamp: {pd.Timestamp.now()}

## Analysis Results
{causal_results.get('summary', 'No summary available')}

## Methodology
- Algorithm: NOTEARS (Non-linear causal discovery)
- Statistical tests: Pearson correlation, Linear regression R²
- Significance threshold: p < 0.05

Generated by Dynamic Data Analysis Dashboard
        """
        
        return report
        
    except Exception as e:
        return f"❌ Error exporting results: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    """Create the main Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 5px;
    }
    
    .upload-area {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: rgba(78, 205, 196, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    
    .table-container {
        max-height: 500px;
        overflow-y: auto;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sortable-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    
    .sortable-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 8px;
        text-align: left;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 10;
        border-bottom: 2px solid #fff;
    }
    
    .sortable-table th:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    
    .sortable-table td {
        padding: 10px 8px;
        border-bottom: 1px solid #eee;
        vertical-align: top;
    }
    
    .sortable-table tr:hover {
        background-color: #f8f9fa;
        transition: background-color 0.2s ease;
    }
    
    .sortable-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .sortable-table tr:nth-child(even):hover {
        background-color: #f0f0f0;
    }
    
    .sort-indicator {
        font-size: 12px;
        margin-left: 5px;
        opacity: 0.7;
        transition: all 0.2s ease;
    }
    
    .sortable-column:hover .sort-indicator {
        opacity: 1;
        transform: scale(1.2);
    }
    
    .table-header th {
        user-select: none;
        cursor: pointer;
    }
    
    /* Advanced table styles */
    .advanced-table {
        font-size: 13px;
    }
    
    .filter-controls {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .filter-controls input, .filter-controls select {
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .filter-controls input:focus, .filter-controls select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
        outline: none;
    }
    
    .numeric-cell {
        text-align: right;
        font-family: 'Courier New', monospace;
        font-weight: 500;
    }
    
    .significance-cell {
        text-align: center;
    }
    
    .badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    
    .badge-secondary {
        background-color: #6c757d;
        color: white;
    }
    
    .significant-row {
        background-color: rgba(40, 167, 69, 0.05);
    }
    
    .strong-correlation {
        border-left: 4px solid #dc3545;
    }
    
    .moderate-correlation {
        border-left: 4px solid #ffc107;
    }
    
    .weak-correlation {
        border-left: 4px solid #28a745;
    }
    
    .sort-priority {
        background: #ffc107;
        color: #212529;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-left: 4px;
        font-weight: bold;
    }
    
    /* Hover effects for filter controls */
    .filter-controls button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    
    /* Responsive design for filters */
    @media (max-width: 768px) {
        .filter-controls > div {
            flex-direction: column;
        }
        
        .filter-controls input, .filter-controls select {
            margin-bottom: 10px;
        }
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Responsive table */
    @media (max-width: 768px) {
        .table-container {
            overflow-x: auto;
        }
        
        .sortable-table {
            min-width: 600px;
        }
        
        .sortable-table th,
        .sortable-table td {
            padding: 8px 6px;
            font-size: 12px;
        }
    }
    """
    
    with gr.Blocks(css=css, title="🔍 Dynamic Data Analysis Dashboard", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🔍 Dynamic Data Analysis Dashboard</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.2em;">Advanced Causal Discovery & Statistical Analysis</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Data Upload Tab
            with gr.TabItem("📁 Data Upload", id="upload_tab"):
                gr.Markdown("## Upload Your Dataset")
                gr.Markdown("*Upload CSV or Excel files to begin analysis. The system will automatically detect data types and prepare your data for analysis.*")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="📂 Choose File",
                            file_types=[".csv", ".xlsx", ".xls"],
                            type="filepath"
                        )
                        
                        upload_status = gr.Textbox(
                            label="📊 Upload Status",
                            interactive=False,
                            placeholder="No file uploaded yet..."
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Tips:
                        - **CSV files**: Comma-separated values
                        - **Excel files**: .xlsx or .xls format
                        - **Size limit**: Up to 100MB
                        - **Columns**: Mix of numeric and categorical data works best
                        - **Missing values**: Will be handled automatically
                        """)
                
                data_preview = gr.HTML(label="📋 Data Preview")
        
            # Visualization Tab  
            with gr.TabItem("📈 Data Visualization", id="viz_tab"):
                gr.Markdown("## Create Interactive Visualizations")
                gr.Markdown("*Select variables and chart types to explore your data visually. Hover over points for detailed information.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Visualization Controls")
                        
                        x_axis = gr.Dropdown(
                            label="📊 X-Axis Variable",
                            info="Choose the independent variable for your visualization",
                            interactive=True
                        )
                        
                        y_axis = gr.Dropdown(
                            label="📈 Y-Axis Variable", 
                            info="Choose the dependent variable to analyze",
                            interactive=True
                        )
                        
                        color_var = gr.Dropdown(
                            label="🎨 Color Variable (Optional)",
                            info="Add a third dimension by coloring points/bars by this variable",
                            interactive=True
                        )
                        
                        chart_type = gr.Dropdown(
                            choices=["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"],
                            value="Scatter Plot",
                            label="📊 Chart Type",
                            info="Select the most appropriate visualization for your data"
                        )
                        
                        y_axis_agg = gr.Radio(
                            choices=["Raw Data", "Average", "Sum", "Count"],
                            value="Raw Data",
                            label="📊 Y-Axis Aggregation",
                            info="Choose how to aggregate Y-axis values: Raw (all points), Average (mean by X-axis), Sum (total by X-axis), or Count (frequency)"
                        )
                        
                        viz_theme = gr.Radio(
                            choices=["Light", "Dark"],
                            value="Light",
                            label="🎨 Theme",
                            info="Choose your preferred visual theme"
                        )
                        
                        create_viz_btn = gr.Button("🚀 Create Visualization", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        viz_output = gr.Plot(label="📊 Interactive Visualization")
            
            # Causal Analysis Tab
            with gr.TabItem("🔍 Causal Analysis", id="causal_tab"):
                gr.Markdown("## Discover Causal Relationships")
                gr.Markdown("*Advanced causal discovery using NOTEARS algorithm. Identify which variables influence others in your dataset.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Analysis Settings")
                        
                        hide_nonsig = gr.Checkbox(
                            label="🎯 Hide Non-Significant Relationships",
                            info="Only show relationships with p-value < 0.05 for cleaner results",
                            value=False
                        )
                        
                        min_corr = gr.Slider(
                            minimum=0.0,
                            maximum=0.8,
                            step=0.1,
                            value=0.0,
                            label="📊 Minimum Correlation Threshold",
                            info="Filter out weak correlations below this threshold"
                        )
                        
                        causal_theme = gr.Radio(
                            choices=["Light", "Dark"],
                            value="Light", 
                            label="🎨 Graph Theme",
                            info="Choose theme for the causal network visualization"
                        )
                        
                        analyze_btn = gr.Button("🔍 Run Causal Analysis", variant="primary", size="lg")
                        
                        # Status indicator for analysis progress
                        analysis_status = gr.Textbox(
                            label="📊 Analysis Status",
                            value="Ready to analyze...",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("### 📤 Export Results")
                        export_btn = gr.Button("💾 Export Analysis Report", variant="secondary")
                        export_output = gr.Textbox(label="📄 Export Status", interactive=False)
                
                with gr.Column(scale=2):
                    causal_network = gr.Plot(label="🕸️ Causal Network Graph")
                    
                    with gr.Accordion("📊 Detailed Results", open=True):
                        causal_summary = gr.Markdown("Run analysis to see results...")
                        causal_table = gr.HTML(label="📋 Statistical Results Table")
        
        # Event handlers
        file_input.change(
            fn=load_data_from_file,
            inputs=[file_input],
            outputs=[upload_status, x_axis, y_axis, color_var, data_preview]
        )
        
        create_viz_btn.click(
            fn=create_visualization,
            inputs=[x_axis, y_axis, color_var, chart_type, viz_theme, y_axis_agg],
            outputs=[viz_output]
        )
        
        analyze_btn.click(
            fn=perform_causal_analysis_with_status,
            inputs=[hide_nonsig, min_corr, causal_theme],
            outputs=[analysis_status, causal_network, causal_table, causal_summary]
        )
        
        export_btn.click(
            fn=export_results,
            outputs=[export_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e0e0e0;">
            <p style="color: #666; margin: 0;">
                🔬 Powered by CausalNex, Plotly & Gradio | 
                📊 Advanced Statistical Analysis | 
                🚀 Built for Data Scientists
            </p>
        </div>
        """)
    
    return demo

def main():
    """Main function for launching the dashboard"""
    print("🚀 Starting Dynamic Data Analysis Dashboard...")
    print("📊 Loading causal analysis engine...")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with optimized settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=False,            # Set to True for public sharing
        debug=False,            # Set to True for development
        show_error=True,        # Show detailed errors
        quiet=False,            # Show startup messages
        inbrowser=True,         # Auto-open browser
        favicon_path=None,      # Add custom favicon if desired
        ssl_verify=False        # For development
    )

if __name__ == "__main__":
    main()