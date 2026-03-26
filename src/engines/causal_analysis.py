#!/usr/bin/env python3
"""
Causal Analysis Module
Handles the main causal discovery workflow.
"""

import pandas as pd
import numpy as np
import html
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
try:
    import gradio as gr
    _progress = gr.Progress()
except ImportError:
    gr = None
    _progress = None

from core import dashboard_config
from core.config import CAUSAL_ANALYSIS_PARAMS
from .causal_network_utils import has_cycles, resolve_cycles, create_network_plot

def perform_causal_analysis(hide_nonsignificant, min_correlation, theme, show_all_relationships=False, progress=_progress, target_vars=None):
    """Perform efficient causal analysis on the data with progress tracking. Optional target_vars restricts analysis."""

    try:
        def create_empty_result(message):
            import plotly.graph_objects as go
            fig = go.Figure()
            clean_msg = message.replace('❌ ', '').strip().replace('. ', '.<br>')
            fig.add_annotation(
                text=clean_msg, xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="gray"), align="center"
            )
            fig.update_layout(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            return fig, "<p class='text-on-surface-variant text-sm text-center py-4'>No relationship data available to tabulate.</p>", f"### ℹ️ Analysis Stopped\n{clean_msg.replace('<br>', ' ')}"

        progress(0.05, desc="🔍 Loading and validating data...")

        if dashboard_config.current_data is None:
            return create_empty_result("❌ No data loaded. Please upload a dataset first.")

        if target_vars:
            # Explicit targeted analysis: include all requested variables
            # Auto-encode categorical strings into numeric factorization indices
            df_subset = dashboard_config.current_data[target_vars].copy()
            for col in target_vars:
                if not pd.api.types.is_numeric_dtype(df_subset[col]):
                    df_subset[col] = pd.factorize(df_subset[col])[0]
            
            df_numeric = df_subset
            if len(df_numeric.columns) < 2:
                return create_empty_result("❌ Causal analysis requires at least two variables (X and Y parameters).")
        else:
            # Global analysis: only search pre-existing numeric columns for performance
            df_numeric = dashboard_config.current_data.select_dtypes(include=[np.number])

        if df_numeric.empty:
            return create_empty_result("❌ No numeric columns found. Please ensure your data contains numeric variables for causal analysis.")

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
            return create_empty_result("❌ No variables with sufficient variation found. Please check your data quality.")

        # Handle NaNs and Infinity
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        if df_numeric.isna().any().any():
            progress(0.18, desc="🧹 Cleaning missing values...")
            df_numeric = df_numeric.fillna(df_numeric.median())

        # Standardize the data for better NOTEARS performance
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index
        )

        progress(0.3, desc="🧠 Building causal structure (NOTEARS)...")

        # Build causal structure using NOTEARS with optimized parameters and cycle handling
        try:
            sm = from_pandas(
                df_scaled,
                max_iter=CAUSAL_ANALYSIS_PARAMS['max_iter'],
                h_tol=CAUSAL_ANALYSIS_PARAMS['h_tol'],
                w_threshold=CAUSAL_ANALYSIS_PARAMS['w_threshold']
            )

            # Check for cycles and resolve if necessary
            if has_cycles(sm):
                print("⚠️ Detected cycles in causal structure, applying resolution...")
                sm = resolve_cycles(sm, df_numeric)

        except Exception as e:
            if "not acyclic" in str(e) or "cycle" in str(e).lower():
                return create_empty_result(f"❌ Causal analysis failed. Cyclic structure detected from bi-directional dependencies. Try adjusting threshold.")
            else:
                raise e

        progress(0.5, desc="🔗 Identifying causal relationships...")

        # Get edges and calculate statistics
        edges = sm.edges()

        if not edges:
            return create_empty_result("❌ No causal relationships found. Try adjusting the minimum correlation threshold or check data quality.")

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
            return create_empty_result("❌ No valid relationships found after statistical analysis.")

        # Filter results based on user preferences
        if hide_nonsignificant:
            results_df = results_df[results_df['p_value'] < 0.05]

        if min_correlation > 0:
            results_df = results_df[results_df['abs_correlation'] >= min_correlation]

        if results_df.empty:
            return create_empty_result(f"❌ No relationships found meeting criteria (p < 0.05, |r| >= {min_correlation}). Try lowering the minimum correlation threshold.")

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
            summary += f"- **{html.escape(str(row['source']))} {direction} {html.escape(str(row['target']))}**: r = {row['correlation']:.3f} (p = {row['p_value']:.3f})\n"

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

        safe_source = html.escape(str(row['source']))
        safe_target = html.escape(str(row['target']))

        causal_rows += f'''
        <tr class="{row_class}"
            data-source="{safe_source.lower()}"
            data-target="{safe_target.lower()}"
            data-correlation="{correlation_val}"
            data-pvalue="{pvalue_val}"
            data-r2="{r2_val}"
            data-significant="{row['significance'].lower()}">
            <td>{safe_source}</td>
            <td>{safe_target}</td>
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
