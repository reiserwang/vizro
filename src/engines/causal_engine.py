#!/usr/bin/env python3
"""
Causal Analysis Engine Module
This module serves as the main entry point for all causal analysis functionalities,
delegating tasks to specialized sub-modules.
"""

import time
from ..core import dashboard_config
from .causal_analysis import perform_causal_analysis
from .causal_intervention import perform_causal_intervention_analysis

def perform_causal_analysis_with_status(hide_nonsignificant, min_correlation, theme, show_all_relationships):
    """Wrapper function to handle causal analysis with status updates"""
    try:
        # Yield initial status
        yield "🔍 Starting causal analysis...", None, None, "Initializing analysis..."
        time.sleep(0.1)
        
        # Run the actual analysis by calling the refactored function
        network_fig, table_html, summary = perform_causal_analysis(
            hide_nonsignificant, min_correlation, theme, show_all_relationships
        )
        
        # Yield final result
        yield summary, network_fig, table_html, "✅ Analysis complete!"
            
    except Exception as e:
        yield f"❌ Analysis failed: {str(e)}", None, None, "Analysis could not be completed."

def export_results():
    """Export analysis results"""
    if dashboard_config.causal_results is None:
        return "❌ No analysis results to export. Please run causal analysis first."
    
    try:
        # Create export data
        results_df = dashboard_config.causal_results['results_df']
        
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"causal_analysis_results_{timestamp}.csv"
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        
        # Create summary
        summary_stats = dashboard_config.causal_results['summary_stats']
        
        export_summary = f"""
        ✅ **Results exported successfully!**
        
        **📁 File:** {filename}  
        **📊 Total relationships:** {summary_stats['total_relationships']}  
        **✅ Significant relationships:** {summary_stats['significant_relationships']}  
        **💪 Strong relationships:** {summary_stats['strong_relationships']}  
        
        **📋 Exported columns:**
        - Source variable
        - Target variable  
        - Correlation coefficient
        - P-value
        - R-squared
        - Significance level
        - Relationship strength
        """
        
        return export_summary
        
    except Exception as e:
        return f"❌ Export failed: {str(e)}"
