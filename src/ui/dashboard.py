#!/usr/bin/env python3
"""
Refactored Gradio-based Dynamic Data Analysis Dashboard
Modular architecture with separated concerns
"""

import gradio as gr
import pandas as pd
import numpy as np

# Import custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.dashboard_config import (
    VIZRO_AVAILABLE, 
    STANDARD_CHART_TYPES, 
    VIZRO_ENHANCED_CHART_TYPES,
    FORECASTING_MODELS,
    Y_AXIS_AGGREGATIONS
)

from core.data_handler import (
    load_data_from_file,
    load_data_from_url,
    update_forecast_dropdowns,
    update_causal_dropdowns
)

from engines.visualization_engine import (
    create_visualization,
    create_vizro_enhanced_visualization,
    create_data_insights_dashboard
)

from engines.forecasting_engine import (
    perform_forecasting
)

from engines.causal_engine import (
    perform_causal_analysis_with_status,
    export_results
)
from engines.causal_intervention import perform_causal_intervention_analysis

from ui.components.config_tab import create_config_tab
from ui.components.visualization_tab import create_visualization_tab
from ui.components.analysis_tab import create_analysis_tabs

def create_gradio_interface():
    """Create the main Gradio interface with modular components"""
    
    # Custom CSS for enhanced styling using Gradio's native CSS variables
    # This automatically supports Day/Night mode (light/dark themes)
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .table-container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        margin: 10px 0;
        background-color: var(--background-fill-primary);
    }
    
    .sortable-table {
        width: 100%;
        border-collapse: collapse;
        color: var(--body-text-color);
    }
    
    .sortable-table th {
        background-color: var(--background-fill-secondary);
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid var(--border-color-primary);
        position: sticky;
        top: 0;
        z-index: 10;
        font-weight: 600;
    }
    
    .sortable-table td {
        padding: 8px 12px;
        border-bottom: 1px solid var(--border-color-primary);
    }
    
    .sortable-table tr:hover {
        background-color: var(--background-fill-secondary);
    }
    
    .sort-indicator {
        float: right;
        color: var(--body-text-color-subdued);
        font-weight: normal;
    }
    
    .correlation-bar {
        height: 20px;
        border-radius: 3px;
        display: inline-block;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    .filter-controls {
        background: var(--background-fill-secondary);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid var(--border-color-primary);
    }
    """
    
    with gr.Blocks(title="🔍 Dynamic Data Analysis Dashboard") as demo:
        gr.HTML(f"<style>{custom_css}</style>", visible=False)
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🔍 Dynamic Data Analysis Dashboard</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Advanced Causal Discovery • Professional Visualizations • Comprehensive Forecasting
            </p>
        </div>
        """)
        
        # Modular Tabs
        with gr.Tabs():
            config_components = create_config_tab()
            viz_components = create_visualization_tab(VIZRO_AVAILABLE, STANDARD_CHART_TYPES, VIZRO_ENHANCED_CHART_TYPES, Y_AXIS_AGGREGATIONS)
            analysis_components = create_analysis_tabs(FORECASTING_MODELS)
            
        # Event handlers
        config_components['file_input'].change(
            fn=load_data_from_file,
            inputs=[config_components['file_input']],
            outputs=[
                config_components['upload_status'], 
                viz_components['x_axis'], 
                viz_components['y_axis'], 
                viz_components['color_var'], 
                config_components['data_preview']
            ]
        ).then(
            fn=lambda file_path: update_forecast_dropdowns() if file_path else (gr.update(), gr.update()),
            inputs=[config_components['file_input']],
            outputs=[analysis_components['forecast_target'], analysis_components['forecast_additional']]
        ).then(
            fn=lambda file_path: update_causal_dropdowns() if file_path else (gr.update(), gr.update(), gr.update(), gr.update()),
            inputs=[config_components['file_input']],
            outputs=[analysis_components['intervention_target'], analysis_components['intervention_var'], analysis_components['intervention_target'], analysis_components['intervention_var']]
        )

        config_components['load_url_btn'].click(
            fn=load_data_from_url,
            inputs=[config_components['url_input']],
            outputs=[
                config_components['upload_status'], 
                viz_components['x_axis'], 
                viz_components['y_axis'], 
                viz_components['color_var'], 
                config_components['data_preview']
            ]
        ).then(
            fn=lambda url: update_forecast_dropdowns() if url else (gr.update(), gr.update()),
            inputs=[config_components['url_input']],
            outputs=[analysis_components['forecast_target'], analysis_components['forecast_additional']]
        ).then(
            fn=lambda url: update_causal_dropdowns() if url else (gr.update(), gr.update(), gr.update(), gr.update()),
            inputs=[config_components['url_input']],
            outputs=[
                analysis_components['intervention_target'], analysis_components['intervention_var'], 
                analysis_components['intervention_target'], analysis_components['intervention_var']
            ]
        )
        
        def update_correlation_window_visibility(chart_type_selection):
            return gr.update(visible=chart_type_selection == 'Correlation Heatmap')

        viz_components['chart_type'].change(
            fn=update_correlation_window_visibility,
            inputs=[viz_components['chart_type']],
            outputs=[viz_components['correlation_window']]
        )

        viz_components['create_viz_btn'].click(
            fn=create_vizro_enhanced_visualization if VIZRO_AVAILABLE else create_visualization,
            inputs=[
                viz_components['x_axis'], viz_components['y_axis'], viz_components['color_var'], 
                viz_components['chart_type'], viz_components['viz_theme'], viz_components['y_axis_agg'], 
                viz_components['correlation_window']
            ],
            outputs=[viz_components['viz_output']]
        )
        
        if VIZRO_AVAILABLE and viz_components.get('insights_btn'):
            viz_components['insights_btn'].click(
                fn=create_data_insights_dashboard,
                outputs=[viz_components['data_insights']]
            )
        
        analysis_components['analyze_btn'].click(
            fn=perform_causal_analysis_with_status,
            inputs=[
                analysis_components['hide_nonsig'], analysis_components['min_corr'], 
                analysis_components['causal_theme'], analysis_components['show_all_relationships']
            ],
            outputs=[
                analysis_components['analysis_status'], analysis_components['causal_network'], 
                analysis_components['causal_table'], analysis_components['causal_summary']
            ]
        )
        
        analysis_components['export_btn'].click(
            fn=export_results,
            outputs=[analysis_components['export_output']]
        )
        

        # Forecasting event handler
        analysis_components['forecast_btn'].click(
            fn=perform_forecasting,
            inputs=[
                analysis_components['forecast_target'], analysis_components['forecast_additional'], 
                analysis_components['forecast_model'], analysis_components['forecast_periods'], 
                analysis_components['seasonal_period'], analysis_components['confidence_level']
            ],
            outputs=[
                analysis_components['forecast_plot'], analysis_components['forecast_summary'], 
                analysis_components['forecast_metrics']
            ]
        )
        
        # Advanced causal analysis event handlers
        analysis_components['intervention_btn'].click(
            fn=perform_causal_intervention_analysis,
            inputs=[
                analysis_components['intervention_target'], analysis_components['intervention_var'], 
                analysis_components['intervention_value']
            ],
            outputs=[analysis_components['intervention_results'], analysis_components['intervention_status']]
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
    """Main function for launching the refactored dashboard"""
    print("🚀 Starting Refactored Dynamic Data Analysis Dashboard...")
    print("📊 Loading modular components...")
    
    # Print module status
    print(f"✅ Configuration loaded")
    print(f"✅ Data handler loaded")
    print(f"✅ Visualization engine loaded")
    print(f"✅ Forecasting engine loaded") 
    print(f"✅ Causal analysis engine loaded")
    
    if VIZRO_AVAILABLE:
        print("✅ Vizro integration enabled - Enhanced visualizations available!")
    else:
        print("⚠️ Vizro not available - Using standard visualizations")
    
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
