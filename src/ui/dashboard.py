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
    
    # Custom CSS for enhanced styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .table-container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .sortable-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .sortable-table th {
        background-color: #f8f9fa;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .sortable-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #dee2e6;
    }
    
    .sortable-table tr:hover {
        background-color: #f5f5f5;
    }
    
    .sort-indicator {
        float: right;
        color: #6c757d;
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
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #dee2e6;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🔍 Dynamic Data Analysis Dashboard") as demo:
        
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
            
            # The config tab is handled by create_config_tab()
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0;">🎨 Visualization Controls</h4>
                            <p>Create professional charts with advanced features</p>
                        </div>
                        """)
                        
                        x_axis = gr.Dropdown(
                            label="📊 X-Axis Variable",
                            choices=[],
                            value=None,
                            info="Select the independent variable"
                        )
                        
                        y_axis = gr.Dropdown(
                            label="📈 Y-Axis Variable", 
                            choices=[],
                            value=None,
                            info="Select the dependent variable"
                        )
                        
                        color_var = gr.Dropdown(
                            label="🎨 Color Variable (Optional)",
                            choices=[],
                            value=None,
                            info="Group data by this variable"
                        )
                        
                        # Chart type selection based on Vizro availability
                        if VIZRO_AVAILABLE:
                            chart_types = STANDARD_CHART_TYPES + VIZRO_ENHANCED_CHART_TYPES
                            default_chart = "Enhanced Scatter Plot"
                        else:
                            chart_types = STANDARD_CHART_TYPES
                            default_chart = "Scatter Plot"
                        
                        chart_type = gr.Dropdown(
                            label="📊 Chart Type",
                            choices=chart_types,
                            value=default_chart,
                            info="Select visualization type"
                        )
                        
                        y_axis_agg = gr.Dropdown(
                            label="📊 Y-Axis Aggregation",
                            choices=Y_AXIS_AGGREGATIONS,
                            value="Raw Data",
                            info="How to aggregate Y-axis data"
                        )
                        
                        viz_theme = gr.Radio(
                            label="🎨 Theme",
                            choices=["Light", "Dark"],
                            value="Light",
                            info="Chart appearance theme"
                        )
                        
                        correlation_window = gr.Slider(
                            label="🪟 Correlation Window Size",
                            minimum=0,
                            maximum=200,
                            value=0,
                            step=1,
                            info="Set window for rolling correlation. 0 to disable.",
                            visible=False
                        )

                        create_viz_btn = gr.Button(
                            "🎨 Create Visualization",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Data insights button (only if Vizro available)
                        if VIZRO_AVAILABLE:
                            insights_btn = gr.Button(
                                "🧠 Generate Data Insights",
                                variant="secondary"
                            )
                    
                    with gr.Column(scale=2):
                        viz_output = gr.Plot(
                            label="📊 Visualization",
                            show_label=True
                        )
                        
                        if VIZRO_AVAILABLE:
                            data_insights = gr.Markdown(
                                value="🧠 Click 'Generate Data Insights' for automated analysis",
                                label="📋 Smart Data Insights"
                            )
            
            # Forecasting Tab
            with gr.Tab("📈 Forecasting", id="forecasting"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0;">🔮 Time Series Forecasting</h4>
                            <p>Predict future values using advanced models</p>
                        </div>
                        """)
                        
                        forecast_target = gr.Dropdown(
                            label="🎯 Target Variable",
                            choices=[],
                            value=None,
                            info="Variable to forecast"
                        )
                        
                        forecast_additional = gr.Dropdown(
                            label="📊 Additional Variables (Optional)",
                            choices=[],
                            value=None,
                            multiselect=True,
                            info="For multivariate models (VAR, Dynamic Factor)"
                        )
                        
                        forecast_model = gr.Dropdown(
                            label="🤖 Forecasting Model",
                            choices=FORECASTING_MODELS,
                            value="Linear Regression",
                            info="Select forecasting algorithm"
                        )
                        
                        forecast_periods = gr.Slider(
                            label="📅 Forecast Periods",
                            minimum=1,
                            maximum=50,
                            value=12,
                            step=1,
                            info="Number of periods to forecast"
                        )
                        
                        seasonal_period = gr.Slider(
                            label="🔄 Seasonal Period (for SARIMA)",
                            minimum=2,
                            maximum=24,
                            value=12,
                            step=1,
                            info="Length of seasonal cycle"
                        )
                        
                        confidence_level = gr.Slider(
                            label="📊 Confidence Level",
                            minimum=0.8,
                            maximum=0.99,
                            value=0.95,
                            step=0.01,
                            info="Confidence interval width"
                        )
                        
                        forecast_btn = gr.Button(
                            "🔮 Generate Forecast",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        forecast_plot = gr.Plot(
                            label="📈 Forecast Visualization",
                            show_label=True
                        )
                        
                        with gr.Row():
                            with gr.Column():
                                forecast_summary = gr.Markdown(
                                    value="📋 Forecast summary will appear here",
                                    label="📊 Forecast Summary"
                                )
                            
                            with gr.Column():
                                forecast_metrics = gr.HTML(
                                    value="<div style='text-align: center; padding: 20px; color: #666;'>📊 Detailed metrics will appear here</div>",
                                    label="📈 Detailed Metrics"
                                )
            
            # Causal Analysis Tab
            with gr.Tab("🔍 Causal Analysis", id="causal_analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0;">🧠 Causal Discovery</h4>
                            <p>Discover true causal relationships in your data</p>
                        </div>
                        """)
                        
                        hide_nonsig = gr.Checkbox(
                            label="🔍 Hide Non-Significant Relationships (p ≥ 0.05)",
                            value=True,
                            info="Show only statistically significant relationships"
                        )
                        
                        min_corr = gr.Slider(
                            label="📊 Minimum Correlation Threshold",
                            minimum=0.0,
                            maximum=0.9,
                            value=0.1,
                            step=0.05,
                            info="Filter weak relationships"
                        )
                        
                        causal_theme = gr.Radio(
                            label="🎨 Network Theme",
                            choices=["Light", "Dark"],
                            value="Light",
                            info="Network visualization theme"
                        )
                        
                        show_all_relationships = gr.Checkbox(
                            label="📊 Show All Relationships",
                            value=False,
                            info="Include non-significant relationships in network"
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Run Causal Analysis",
                            variant="primary",
                            size="lg"
                        )
                        
                        export_btn = gr.Button(
                            "📥 Export Results",
                            variant="secondary"
                        )
                    
                    with gr.Column(scale=2):
                        analysis_status = gr.Markdown(
                            value="📋 Click 'Run Causal Analysis' to start",
                            label="📊 Analysis Status"
                        )
                        
                        causal_network = gr.Plot(
                            label="🕸️ Causal Network",
                            show_label=True
                        )
                        
                        causal_table = gr.HTML(
                            value="<div style='text-align: center; padding: 20px; color: #666;'>📊 Results table will appear here</div>",
                            label="📋 Detailed Results"
                        )
                        
                        causal_summary = gr.Markdown(
                            value="📋 Analysis summary will appear here",
                            label="📊 Summary"
                        )
                        
                        export_output = gr.Markdown(
                            value="",
                            label="📥 Export Status"
                        )
            
            # Advanced Causal Analysis Tab
            with gr.Tab("🎯 Advanced Causal Analysis", id="advanced_causal"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div style="background: #fce4ec; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0;">🎯 Intervention Analysis</h4>
                            <p>Analyze "what-if" scenarios using do-calculus</p>
                        </div>
                        """)
                        
                        intervention_target = gr.Dropdown(
                            label="🎯 Target Variable",
                            choices=[],
                            value=None,
                            info="Variable to analyze the effect on"
                        )
                        
                        intervention_var = gr.Dropdown(
                            label="🔧 Intervention Variable", 
                            choices=[],
                            value=None,
                            info="Variable to intervene on"
                        )
                        
                        intervention_value = gr.Number(
                            label="💰 Intervention Value",
                            value=0,
                            info="New value to set for intervention variable"
                        )
                        
                        intervention_btn = gr.Button(
                            "🎯 Run Intervention Analysis",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        intervention_results = gr.HTML(
                            value="<div style='text-align: center; padding: 20px; color: #666;'>🎯 Intervention results will appear here</div>",
                            label="🎯 Intervention Results"
                        )
                        
                        intervention_status = gr.Markdown(
                            value="📋 Configure intervention and click 'Run Analysis'",
                            label="📊 Status"
                        )
        
=======
>>>>>>> origin/main
        # Event handlers
        config_components['file_input'].change(
            fn=load_data_from_file,
<<<<<<< HEAD
            inputs=[file_input],
            outputs=[upload_status, x_axis, y_axis, color_var, data_preview]
        ).then(
            fn=lambda file_path: update_forecast_dropdowns() if file_path else (gr.update(), gr.update()),
            inputs=[file_input],
            outputs=[forecast_target, forecast_additional]
        ).then(
            fn=lambda file_path: update_causal_dropdowns() if file_path else (gr.update(), gr.update(), gr.update(), gr.update()),
            inputs=[file_input],
            outputs=[intervention_target, intervention_var, intervention_target, intervention_var]
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
        
        # Update forecasting dropdowns when data is loaded
        config_components['file_input'].change(
            fn=lambda file_path: update_forecast_dropdowns() if file_path else (gr.update(), gr.update()),
            inputs=[config_components['file_input']],
            outputs=[analysis_components['forecast_target'], analysis_components['forecast_additional']]
        )
        
        # Update causal analysis dropdowns when data is loaded
        config_components['file_input'].change(
            fn=lambda file_path: update_causal_dropdowns() if file_path else (gr.update(), gr.update(), gr.update(), gr.update()),
            inputs=[config_components['file_input']],
            outputs=[
                analysis_components['intervention_target'], analysis_components['intervention_var'], 
                analysis_components['intervention_target'], analysis_components['intervention_var']
            ]
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
