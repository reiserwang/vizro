import gradio as gr

def create_visualization_tab(VIZRO_AVAILABLE, STANDARD_CHART_TYPES, VIZRO_ENHANCED_CHART_TYPES, Y_AXIS_AGGREGATIONS):
    with gr.Tab("📊 Data Visualization", id="visualization"):
        with gr.Row():
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
                
                insights_btn = None
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
                
                data_insights = None
                if VIZRO_AVAILABLE:
                    data_insights = gr.Markdown(
                        value="🧠 Click 'Generate Data Insights' for automated analysis",
                        label="📋 Smart Data Insights"
                    )
                    
    return {
        'x_axis': x_axis,
        'y_axis': y_axis,
        'color_var': color_var,
        'chart_type': chart_type,
        'y_axis_agg': y_axis_agg,
        'viz_theme': viz_theme,
        'correlation_window': correlation_window,
        'create_viz_btn': create_viz_btn,
        'insights_btn': insights_btn,
        'viz_output': viz_output,
        'data_insights': data_insights
    }
