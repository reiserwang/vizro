import gradio as gr

def create_analysis_tabs(FORECASTING_MODELS):
    components = {}
    
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
                
                components['forecast_target'] = gr.Dropdown(label="🎯 Target Variable", choices=[], value=None, info="Variable to forecast")
                
                components['forecast_additional'] = gr.Dropdown(label="📊 Additional Variables (Optional)", choices=[], value=None, multiselect=True, info="For multivariate models (VAR, Dynamic Factor)")
                
                components['forecast_model'] = gr.Dropdown(label="🤖 Forecasting Model", choices=FORECASTING_MODELS, value="Linear Regression", info="Select forecasting algorithm")
                
                components['forecast_periods'] = gr.Slider(label="📅 Forecast Periods", minimum=1, maximum=50, value=12, step=1, info="Number of periods to forecast")
                
                components['seasonal_period'] = gr.Slider(label="🔄 Seasonal Period (for SARIMA)", minimum=2, maximum=24, value=12, step=1, info="Length of seasonal cycle")
                
                components['confidence_level'] = gr.Slider(label="📊 Confidence Level", minimum=0.8, maximum=0.99, value=0.95, step=0.01, info="Confidence interval width")
                
                components['forecast_btn'] = gr.Button("🔮 Generate Forecast", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                components['forecast_plot'] = gr.Plot(label="📈 Forecast Visualization", show_label=True)
                with gr.Row():
                    with gr.Column():
                        components['forecast_summary'] = gr.Markdown(value="📋 Forecast summary will appear here", label="📊 Forecast Summary")
                    with gr.Column():
                        components['forecast_metrics'] = gr.HTML(value="<div style='text-align: center; padding: 20px; color: #666;'>📊 Detailed metrics will appear here</div>", label="📈 Detailed Metrics")
                        
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
                
                components['hide_nonsig'] = gr.Checkbox(label="🔍 Hide Non-Significant Relationships (p ≥ 0.05)", value=True, info="Show only statistically significant relationships")
                
                components['min_corr'] = gr.Slider(label="📊 Minimum Correlation Threshold", minimum=0.0, maximum=0.9, value=0.1, step=0.05, info="Filter weak relationships")
                
                components['causal_theme'] = gr.Radio(label="🎨 Network Theme", choices=["Light", "Dark"], value="Light", info="Network visualization theme")
                
                components['show_all_relationships'] = gr.Checkbox(label="📊 Show All Relationships", value=False, info="Include non-significant relationships in network")
                
                components['analyze_btn'] = gr.Button("🔍 Run Causal Analysis", variant="primary", size="lg")
                
                components['export_btn'] = gr.Button("📥 Export Results", variant="secondary")
            
            with gr.Column(scale=2):
                components['analysis_status'] = gr.Markdown(value="📋 Click 'Run Causal Analysis' to start", label="📊 Analysis Status")
                components['causal_network'] = gr.Plot(label="🕸️ Causal Network", show_label=True)
                components['causal_table'] = gr.HTML(value="<div style='text-align: center; padding: 20px; color: #666;'>📊 Results table will appear here</div>", label="📋 Detailed Results")
                components['causal_summary'] = gr.Markdown(value="📋 Analysis summary will appear here", label="📊 Summary")
                components['export_output'] = gr.Markdown(value="", label="📥 Export Status")
                
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
                
                components['intervention_target'] = gr.Dropdown(label="🎯 Target Variable", choices=[], value=None, info="Variable to analyze the effect on")
                
                components['intervention_var'] = gr.Dropdown(label="🔧 Intervention Variable", choices=[], value=None, info="Variable to intervene on")
                
                components['intervention_value'] = gr.Number(label="💰 Intervention Value", value=0, info="New value to set for intervention variable")
                
                components['intervention_btn'] = gr.Button("🎯 Run Intervention Analysis", variant="primary")
            
            with gr.Column():
                components['intervention_results'] = gr.HTML(value="<div style='text-align: center; padding: 20px; color: #666;'>🎯 Intervention results will appear here</div>", label="🎯 Intervention Results")
                components['intervention_status'] = gr.Markdown(value="📋 Configure intervention and click 'Run Analysis'", label="📊 Status")

    return components
