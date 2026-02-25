import gradio as gr

def create_config_tab():
    with gr.Tab("📁 Data Upload", id="data_upload"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("File Upload", id="tab_file"):
                        gr.HTML("""
                        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="margin-top: 0;">📊 Upload Your Dataset</h3>
                            <p><strong>Supported formats:</strong> CSV, Excel (.xlsx, .xls), JSON, Parquet</p>
                            <p><strong>Requirements:</strong> Numeric columns for analysis</p>
                            <p><strong>Recommended:</strong> 100+ rows, 5+ variables</p>
                        </div>
                        """)
                        
                        file_input = gr.File(
                            label="📁 Choose File",
                            file_types=[".csv", ".xlsx", ".xls", ".json", ".parquet"],
                            type="filepath"
                        )
                    
                    with gr.Tab("URL Import", id="tab_url"):
                        gr.HTML("""
                        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="margin-top: 0;">🌐 Load from URL</h3>
                            <p>Enter a direct link to a CSV, JSON, or Parquet file.</p>
                        </div>
                        """)
                        url_input = gr.Textbox(
                            label="🌐 Dataset URL",
                            placeholder="https://example.com/data.csv",
                            lines=1
                        )
                        load_url_btn = gr.Button("⬇️ Load from URL", variant="primary")
                        
                upload_status = gr.Markdown("📋 No file uploaded yet")
                
            with gr.Column(scale=2):
                data_preview = gr.HTML(
                    value="<div style='text-align: center; padding: 50px; color: #666;'>📊 Data preview will appear here after upload</div>",
                    label="Data Preview"
                )
    
    return {
        'file_input': file_input,
        'url_input': url_input,
        'load_url_btn': load_url_btn,
        'upload_status': upload_status,
        'data_preview': data_preview
    }
