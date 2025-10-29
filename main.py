#!/usr/bin/env python3
"""
Advanced Analytics Dashboard - Main Application Entry Point

This is the main entry point for the Advanced Analytics Dashboard application.
Run this file to start the Gradio web interface.

Usage:
    python main.py
    
    or with uv:
    uv run python main.py
"""

import sys
import os

def main():
    """Main application entry point."""
    print("🚀 Starting Advanced Analytics Dashboard...")
    
    try:
        # Add src directory to path for modular imports
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        print("📊 Loading modular dashboard interface...")
        
        # Import from the restructured dashboard
        from ui.dashboard import create_gradio_interface
        
        print("✅ Modular dashboard loaded successfully!")
        
        # Create and launch the dashboard
        dashboard = create_gradio_interface()
        
        # Launch with appropriate settings
        dashboard.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Trying fallback to original dashboard...")
        
        try:
            # Fallback to original dashboard
            print("🔄 Loading original dashboard...")
            sys.path.insert(0, os.path.dirname(__file__))
            
            # Import and run the original dashboard directly
            import gradio_dashboard
            print("✅ Original dashboard loaded! Starting server...")
            
            # The original dashboard should have its own launch mechanism
            # Let's try to find and call it
            if hasattr(gradio_dashboard, 'main'):
                gradio_dashboard.main()
            elif hasattr(gradio_dashboard, 'launch'):
                gradio_dashboard.launch()
            else:
                print("⚠️ Original dashboard structure unknown, please run 'python gradio_dashboard.py' directly")
                sys.exit(1)
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()