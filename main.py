#!/usr/bin/env python3
"""
Advanced Analytics Dashboard - Main Application Entry Point

This is the main entry point for the Advanced Analytics Dashboard application.
Run this file to start the Gradio web interface, the FastAPI server, or both.

Usage:
    python main.py --mode ui        # Run only the Gradio UI (default)
    python main.py --mode api       # Run only the FastAPI server
    python main.py --mode both      # Run both API and UI
    
    or with uv:
    uv run python main.py
"""

import sys
import os
import argparse

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Advanced Analytics Dashboard")
    
    # Default to native mode if running as a PyInstaller packaged app
    is_frozen = getattr(sys, 'frozen', False)
    default_mode = "native" if is_frozen else "ui"
    
    parser.add_argument("--mode", type=str, choices=["ui", "api", "both", "native"], default=default_mode, 
                        help=f"Run UI, API, Both, or Native Mac App (default: {default_mode})")
    args = parser.parse_args()
    
    print(f"🚀 Starting Advanced Analytics Dashboard in '{args.mode.upper()}' mode...")
    
    try:
        # Add src directory to path for modular imports
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        if args.mode in ["api", "both"]:
            print("🌐 Starting API server...")
            import uvicorn
            from api.routes import app
            
            if args.mode == "both":
                import threading
                # Start API in a daemon thread so it runs alongside UI
                api_thread = threading.Thread(
                    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info"),
                    daemon=True
                )
                api_thread.start()
                print("✅ API server started on http://localhost:8000")
            else:
                # Run API blocking
                uvicorn.run(app, host="0.0.0.0", port=8000)
                
        if args.mode == "native":
            import webview
            from webview.menu import Menu, MenuAction
            from threading import Thread
            from api.routes import app
            import uvicorn
            import time
            
            print("🍏 Starting macOS Native Application Wrapper...")
            
            def start_api():
                uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
                
            Thread(target=start_api, daemon=True).start()
            time.sleep(1) # wait for API to start
            
            webview.create_window(
                "Vizro Analytics Dashboard", 
                "http://127.0.0.1:8000/ui/data_source.html",
                width=1200, 
                height=800,
                min_size=(800, 600),
                transparent=True,
                frameless=True,
                easy_drag=True
            )
            
            # Recreate the application menu for macOS to restore standard keyboard shortcuts
            # (copy, paste, cut, select all, undo, redo, quit) 
            # In frameless mode, the default menu is often stripped, which disables these shortcuts natively.
            def void(): pass
            
            menu_items = [
                Menu('App', [
                    MenuAction('Quit', webview.windows[0].destroy)
                ]),
                Menu('Edit', [
                    MenuAction('Undo', void),
                    MenuAction('Redo', void),
                    MenuAction('Cut', void),
                    MenuAction('Copy', void),
                    MenuAction('Paste', void),
                    MenuAction('Select All', void),
                ])
            ]
            
            webview.start(menu=menu_items)
            return

        if args.mode in ["ui", "both"]:
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
        
        if args.mode != "ui":
            print("⚠️ API mode not supported in fallback. Exiting.")
            sys.exit(1)
            
        try:
            # Fallback to original dashboard
            print("🔄 Loading original dashboard...")
            sys.path.insert(0, os.path.dirname(__file__))
            
            # Import and run the original dashboard directly
            import gradio_dashboard
            print("✅ Original dashboard loaded! Starting server...")
            
            # The original dashboard should have its own launch mechanism
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