#!/usr/bin/env python3
"""
Simple script to run the Dynamic Data Analysis Dashboard
"""

import sys
import subprocess
import importlib.util

def check_and_install_requirements():
    """Check if required packages are installed, install if missing"""
    required_packages = [
        'gradio',
        'pandas', 
        'numpy',
        'plotly',
        'scikit-learn',
        'scipy',
        'causalnex',
        'networkx',
        'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            # Try UV first (faster)
            subprocess.run([
                'uv', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            print("✅ Packages installed successfully with UV")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to pip
            print("⚠️ UV not found, using pip...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            print("✅ Packages installed successfully with pip")
    else:
        print("✅ All required packages are already installed")

def main():
    """Main function to run the dashboard"""
    print("🚀 Starting Dynamic Data Analysis Dashboard...")
    print("=" * 50)
    
    # Check and install requirements
    try:
        check_and_install_requirements()
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        print("💡 Try running manually: pip install -r requirements.txt")
        return
    
    # Import and run the dashboard
    try:
        print("🔍 Loading dashboard...")
        import gradio_dashboard
        
        print("🌐 Starting web interface...")
        print("📱 Dashboard will open at: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop the dashboard")
        print("=" * 50)
        
        # Run the dashboard
        gradio_dashboard.main()
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except ImportError as e:
        print(f"❌ Error importing dashboard: {e}")
        print("💡 Make sure gradio_dashboard.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Check the error message above for details")

if __name__ == "__main__":
    main()