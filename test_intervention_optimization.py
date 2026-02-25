#!/usr/bin/env python3
"""
Test script for optimized intervention analysis
"""

import pandas as pd
import time
import sys

# Load the data
print("🔄 Loading test data...")
df = pd.read_csv('sales_data.csv')
print(f"📊 Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

# Import the optimized function
sys.path.append('.')
from gradio_dashboard import perform_causal_intervention_analysis

# Mock progress function for testing
class MockProgress:
    def __call__(self, value, desc=""):
        print(f"Progress: {value*100:.0f}% - {desc}")

# Set up test parameters
target_var = "Revenue"
intervention_var = "Marketing_Spend"
intervention_value = 50000.0

print(f"\n🧪 Testing intervention analysis:")
print(f"   Target: {target_var}")
print(f"   Intervention: {intervention_var} = {intervention_value}")

# Set global data (simulate dashboard state)
import gradio_dashboard
gradio_dashboard.current_data = df

# Run the test
print("\n🚀 Starting optimized intervention analysis...")
start_time = time.time()

try:
    result_html, result_msg = perform_causal_intervention_analysis(
        target_var, 
        intervention_var, 
        intervention_value, 
        progress=MockProgress()
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n✅ Test completed in {elapsed:.1f} seconds!")
    print(f"📊 Result message: {result_msg}")
    
    if "✅" in result_msg:
        print("🎉 SUCCESS: Intervention analysis completed successfully!")
        print(f"⚡ Performance: {elapsed:.1f}s (target: <30s)")
    else:
        print("❌ FAILED: Intervention analysis did not complete successfully")
        print(f"Error details: {result_msg}")
        
except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n❌ Test failed after {elapsed:.1f} seconds")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()