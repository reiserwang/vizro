# Lag Analysis Extraction Plan

## 🎯 Objective
Extract lag analysis functions from `gradio_dashboard.py` (5,356 lines) into a modular structure.

## 📦 New Module Structure

### **File**: `src/engines/lag_analysis_engine.py`

**Functions to Extract:**
1. `perform_lag_analysis()` - Lines 3480-3969 (~490 lines)
2. `auto_discover_lagged_relationships()` - Lines 3970-4300 (~330 lines)

**Total**: ~820 lines to extract

## 🔧 Implementation Steps

### **Step 1: Create Module File** ✅
- Created: `src/engines/lag_analysis_engine.py`
- Added: Module docstring and imports

### **Step 2: Extract Core Functions** (In Progress)
Due to function size (490+ lines), need to:
1. Copy `perform_lag_analysis()` function
2. Copy `auto_discover_lagged_relationships()` function
3. Add helper functions if needed

### **Step 3: Update Imports**
Add to `gradio_dashboard.py`:
```python
from src.engines.lag_analysis_engine import (
    perform_lag_analysis,
    auto_discover_lagged_relationships,
    set_current_data
)
```

### **Step 4: Remove Original Functions**
Delete lines 3480-4300 from `gradio_dashboard.py`

### **Step 5: Update Data Handling**
Add after data loading in dashboard:
```python
from src.engines import lag_analysis_engine
lag_analysis_engine.set_current_data(current_data)
```

## 📊 Benefits

### **Before Extraction:**
- `gradio_dashboard.py`: 5,356 lines
- Monolithic structure
- Hard to maintain

### **After Extraction:**
- `gradio_dashboard.py`: ~4,536 lines (-820 lines, -15%)
- `src/engines/lag_analysis_engine.py`: ~850 lines (new)
- Modular, testable, maintainable

## 🔄 Alternative Approach

Given the function size and complexity, I recommend:

### **Option A: Manual Extraction** (Recommended)
1. I provide the complete module file content
2. You create the file manually
3. Update imports in dashboard
4. Test functionality

### **Option B: Incremental Extraction**
1. Extract one function at a time
2. Test after each extraction
3. Gradually build the module

### **Option C: Keep Current Structure**
1. Add comments and documentation
2. Improve existing code organization
3. Plan future refactoring

## 📝 Next Steps

**Immediate Action Required:**
Choose one of the options above, and I'll proceed accordingly.

**Recommended**: Option A - I'll provide the complete, well-documented module file that you can create manually.

This ensures:
- ✅ No truncation issues
- ✅ Complete functionality
- ✅ Proper error handling
- ✅ Comprehensive documentation
- ✅ Easy testing and validation

Would you like me to provide the complete module file content for manual creation?
