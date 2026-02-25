# 📁 Project Structure

## 🏗️ Directory Organization

```
advanced-analytics-dashboard/
├── 📁 src/                          # Source code
│   ├── 📁 core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration management
│   │   ├── data_handler.py          # Data operations
│   │   └── dashboard_config.py      # Legacy config (migration)
│   ├── 📁 engines/                  # Analysis engines
│   │   ├── __init__.py
│   │   ├── causal_engine.py         # Causal analysis
│   │   ├── forecasting_engine.py    # Time series forecasting
│   │   └── visualization_engine.py  # Data visualization
│   ├── 📁 ui/                       # User interface
│   │   ├── __init__.py
│   │   ├── dashboard.py             # Main Gradio interface
│   │   └── settings_manager.py      # Settings management
│   └── 📁 utils/                    # Utilities
│       ├── __init__.py
│       └── data_generator.py        # Sample data generation
├── 📁 tests/                        # Test suite
│   ├── __init__.py
│   ├── 📁 unit/                     # Unit tests
│   │   ├── test_*.py                # Individual test files
│   ├── 📁 integration/              # Integration tests
│   └── 📁 fixtures/                 # Test data and fixtures
├── 📁 docs/                         # Documentation
│   ├── __init__.py
│   ├── PROJECT_OVERVIEW.md          # Project overview
│   ├── 📁 user-guide/               # User documentation
│   │   └── GETTING_STARTED.md       # Getting started guide
│   ├── 📁 technical/                # Technical documentation
│   │   ├── ARCHITECTURE.md          # System architecture
│   │   ├── *_SUMMARY.md             # Technical summaries
│   │   └── *_FIX.md                 # Fix documentation
│   └── 📁 api/                      # API documentation
│       └── API_REFERENCE.md         # API reference
├── 📁 config/                       # Configuration files
│   ├── __init__.py
│   └── dashboard_settings.json      # Dashboard settings
├── 📄 main.py                       # Application entry point
├── 📄 PROJECT_STRUCTURE.md          # This file
├── 📄 README.md                     # Project README
├── 📄 requirements.txt              # Python dependencies
├── 📄 pyproject.toml               # Project configuration
└── 📄 .gitignore                   # Git ignore rules
```

## 📋 File Descriptions

### 🎯 Main Application
- **`main.py`**: Primary entry point for the application
- **`README.md`**: Project overview and quick start guide

### 🏗️ Source Code (`src/`)

#### Core Module (`src/core/`)
- **`config.py`**: Centralized configuration management and global state
- **`data_handler.py`**: Data loading, validation, and preprocessing
- **`dashboard_config.py`**: Legacy configuration (to be migrated)

#### Analysis Engines (`src/engines/`)
- **`causal_engine.py`**: Causal discovery and intervention analysis
- **`forecasting_engine.py`**: Time series forecasting with multiple models
- **`visualization_engine.py`**: Interactive data visualization creation

#### User Interface (`src/ui/`)
- **`dashboard.py`**: Main Gradio web interface
- **`settings_manager.py`**: User preferences and settings management

#### Utilities (`src/utils/`)
- **`data_generator.py`**: Sample dataset generation for testing

### 🧪 Tests (`tests/`)
- **`unit/`**: Unit tests for individual components
- **`integration/`**: End-to-end integration tests
- **`fixtures/`**: Test data and mock objects

### 📚 Documentation (`docs/`)
- **`PROJECT_OVERVIEW.md`**: Comprehensive project overview
- **`user-guide/`**: User-facing documentation
- **`technical/`**: Technical implementation details
- **`api/`**: API reference and examples

### ⚙️ Configuration (`config/`)
- **`dashboard_settings.json`**: Default dashboard configuration

## 🔗 Module Dependencies

### Import Structure
```python
# Core modules
from src.core.config import dashboard_config
from src.core.data_handler import DataHandler

# Analysis engines
from src.engines.causal_engine import CausalAnalysisEngine
from src.engines.forecasting_engine import ForecastingEngine
from src.engines.visualization_engine import VisualizationEngine

# UI components
from src.ui.dashboard import create_dashboard
from src.ui.settings_manager import SettingsManager

# Utilities
from src.utils.data_generator import DataGenerator
```

### Dependency Graph
```
main.py
└── src.ui.dashboard
    ├── src.core.config
    ├── src.core.data_handler
    ├── src.engines.causal_engine
    ├── src.engines.forecasting_engine
    ├── src.engines.visualization_engine
    └── src.ui.settings_manager
```

## 🚀 Usage Patterns

### Running the Application
```bash
# From project root
python main.py

# Or with uv
uv run python main.py
```

### Importing Components
```python
# For custom analysis scripts
import sys
sys.path.append('src')

from core.config import dashboard_config
from engines.causal_engine import perform_causal_analysis
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

## 📦 Package Structure

### Core Package (`src.core`)
```python
src.core/
├── __init__.py              # Package initialization
├── config.py               # DashboardConfig class
└── data_handler.py         # DataHandler class
```

### Engines Package (`src.engines`)
```python
src.engines/
├── __init__.py              # Package initialization
├── causal_engine.py        # Causal analysis functions
├── forecasting_engine.py   # Forecasting functions
└── visualization_engine.py # Visualization functions
```

### UI Package (`src.ui`)
```python
src.ui/
├── __init__.py              # Package initialization
├── dashboard.py            # Dashboard creation functions
└── settings_manager.py     # Settings management class
```

## 🔧 Configuration Management

### Settings Hierarchy
1. **Default Settings**: Hardcoded defaults in `src/core/config.py`
2. **Configuration File**: `config/dashboard_settings.json`
3. **Environment Variables**: Runtime overrides
4. **User Preferences**: Session-specific settings

### Configuration Files
```json
// config/dashboard_settings.json
{
    "theme": "Light",
    "performance": {
        "max_samples": 1500,
        "enable_caching": true
    },
    "analysis": {
        "causal": {
            "max_variables": 12,
            "significance_threshold": 0.05
        },
        "forecasting": {
            "default_periods": 12,
            "confidence_level": 0.95
        }
    }
}
```

## 📊 Data Flow

### Application Startup
1. **`main.py`** → Initialize application
2. **`src.ui.dashboard`** → Create Gradio interface
3. **`src.core.config`** → Load configuration
4. **`config/dashboard_settings.json`** → Load settings

### Analysis Workflow
1. **User Upload** → `src.core.data_handler.load_data()`
2. **Data Validation** → `src.core.data_handler.validate_data()`
3. **Analysis Request** → `src.engines.*_engine.analyze()`
4. **Results Display** → `src.ui.dashboard` components

## 🛠️ Development Workflow

### Adding New Features
1. **Core Logic**: Add to appropriate engine in `src/engines/`
2. **UI Integration**: Update `src/ui/dashboard.py`
3. **Configuration**: Add settings to `src/core/config.py`
4. **Tests**: Add tests to `tests/unit/` or `tests/integration/`
5. **Documentation**: Update relevant docs in `docs/`

### Code Organization Principles
- **Single Responsibility**: Each module has a clear, focused purpose
- **Loose Coupling**: Minimal dependencies between modules
- **High Cohesion**: Related functionality grouped together
- **Clear Interfaces**: Well-defined APIs between components

## 📝 Naming Conventions

### Files and Directories
- **Modules**: `snake_case.py`
- **Packages**: `snake_case/`
- **Documentation**: `UPPER_CASE.md`
- **Tests**: `test_*.py`

### Code Elements
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_CASE`

### Documentation
- **User Guides**: `GETTING_STARTED.md`
- **Technical Docs**: `ARCHITECTURE.md`
- **API Reference**: `API_REFERENCE.md`
- **Fix Documentation**: `*_FIX.md`

## 🔄 Migration Notes

### From Legacy Structure
The project has been restructured from a flat file organization to a modular package structure:

**Old Structure** → **New Structure**
- `gradio_dashboard_refactored.py` → `src/ui/dashboard.py`
- `causal_analysis_engine.py` → `src/engines/causal_engine.py`
- `forecasting_engine.py` → `src/engines/forecasting_engine.py`
- `visualization_engine.py` → `src/engines/visualization_engine.py`
- `settings_manager.py` → `src/ui/settings_manager.py`
- `data_handler.py` → `src/core/data_handler.py`
- `dashboard_config.py` → `src/core/config.py` (merged)

### Import Updates Required
Update any existing scripts to use the new import paths:
```python
# Old imports
from causal_analysis_engine import perform_causal_analysis

# New imports
from src.engines.causal_engine import perform_causal_analysis
```