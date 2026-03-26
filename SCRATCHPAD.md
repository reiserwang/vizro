# SCRATCHPAD

## Shared State
- **Project**: Vizro
- **Goal**: Implement Mac Native wrapper for web dashboard UI
- **Active Conductor Track**: `mac-native-ui-migration`
- **UI Design**: Generated via Stitch MCP, including main dashboard with tabs, data source management, and forecasting detail view. 
- **Tech Stack**: Python (backend/engines like Causal Engine, Forecasting Engine) + Web UI -> macOS WebView Native Wrapper (e.g., PyQt6/PySide6 QWebEngineView, Tauri, or PyWebView).

## Handoff Queue
- Orchestrator to claim track `mac-native-ui-migration`.
- Assess current `main.py` and `routes.py` for API service mode vs. Native app bundling.
- Planner to refine `workspace/conductor/tracks/mac-native-ui-migration/plan.md`.
