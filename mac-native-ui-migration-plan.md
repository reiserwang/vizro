# Mac Native App Implementation Plan

## Track: `mac-native-ui-migration`
## Goal: Implement the newly designed UI as a macOS native app.

### Tasks
- [x] 1. Identify optimal Python-to-macOS WebView framework (Tauri, PyWebView, or PySide6).
- [x] 2. Update `main.py` to launch the native app window instead of serving web directly.
- [x] 3. Refactor Python data engines (`causal_engine.py`, `forecasting_engine.py`) to serve data securely to the local frontend.
- [x] 4. Build the React/HTML/JS frontend based on the Stitch MCP "Alloy Desktop" design.
- [x] 5. Implement the transparent macOS sidebar (Vibrance effect).
- [x] 6. Package application for macOS distribution.
