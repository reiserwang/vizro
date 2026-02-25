## Gemini Notes

### 2025-08-25

-   Updated `dashboard.py` to:
    -   Change the "Sales Volume" chart from a scatter plot to a stacked bar chart.
    -   Add trend line labels to the "Accumulated and Active Volume" chart.
    -   Add node labels to the "CausalNex Time Series Analysis" chart.
    -   Add a placeholder "Apply Filters" button.
-   Updated `README.md` to reflect the new features of the dashboard.

### UV Usage Instructions

To ensure consistent dependency management and script execution, please use `uv` as follows:

1.  **Activate Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```
2.  **Install Dependencies:**
    Use `uv pip install` instead of `pip install`.
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Run Python Scripts:**
    Use `uv run python` instead of `python`.
    ```bash
    uv run python your_script_name.py
    ```