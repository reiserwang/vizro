# Dynamic CSV Analysis Dashboard

This project contains a Python script that creates an interactive dashboard for analyzing any CSV data.

## Features

The dashboard provides the following functionalities:

*   **Upload any CSV file**: Users can upload any CSV file for analysis.
*   **Dynamic Plotting**: The dashboard dynamically generates dropdowns based on the columns of the uploaded CSV. Users can select columns for the X and Y axes, and a column for color-coding the data points in a scatter plot.
*   **Time-based Filtering**: If a date/time column is detected, users can filter the data by time ranges like "Last 3 Months", "Last 6 Months", "Last Year", and "YTD".
*   **CausalNex Analysis**: A causal graph that shows the relationships between different numeric columns in the dataset.

## Setup

1.  Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the dashboard

To run the interactive dashboard:

```bash
python dashboard.py
```

This will start a web server, and you can view the dashboard by opening the URL provided in the console (usually `http://127.0.0.1:8050`).