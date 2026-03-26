"""
API-layer tests for /api/v1/upload and /api/v1/visualize using FastAPI TestClient.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi.testclient import TestClient
from src.api.routes import app
from tests.conftest import df_to_csv_bytes, _numeric_df, _categorical_df, _datetime_df


client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_csv(df):
    """POST a DataFrame as CSV bytes to /api/v1/upload."""
    content = df_to_csv_bytes(df)
    response = client.post(
        "/api/v1/upload",
        files={"file": ("data.csv", content, "text/csv")},
    )
    return response


def visualize(x, y, chart_type, color=None, y_agg="Raw Data", theme="Light", window=0):
    """POST to /api/v1/visualize."""
    return client.post("/api/v1/visualize", json={
        "x_axis": x,
        "y_axis": y,
        "color_var": color,
        "chart_type": chart_type,
        "theme": theme,
        "y_axis_agg": y_agg,
        "correlation_window": window,
    })


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------

class TestUploadEndpoint:
    def test_upload_numeric_csv_returns_200(self):
        r = upload_csv(_numeric_df())
        assert r.status_code == 200, r.text

    def test_upload_returns_column_lists(self):
        r = upload_csv(_numeric_df())
        body = r.json()
        assert "columns" in body
        assert "numeric_columns" in body
        assert "categorical_columns" in body

    def test_upload_numeric_df_correct_columns(self):
        r = upload_csv(_numeric_df())
        body = r.json()
        assert set(body["columns"]) == {"x", "y", "value_a", "value_b", "value_c"}
        assert set(body["numeric_columns"]) == {"x", "y", "value_a", "value_b", "value_c"}
        assert body["categorical_columns"] == []

    def test_upload_categorical_df_separates_types(self):
        r = upload_csv(_categorical_df())
        body = r.json()
        assert "region" in body["categorical_columns"]
        assert "sales" in body["numeric_columns"]

    def test_upload_replaces_previous_dataset(self):
        upload_csv(_numeric_df())
        r2 = upload_csv(_categorical_df())
        body = r2.json()
        assert "region" in body["columns"]


# ---------------------------------------------------------------------------
# Visualize error cases (no data / bad inputs)
# ---------------------------------------------------------------------------

class TestVisualizeErrorCases:
    def setup_method(self):
        """Clear data from both the routes module and the engine module."""
        from src.core import dashboard_config as _dc
        _dc.current_data = None

    def test_visualize_before_upload_returns_400(self):
        r = visualize("x", "y", "Enhanced Scatter Plot")
        assert r.status_code == 400

    def test_visualize_bad_chart_type_returns_400(self):
        upload_csv(_numeric_df())
        r = visualize("x", "y", "Totally Unknown Chart")
        # Engine returns None for unknown chart → 400
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Round-trip: upload → visualize (all chart types, numeric data)
# ---------------------------------------------------------------------------

NUMERIC_CHART_CASES = [
    ("Enhanced Scatter Plot", "x", "y", None),
    ("Statistical Box Plot", "x", "y", None),
    ("Correlation Heatmap", "x", "y", None),
    ("Distribution Analysis", "x", "y", None),
    ("Time Series Analysis", "x", "y", None),
    ("Advanced Bar Chart", "x", "y", None),
]


class TestRoundTripNumeric:
    def setup_method(self):
        upload_csv(_numeric_df())

    @pytest.mark.parametrize("chart_type,x,y,color", NUMERIC_CHART_CASES)
    def test_chart_renders(self, chart_type, x, y, color):
        r = visualize(x, y, chart_type, color=color)
        assert r.status_code == 200, (
            f"chart={chart_type} failed: {r.status_code} {r.text}"
        )
        body = r.json()
        assert body.get("status") == "success"
        assert "plot" in body
        assert "data" in body["plot"]
        assert "layout" in body["plot"]


# ---------------------------------------------------------------------------
# Round-trip: upload → visualize (categorical data)
# ---------------------------------------------------------------------------

CATEGORICAL_CHART_CASES = [
    ("Enhanced Scatter Plot", "sales", "profit", None),
    ("Enhanced Scatter Plot", "region", "sales", "product"),  # categorical x
    ("Statistical Box Plot", "region", "sales", None),
    ("Advanced Bar Chart", "region", "sales", None),
    ("Advanced Bar Chart", "region", "sales", "product"),
    ("Distribution Analysis", "region", "sales", None),
]


class TestRoundTripCategorical:
    def setup_method(self):
        upload_csv(_categorical_df())

    @pytest.mark.parametrize("chart_type,x,y,color", CATEGORICAL_CHART_CASES)
    def test_chart_renders(self, chart_type, x, y, color):
        r = visualize(x, y, chart_type, color=color)
        assert r.status_code == 200, (
            f"chart={chart_type} x={x} color={color} failed: {r.status_code} {r.text}"
        )
        body = r.json()
        assert body.get("status") == "success"
        assert "plot" in body


# ---------------------------------------------------------------------------
# Round-trip: upload → visualize (datetime data)
# ---------------------------------------------------------------------------

class TestRoundTripDatetime:
    def setup_method(self):
        upload_csv(_datetime_df())

    @pytest.mark.parametrize("chart_type,x,y", [
        ("Enhanced Scatter Plot", "date", "revenue"),
        ("Time Series Analysis", "date", "revenue"),
        ("Distribution Analysis", "date", "revenue"),
        ("Line Chart", "date", "revenue"),
    ])
    def test_chart_renders(self, chart_type, x, y):
        r = visualize(x, y, chart_type)
        assert r.status_code == 200, (
            f"chart={chart_type} failed: {r.status_code} {r.text}"
        )
        body = r.json()
        assert body.get("status") == "success"
        assert "plot" in body


# ---------------------------------------------------------------------------
# Round-trip: taipei.csv (real file, string x-axis)
# ---------------------------------------------------------------------------

class TestRoundTripTaipei:
    def setup_method(self):
        import pandas as pd
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        df = pd.read_csv(os.path.join(root, "taipei.csv"))
        upload_csv(df)

    @pytest.mark.parametrize("chart_type,x,y,color", [
        ("Enhanced Scatter Plot", "年別", "每戶人數", "行政區"),
        ("Enhanced Scatter Plot", "年別", "每戶人數", None),
        ("Statistical Box Plot", "行政區", "每戶人數", None),
        ("Advanced Bar Chart", "行政區", "每戶人數", None),
        ("Distribution Analysis", "每戶人數", "家庭戶數[戶]", None),
    ])
    def test_taipei_chart_renders(self, chart_type, x, y, color):
        r = visualize(x, y, chart_type, color=color)
        assert r.status_code == 200, (
            f"taipei chart={chart_type} x={x} failed: {r.status_code} {r.text}"
        )
        body = r.json()
        assert body.get("status") == "success"
        assert "plot" in body

    def test_heatmap_taipei(self):
        """Heatmap needs ≥2 numeric columns — taipei has many."""
        r = visualize("年別", "每戶人數", "Correlation Heatmap")
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Round-trip: sales_data.csv
# ---------------------------------------------------------------------------

class TestRoundTripSales:
    def setup_method(self):
        import pandas as pd
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        df = pd.read_csv(os.path.join(root, "sales_data.csv"))
        upload_csv(df)

    @pytest.mark.parametrize("chart_type,x,y,color", [
        ("Enhanced Scatter Plot", "Revenue", "Sales_Volume", None),
        ("Enhanced Scatter Plot", "Revenue", "Sales_Volume", "Region"),
        ("Statistical Box Plot", "Region", "Revenue", "Product_Category"),
        ("Correlation Heatmap", "Revenue", "Sales_Volume", None),
        ("Distribution Analysis", "Revenue", "Sales_Volume", None),
        ("Time Series Analysis", "Date", "Revenue", None),
        ("Advanced Bar Chart", "Region", "Revenue", None),
        ("Advanced Bar Chart", "Region", "Revenue", "Product_Category"),
    ])
    def test_sales_chart_renders(self, chart_type, x, y, color):
        r = visualize(x, y, chart_type, color=color)
        assert r.status_code == 200, (
            f"sales chart={chart_type} x={x} failed: {r.status_code} {r.text}"
        )
        body = r.json()
        assert body.get("status") == "success"
        assert "plot" in body

    def test_aggregation_average(self):
        r = visualize("Region", "Revenue", "Advanced Bar Chart", y_agg="Average")
        assert r.status_code == 200

    def test_aggregation_sum(self):
        r = visualize("Region", "Revenue", "Advanced Bar Chart", y_agg="Sum")
        assert r.status_code == 200

    def test_dark_theme(self):
        r = visualize("Revenue", "Sales_Volume", "Enhanced Scatter Plot", theme="Dark")
        assert r.status_code == 200
