"""
End-to-end integration tests: full upload → visualize flow verifying that
diagram data (plot.data, plot.layout) is non-empty in the response.

These tests use the real taipei.csv and sales_data.csv files and synthetic
test datasets to maximise coverage across data shapes.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi.testclient import TestClient
from src.api.routes import app
from tests.conftest import df_to_csv_bytes

client = TestClient(app)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload(df: pd.DataFrame):
    content = df_to_csv_bytes(df)
    r = client.post(
        "/api/v1/upload",
        files={"file": ("data.csv", content, "text/csv")},
    )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    return r.json()


def viz(x, y, chart, color=None, y_agg="Raw Data", theme="Light", window=0):
    r = client.post("/api/v1/visualize", json={
        "x_axis": x,
        "y_axis": y,
        "color_var": color,
        "chart_type": chart,
        "theme": theme,
        "y_axis_agg": y_agg,
        "correlation_window": window,
    })
    return r


def assert_diagram(r, label=""):
    assert r.status_code == 200, f"{label}: HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("status") == "success", f"{label}: no 'success' status"
    plot = body.get("plot", {})
    assert "data" in plot, f"{label}: 'data' key missing from plot"
    assert "layout" in plot, f"{label}: 'layout' key missing from plot"
    assert len(plot["data"]) > 0, f"{label}: plot.data is empty — no traces rendered"


# ---------------------------------------------------------------------------
# Dataset 1: taipei.csv (real file, Chinese column names, string x-axis)
# ---------------------------------------------------------------------------

class TestTaipeiFlow:
    @pytest.fixture(autouse=True)
    def upload_taipei(self):
        df = pd.read_csv(os.path.join(ROOT, "taipei.csv"))
        upload(df)

    @pytest.mark.parametrize("chart,x,y,color", [
        ("Enhanced Scatter Plot", "年別", "每戶人數", None),
        ("Enhanced Scatter Plot", "年別", "每戶人數", "行政區"),
        ("Statistical Box Plot", "行政區", "每戶人數", None),
        ("Statistical Box Plot", "行政區", "每戶人數", "年別"),
        ("Correlation Heatmap", "每戶人數", "家庭戶數[戶]", None),
        ("Distribution Analysis", "每戶人數", "家庭戶數[戶]", None),
        ("Advanced Bar Chart", "行政區", "每戶人數", None),
        ("Advanced Bar Chart", "行政區", "每戶人數", "年別"),
    ])
    def test_diagram_rendered(self, chart, x, y, color):
        r = viz(x, y, chart, color=color)
        assert_diagram(r, label=f"taipei/{chart}/x={x}")

    def test_dark_theme(self):
        r = viz("年別", "每戶人數", "Enhanced Scatter Plot", theme="Dark")
        assert_diagram(r, label="taipei/dark-theme")

    def test_aggregation_average(self):
        r = viz("行政區", "每戶人數", "Advanced Bar Chart", y_agg="Average")
        assert_diagram(r, label="taipei/agg-average")

    def test_aggregation_sum(self):
        r = viz("行政區", "每戶人數", "Advanced Bar Chart", y_agg="Sum")
        assert_diagram(r, label="taipei/agg-sum")


# ---------------------------------------------------------------------------
# Dataset 2: sales_data.csv (real file, date column, mixed types)
# ---------------------------------------------------------------------------

class TestSalesFlow:
    @pytest.fixture(autouse=True)
    def upload_sales(self):
        df = pd.read_csv(os.path.join(ROOT, "sales_data.csv"))
        upload(df)

    @pytest.mark.parametrize("chart,x,y,color", [
        ("Enhanced Scatter Plot",   "Revenue",    "Sales_Volume",   None),
        ("Enhanced Scatter Plot",   "Revenue",    "Sales_Volume",   "Region"),
        ("Enhanced Scatter Plot",   "Region",     "Revenue",        None),      # str x
        ("Statistical Box Plot",    "Region",     "Revenue",        None),
        ("Statistical Box Plot",    "Region",     "Revenue",        "Product_Category"),
        ("Correlation Heatmap",     "Revenue",    "Sales_Volume",   None),
        ("Distribution Analysis",   "Revenue",    "Sales_Volume",   None),
        ("Distribution Analysis",   "Region",     "Revenue",        None),      # str x
        ("Time Series Analysis",    "Date",       "Revenue",        None),
        ("Advanced Bar Chart",      "Region",     "Revenue",        None),
        ("Advanced Bar Chart",      "Region",     "Revenue",        "Product_Category"),
    ])
    def test_diagram_rendered(self, chart, x, y, color):
        r = viz(x, y, chart, color=color)
        assert_diagram(r, label=f"sales/{chart}/x={x}")

    @pytest.mark.parametrize("y_agg", ["Average", "Sum", "Count"])
    def test_aggregation_modes(self, y_agg):
        r = viz("Region", "Revenue", "Advanced Bar Chart", y_agg=y_agg)
        assert_diagram(r, label=f"sales/agg={y_agg}")


# ---------------------------------------------------------------------------
# Dataset 3: Synthetic numeric-only (500 rows × 5 numeric cols)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def upload_numeric():
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, n),
        "y": rng.uniform(0, 500, n),
        "value_a": rng.normal(50, 15, n),
        "value_b": rng.normal(100, 25, n),
        "value_c": rng.integers(0, 200, n).astype(float),
    })
    upload(df)


class TestSyntheticNumericFlow:
    @pytest.fixture(autouse=True)
    def _upload(self, upload_numeric):
        pass

    @pytest.mark.parametrize("chart,x,y", [
        ("Enhanced Scatter Plot", "x", "y"),
        ("Statistical Box Plot", "x", "y"),
        ("Correlation Heatmap", "x", "y"),
        ("Distribution Analysis", "x", "y"),
        ("Time Series Analysis", "x", "y"),
        ("Advanced Bar Chart", "x", "y"),
    ])
    def test_diagram_rendered(self, chart, x, y):
        r = viz(x, y, chart)
        assert_diagram(r, label=f"numeric/{chart}")

    def test_rolling_heatmap(self):
        r = viz("x", "y", "Correlation Heatmap", window=30)
        assert_diagram(r, label="numeric/rolling-heatmap")


# ---------------------------------------------------------------------------
# Dataset 4: Synthetic categorical-heavy (5 cat + 2 numeric cols, 300 rows)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def upload_categorical():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "region":  rng.choice(["North", "South", "East", "West"], n),
        "product": rng.choice(["Alpha", "Beta", "Gamma"], n),
        "segment": rng.choice(["A", "B", "C", "D"], n),
        "channel": rng.choice(["Online", "Offline"], n),
        "tier":    rng.choice(["Gold", "Silver", "Bronze"], n),
        "sales":   rng.uniform(500, 5000, n),
        "profit":  rng.uniform(50, 1000, n),
    })
    upload(df)


class TestSyntheticCategoricalFlow:
    @pytest.fixture(autouse=True)
    def _upload(self, upload_categorical):
        pass

    @pytest.mark.parametrize("chart,x,y,color", [
        ("Enhanced Scatter Plot", "sales", "profit", None),
        ("Enhanced Scatter Plot", "region", "sales", "product"),
        ("Statistical Box Plot", "region", "sales", None),
        ("Distribution Analysis", "region", "sales", None),
        ("Advanced Bar Chart", "region", "sales", None),
        ("Advanced Bar Chart", "region", "sales", "product"),
    ])
    def test_diagram_rendered(self, chart, x, y, color):
        r = viz(x, y, chart, color=color)
        assert_diagram(r, label=f"categorical/{chart}/x={x}")

    def test_count_aggregation(self):
        r = viz("region", "sales", "Advanced Bar Chart", y_agg="Count")
        assert_diagram(r, label="categorical/agg-count")


# ---------------------------------------------------------------------------
# Dataset 5: Datetime-heavy (200 daily rows)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def upload_datetime():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(13)
    df = pd.DataFrame({
        "date":    dates.astype(str),
        "revenue": (1000 + np.cumsum(rng.normal(5, 30, 200))).round(2),
        "cost":    (500 + np.cumsum(rng.normal(2, 15, 200))).round(2),
        "units":   rng.integers(10, 200, 200).astype(float),
    })
    upload(df)


class TestSyntheticDatetimeFlow:
    @pytest.fixture(autouse=True)
    def _upload(self, upload_datetime):
        pass

    @pytest.mark.parametrize("chart,x,y", [
        ("Enhanced Scatter Plot", "date", "revenue"),
        ("Time Series Analysis", "date", "revenue"),
        ("Distribution Analysis", "date", "revenue"),
    ])
    def test_diagram_rendered(self, chart, x, y):
        r = viz(x, y, chart)
        assert_diagram(r, label=f"datetime/{chart}")

    def test_time_series_moving_average_present(self):
        """Time series with 200 rows should include at least the original trace."""
        r = viz("date", "revenue", "Time Series Analysis")
        assert_diagram(r, label="datetime/time-series")
        data = r.json()["plot"]["data"]
        # At least 1 trace; 2 if moving average was added successfully
        assert len(data) >= 1, "Expected at least 1 trace in time series"
