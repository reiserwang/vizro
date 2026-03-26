"""
tests/unit/test_axis_reactivity.py

Tests verifying that:
1. Changing X/Y/color/chart-type produces genuinely different plot data (API-layer)
2. Y-axis values in the returned plot match actual DataFrame column ranges (Bug 2 regression)
3. Color variable correctly splits a scatter into N separate color groups
4. Chart type changes are reflected in the trace type returned
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi.testclient import TestClient
from src.api.routes import app
from tests.conftest import df_to_csv_bytes, _numeric_df, _categorical_df

client = TestClient(app)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def upload(df):
    content = df_to_csv_bytes(df)
    r = client.post("/api/v1/upload", files={"file": ("data.csv", content, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def viz(x, y, chart="Enhanced Scatter Plot", color=None, theme="Light", y_agg="Raw Data"):
    r = client.post("/api/v1/visualize", json={
        "x_axis": x, "y_axis": y, "color_var": color,
        "chart_type": chart, "theme": theme,
        "y_axis_agg": y_agg, "correlation_window": 0,
    })
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()["plot"]

def _decode_plotly_y(y_field):
    """
    Plotly's to_json() serializes large float arrays as a binary-encoded dict:
        {'dtype': 'f8', 'bdata': '<base64>'}
    This helper decodes it back to a Python list of floats.
    Falls back to treating it as a plain list for small arrays.
    """
    if isinstance(y_field, list):
        return [float(v) for v in y_field if v is not None]
    if isinstance(y_field, dict) and "bdata" in y_field:
        import base64, struct
        dtype = y_field.get("dtype", "f8")
        raw = base64.b64decode(y_field["bdata"])
        # dtype 'f8' = float64 (8 bytes per element)
        fmt_map = {"f8": ("d", 8), "f4": ("f", 4), "i4": ("i", 4), "i8": ("q", 8)}
        fmt_char, size = fmt_map.get(dtype, ("d", 8))
        n = len(raw) // size
        return list(struct.unpack_from(f"<{n}{fmt_char}", raw))
    return []



def marker_y_values(plot):
    """
    Collect numeric y values from scatter/scattergl traces with mode='markers'.
    Handles Plotly binary-encoded {dtype, bdata} y-arrays (used for large float arrays).
    """
    vals = []
    for trace in plot.get("data", []):
        if trace.get("type") not in ("scatter", "scattergl"):
            continue
        mode = trace.get("mode", "")
        if "markers" not in mode:
            continue
        vals.extend(_decode_plotly_y(trace.get("y")))
    return vals


def color_group_names(plot):

    """
    Extracts the set of distinct legend group names from scatter-marker traces.
    One name per unique color category.
    """
    groups = set()
    for trace in plot.get("data", []):
        if trace.get("type") not in ("scatter", "scattergl"):
            continue
        mode = trace.get("mode", "")
        if "markers" not in mode:
            continue
        name = trace.get("legendgroup") or trace.get("name") or ""
        if name:
            groups.add(name)
    return groups


def all_scatter_trace_count(plot):
    """Count all traces (any mode) of type scatter/scattergl."""
    return sum(1 for t in plot.get("data", []) if t.get("type") in ("scatter", "scattergl"))


# ===========================================================================
# Bug 1 verification: different x/y → different plot data from the API
# ===========================================================================

class TestAxisChangeProducesDifferentPlot:
    """
    Proves that the API returns different chart data when x or y changes.
    If the frontend re-called the API on every dropdown change the user would
    see a different chart — confirming the back-end is correct and that the 
    stale-render was purely a missing frontend `change` listener (Bug 1).
    """

    @pytest.fixture(autouse=True)
    def load_numeric(self):
        upload(_numeric_df())

    def test_changing_x_returns_different_xaxis_title(self):
        plot_a = viz("x", "y")
        plot_b = viz("value_a", "y")
        title_a = plot_a.get("layout", {}).get("xaxis", {}).get("title", {}).get("text", "")
        title_b = plot_b.get("layout", {}).get("xaxis", {}).get("title", {}).get("text", "")
        assert title_a != title_b, "Different X columns should produce different xaxis titles"

    def test_changing_y_returns_different_yaxis_title(self):
        plot_a = viz("x", "y")
        plot_b = viz("x", "value_a")
        title_a = plot_a.get("layout", {}).get("yaxis", {}).get("title", {}).get("text", "")
        title_b = plot_b.get("layout", {}).get("yaxis", {}).get("title", {}).get("text", "")
        assert title_a != title_b, "Different Y columns should produce different yaxis titles"

    def test_changing_x_changes_chart_title(self):
        plot_a = viz("x", "y")
        plot_b = viz("value_a", "y")
        title_a = plot_a.get("layout", {}).get("title", {}).get("text", "")
        title_b = plot_b.get("layout", {}).get("title", {}).get("text", "")
        assert title_a != title_b, "Chart titles should differ when X column changes"

    def test_changing_chart_type_changes_trace_type(self):
        """Scatter vs Bar must use different Plotly trace types."""
        plot_scatter = viz("x", "y", chart="Enhanced Scatter Plot")
        plot_bar     = viz("x", "y", chart="Advanced Bar Chart")
        type_scatter = plot_scatter["data"][0]["type"]
        type_bar     = plot_bar["data"][0]["type"]
        assert type_scatter != type_bar, (
            f"Scatter vs Bar should produce different trace types "
            f"(got {type_scatter!r} vs {type_bar!r})"
        )

    def test_changing_theme_changes_template(self):
        plot_light = viz("x", "y", theme="Light")
        plot_dark  = viz("x", "y", theme="Dark")
        tmpl_light = str(plot_light.get("layout", {}).get("template", ""))
        tmpl_dark  = str(plot_dark.get("layout", {}).get("template", ""))
        assert tmpl_light != tmpl_dark, "Light vs Dark theme should yield different templates"

    def test_chart_types_that_are_supported_return_200(self):
        """Documented chart types must render without server error."""
        for ct in ("Enhanced Scatter Plot", "Advanced Bar Chart", "Statistical Box Plot"):
            r = client.post("/api/v1/visualize", json={
                "x_axis": "x", "y_axis": "y", "color_var": None,
                "chart_type": ct, "theme": "Light",
                "y_axis_agg": "Raw Data", "correlation_window": 0,
            })
            assert r.status_code == 200, f"Chart type '{ct}' failed: {r.text}"


# ===========================================================================
# Bug 2 verification: y-values must come from actual data, not ordinal indices
# Use categorical X so the engine skips OLS and returns raw row-level markers.
# ===========================================================================

class TestYValuesMatchDataRange:
    """
    Regression for the screenshot showing y-axis values 0–5 instead of actual
    financial values. Using categorical x-axis avoids the OLS path so that
    all raw marker data points can be inspected directly.
    """

    @pytest.fixture(autouse=True)
    def load_categorical(self):
        # Use categorical dataset so scatter skips OLS (categorical x-axis)
        upload(_categorical_df())

    def test_marker_y_values_are_present(self):
        """Marker traces must provide y-values for a scatter with categorical x."""
        plot = viz("region", "sales", chart="Enhanced Scatter Plot")
        y_vals = marker_y_values(plot)
        assert len(y_vals) > 0, "No numeric y-values found in marker traces"

    def test_y_values_within_column_range(self):
        """Marker y-values must lie within the actual column's min/max."""
        df = _categorical_df()
        col_min = float(df["sales"].min())
        col_max = float(df["sales"].max())

        plot = viz("region", "sales", chart="Enhanced Scatter Plot")
        y_vals = marker_y_values(plot)
        assert y_vals, "No y-values found in marker traces"
        for v in y_vals:
            assert col_min - 1 <= v <= col_max + 1, (
                f"y-value {v:.2f} is outside column range [{col_min:.2f}, {col_max:.2f}]"
            )

    def test_y_values_not_sequential_zero_to_n(self):
        """
        Bug 2 regression: 0,1,2,3,...N-1 would indicate ordinal encoding of
        x-axis categories leaking into y positions. Should never happen.
        """
        plot = viz("region", "sales", chart="Enhanced Scatter Plot")
        y_vals = sorted(marker_y_values(plot))
        if len(y_vals) < 2:
            return
        sequential = list(range(len(y_vals)))
        assert y_vals != sequential, (
            "y-values are exactly sequential integers starting at 0 — "
            "ordinal encoding of x-axis categories may have leaked into y-axis"
        )

    def test_y_mean_close_to_column_mean(self):
        """Mean of marker y-values should be close to actual column mean."""
        df = _categorical_df()
        expected_mean = float(df["sales"].mean())

        plot = viz("region", "sales", chart="Enhanced Scatter Plot")
        y_vals = marker_y_values(plot)
        assert y_vals, "No y-values found"
        actual_mean = sum(y_vals) / len(y_vals)
        # Allow 30% tolerance for subsampled/aggregated views
        assert abs(actual_mean - expected_mean) < expected_mean * 0.3, (
            f"Plot mean ({actual_mean:.1f}) too far from column mean ({expected_mean:.1f})"
        )


class TestTaipeiYValuesMatchData:
    """
    Regression using the real taipei.csv — the exact scenario reported:
    X=年別 (categorical), Y=3.財產所得收入[NT], Color=行政區.
    The scatter must show millions-of-NT values, NOT sequential 0-5.
    """

    @pytest.fixture(autouse=True)
    def load_taipei(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "taipei.csv"))
        upload(df)

    def test_financial_column_y_values_not_zero_to_five(self):
        """
        Bug 2 regression: 3.財產所得收入[NT] ranges 3.4M–28.4M.
        No marker y-values should be ≤ 10 (that would indicate ordinal encoding).
        """
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",           # categorical — skips OLS, gets raw data
            "y_axis": "3.財產所得收入[NT]",
            "color_var": "行政區",
            "chart_type": "Enhanced Scatter Plot",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        assert r.status_code == 200, r.text
        plot = r.json()["plot"]
        y_vals = marker_y_values(plot)
        assert y_vals, "No numeric y-values found in marker traces"

        small_count = sum(1 for v in y_vals if v <= 10)
        assert small_count == 0, (
            f"{small_count}/{len(y_vals)} y-values ≤ 10 — financial column should "
            f"have values in the millions. Ordinal encoding may have leaked into y."
        )

    def test_financial_column_y_min_above_one_million(self):
        """y-min for 財產所得 must be >> 1M."""
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "3.財產所得收入[NT]",
            "color_var": None,
            "chart_type": "Enhanced Scatter Plot",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        assert r.status_code == 200, r.text
        y_vals = marker_y_values(r.json()["plot"])
        assert y_vals, "No numeric y-values found"
        assert min(y_vals) > 1_000_000, (
            f"Min y={min(y_vals):,.0f} — expected > 1,000,000 for 財產所得 column"
        )

    def test_color_var_produces_12_color_groups_for_12_districts(self):
        """
        行政區 has 12 unique districts → scatter must produce 12 color groups.
        Bug 2: if color_var is silently ignored, only 1 group appears.
        """
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "每戶人數",
            "color_var": "行政區",
            "chart_type": "Enhanced Scatter Plot",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        assert r.status_code == 200, r.text
        groups = color_group_names(r.json()["plot"])
        assert len(groups) == 12, (
            f"Expected 12 color groups (1 per district), got {len(groups)}: {groups}"
        )

    def test_color_none_produces_single_color_group(self):
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "每戶人數",
            "color_var": None,
            "chart_type": "Enhanced Scatter Plot",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        assert r.status_code == 200, r.text
        groups = color_group_names(r.json()["plot"])
        assert len(groups) <= 1, (
            f"Expected ≤1 group with no color grouping, got {len(groups)}: {groups}"
        )

    def test_all_numeric_columns_as_y_return_200_and_data(self):
        """Every numeric column in taipei.csv must render without error."""
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "taipei.csv"))
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols[:10]:      # first 10 for speed
            r = client.post("/api/v1/visualize", json={
                "x_axis": "年別",
                "y_axis": col,
                "color_var": "行政區",
                "chart_type": "Enhanced Scatter Plot",
                "theme": "Light",
                "y_axis_agg": "Raw Data",
                "correlation_window": 0,
            })
            assert r.status_code == 200, f"Column '{col}' failed: {r.text}"
            y_vals = marker_y_values(r.json()["plot"])
            assert y_vals, f"No numeric y-values for column '{col}'"

class TestAdvancedBarChartYValues:
    """
    Regression for the 'Y-axis shows counts instead of volumes' bug in Advanced Bar Chart.
    Ensures that when y_axis_agg is 'Raw Data', the actual column values are used.
    """
    
    @pytest.fixture(autouse=True)
    def load_taipei(self):
        import pandas as pd
        import os
        df = pd.read_csv(os.path.join(ROOT, "taipei.csv"))
        upload(df)
        
    def test_raw_data_bar_chart_uses_actual_values(self):
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "[1]本業薪資[NT]",
            "color_var": "行政區",
            "chart_type": "Advanced Bar Chart",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        assert r.status_code == 200, r.text
        plot = r.json()["plot"]
        
        # Verify y-axis label is the column name, not "mean" or "count"
        y_title = plot.get("layout", {}).get("yaxis", {}).get("title", {}).get("text", "")
        # The new logic wraps it in "Title by X", but we can check if it at least contains the column name
        assert "[1]本業薪資[NT]" in y_title, f"Expected y-axis title to contain the column name, got {y_title}"
        
        # Find 109年 中山區 (which has a known value ~72M)
        found_val = None
        for trace in plot.get("data", []):
            if trace.get("name") == "中山區":
                # Assuming first X value is 109年 (after sorting)
                y_arr = _decode_plotly_y(trace.get("y", []))
                x_arr = trace.get("x", [])
                for x, y in zip(x_arr, y_arr):
                    if x == "109年":
                        found_val = y
        
        assert found_val is not None, "Could not find 109年 中山區 in plot data"
        assert found_val > 70_000_000, f"Expected 109年 中山區 salary to be > 70M, got {found_val}"
        
    def test_missing_data_years_are_nan_not_zero(self):
        """113年 and 114年 have NaN for salary. They should not be plotted as zeroes/counts."""
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "[1]本業薪資[NT]",
            "color_var": "行政區",
            "chart_type": "Advanced Bar Chart",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        plot = r.json()["plot"]
        
        for trace in plot.get("data", []):
            x_arr = trace.get("x", [])
            assert "113年" not in x_arr, "113年 should be dropped entirely since it has NaN salary"
            assert "114年" not in x_arr, "114年 should be dropped entirely since it has NaN salary"



# ===========================================================================
# Color variable rendering: categorical splits
# ===========================================================================

class TestColorVariableRendering:
    """Verify that color_var correctly separates data into N color groups."""

    @pytest.fixture(autouse=True)
    def load_categorical(self):
        upload(_categorical_df())

    def test_color_region_produces_4_groups(self):
        """_categorical_df has 4 regions → 4 legend groups in scatter markers."""
        plot = viz("region", "sales", color="region")
        groups = color_group_names(plot)
        assert len(groups) == 4, (
            f"Expected 4 color groups for 4 regions, got {len(groups)}: {groups}"
        )

    def test_no_color_produces_at_most_one_group(self):
        plot = viz("region", "sales", color=None)
        groups = color_group_names(plot)
        assert len(groups) <= 1, (
            f"Expected ≤1 group with no color grouping, got {len(groups)}: {groups}"
        )

    def test_color_trace_names_match_unique_values(self):
        """Legend group names must correspond to region values."""
        plot = viz("region", "sales", color="region")
        groups = color_group_names(plot)
        expected = {"North", "South", "East", "West"}
        assert groups == expected, (
            f"Legend groups {groups} don't match expected regions {expected}"
        )

    def test_different_color_columns_produce_different_n_groups(self):
        """region (4 values) vs product (3 values) → different group counts."""
        plot_region  = viz("region", "sales", color="region")
        plot_product = viz("region", "sales", color="product")
        n_region  = len(color_group_names(plot_region))
        n_product = len(color_group_names(plot_product))
        assert n_region != n_product, "Color columns with different cardinalities must yield different group counts"
        assert n_region  == 4, f"Expected 4 for region, got {n_region}"
        assert n_product == 3, f"Expected 3 for product, got {n_product}"

    def test_changing_color_from_categorical_to_none_reduces_groups(self):
        """Removing color grouping must reduce legend diversity."""
        n_with    = len(color_group_names(viz("region", "sales", color="region")))
        n_without = len(color_group_names(viz("region", "sales", color=None)))
        assert n_with > n_without, (
            f"Grouped ({n_with}) should have more groups than ungrouped ({n_without})"
        )

class TestAdvancedBarChartYValues:
    """
    Regression for the 'Y-axis shows counts instead of volumes' bug in Advanced Bar Chart.
    Ensures that when y_axis_agg is 'Raw Data', the actual column values are used.
    """
    
    @pytest.fixture(autouse=True)
    def load_taipei(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "taipei.csv"))
        upload(df)
        
    def test_raw_data_bar_chart_uses_actual_values(self):
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "[1]本業薪資[NT]",
            "color_var": "行政區",
            "chart_type": "Advanced Bar Chart",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        assert r.status_code == 200, r.text
        plot = r.json()["plot"]
        
        # Verify y-axis label is the column name, not "mean" or "count"
        y_title = plot.get("layout", {}).get("yaxis", {}).get("title", {}).get("text", "")
        assert y_title == "[1]本業薪資[NT]", f"Expected y-axis title to be the column name, got {y_title}"
        
        # Find 109年 中山區 (which has a known value ~72M)
        found_val = None
        for trace in plot.get("data", []):
            if trace.get("name") == "中山區":
                # Assuming first X value is 109年 (after sorting)
                y_arr = _decode_plotly_y(trace.get("y", []))
                x_arr = trace.get("x", [])
                for x, y in zip(x_arr, y_arr):
                    if x == "109年":
                        found_val = y
        
        assert found_val is not None, "Could not find 109年 中山區 in plot data"
        assert found_val > 70_000_000, f"Expected 109年 中山區 salary to be > 70M, got {found_val}"
        
    def test_missing_data_years_are_nan_not_zero(self):
        """113年 and 114年 have NaN for salary. They should not be plotted as zeroes/counts."""
        r = client.post("/api/v1/visualize", json={
            "x_axis": "年別",
            "y_axis": "[1]本業薪資[NT]",
            "color_var": "行政區",
            "chart_type": "Advanced Bar Chart",
            "theme": "Light",
            "y_axis_agg": "Raw Data",
            "correlation_window": 0,
        })
        plot = r.json()["plot"]
        
        for trace in plot.get("data", []):
            x_arr = trace.get("x", [])
            assert "113年" not in x_arr, "113年 should be dropped entirely since it has NaN salary"
            assert "114年" not in x_arr, "114年 should be dropped entirely since it has NaN salary"

