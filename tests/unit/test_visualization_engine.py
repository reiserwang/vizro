"""
Unit tests for src/engines/visualization_engine.py.

All tests patch `dashboard_config.current_data` directly so no server or
HTTP layer is required.
"""
import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.engines.visualization_engine import (
    create_visualization,
    create_vizro_enhanced_visualization,
    create_enhanced_scatter_plot,
    create_statistical_box_plot,
    create_correlation_heatmap,
    create_distribution_analysis,
    create_time_series_analysis,
    create_advanced_bar_chart,
)

# The visualization engine imports dashboard_config at module level;
# we must patch the attribute on the *object already bound* inside the engine.
import src.engines.visualization_engine as _viz_engine
_dashboard_config = _viz_engine.dashboard_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_data(df):
    """Patch current_data on the dashboard_config object bound inside the engine module."""
    return patch.object(_dashboard_config, "current_data", df)


def _is_valid_figure(fig) -> bool:
    return fig is not None and isinstance(fig, go.Figure)


# ===========================================================================
# create_visualization (standard charts)
# ===========================================================================

class TestCreateVisualization:
    """Tests for the standard (non-Vizro) visualization function."""

    def test_scatter_numeric(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_visualization("x", "y", None, "Scatter Plot", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_line_numeric(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_visualization("x", "y", None, "Line Chart", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_bar_categorical(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_visualization("region", "sales", None, "Bar Chart", "Dark", "Average")
        assert _is_valid_figure(fig)

    def test_histogram(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_visualization("x", "y", None, "Histogram", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_no_data_returns_none(self):
        with _patch_data(None):
            fig = create_visualization("x", "y", None, "Scatter Plot", "Light", "Raw Data")
        assert fig is None

    def test_unknown_chart_type_returns_none(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_visualization("x", "y", None, "Unknown Chart", "Light", "Raw Data")
        assert fig is None

    def test_missing_x_axis_returns_none(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_visualization("", "y", None, "Scatter Plot", "Light", "Raw Data")
        assert fig is None

    def test_aggregation_average(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_visualization("region", "sales", None, "Bar Chart", "Light", "Average")
        assert _is_valid_figure(fig)

    def test_aggregation_sum_with_color(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_visualization("region", "sales", "product", "Bar Chart", "Light", "Sum")
        assert _is_valid_figure(fig)

    def test_aggregation_count(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_visualization("region", "sales", None, "Bar Chart", "Light", "Count")
        assert _is_valid_figure(fig)

    def test_datetime_x_axis_line(self, datetime_df):
        with _patch_data(datetime_df):
            fig = create_visualization("date", "revenue", None, "Line Chart", "Light", "Raw Data")
        assert _is_valid_figure(fig)


# ===========================================================================
# create_enhanced_scatter_plot
# ===========================================================================

class TestEnhancedScatterPlot:
    """Tests specifically for the enhanced scatter function."""

    def test_numeric_x_numeric_y(self, numeric_df):
        fig = create_enhanced_scatter_plot(numeric_df, "x", "y", None, "y", "plotly_white")
        assert _is_valid_figure(fig)

    def test_categorical_x_no_ols_crash(self, categorical_df):
        """Non-numeric x must NOT trigger trendline='ols' (would crash)."""
        fig = create_enhanced_scatter_plot(categorical_df, "region", "sales", None, "sales", "plotly_dark")
        assert _is_valid_figure(fig)

    def test_datetime_x(self, datetime_df):
        fig = create_enhanced_scatter_plot(datetime_df, "date", "revenue", None, "revenue", "plotly_white")
        assert _is_valid_figure(fig)

    def test_with_color_var(self, categorical_df):
        fig = create_enhanced_scatter_plot(categorical_df, "sales", "profit", "region", "profit", "plotly_white")
        assert _is_valid_figure(fig)

    def test_taipei_string_x(self, taipei_df):
        """Real taipei.csv data: 年別 is string — must not crash."""
        fig = create_enhanced_scatter_plot(taipei_df, "年別", "每戶人數", "行政區", "每戶人數", "plotly_dark")
        assert _is_valid_figure(fig)


# ===========================================================================
# create_statistical_box_plot
# ===========================================================================

class TestBoxPlot:
    def test_basic_box(self, categorical_df):
        fig = create_statistical_box_plot(categorical_df, "region", "sales", None, "Sales", "plotly_white")
        assert _is_valid_figure(fig)

    def test_with_color(self, categorical_df):
        fig = create_statistical_box_plot(categorical_df, "region", "sales", "product", "Sales", "plotly_dark")
        assert _is_valid_figure(fig)

    def test_sales_data(self, sales_df):
        fig = create_statistical_box_plot(sales_df, "Region", "Revenue", "Product_Category", "Revenue", "plotly_white")
        assert _is_valid_figure(fig)


# ===========================================================================
# create_correlation_heatmap
# ===========================================================================

class TestCorrelationHeatmap:
    def test_sufficient_numeric_columns(self, numeric_df):
        fig = create_correlation_heatmap(numeric_df, "plotly_white")
        assert _is_valid_figure(fig)

    def test_only_one_numeric_column_returns_none(self, single_numeric_col_df):
        fig = create_correlation_heatmap(single_numeric_col_df, "plotly_white")
        assert fig is None

    def test_rolling_window(self, numeric_df):
        fig = create_correlation_heatmap(numeric_df, "plotly_dark", window_size=30)
        assert _is_valid_figure(fig)

    def test_sales_heatmap(self, sales_df):
        numeric_sales = sales_df.select_dtypes(include=[np.number])
        fig = create_correlation_heatmap(numeric_sales, "plotly_white")
        assert _is_valid_figure(fig)


# ===========================================================================
# create_distribution_analysis
# ===========================================================================

class TestDistributionAnalysis:
    def test_numeric_x_and_y(self, numeric_df):
        fig = create_distribution_analysis(numeric_df, "x", "y", "y", "plotly_white")
        assert _is_valid_figure(fig)

    def test_categorical_x(self, categorical_df):
        fig = create_distribution_analysis(categorical_df, "region", "sales", "sales", "plotly_dark")
        assert _is_valid_figure(fig)

    def test_datetime_x(self, datetime_df):
        fig = create_distribution_analysis(datetime_df, "date", "revenue", "revenue", "plotly_white")
        assert _is_valid_figure(fig)


# ===========================================================================
# create_time_series_analysis
# ===========================================================================

class TestTimeSeriesAnalysis:
    def test_datetime_x(self, datetime_df):
        fig = create_time_series_analysis(datetime_df, "date", "revenue", None, "Revenue", "plotly_white")
        assert _is_valid_figure(fig)

    def test_numeric_fallback(self, numeric_df):
        """Non-datetime x falls back to regular line."""
        fig = create_time_series_analysis(numeric_df, "x", "y", None, "y", "plotly_dark")
        assert _is_valid_figure(fig)

    def test_with_color(self, sales_df):
        fig = create_time_series_analysis(sales_df, "Date", "Revenue", "Region", "Revenue", "plotly_white")
        assert _is_valid_figure(fig)

    def test_moving_average_added_for_large_data(self, datetime_df):
        """With 200 rows, a moving average trace should be appended."""
        fig = create_time_series_analysis(datetime_df, "date", "revenue", None, "Revenue", "plotly_white")
        assert _is_valid_figure(fig)
        # At least 2 traces: original + moving average
        assert len(fig.data) >= 2


# ===========================================================================
# create_advanced_bar_chart
# ===========================================================================

class TestAdvancedBarChart:
    def test_no_color_average(self, categorical_df):
        fig = create_advanced_bar_chart(categorical_df, "region", "sales", None, "Sales", "plotly_white", "Average")
        assert _is_valid_figure(fig)

    def test_with_color(self, categorical_df):
        fig = create_advanced_bar_chart(categorical_df, "region", "sales", "product", "Sales", "plotly_dark", "Average")
        assert _is_valid_figure(fig)

    def test_count_aggregation(self, categorical_df):
        fig = create_advanced_bar_chart(categorical_df, "region", "sales", None, "Count", "plotly_white", "Count")
        assert _is_valid_figure(fig)


# ===========================================================================
# create_vizro_enhanced_visualization  (full dispatch + fallback)
# ===========================================================================

CHART_TYPES = [
    ("Enhanced Scatter Plot", "x", "y"),
    ("Statistical Box Plot", "region", "sales"),
    ("Correlation Heatmap", "x", "y"),
    ("Distribution Analysis", "x", "y"),
    ("Advanced Bar Chart", "region", "sales"),
]


class TestVizroEnhancedVisualization:

    def test_no_data_returns_none(self):
        with _patch_data(None):
            fig = create_vizro_enhanced_visualization("x", "y", None, "Enhanced Scatter Plot", "Light", "Raw Data")
        assert fig is None

    def test_missing_axes_returns_none(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_vizro_enhanced_visualization("", "", None, "Enhanced Scatter Plot", "Light", "Raw Data")
        assert fig is None

    def test_enhanced_scatter_numeric(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_vizro_enhanced_visualization("x", "y", None, "Enhanced Scatter Plot", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_enhanced_scatter_categorical_x(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_vizro_enhanced_visualization("region", "sales", None, "Enhanced Scatter Plot", "Dark", "Raw Data")
        assert _is_valid_figure(fig)

    def test_enhanced_scatter_datetime_x(self, datetime_df):
        with _patch_data(datetime_df):
            fig = create_vizro_enhanced_visualization("date", "revenue", None, "Enhanced Scatter Plot", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_box_plot(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_vizro_enhanced_visualization("region", "sales", None, "Statistical Box Plot", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_heatmap(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_vizro_enhanced_visualization("x", "y", None, "Correlation Heatmap", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_distribution_analysis(self, numeric_df):
        with _patch_data(numeric_df):
            fig = create_vizro_enhanced_visualization("x", "y", None, "Distribution Analysis", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_time_series(self, datetime_df):
        with _patch_data(datetime_df):
            fig = create_vizro_enhanced_visualization("date", "revenue", None, "Time Series Analysis", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_advanced_bar(self, categorical_df):
        with _patch_data(categorical_df):
            fig = create_vizro_enhanced_visualization("region", "sales", None, "Advanced Bar Chart", "Light", "Average")
        assert _is_valid_figure(fig)

    def test_taipei_enhanced_scatter(self, taipei_df):
        """Real taipei.csv: 年別 (string) x-axis must not crash."""
        with _patch_data(taipei_df):
            fig = create_vizro_enhanced_visualization("年別", "每戶人數", "行政區", "Enhanced Scatter Plot", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_taipei_box_plot(self, taipei_df):
        with _patch_data(taipei_df):
            fig = create_vizro_enhanced_visualization("行政區", "每戶人數", None, "Statistical Box Plot", "Dark", "Raw Data")
        assert _is_valid_figure(fig)

    def test_taipei_heatmap(self, taipei_df):
        with _patch_data(taipei_df):
            fig = create_vizro_enhanced_visualization("年別", "每戶人數", None, "Correlation Heatmap", "Light", "Raw Data")
        assert _is_valid_figure(fig)

    def test_sales_all_basic_chart_types(self, sales_df):
        """sales_data.csv: smoke-test all 6 enhanced chart types."""
        configs = [
            ("Enhanced Scatter Plot", "Revenue", "Sales_Volume", None),
            ("Statistical Box Plot", "Region", "Revenue", "Product_Category"),
            ("Correlation Heatmap", "Revenue", "Sales_Volume", None),
            ("Distribution Analysis", "Revenue", "Sales_Volume", None),
            ("Time Series Analysis", "Date", "Revenue", None),
            ("Advanced Bar Chart", "Region", "Revenue", None),
        ]
        with _patch_data(sales_df):
            for chart_type, x, y, color in configs:
                fig = create_vizro_enhanced_visualization(x, y, color, chart_type, "Light", "Raw Data")
                assert _is_valid_figure(fig), f"Failed for {chart_type} with x={x}, y={y}"
