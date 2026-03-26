"""
Shared pytest fixtures for the vizro test suite.
"""
import pytest
import pandas as pd
import numpy as np
from io import BytesIO


# ---------------------------------------------------------------------------
# Synthetic DataFrame factories
# ---------------------------------------------------------------------------

def _numeric_df() -> pd.DataFrame:
    """500-row, 5-column purely numeric dataset."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "x": rng.uniform(0, 100, n),
        "y": rng.uniform(0, 500, n),
        "value_a": rng.normal(50, 15, n),
        "value_b": rng.normal(100, 25, n),
        "value_c": rng.integers(0, 200, n).astype(float),
    })


def _categorical_df() -> pd.DataFrame:
    """Dataset with 5 categorical + 2 numeric columns."""
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "region": rng.choice(["North", "South", "East", "West"], n),
        "product": rng.choice(["Alpha", "Beta", "Gamma"], n),
        "segment": rng.choice(["A", "B", "C", "D"], n),
        "channel": rng.choice(["Online", "Offline"], n),
        "tier": rng.choice(["Gold", "Silver", "Bronze"], n),
        "sales": rng.uniform(500, 5000, n),
        "profit": rng.uniform(50, 1000, n),
    })


def _datetime_df() -> pd.DataFrame:
    """Time-series dataset with a datetime x-axis."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(13)
    return pd.DataFrame({
        "date": dates,
        "revenue": 1000 + np.cumsum(rng.normal(5, 30, 200)),
        "cost": 500 + np.cumsum(rng.normal(2, 15, 200)),
        "units": rng.integers(10, 200, 200).astype(float),
    })


def _tiny_df() -> pd.DataFrame:
    """Minimal 3-row dataset for edge-case tests."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0],
    })


def _single_numeric_col_df() -> pd.DataFrame:
    """Only one numeric column — heatmap should return None."""
    return pd.DataFrame({
        "category": ["A", "B", "C", "D"] * 25,
        "value": np.random.default_rng(99).uniform(0, 1, 100),
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    return _numeric_df()


@pytest.fixture
def categorical_df():
    return _categorical_df()


@pytest.fixture
def datetime_df():
    return _datetime_df()


@pytest.fixture
def tiny_df():
    return _tiny_df()


@pytest.fixture
def single_numeric_col_df():
    return _single_numeric_col_df()


@pytest.fixture
def taipei_df(tmp_path):
    """Load the real taipei.csv from the project root."""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(root, "taipei.csv")
    return pd.read_csv(csv_path)


@pytest.fixture
def sales_df(tmp_path):
    """Load the real sales_data.csv from the project root."""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(root, "sales_data.csv")
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# CSV bytes helpers (for API upload tests)
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
