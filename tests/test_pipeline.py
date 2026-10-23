"""Initial sanity test for HW2 pipeline environment."""

import numpy as np
import pandas as pd
import sklearn
import statsmodels
import streamlit

def test_environment_packages():
    """Verify core packages are imported and working."""
    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None
    assert statsmodels.__version__ is not None
    assert streamlit.__version__ is not None

def test_random_seed():
    """Verify reproducible random seed."""
    np.random.seed(42)
    val = np.random.rand(3)
    np.random.seed(42)
    val2 = np.random.rand(3)
    assert np.allclose(val, val2)

def test_gate1_flood_dataset_spec():
    """Verify Flood dataset conforms to HW2 Gate 1 requirements (>13 features, continuous target)."""
    from pathlib import Path
    data_path = Path(__file__).resolve().parent.parent / "data" / "flood" / "train.csv"
    assert data_path.exists(), f"Flood train dataset missing: {data_path}"
    df = pd.read_csv(data_path, nrows=50)
    assert "FloodProbability" in df.columns
    assert "id" in df.columns
    feature_cols = [c for c in df.columns if c not in ["FloodProbability", "id", "Unnamed: 0"]]
    assert len(feature_cols) == 20, f"Expected 20 features, got {len(feature_cols)}"
    assert 14 <= len(feature_cols) <= 39

def test_gate1_insurance_dataset_spec():
    """Verify Insurance dataset conforms to HW2 Gate 1 requirements (>13 features, continuous target)."""
    from pathlib import Path
    data_path = Path(__file__).resolve().parent.parent / "data" / "insurance" / "train.csv"
    assert data_path.exists(), f"Insurance train dataset missing: {data_path}"
    df = pd.read_csv(data_path, nrows=50)
    assert "Premium Amount" in df.columns
    assert "id" in df.columns
    feature_cols = [c for c in df.columns if c not in ["Premium Amount", "id", "Unnamed: 0"]]
    assert len(feature_cols) == 19, f"Expected 19 features, got {len(feature_cols)}"
    assert 14 <= len(feature_cols) <= 39

