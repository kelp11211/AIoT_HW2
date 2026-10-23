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
