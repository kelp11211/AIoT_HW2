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

def test_gate2_data_split_no_leakage():
    """Verify train_test_split has no index overlap and matches 80/20 ratio."""
    from src.preprocessing import split_data
    df = pd.DataFrame({
        "feat1": np.arange(100),
        "feat2": np.arange(100) * 2,
        "target": np.arange(100) * 0.5
    })
    X_train, X_test, y_train, y_test = split_data(df, "target", ["feat1", "feat2"], test_size=0.2, random_state=42)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_gate2_preprocessor_pipeline():
    """Verify ColumnTransformer handles numerical and categorical without data leakage."""
    from src.preprocessing import build_preprocessor
    df_train = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "cat1": ["A", "B", "A", None]
    })
    df_test = pd.DataFrame({
        "num1": [5.0, np.nan],
        "cat1": ["B", "NEW_CATEGORY"]  # Test unseen category handling
    })
    preprocessor = build_preprocessor(num_cols=["num1"], cat_cols=["cat1"])
    X_train_trans = preprocessor.fit_transform(df_train)
    X_test_trans = preprocessor.transform(df_test)

    assert X_train_trans.shape[0] == 4
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()

def test_gate3_feature_selector():
    """Verify SelectKBest correctly selects top k features without error."""
    from src.feature_selection import get_feature_selector
    selector = get_feature_selector(k=2)
    X = np.random.RandomState(42).randn(50, 5)
    y = X[:, 0] * 3.0 + X[:, 1] * 2.0 + np.random.RandomState(42).randn(50) * 0.1
    X_selected = selector.fit_transform(X, y)
    assert X_selected.shape == (50, 2)
    assert selector.get_support().sum() == 2

def test_gate3_modeling_pipelines():
    """Verify all model pipelines (linear, ridge, lasso, dummy) train and predict cleanly."""
    from src.modeling import build_pipeline
    from src.preprocessing import build_preprocessor
    df_train = pd.DataFrame({
        "num1": np.linspace(0, 10, 60),
        "num2": np.linspace(10, 20, 60),
    })
    y_train = df_train["num1"] * 2.0 + df_train["num2"] * 0.5 + 1.0

    preprocessor = build_preprocessor(num_cols=["num1", "num2"])
    for m_type in ["dummy", "linear", "ridge", "lasso"]:
        pipe = build_pipeline(preprocessor, model_type=m_type)
        pipe.fit(df_train, y_train)
        preds = pipe.predict(df_train)
        assert len(preds) == len(df_train)
        assert not np.isnan(preds).any()

def test_gate4_conformal_predictor():
    """Verify SplitConformalPredictor calculates calibrated intervals and valid coverage."""
    from src.intervals import SplitConformalPredictor
    predictor = SplitConformalPredictor(alpha=0.05)
    
    # Generate calibrated synthetic data
    rng = np.random.RandomState(42)
    y_cal_true = rng.randn(1000)
    y_cal_pred = y_cal_true + rng.normal(0, 0.1, size=1000)
    
    predictor.calibrate(y_cal_true, y_cal_pred)
    assert predictor.q_hat is not None
    assert predictor.q_hat > 0

    # Test coverage
    y_test_true = rng.randn(500)
    y_test_pred = y_test_true + rng.normal(0, 0.1, size=500)
    summary = predictor.evaluate_coverage(y_test_true, y_test_pred)
    assert 0.90 <= summary["empirical_coverage"] <= 1.0
    assert np.isclose(summary["average_width"], 2 * predictor.q_hat)


def test_gate4_regression_metrics():
    """Verify regression metrics calculation on exact synthetic data."""
    from src.evaluation import compute_regression_metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_regression_metrics(y_true, y_pred, n_features=2)
    assert metrics["MSE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0
    assert metrics["Adjusted_R2"] == 1.0




