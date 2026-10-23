"""Modeling Module for AIoT HW2.

Constructs sklearn Pipelines for:
1. Baseline (DummyRegressor)
2. Full Features Multiple Linear Regression
3. Selected Features Multiple Linear Regression
4. Ridge Regression (L2 regularization with cross-validated alpha)
5. Lasso Regression (L1 regularization with cross-validated alpha)
"""

from typing import Dict, Any, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate


def build_pipeline(
    preprocessor: ColumnTransformer,
    model_type: str = "linear",
    selector: Optional[SelectKBest] = None,
    alpha_cv: int = 5,
    random_state: int = 42
) -> Pipeline:
    """Build a complete preprocessing + (optional feature selection) + regression pipeline."""
    steps = [("preprocessor", preprocessor)]

    if selector is not None:
        steps.append(("selector", selector))

    if model_type == "dummy":
        regressor = DummyRegressor(strategy="mean")
    elif model_type == "linear":
        regressor = LinearRegression()
    elif model_type == "ridge":
        alphas = np.logspace(-2, 3, 6)
        cv = KFold(n_splits=alpha_cv, shuffle=True, random_state=random_state)
        regressor = RidgeCV(alphas=alphas, cv=cv)
    elif model_type == "lasso":
        alphas = [1e-4, 1e-3, 1e-2, 1e-1]
        cv = KFold(n_splits=alpha_cv, shuffle=True, random_state=random_state)
        regressor = LassoCV(alphas=alphas, cv=cv, random_state=random_state, max_iter=3000)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    steps.append(("regressor", regressor))
    return Pipeline(steps)


def evaluate_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    sample_size: int = 100000,
    random_state: int = 42
) -> Dict[str, float]:
    """Perform K-fold Cross Validation exclusively on training set to obtain CV metrics."""
    if len(X_train) > sample_size:
        idx = np.random.RandomState(random_state).choice(len(X_train), size=sample_size, replace=False)
        X_eval = X_train.iloc[idx]
        y_eval = y_train.iloc[idx]
    else:
        X_eval = X_train
        y_eval = y_train

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error"
    }

    cv_results = cross_validate(pipeline, X_eval, y_eval, cv=cv, scoring=scoring)

    mean_r2 = float(np.mean(cv_results["test_r2"]))
    mean_mae = float(-np.mean(cv_results["test_neg_mae"]))
    mean_mse = float(-np.mean(cv_results["test_neg_mse"]))
    mean_rmse = float(np.sqrt(mean_mse))

    return {
        "CV_MAE": mean_mae,
        "CV_MSE": mean_mse,
        "CV_RMSE": mean_rmse,
        "CV_R2": mean_r2
    }


def extract_coefficients(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract linear regression weights (coefficients) and intercept from fitted pipeline."""
    regressor = pipeline.named_steps["regressor"]
    
    # Check if feature selector was used
    if "selector" in pipeline.named_steps:
        selector = pipeline.named_steps["selector"]
        mask = selector.get_support()
        used_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    else:
        used_features = feature_names

    if hasattr(regressor, "coef_"):
        coefs = regressor.coef_
        df_coef = pd.DataFrame({
            "Feature": used_features,
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs)
        }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
        return df_coef
    else:
        return pd.DataFrame({"Feature": used_features, "Coefficient": [0.0]*len(used_features)})
