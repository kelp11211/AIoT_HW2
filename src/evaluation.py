"""Evaluation and Diagnostics Module for AIoT HW2.

Computes standard regression metrics (MAE, MSE, RMSE, R2, Adjusted R2)
and performs statistical residual assumption checks.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int = 1
) -> Dict[str, float]:
    """Calculate MAE, MSE, RMSE, R2, and Adjusted R2 on test set."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    n = len(y_t)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    if n > n_features + 1 and n_features > 0:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))
    else:
        adj_r2 = r2

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2
    }


def diagnose_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Check regression error assumptions:

    1. Error Mean ~ 0
    2. Durbin-Watson statistic (Independence, ideal ~ 2.0)
    3. Residual Skewness & Kurtosis (Normality)
    """
    residuals = np.array(y_true) - np.array(y_pred)
    dw = float(durbin_watson(residuals))
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    
    # Skewness and Kurtosis
    from scipy.stats import skew, kurtosis
    res_skew = float(skew(residuals))
    res_kurt = float(kurtosis(residuals))

    return {
        "residual_mean": res_mean,
        "residual_std": res_std,
        "durbin_watson": dw,
        "residual_skewness": res_skew,
        "residual_kurtosis": res_kurt,
        "is_independent": bool(1.8 <= dw <= 2.2)
    }
