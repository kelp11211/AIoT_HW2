"""Uncertainty Quantification and Prediction Intervals for AIoT HW2.

Implements:
1. Split Conformal Prediction (Distribution-Free, Non-parametric, Finite-sample coverage guarantee)
2. Parametric OLS Prediction Interval (Individual observation) vs Confidence Interval (Conditional mean)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import statsmodels.api as sm


class SplitConformalPredictor:
    """Split Conformal Predictor for calibrated 1 - alpha prediction intervals."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.n_cal: Optional[int] = None

    def calibrate(self, y_cal: np.ndarray, y_cal_pred: np.ndarray):
        """Calibrate on hold-out calibration set strictly isolated from test."""
        residuals = np.abs(np.array(y_cal) - np.array(y_cal_pred))
        self.n_cal = len(residuals)
        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((self.n_cal + 1) * (1 - self.alpha)) / self.n_cal
        level = min(1.0, max(0.0, level))
        self.q_hat = float(np.quantile(residuals, level, method="higher"))

    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals [lower, upper] for point predictions."""
        if self.q_hat is None:
            raise ValueError("ConformalPredictor must be calibrated before predicting intervals.")
        y_arr = np.array(y_pred)
        lower = y_arr - self.q_hat
        upper = y_arr + self.q_hat
        return lower, upper

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate empirical test set coverage and average width."""
        lower, upper = self.predict_intervals(y_pred)
        y_arr = np.array(y_true)
        in_interval = (y_arr >= lower) & (y_arr <= upper)
        coverage = float(np.mean(in_interval))
        avg_width = float(np.mean(upper - lower))
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": coverage,
            "average_width": avg_width,
            "quantile_q_hat": self.q_hat,
            "calibration_samples": self.n_cal
        }


def compute_ols_intervals(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Compute parametric OLS Confidence Intervals (mean) and Prediction Intervals (observations)."""
    X_train_sm = sm.add_constant(X_train_scaled)
    ols_model = sm.OLS(y_train, X_train_sm).fit()

    X_test_sm = sm.add_constant(X_test_scaled)
    pred_summary = ols_model.get_prediction(X_test_sm).summary_frame(alpha=alpha)
    return pred_summary
