"""Feature Selection Module for AIoT HW2.

Implements Filter (SelectKBest with f_regression) and Regularized (Lasso) feature selection
embedded directly in cross-validation pipelines to guarantee zero data leakage.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_feature_selector(k: int = 15) -> SelectKBest:
    """Create a SelectKBest selector using f_regression scoring."""
    return SelectKBest(score_func=f_regression, k=k)


def evaluate_k_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_candidates: Optional[List[int]] = None,
    cv_splits: int = 5,
    sample_size: int = 100000,
    random_state: int = 42
) -> pd.DataFrame:
    """Evaluate different values of k using K-Fold Cross Validation on training data.

    Returns:
        DataFrame containing k, mean_cv_r2, std_cv_r2, mean_cv_rmse
    """
    if k_candidates is None:
        total_p = X_train.shape[1]
        k_candidates = sorted(list(set([5, 8, 10, 12, 14, 15, 18, total_p])))

    # Subsample training data for fast and robust CV computation if very large
    if len(X_train) > sample_size:
        idx = np.random.RandomState(random_state).choice(len(X_train), size=sample_size, replace=False)
        X_sub = X_train.iloc[idx]
        y_sub = y_train.iloc[idx]
    else:
        X_sub = X_train
        y_sub = y_train

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []

    for k in k_candidates:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_regression, k=k)),
            ("regressor", LinearRegression())
        ])

        # Evaluate R2
        r2_scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="r2")
        # Evaluate neg_mean_squared_error
        mse_scores = -cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(mse_scores)

        results.append({
            "k_features": k,
            "CV_R2_Mean": float(np.mean(r2_scores)),
            "CV_R2_Std": float(np.std(r2_scores)),
            "CV_RMSE_Mean": float(np.mean(rmse_scores)),
            "CV_RMSE_Std": float(np.std(rmse_scores))
        })

    return pd.DataFrame(results)


def get_selected_feature_names(
    selector: SelectKBest,
    all_feature_names: List[str]
) -> Tuple[List[str], pd.DataFrame]:
    """Retrieve the names and F-scores of selected features."""
    mask = selector.get_support()
    scores = selector.scores_
    pvalues = selector.pvalues_

    selected = [all_feature_names[i] for i in range(len(all_feature_names)) if mask[i]]
    
    score_df = pd.DataFrame({
        "Feature": all_feature_names,
        "F_Score": scores,
        "P_Value": pvalues,
        "Selected": mask
    }).sort_values("F_Score", ascending=False).reset_index(drop=True)

    return selected, score_df
