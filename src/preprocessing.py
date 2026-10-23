"""Data Preprocessing and Pipeline Construction for AIoT HW2.

Ensures zero data leakage:
- Preprocessor is fit exclusively on X_train.
- Handles missing values, standard scaling, and one-hot encoding for unseen categories.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Strict 80/20 train/test split. Test set is strictly isolated."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    num_cols: List[str],
    cat_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """Build a ColumnTransformer with separate numerical and categorical pipelines.

    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder(handle_unknown='ignore')
    """
    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from fitted ColumnTransformer."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            # If the pipeline or final step has get_feature_names_out
            names.extend(trans.get_feature_names_out(cols).tolist())
        elif hasattr(trans, "named_steps") and hasattr(trans.named_steps.get("encoder"), "get_feature_names_out"):
            enc_names = trans.named_steps["encoder"].get_feature_names_out(cols).tolist()
            names.extend(enc_names)
        else:
            names.extend(cols)
    return names
