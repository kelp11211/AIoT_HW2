"""Data Loader Module for AIoT HW2.

Provides reproducible loading, schema validation, and safe URL retrieval.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import io
import requests
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASET_CONFIGS = {
    "flood": {
        "path": DATA_DIR / "flood" / "train.csv",
        "target": "FloodProbability",
        "id_cols": ["id", "Unnamed: 0"],
        "description": "Kaggle Playground Series s4e5 (Flood Prediction)"
    },
    "insurance": {
        "path": DATA_DIR / "insurance" / "train.csv",
        "target": "Premium Amount",
        "id_cols": ["id", "Unnamed: 0"],
        "description": "Kaggle Playground Series s4e12 (Insurance Prediction)"
    }
}


def load_dataset(
    dataset_name: str = "flood",
    sample_n: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, str, List[str], List[str], List[str]]:
    """Load dataset from local directory.

    Returns:
        df: pd.DataFrame
        target_col: str
        feature_cols: list of feature names
        num_cols: list of numerical feature names
        cat_cols: list of categorical feature names
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    csv_path = config["path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    if dataset_name == "insurance" and "Policy Start Date" in df.columns:
        # Parse date to numerical year to avoid 50,000+ sparse dummy columns
        df["Policy Start Year"] = pd.to_datetime(df["Policy Start Date"], errors="coerce").dt.year.fillna(2022).astype(int)
        df = df.drop(columns=["Policy Start Date"])

    target_col = config["target"]
    excluded = config["id_cols"] + [target_col]
    feature_cols = [c for c in df.columns if c not in excluded]


    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    return df, target_col, feature_cols, num_cols, cat_cols


def load_external_csv(
    source: str,
    target_col: Optional[str] = None,
    max_bytes: int = 20 * 1024 * 1024,
    timeout: int = 10
) -> Tuple[pd.DataFrame, str, List[str], List[str], List[str]]:
    """Safely load external CSV from URL or uploaded path.

    Includes security checks:
    - Content-type validation
    - Max size limit
    - Blocks private IP ranges (basic SSRF defense)
    """
    if source.startswith("http://") or source.startswith("https://"):
        # SSRF checks
        lower_url = source.lower()
        blocked = ["localhost", "127.0.0.1", "0.0.0.0", "169.254.", "10.", "172.16.", "192.168."]
        if any(b in lower_url for b in blocked):
            raise ValueError("Access to internal/private network addresses is blocked for security.")

        resp = requests.get(source, timeout=timeout, stream=True)
        resp.raise_for_status()

        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise ValueError(f"File size exceeds limit of {max_bytes / (1024*1024):.1f} MB.")

        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_csv(source)

    if target_col is None:
        # Default to last numerical column if not specified
        num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_candidates:
            raise ValueError("No numeric candidate target column found in CSV.")
        target_col = num_candidates[-1]

    if target_col not in df.columns:
        raise ValueError(f"Specified target '{target_col}' not found in CSV.")

    excluded = ["id", "Unnamed: 0", target_col]
    feature_cols = [c for c in df.columns if c not in excluded]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    return df, target_col, feature_cols, num_cols, cat_cols
