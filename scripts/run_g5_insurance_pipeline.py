"""Support Script for Gate 5: Train and Precompute Topic B (Insurance Dataset) Pipeline.

Enables seamless switching in the Streamlit Web Application for Bonus Topic 1 (+1 point).
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_dataset
from src.intervals import SplitConformalPredictor
from src.modeling import build_pipeline
from src.preprocessing import build_preprocessor, split_data

FIG_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_insurance_pipeline():
    print("--- Running Insurance Dataset (Topic B) Pipeline for Gate 5 Web App ---")
    
    # Load 100,000 sample of Insurance dataset for fast & efficient web modeling
    print("Loading 100,000 samples of Insurance dataset...")
    df, target_col, feature_cols, num_cols, cat_cols = load_dataset("insurance", sample_n=100000, random_state=42)
    print(f"Loaded: {len(df)} rows. Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")

    # 1. EDA Figure: Target Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sns.histplot(df[target_col], kde=True, bins=50, color="#059669", ax=ax, edgecolor="#047857", alpha=0.6)
    ax.set_title(f"Target Distribution: {target_col} (Topic B)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Premium Amount ($)", fontsize=10)
    plt.tight_layout()
    fig1_path = FIG_DIR / "eda_insurance_target_dist.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. EDA Figure: Correlation Heatmap for Numerical Features
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    corr_num = df[num_cols + [target_col]].corr()
    sns.heatmap(corr_num, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_title("Correlation Heatmap: Numerical Features vs Premium Amount", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2_path = FIG_DIR / "eda_insurance_corr_heatmap.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Preprocessing & Split 80/20
    X_train, X_test, y_train, y_test = split_data(df, target_col, feature_cols, test_size=0.20, random_state=42)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 4. Train Linear Regression & Ridge
    pipe_lr = build_pipeline(preprocessor, model_type="linear")
    pipe_lr.fit(X_train, y_train)

    y_pred = pipe_lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics_df = pd.DataFrame([{
        "Dataset": "Insurance Premium (s4e12)",
        "Model": "LinearRegression (ColumnTransformer)",
        "Features_Raw": len(feature_cols),
        "Test_MAE": round(mae, 2),
        "Test_MSE": round(mse, 2),
        "Test_RMSE": round(rmse, 2),
        "Test_R2": round(r2, 4)
    }])
    metrics_path = REPORTS_DIR / "gate5_insurance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Insurance Test Metrics:")
    print(metrics_df.to_string(index=False))

    # 5. Conformal Prediction Interval
    cal_size = int(len(X_train) * 0.2)
    X_core, y_core = X_train.iloc[cal_size:], y_train.iloc[cal_size:]
    X_cal, y_cal = X_train.iloc[:cal_size], y_train.iloc[:cal_size:]

    pipe_cal = build_pipeline(preprocessor, model_type="linear")
    pipe_cal.fit(X_core, y_core)
    y_cal_pred = pipe_cal.predict(X_cal)

    conformal = SplitConformalPredictor(alpha=0.05)
    conformal.calibrate(y_cal.values, y_cal_pred)
    coverage_res = conformal.evaluate_coverage(y_test.values, y_pred)
    print(f"Insurance 95% PI Coverage: {coverage_res['empirical_coverage']*100:.2f}%, Width: {coverage_res['average_width']:.2f}")

    # 6. Save Model & Metadata
    model_path = MODELS_DIR / "insurance_lr_pipeline.joblib"
    joblib.dump(pipe_lr, model_path)
    print(f"Saved: {model_path}")

    # Save small sample CSV for quick app demo
    sample_demo_path = BASE_DIR / "data" / "insurance" / "demo_sample.csv"
    df.head(500).to_csv(sample_demo_path, index=False)
    print(f"Saved: {sample_demo_path}")

    print("--- Topic B Insurance Pipeline Complete! ---")


if __name__ == "__main__":
    run_insurance_pipeline()
