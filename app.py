"""AIoT HW2: CRISP-DM Multiple Linear Regression Web Application.

Author: AIoT HW2 Team
Framework: Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AIoT HW2: CRISP-DM Multiple Linear Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 80%;
        font-weight: 700;
        border-radius: 0.25rem;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 AIoT HW2: CRISP-DM 多元線性回歸分析系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">嚴謹回歸建模 · 特徵選擇 (Filter/Wrapper) · 95% 預測區間 · CRISP-DM 六大階段完整剖析</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📌 導覽與設定")
problem_choice = st.sidebar.selectbox(
    "選擇分析資料集 (Problem Selection)",
    [
        "Problem A: 洪水機率預測 (Flood Prediction - 20 Features)",
        "Problem B: 車輛與個人保費預測 (Insurance Premium - 19 Features)",
        "Bonus: 自訂 CSV 資料集 (URL / 上傳)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 系統規格")
st.sidebar.info("""
- **Python**: 3.14.6
- **核心算法**: `LinearRegression`
- **特徵選擇**: SelectKBest / RFE / Lasso
- **區間估計**: 95% Prediction Interval (PI)
- **方法論**: CRISP-DM 6 階段
""")

# CRISP-DM Tabs
tabs = st.tabs([
    "1. 商業理解 (Business)",
    "2. 資料理解 (Data EDA)",
    "3. 資料前處理 (Preparation)",
    "4. 建模與特徵選擇 (Modeling)",
    "5. 模型評估與預測區間 (Evaluation)",
    "6. 部署與即時預測 (Deployment)"
])

with tabs[0]:
    st.header("1. Business Understanding (商業理解)")
    st.markdown("""
    ### 專案目標
    運用 Kaggle 公開真實競賽資料集，建立標準化**多元線性回歸 (Multiple Linear Regression)** 預測管線，嚴格遵循 CRISP-DM 流程，並落實無資料洩漏 (No Data Leakage) 之特徵選擇與 95% 預測區間量化。
    """)
    if "Flood" in problem_choice:
        st.success("**主題 A：Predict the Flood** (Kaggle Playground Series s4e5)")
        st.markdown("""
        - **Target**: `FloodProbability` (連續值 `0.285` - `0.725`)
        - **特徵數**: 20 個原始連續環境與治理評分指標（無洩漏、排除 ID）
        - **決策價值**: 提前評估季風、排水、都市化與基建老化之致災機率，提供防災資源調度參考。
        """)
    elif "Insurance" in problem_choice:
        st.success("**主題 B：Insurance Premium Prediction** (Kaggle Playground Series s4e12)")
        st.markdown("""
        - **Target**: `Premium Amount` (連續數值，保險費用)
        - **特徵數**: 19 個混合特徵（8 個數值、11 個類別）
        - **決策價值**: 依投保人年齡、收入、健康分數、車齡與理賠紀錄，自動化核保定價。
        """)
    else:
        st.info("**加分功能：自訂 CSV 資料集動態預測**")

with tabs[1]:
    st.header("2. Data Understanding (資料理解與 EDA)")
    st.write("探索性資料分析（包含資料形狀、缺失值檢查、目標分布圖、相關性熱圖）。")

with tabs[2]:
    st.header("3. Data Preparation (資料準備)")
    st.write("80/20 訓練/測試切分，僅在訓練集擬合 Standard Scaler / Imputer / OneHotEncoder。")

with tabs[3]:
    st.header("4. Modeling & Feature Selection (建模與特徵選擇)")
    st.write("比較 Baseline (Dummy) vs 全特徵多元線性回歸 vs SelectKBest (F-regression) vs Ridge / Lasso。")

with tabs[4]:
    st.header("5. Evaluation & Prediction Intervals (評估與 95% 預測區間)")
    st.write("在獨立測試集計算 MAE、MSE、RMSE、R²，並呈現殘差檢驗與 95% 預測區間視覺化。")

with tabs[5]:
    st.header("6. Deployment & Prediction (部署與互動預測)")
    st.write("單筆特徵手動調節與即時預測功能。")
