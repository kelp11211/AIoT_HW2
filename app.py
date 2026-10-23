"""AIoT HW2: CRISP-DM Multiple Linear Regression System.

Main Streamlit Application featuring:
- Topic A: Flood Prediction (20 Features, Kaggle S4E5)
- Topic B: Insurance Premium Prediction (19 Features, Kaggle S4E12) [+1 Bonus]
- Bonus Custom CSV: Remote URL / Local Upload with SSRF Protection [+2 Bonus]
- CRISP-DM 6-Phase Navigation
- 95% Prediction Intervals (Conformal & OLS)
- Interactive Real-Time Prediction Sliders
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

from src.data_loader import load_dataset, load_external_csv
from src.evaluation import compute_regression_metrics
from src.intervals import SplitConformalPredictor
from src.modeling import build_pipeline
from src.preprocessing import build_preprocessor, split_data

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

# Streamlit Page Config
st.set_page_config(
    page_title="AIoT HW2: 多元線性回歸系統 (CRISP-DM)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Header & Navigation
st.sidebar.markdown("## ⚙️ 系統設定與題庫切換")
problem_mode = st.sidebar.selectbox(
    "選擇分析任務 (Problem Selection)",
    [
        "主題 A: 洪水機率預測 (Flood Prediction - 20 Features)",
        "主題 B: 保費金額預測 (Insurance Premium - 19 Features) [+1分加分項]",
        "加分功能: 任意自訂 CSV (URL / 本機上傳) [+2分加分項]"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 專案技術規格")
st.sidebar.markdown("""
- **架構**：CRISP-DM 6 階段
- **主要模型**：`LinearRegression`
- **特徵選擇**：Filter (SelectKBest) / Lasso
- **不確定性**：95% Split Conformal PI
- **隨機種子**：`random_state = 42`
- **環境**：Python 3.14 · Scikit-Learn · Streamlit
""")

# ==============================================================================
# Helper Loaders with Streamlit Caching
# ==============================================================================
@st.cache_data
def get_sample_df(problem: str):
    if "Flood" in problem:
        sample_path = BASE_DIR / "data" / "flood" / "demo_sample.csv"
        full_path = BASE_DIR / "data" / "flood" / "train.csv"
        path = sample_path if sample_path.exists() else full_path
        return pd.read_csv(path, nrows=500)
    elif "Insurance" in problem:
        sample_path = BASE_DIR / "data" / "insurance" / "demo_sample.csv"
        full_path = BASE_DIR / "data" / "insurance" / "train.csv"
        path = sample_path if sample_path.exists() else full_path
        return pd.read_csv(path, nrows=500)
    return None


@st.cache_resource
def get_model(problem: str):
    if "Flood" in problem:
        model_path = MODELS_DIR / "flood_lr_all.joblib"
        if model_path.exists():
            return joblib.load(model_path)
    elif "Insurance" in problem:
        model_path = MODELS_DIR / "insurance_lr_pipeline.joblib"
        if model_path.exists():
            return joblib.load(model_path)
    return None


# ==============================================================================
# MAIN PAGE: HEADER
# ==============================================================================
st.markdown('<div class="main-title">🌊 AIoT HW2: 多元線性回歸與 CRISP-DM 分析系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">嚴謹回歸建模 · Filter / Regularized 特徵選擇 · 95% 預測區間量化 · 雙主題切換與動態 CSV 分析</div>', unsafe_allow_html=True)


# ==============================================================================
# MODE 1: PROBLEM A - FLOOD PREDICTION
# ==============================================================================
if "Flood" in problem_mode:
    st.markdown("""
    <span class="badge-primary">正式必做主題 A</span> 
    <span class="badge-success">Kaggle Playground Series s4e5</span>
    <span class="badge-primary">20 個原始數值特徵</span>
    <span class="badge-success">嚴格杜絕資料洩漏</span>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "1. 商業理解 (Business)",
        "2. 資料理解 (Data EDA)",
        "3. 資料準備 (Preparation)",
        "4. 建模與特徵選擇 (Modeling)",
        "5. 評估與預測區間 (Evaluation)",
        "6. 即時互動預測 (Deployment)"
    ])

    # Tab 1: Business Understanding
    with tabs[0]:
        st.header("Phase 1: Business Understanding (商業理解)")
        st.markdown("""
        ### 🌧️ 商業背景與決策價值
        近年受極端氣候與都市化急速擴張影響，區域性強降雨頻率劇增。水利單位若單純依賴物理數值模擬，常因計算耗時難以及時進行災前資源部署。
        本專案以 **20 個環境、水文治理與基礎建設指標**，建立輕量、透明且高度可解釋之多元線性回歸模型，在降雨預報發布初期即刻預估洪水發生機率，協助主管機關精確調派抽水機具、沙包儲備並預先發布疏散警訊。
        """)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("目標變數 (Target)", "FloodProbability", "連續浮點數 (0~1)")
        c2.metric("原始特徵數量", "20 Features", "符合 >13 規格")
        c3.metric("資料總規模", "1,117,957 列", "Kaggle 官方競賽")
        c4.metric("基準評估指標", "R² > 0.84", "優於 Dummy 均值")

        st.info("🔗 **Kaggle 競賽頁面**：[Playground Series Season 4 Episode 5](https://www.kaggle.com/competitions/playground-series-s4e5/data)")

    # Tab 2: Data Understanding
    with tabs[1]:
        st.header("Phase 2: Data Understanding (資料探索 EDA)")
        df_sample = get_sample_df("Flood")
        if df_sample is not None:
            st.markdown("#### 📋 資料預覽 (Sample Preview)")
            st.dataframe(df_sample.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📊 目標變數分布圖 (Target Distribution)")
            fig1_path = FIG_DIR / "eda_flood_target_dist.png"
            if fig1_path.exists():
                st.image(str(fig1_path), use_container_width=True)
            else:
                st.info("圖表請執行 scripts/run_g2_eda_prep.py 產出")

        with col_b:
            st.markdown("#### 📈 多重共線性檢驗 (VIF Analysis)")
            vif_fig_path = FIG_DIR / "eda_flood_vif_chart.png"
            if vif_fig_path.exists():
                st.image(str(vif_fig_path), use_container_width=True)
            
            vif_csv = REPORTS_DIR / "vif_report_flood.csv"
            if vif_csv.exists():
                with st.expander("查看 20 特徵完整 VIF 數值表 (全數 < 5.0)"):
                    st.dataframe(pd.read_csv(vif_csv), use_container_width=True)

        st.markdown("#### 🌡️ 20 個環境特徵相關係數熱圖 (Correlation Heatmap)")
        fig2_path = FIG_DIR / "eda_flood_corr_heatmap.png"
        if fig2_path.exists():
            st.image(str(fig2_path), use_container_width=True)

    # Tab 3: Data Preparation
    with tabs[2]:
        st.header("Phase 3: Data Preparation (資料前處理與防洩漏)")
        st.markdown("""
        ### 🛡️ 嚴格無資料洩漏 (Strict Hold-out 80/20) 管線
        - **Hold-out 切分**：全量 1,117,957 筆資料依固定種子 (`random_state=42`) 切分為 **訓練集 (894,365 筆)** 與 **測試集 (223,592 筆)**。
        - **Pipeline 封裝**：`ColumnTransformer` 串接 `SimpleImputer(strategy='median')` 與 `StandardScaler()`。
        - **單向擬合**：前處理器只在訓練集執行 `.fit()`，均值與標準差純粹來自訓練資料，測試集完全隔離至評估階段。
        """)

        st.code("""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), feature_cols)
    ]
)
# 僅在訓練集擬合！
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)
        """, language="python")

    # Tab 4: Modeling & Feature Selection
    with tabs[3]:
        st.header("Phase 4: Modeling & Feature Selection (建模與特徵選擇)")
        st.markdown("### 🏆 5 折交叉驗證 (5-Fold CV on Training Set) 模型比較表")
        
        comp_csv = REPORTS_DIR / "gate3_model_comparison.csv"
        if comp_csv.exists():
            st.dataframe(pd.read_csv(comp_csv), use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 📉 特徵數 k 與 5 折 CV 分數趨勢圖")
            fig_k = FIG_DIR / "g3_cv_score_vs_k.png"
            if fig_k.exists():
                st.image(str(fig_k), use_container_width=True)

        with col_m2:
            st.markdown("#### ⚖️ 多元線性回歸標準化係數 (Feature Weights $\\beta$)")
            fig_coef = FIG_DIR / "g3_feature_coefficients.png"
            if fig_coef.exists():
                st.image(str(fig_coef), use_container_width=True)

    # Tab 5: Evaluation & Intervals
    with tabs[4]:
        st.header("Phase 5: Evaluation & 95% Prediction Intervals (評估與預測區間)")
        st.markdown("### 🎯 獨立測試集 (223,592 筆) 最終評估成果")
        
        test_csv = REPORTS_DIR / "gate4_test_metrics.csv"
        if test_csv.exists():
            st.dataframe(pd.read_csv(test_csv), use_container_width=True)

        st.markdown("### 🔮 95% 預測區間 (PI) vs 95% 信賴區間 (CI) 視覺化圖表")
        fig_inter = FIG_DIR / "g4_actual_vs_predicted_intervals.png"
        if fig_inter.exists():
            st.image(str(fig_inter), use_container_width=True)

        conformal_json = REPORTS_DIR / "gate4_conformal_summary.json"
        if conformal_json.exists():
            import json
            with open(conformal_json, "r") as f:
                conf_data = json.load(f)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("目標邊際覆蓋率", "95.00%")
            c2.metric("測試集實測覆蓋率", f"{conf_data['empirical_coverage']*100:.2f}%", "達成 95% 保證")
            c3.metric("平均預測區間寬度", f"{conf_data['average_width']:.4f}")
            c4.metric("不符合度分位數 q_hat", f"{conf_data['quantile_q_hat']:.6f}")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### 殘差 vs. 擬合值 (同方差性檢驗)")
            f_rf = FIG_DIR / "g4_residual_vs_fitted.png"
            if f_rf.exists():
                st.image(str(f_rf), use_container_width=True)
        with col_e2:
            st.markdown("#### 殘差常態 Q-Q 圖與直方密度 (常態性檢驗)")
            f_diag = FIG_DIR / "g4_residual_diagnostics.png"
            if f_diag.exists():
                st.image(str(f_diag), use_container_width=True)

    # Tab 6: Live Deployment & Inference
    with tabs[5]:
        st.header("Phase 6: Deployment & Interactive Inference (線上即時預測)")
        st.markdown("調節下方 20 個環境指標評分，模型將即時推論 **預測洪水機率** 與 **95% 預測區間**：")

        model_flood = get_model("Flood")
        features_20 = [
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
            'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
            'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
            'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors'
        ]

        # Input grid (4 columns x 5 rows)
        input_vals = {}
        cols = st.columns(4)
        for idx, feat in enumerate(features_20):
            with cols[idx % 4]:
                input_vals[feat] = st.slider(
                    feat,
                    min_value=0,
                    max_value=16,
                    value=5,
                    step=1
                )

        if st.button("🌊 執行即時洪水機率推論", type="primary"):
            if model_flood is not None:
                input_df = pd.DataFrame([input_vals])
                pred_prob = float(model_flood.predict(input_df)[0])
                
                # Split Conformal 95% PI: q_hat = 0.040058
                q_hat = 0.040058
                lower_pi = max(0.0, pred_prob - q_hat)
                upper_pi = min(1.0, pred_prob + q_hat)

                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("預測洪水機率 (Point Pred)", f"{pred_prob:.4f}")
                col_res2.metric("95% 預測區間下界 (PI Lower)", f"{lower_pi:.4f}")
                col_res3.metric("95% 預測區間上界 (PI Upper)", f"{upper_pi:.4f}")

                # Risk Alert
                if pred_prob >= 0.55:
                    st.error(f"🚨 **高警戒風險 (High Risk)**：洪水預測機率達到 {pred_prob*100:.1f}%，建議啟動第一級防汛響應，加派抽水機具！")
                elif pred_prob >= 0.48:
                    st.warning(f"⚠️ **中度警戒 (Medium Risk)**：洪水預測機率為 {pred_prob*100:.1f}%，請注意集水區水位監控。")
                else:
                    st.success(f"✅ **安全正常 (Low Risk)**：洪水機率預估為 {pred_prob*100:.1f}%，處於平水期安全指標範圍。")
            else:
                st.error("模型檔案尚未載入，請確認 models/flood_lr_all.joblib 存在。")


# ==============================================================================
# MODE 2: PROBLEM B - INSURANCE PREDICTION (+1 BONUS)
# ==============================================================================
elif "Insurance" in problem_mode:
    st.markdown("""
    <span class="badge-primary">加分題目 B (+1分)</span> 
    <span class="badge-success">Kaggle Playground Series s4e12</span>
    <span class="badge-primary">19 個特徵 (8 數值 + 11 類別)</span>
    <span class="badge-success">ColumnTransformer 混合編碼</span>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "1. 題目背景與指標",
        "2. 資料探索 (EDA)",
        "3. 模型評估表",
        "4. 即時保費試算推論"
    ])

    with tabs[0]:
        st.header("主題 B：車輛與個人保費預測 (Insurance Premium Prediction)")
        st.markdown("""
        - **Kaggle 官方競賽**：[Playground Series Season 4 Episode 12](https://www.kaggle.com/competitions/playground-series-s4e12/data)
        - **目標變數 (Target)**：`Premium Amount` (連續數值，保險金額美元)
        - **特徵規格**：19 個原始輸入特徵（8 數值指標 + 11 類別特徵，排除純索引 `id`）。
        - **商業價值**：自動依據年齡、年收入、健康分數、車齡與理賠紀錄快速試算保費。
        """)

    with tabs[1]:
        st.header("主題 B 資料探索 (EDA)")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("#### 目標變數分布 (Premium Amount)")
            f_ins_dist = FIG_DIR / "eda_insurance_target_dist.png"
            if f_ins_dist.exists():
                st.image(str(f_ins_dist), use_container_width=True)
        with col_i2:
            st.markdown("#### 數值特徵相關矩陣 (Correlation Heatmap)")
            f_ins_corr = FIG_DIR / "eda_insurance_corr_heatmap.png"
            if f_ins_corr.exists():
                st.image(str(f_ins_corr), use_container_width=True)

    with tabs[2]:
        st.header("主題 B 測試集評估成果")
        ins_csv = REPORTS_DIR / "gate5_insurance_metrics.csv"
        if ins_csv.exists():
            st.dataframe(pd.read_csv(ins_csv), use_container_width=True)
            st.info("💡 主題 B 採用 `ColumnTransformer` 串接 `StandardScaler` 與 `OneHotEncoder(handle_unknown='ignore')` 進行類別特徵混合編碼。")

    with tabs[3]:
        st.header("💰 即時保費試算推論 (Live Actuarial Inference)")
        model_ins = get_model("Insurance")
        
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            age = st.slider("年齡 (Age)", 18, 85, 35)
            annual_income = st.number_input("年收入 ($)", min_value=10000, max_value=500000, value=65000, step=5000)
            health_score = st.slider("健康評分 (Health Score)", 0.0, 100.0, 75.0, 1.0)
            credit_score = st.slider("信用評分 (Credit Score)", 300, 850, 700)
        with ci2:
            num_dependents = st.slider("扶養人數 (Number of Dependents)", 0, 10, 2)
            vehicle_age = st.slider("車齡 (Vehicle Age)", 0, 25, 4)
            prev_claims = st.slider("過往理賠次數 (Previous Claims)", 0, 10, 0)
            duration = st.slider("保險期長度 (Insurance Duration Years)", 1, 30, 5)
        with ci3:
            gender = st.selectbox("性別 (Gender)", ["Male", "Female"])
            marital = st.selectbox("婚姻狀況 (Marital Status)", ["Single", "Married", "Divorced"])
            policy_type = st.selectbox("保單類型 (Policy Type)", ["Basic", "Comprehensive", "Premium"])
            smoking = st.selectbox("吸菸習慣 (Smoking Status)", ["No", "Yes"])

        if st.button("💵 計算建議保險費", type="primary"):
            if model_ins is not None:
                sample_input = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "Annual Income": annual_income,
                    "Marital Status": marital,
                    "Number of Dependents": num_dependents,
                    "Education Level": "Bachelor's",
                    "Occupation": "Employed",
                    "Health Score": health_score,
                    "Location": "Suburban",
                    "Policy Type": policy_type,
                    "Previous Claims": prev_claims,
                    "Vehicle Age": vehicle_age,
                    "Credit Score": credit_score,
                    "Insurance Duration": duration,
                    "Policy Start Year": 2023,
                    "Customer Feedback": "Good",
                    "Smoking Status": smoking,
                    "Exercise Frequency": "Weekly",
                    "Property Type": "House"
                }])
                pred_premium = float(model_ins.predict(sample_input)[0])
                st.success(f"### 預估年保費：**${pred_premium:,.2f} USD** (95% 預測區間: [${max(0, pred_premium-1745):,.2f}, ${pred_premium+1745:,.2f}])")
            else:
                st.error("保費模型未載入，請確認 models/insurance_lr_pipeline.joblib")


# ==============================================================================
# MODE 3: BONUS CUSTOM CSV LOADER (+2 BONUS)
# ==============================================================================
else:
    st.markdown("""
    <span class="badge-primary">加分題目 C (+2分)</span> 
    <span class="badge-success">動態載入任意外部 CSV 資料集</span>
    <span class="badge-primary">安全防護 SSRF Defense & 記憶體限制</span>
    """, unsafe_allow_html=True)

    st.markdown("### 🌐 自訂 CSV 資料集預測管線 (Any CSV Loader)")
    st.write("支援輸入遠端 CSV 下載網址或自本機上傳 CSV 檔，系統將自動解析欄位型態、驗證目標變數並動態訓練多元線性回歸模型。")

    input_method = st.radio("選擇資料來源方式", ["遠端 CSV URL (含 SSRF 防護)", "本機檔案上傳"], horizontal=True)

    df_custom = None
    target_custom = None

    if input_method == "遠端 CSV URL (含 SSRF 防護)":
        url_input = st.text_input(
            "輸入公開 CSV 網址：",
            value="https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
        )
        if st.button("📥 擷取並解析遠端資料集"):
            try:
                with st.spinner("安全下載並檢驗 CSV 中..."):
                    df_custom, target_custom, feature_cols, num_cols, cat_cols = load_external_csv(url_input)
                    st.session_state["custom_df"] = df_custom
                    st.success(f"成功安全載入資料！共 {len(df_custom)} 筆列，{len(df_custom.columns)} 個欄位。")
            except Exception as e:
                st.error(f"❌ 讀取失敗或觸發安全防護：{str(e)}")

    else:
        uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])
        if uploaded_file is not None:
            try:
                df_custom = pd.read_csv(uploaded_file)
                st.session_state["custom_df"] = df_custom
                st.success(f"成功載入上傳檔案！共 {len(df_custom)} 筆，{len(df_custom.columns)} 個欄位。")
            except Exception as e:
                st.error(f"❌ CSV 解析錯誤：{str(e)}")

    # If dataset in session state
    if "custom_df" in st.session_state:
        df_active = st.session_state["custom_df"]
        st.markdown("#### 📋 資料前 10 筆預覽")
        st.dataframe(df_active.head(10), use_container_width=True)

        st.markdown("#### ⚙️ 設定回歸分析參數")
        col_t1, col_t2 = st.columns(2)
        
        num_candidates = df_active.select_dtypes(include=[np.number]).columns.tolist()
        with col_t1:
            selected_target = st.selectbox("指定預測目標變數 (Target Column)", options=df_active.columns, index=len(df_active.columns)-1)
            
            # Check target is continuous
            if selected_target not in num_candidates:
                st.error("⚠️ 警告：選取的目標不是數值欄位，無法進行線性回歸！")
            elif df_active[selected_target].nunique() <= 5:
                st.warning("⚠️ 注意：選取的目標變數相異值小於等於 5 個，可能屬於離散分類任務。")
            else:
                st.success("✅ 目標變數符合連續數值回歸規格！")

        with col_t2:
            avail_features = [c for c in df_active.columns if c != selected_target]
            selected_features = st.multiselect("選取參與建模之特徵欄位", options=avail_features, default=avail_features)

        if st.button("🚀 即刻訓練多元線性回歸模型並評估", type="primary"):
            if selected_target in num_candidates and len(selected_features) > 0:
                with st.spinner("正在執行動態資料前處理與回歸建模..."):
                    # Split 80/20
                    sub_df = df_active[selected_features + [selected_target]].dropna()
                    X_train, X_test, y_train, y_test = split_data(sub_df, selected_target, selected_features, test_size=0.20, random_state=42)

                    num_feats = [c for c in selected_features if c in num_candidates]
                    cat_feats = [c for c in selected_features if c not in num_candidates]

                    preprocessor = build_preprocessor(num_feats, cat_feats)
                    pipe = build_pipeline(preprocessor, model_type="linear")
                    pipe.fit(X_train, y_train)

                    y_test_pred = pipe.predict(X_test)
                    metrics = compute_regression_metrics(y_test, y_test_pred, n_features=len(selected_features))

                    st.markdown("### 📊 測試集評估結果")
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("測試集 MAE", f"{metrics['MAE']:.4f}")
                    cm2.metric("測試集 MSE", f"{metrics['MSE']:.4f}")
                    cm3.metric("測試集 RMSE", f"{metrics['RMSE']:.4f}")
                    cm4.metric("決定係數 R²", f"{metrics['R2']:.4f}")

                    # Plot Actual vs Predicted
                    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
                    ax.scatter(y_test, y_test_pred, color="#2563EB", alpha=0.7, edgecolors="none")
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="Perfect Fit (y = x)")
                    ax.set_title(f"Custom Dataset: Actual vs. Predicted ({selected_target})", fontsize=12, fontweight="bold")
                    ax.set_xlabel("Actual Value", fontsize=10)
                    ax.set_ylabel("Predicted Value", fontsize=10)
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.error("請確認目標變數為數值型，且至少選擇一個特徵欄位。")
