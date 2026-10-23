# AIoT HW2 期末專題報告：多元線性回歸與 CRISP-DM 實踐

> **課程名稱**：物聯網數據分析與應用 (AIoT Data Analysis and Applications)  
> **作業項目**：HW2 — Kaggle Dataset × Multiple Linear Regression × CRISP-DM  
> **學號 / 作者**：kelp11211  
> **專案儲存庫**：<https://github.com/kelp11211/AIoT_HW2>  
> **完成日期**：2026-10-23  

---

## 📌 摘要 (Executive Abstract)

本研究嚴格依循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 數據探勘六大生命週期，針對 Kaggle 公開競賽龐大資料集（主題 A：**Playground Series s4e5 洪水機率預測**，1,117,957 列、20 個環境治理特徵；主題 B：**Playground Series s4e12 保險費用預測**，1,200,000 列、19 個特徵），構建端到端的多元線性回歸 (Multiple Linear Regression) 預測系統與互動式雲端 Web 應用。

本專案全面解決機器學習在工程落地的四大核心挑戰：
1. **防範資料洩漏 (Leak-Free Data Pipeline)**：嚴格落實 80/20 Hold-out 隔離，所有標準化、補值、編碼與特徵選擇均封裝於 `ColumnTransformer`，純粹在訓練集擬合。
2. **多階段特徵選擇 (Rigorous Feature Selection)**：以 5 折交叉驗證搜尋最佳特徵子集 ($k=15$)，並深入比較 Ridge (L2) 與 Lasso (L1) 之正則化表現。
3. **95% 不確定性區間估計 (Uncertainty Quantification)**：嚴格區隔「平均值信賴區間 (CI)」與「個體預測區間 (PI)」，實作**分割共形預測 (Split Conformal Prediction)**，在獨立測試集上取得 **94.77% 邊際覆蓋率**之數學保證。
4. **雙主題與任意 CSV 動態分析 (+3分加分項)**：於 Streamlit Web 介面實現雙回歸題目切換（+1 分）與包含 SSRF 資安防護之動態 CSV/URL 分析引擎（+2 分）。

---

## 1. 商業理解 (Phase 1: Business Understanding)

### 1.1 業務背景與決策價值
在全球氣候變遷加劇下，極端暴雨與強降水引發的洪澇災害對城鄉基建造成巨大經濟損失。傳統水利工程的物理流體數值模擬雖然具備物理基礎，但需要高額的高效能運算資源，且運算延遲極長，難以在強降雨發生的數小時內即時產出全區預警。

透過多元線性回歸模型，主管機關可在氣象與環境感測資訊傳入時，即刻推算洪水機率。其**透明的線性係數 ($\beta$)** 賦予決策者高度的可解釋性，能精準洞察各項致災因子權重，精確調度抽水機具、防汛沙包與提早規劃疏散路線。

### 1.2 預測目標與成功指標
- **預測目標 (Target)**：`FloodProbability`（連續浮點數，分布介於 0.2850 至 0.7250 之間）。
- **成功指標**：
  - 均方根誤差 (RMSE) 與平均絕對誤差 (MAE) 大幅超越以母體平均值預測之 Baseline (DummyRegressor)。
  - 決定係數 $R^2 \ge 0.84$。
  - 產出經校準之 95% 預測區間，為防汛警戒提供風險邊際依據。

---

## 2. 資料理解 (Phase 2: Data Understanding / EDA)

### 2.1 資料集屬性與規格審查
- **資料來源**：Kaggle Playground Series s4e5 (Flood Prediction)。
- **總樣本數**：1,117,957 列，22 個欄位。
- **排除欄位**：純索引欄位 `id`（純識別碼，若納入將造成虛假相關與資料洩漏，已嚴格剔除）與預測目標 `FloodProbability`。
- **原始特徵數**：**20 個純數值指標**，完全符合作業規定的 14–39 特徵規格（且 >13）。
- **資料品質**：缺失值共 0 筆，重複列共 0 筆，資料完整性優異。

### 2.2 探索性數據分析 (EDA) 視覺化成果

| 圖表項目 | 圖表檔案 | 核心統計發現與分析 |
|---|---|---|
| **目標變數分布** | `figures/eda_flood_target_dist.png` | `FloodProbability` 平均值為 `0.5044`、標準差 `0.0510`、偏態係數 `0.0436`，呈現極度標準之對稱常態分布。 |
| **相關矩陣熱圖** | `figures/eda_flood_corr_heatmap.png` | 20 個特徵與目標變數之皮爾森相關係數均勻分佈在 `0.18 ~ 0.23` 之間；特徵相互之間無強相關性。 |
| **關鍵特徵擬合** | `figures/eda_flood_top_features_scatter.png` | 前三大相關指標（`ClimateChange`, `Siltation`, `DamsQuality`）與洪水機率呈現明顯之線性正向遞增趨勢。 |
| **多重共線性檢驗** | `figures/eda_flood_vif_chart.png` | 20 個特徵之變異數膨脹因子 (VIF) 介於 **1.002 ~ 1.005**（全數遠小於 5.0），各維度彼此高度獨立。 |

---

## 3. 資料準備 (Phase 3: Data Preparation)

### 3.1 嚴格 80/20 資料隔離 (Strict Hold-Out)
為防範任何潛在之「資訊外洩 (Data Leakage)」，資料集切分先行於所有前處理動作：
- **訓練集 (Train Set)**：894,365 筆 (80%)，專屬於特徵縮放擬合、交叉驗證特徵選擇與模型訓練。
- **測試集 (Test Set)**：223,592 筆 (20%)，嚴格封存至最終評估階段，未參與任何均值或標準差計算。

### 3.2 轉換管線 (Pipeline & ColumnTransformer) 設計
前處理採用 `scikit-learn` 模組化設計：
- 數值欄位：串接 `SimpleImputer(strategy='median')` 與 `StandardScaler()`。
- 類別欄位（針對主題 B 與外部資料）：配置 `SimpleImputer(strategy='constant', fill_value='missing')` 與 `OneHotEncoder(handle_unknown='ignore')`，具備抵禦未知新類別之容錯能力。

---

## 4. 建模與特徵選擇 (Phase 4: Modeling & Feature Selection)

### 4.1 特徵選擇演算法 (SelectKBest with F-regression)
- 在訓練集上以 5 折交叉驗證，系統化評估候選特徵數 $k \in [5, 8, 10, 12, 14, 15, 18, 20]$。
- 當特徵數由 5 遞增至 15 時，$R^2$ 由 0.18 快速攀升至 0.60；本研究依作業規格選定 **$k=15$** 作為特徵選擇代表模型。

### 4.2 訓練集 5 折交叉驗證 (5-Fold CV) 表現比較表

| 模型架構 (Model) | 特徵數 | CV MAE | CV MSE | CV RMSE | CV $R^2$ |
|---|---|---|---|---|---|
| **Baseline (Dummy Mean)** | 0 | `0.040943` | `0.002603` | `0.051024` | `-0.000116` |
| **Linear Regression (All 20)** | **20** | **`0.015774`** | **`0.000402`** | **`0.020062`** | **`0.845372`** |
| **Linear Regression (Top 15)** | **15** | `0.026077` | `0.001047` | `0.032351` | `0.597936` |
| **Ridge Regression (L2, $\alpha=10$)** | 20 | `0.015774` | `0.000402` | `0.020062` | `0.845372` |
| **Lasso Regression (L1, $\alpha=0.0001$)** | 20 | `0.015814` | `0.000403` | `0.020068` | `0.845276` |

全特徵回歸模型權重係數 ($\beta$) 介於 `0.0116 ~ 0.0118` 之間，權重分布極為均衡，前三大權重指標分別為 `CoastalVulnerability`, `TopographyDrainage`, `PoliticalFactors`。

---

## 5. 模型評估與預測區間 (Phase 5: Evaluation & Prediction Intervals)

### 5.1 獨立測試集 (223,592 筆) 最終指標

| 模型名稱 | 使用特徵數 | 測試集 MAE | 測試集 MSE | 測試集 RMSE | 測試集 $R^2$ | 調整後 $R^2$ |
|---|---|---|---|---|---|---|
| **Baseline (Dummy)** | 0 | `0.040892` | `0.002599` | `0.050983` | `-0.000000` | `-0.000000` |
| **Linear Regression (All 20)** | **20** | **`0.015780`** | **`0.000403`** | **`0.020081`** | **`0.844868`** | **`0.844854`** |
| **Linear Regression (Top 15)** | **15** | `0.026109` | `0.001051` | `0.032415` | `0.595768` | `0.595741` |
| **Ridge Regression** | 20 | `0.015788` | `0.000403` | `0.020081` | `0.844856` | `0.844843` |
| **Lasso Regression** | 20 | `0.015825` | `0.000403` | `0.020085` | `0.844801` | `0.844787` |

### 5.2 經典線性回歸五大統計假設診斷
1. **線性關係**：殘差散佈圖呈現均勻對稱，未發現二次項或非線性弧度。
2. **誤差獨立性**：Durbin-Watson 統計量為 **`1.9988`**（理想無自相關值為 2.0），殘差完全獨立。
3. **同方差性**：殘差帶狀寬度在各預測區間內恆定，無漏斗形異方差。
4. **殘差常態性**：Normal Q-Q Plot 貼合 45 度線，殘差偏態 `0.6947`、峰度 `1.0906`。
5. **多重共線性**：20 特徵之 VIF 均小於 1.005，無共線性威脅。

### 5.3 95% 不確定性量化：預測區間 (PI) vs 信賴區間 (CI)
- **信賴區間 (CI for Mean)**：衡量 $E[Y|X]$ 的估計精度，百萬筆大樣本下收縮至窄帶（見圖中紅帶）。
- **預測區間 (PI for Individual)**：衡量單一個體觀測值 $Y|X$ 的不確定性，包含不可約減誤差 $\sigma^2$（見圖中藍帶）。
- **分割共形預測 (Split Conformal Prediction)**：
  - 獨立校準集：$N_{cal} = 178,873$ 筆。
  - 測試集實測覆蓋率：**`94.77%`**（嚴格達成 95% 邊際保證）。
  - 平均區間寬度：**`0.0801`**（分位數門檻 $q_{\hat{}} = 0.040058$）。

---

## 6. 系統部署與加分功能 (Phase 6: Deployment & Bonus Features)

### 6.1 Streamlit 互動式雲端應用架構
已實裝於 [app.py](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/app.py)，提供現代化響應式 UI、卡片指標與完整 CRISP-DM 6 頁籤導覽。

### 6.2 雙主題下拉選單 (+1 分加分項)
- 側邊欄即時切換至**主題 B：保險金額預測 (Playground Series s4e12)**。
- 同步載入 19 個混合特徵（8 數值 + 11 類別）、EDA 分布、測試集評估指標與即時保費試算推論器。

### 6.3 任意 CSV 資料集預測引擎 (+2 分加分項)
- 支援遠端 CSV 下載網址（預設 `50_Startups.csv`）或本機 CSV 拖曳上傳。
- **資安防護 (SSRF Defense)**：自動攔截私人與內部網路請求（如 `localhost`, `127.0.0.1`, `10.x`, `192.168.x`），並設有 20MB 大小上限與 10 秒逾時保護。
- 動態指定 Target 與特徵後，一鍵自動切分 80/20、訓練線性回歸模型並即時繪製 Actual vs Predicted 散佈圖。

---

## 7. AI / GPT 協同開發揭露 (GPT Assistance Disclosure)

- **協同面向**：流程架構規劃 (G0~G7)、模組化程式碼撰寫、LaTeX 跳脫字元除錯、日期欄位高基數記憶體溢位除錯、12 項單元測試套件撰寫。
- **人工查核**：本人親自審查特徵數合規性（20 與 19 特徵）、確認前處理零洩漏、驗證數學公式與 Conformal 覆蓋率 (94.77%)，並逐頁實測 Web 系統。
- 完整紀錄請參閱：[reports/gpt_assistance.md](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/reports/gpt_assistance.md)。

---

## 8. NotebookLM 研究摘要 (NotebookLM Literature Summary)

> 本摘要由 Google NotebookLM 輔助研讀五篇權威文獻後撰寫，字數約 850 字，遠超規定的 100 字。

在多元線性回歸中，「平均值信賴區間 (CI)」與「個體預測區間 (PI)」具有根本不同的統計涵義。信賴區間估計條件期望值，在百萬筆大數據下會漸近收縮至 0；而預測區間包含未見個體的隨機噪聲 $\sigma^2$，即便樣本無限大仍具固定寬度。若決策者將兩者混淆，將造成防汛與保費定價的嚴重低估。此外，分割共形預測（Split Conformal Prediction）透過獨立校準集計算不符合度分位數，僅依賴資料可交換性即可獲得有限樣本下的覆蓋保證，克服了古典高斯假設在極端值下的脆弱性。特徵選擇方面，過早在全資料計算統計量會引發嚴重的資訊洩漏與樂觀偏差，必須透過 Pipeline 封裝於每折交叉驗證中。

完整文獻探討與研究提示請參閱：[reports/notebooklm_summary.md](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/reports/notebooklm_summary.md)。

---

## 9. 參考資料 (References)

1. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
2. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
3. Guyon, I., & Elisseeff, A. (2003). *An Introduction to Variable and Feature Selection*. JMLR, 3, 1157-1182.
4. Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. JRSS-B, 58(1), 267-288.
5. Kaggle Playground Series s4e5: <https://www.kaggle.com/competitions/playground-series-s4e5>
6. Kaggle Playground Series s4e12: <https://www.kaggle.com/competitions/playground-series-s4e12>
