# Gate 2 驗收報告：資料理解 (EDA) 與資料準備 (Data Preparation)

> **審查日期**：2026-10-23  
> **驗收狀態**：**PASS ✅**  
> **核心標準**：EDA 可重現產出高解析圖表、無任何資料洩漏 (Strict Hold-out 80/20)、多重共線性 (VIF) 與前處理驗證通過。

---

## 1. 資料概況與品質指標 (Data Understanding)

- **資料集名稱**：Kaggle Playground Series s4e5 (Predict the Flood)
- **總樣本數**：1,117,957 列
- **原始輸入特徵數**：20 個（排除 `id` 與 `FloodProbability`）
- **缺失值個數**：0 個（完整無缺失）
- **重複列數**：0 列（無重複觀測值）
- **目標變數統計特徵**：
  - 最小值：`0.2850`
  - 最大值：`0.7250`
  - 平均值：`0.5045`
  - 標準差：`0.0510`
  - 偏態 (Skewness)：`0.0472`（接近對稱常態分布）
  - 峰度 (Kurtosis)：`-0.0396`

---

## 2. 探索性分析視覺化圖表 (EDA Figures)

本階段已自動生成並儲存以下分析圖表至 `figures/` 目錄：
1. **目標變數分布圖** (`figures/eda_flood_target_dist.png`)：呈現 `FloodProbability` 呈高度對稱之鐘型常態分布，中心集中於 0.5044。
2. **20 個環境特徵相關係數熱圖** (`figures/eda_flood_corr_heatmap.png`)：各特徵與目標變數呈現穩定之正相關性 (約 0.18 ~ 0.23)，特徵間互相獨立性高。
3. **關鍵特徵散佈與回歸趨勢圖** (`figures/eda_flood_top_features_scatter.png`)：展示前三大相關特徵之線性回歸適配關係。
4. **VIF 多重共線性長條圖** (`figures/eda_flood_vif_chart.png`)：驗證共線性狀況。

---

## 3. 多重共線性檢驗 (VIF Analysis)

- **檢驗方法**：標準化特徵後計算變異數膨脹因子 (Variance Inflation Factor, VIF)。
- **檢驗結果**：所有 20 個特徵之 VIF 均介於 `1.002` 至 `1.005` 之間。
- **結論**：遠低於共線性警戒閾值 5.0，特徵間不存在嚴重多重共線性，完全適合進行多元線性回歸建模。
- 詳細數據報表參閱：[reports/vif_report_flood.csv](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/reports/vif_report_flood.csv)。

---

## 4. 資料前處理與杜絕資料洩漏 (Data Preparation & No Leakage)

1. **資料集切分**：
   - 採用 `train_test_split(test_size=0.20, random_state=42)`。
   - 訓練集：894,365 筆 (80%)。
   - 測試集：223,592 筆 (20%，嚴格保留至 Gate 4/5 評估)。
2. **前處理管線設計**：
   - 使用 `scikit-learn` 之 `ColumnTransformer`。
   - 數值特徵以 `SimpleImputer(strategy='median')` 補值，並串接 `StandardScaler()`。
   - 類別特徵（針對主題 B 及外部資料）配置 `OneHotEncoder(handle_unknown='ignore')`。
3. **無洩漏驗收**：
   - Preprocessor 僅呼叫 `.fit(X_train)`，均值與標準差純粹來自訓練集。
   - 測試集僅執行 `.transform(X_test)`，未參與任何統計量計算。

---

**Gate 2 驗收狀態：PASS ✅**
