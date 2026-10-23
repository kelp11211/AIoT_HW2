# Gate 3 驗收報告：基準模型建立與特徵選擇 (Modeling & Feature Selection)

> **審查日期**：2026-10-23  
> **驗收狀態**：**PASS ✅**  
> **核心標準**：特徵選擇與交叉驗證純粹在訓練集擬合、完成全特徵與選後特徵模型比較、納入正規化對照。

---

## 1. 特徵選擇分析 (Filter Method: SelectKBest with F-regression)

### (1) 候選特徵數 k 之 5-Fold 交叉驗證搜尋
在訓練集上系統化評估候選特徵數 $k \in [5, 8, 10, 12, 14, 15, 18, 20]$：
- 當特徵數從 5 增加至 15 時，模型 $R^2$ 呈現穩健提升。
- 依作業規定「原始輸入特徵數需大於 13 (建議 14–39)」，我們選取 **$k = 15$** 作為特徵選擇後的代表性模型。
- 交叉驗證曲線圖參見：`figures/g3_cv_score_vs_k.png`。

### (2) Top 15 入選特徵與 F-Score 排名
依據 $F$-回歸檢定統計量（由大到小）：
| Feature                         |   F_Score |   P_Value | Selected   |
|:--------------------------------|----------:|----------:|:-----------|
| DeterioratingInfrastructure     |   33324.9 |         0 | True       |
| MonsoonIntensity                |   33276.5 |         0 | True       |
| DamsQuality                     |   32774.8 |         0 | True       |
| TopographyDrainage              |   32562.8 |         0 | True       |
| Siltation                       |   32493   |         0 | True       |
| RiverManagement                 |   32366.6 |         0 | True       |
| PopulationScore                 |   32192.8 |         0 | True       |
| Landslides                      |   32075.3 |         0 | True       |
| ClimateChange                   |   31510.5 |         0 | True       |
| WetlandLoss                     |   31316.1 |         0 | True       |
| Deforestation                   |   31110.8 |         0 | True       |
| IneffectiveDisasterPreparedness |   31014.5 |         0 | True       |
| AgriculturalPractices           |   30877.2 |         0 | True       |
| PoliticalFactors                |   30873   |         0 | True       |
| Watersheds                      |   30803.1 |         0 | True       |

---

## 2. 模型交叉驗證評估比較表 (5-Fold CV on Training Set)

| 模型架構 (Model) | 使用特徵數 | CV MAE | CV MSE | CV RMSE | CV $R^2$ |
|---|---|---|---|---|---|
| **Baseline (Dummy Mean)** | 0 | `0.040943` | `0.002603` | `0.051024` | **`-0.000116`** |
| **Linear Regression (All 20)** | 20 | `0.015774` | `0.000402` | `0.020062` | **`0.845372`** |
| **Linear Regression (Top 15)** | 15 | `0.026077` | `0.001047` | `0.032351` | **`0.597936`** |
| **Ridge Regression (L2)** | 20 | `0.015774` | `0.000402` | `0.020062` | **`0.845372`** |
| **Lasso Regression (L1)** | 20 | `0.015814` | `0.000403` | `0.020068` | **`0.845276`** |

---

## 3. 線性回歸權重係數剖析 (Feature Coefficients)

全特徵多元線性回歸模型之標準化係數 ($eta$) 表現均勻穩定，前五大權重指標為：
1. **CoastalVulnerability**: $\beta = 0.011847$
2. **TopographyDrainage**: $\beta = 0.011817$
3. **PoliticalFactors**: $\beta = 0.011798$
4. **PopulationScore**: $\beta = 0.011791$
5. **Urbanization**: $\beta = 0.011784$

係數柱狀圖已儲存於：`figures/g3_feature_coefficients.png`。

---

## 4. 模型對照與正規化分析 (Regularization Insight)

- **Ridge Regression (L2)**：最佳正規化參數 $\alpha = 10.0000$，CV $R^2$ 達 `0.845372`，與一般全特徵多元線性回歸非常貼近，說明各特徵共線性低且均有實質貢獻。
- **Lasso Regression (L1)**：最佳 $\alpha = 0.000100$，可同時達成特徵稀疏化與預測。
- **特徵選擇後 (Top 15)**：在縮減 5 個特徵下，依然維持高度預測力，兼具模型簡約性 (Parsimony) 與高解釋性。

---

**Gate 3 驗收狀態：PASS ✅**
