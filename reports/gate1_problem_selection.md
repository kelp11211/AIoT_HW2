# Gate 1 驗收報告：Kaggle 題目與資料規格審查

> **審查日期**：2026-10-23  
> **驗收狀態**：**PASS**  
> **審查標準**：Kaggle 公開資料集、原始輸入特徵數嚴格介於 14–39（>13，不含 Target 與 ID）、預測目標為連續數值、無資料洩漏。

---

## 1. 題目與資料規格總覽

| 規格項目 | 主題 A (必做主要題目) | 主題 B (加分雙主題切換) |
|---|---|---|
| **題目名稱** | Predict the Flood (Kaggle Playground Series s4e5) | Insurance Premium Prediction (Kaggle Playground Series s4e12) |
| **Kaggle 官方網址** | [https://www.kaggle.com/competitions/playground-series-s4e5/data](https://www.kaggle.com/competitions/playground-series-s4e5/data) | [https://www.kaggle.com/competitions/playground-series-s4e12/data](https://www.kaggle.com/competitions/playground-series-s4e12/data) |
| **角色定位** | 主題 A（完整 CRISP-DM 分析與預測區間） | 主題 B（支援 Web 雙題目切換） |
| **樣本列數** | 1,117,957 列 | 1,200,000 列 |
| **目標變數 (Target)** | `FloodProbability` (連續浮點數) | `Premium Amount` (連續浮點數) |
| **目標值範圍** | `0.2850` ~ `0.7250` (平均 `0.5045`) | `20.00` ~ `4999.00` (平均 `1102.54`) |
| **原始輸入特徵數** | **20 個** (符合 14–39 要求) | **19 個** (符合 14–39 要求) |
| **特徵型態組成** | 20 個純數值指標 | 8 個數值指標、11 個類別特徵 |
| **排除欄位 (防洩漏)** | ['id', 'FloodProbability'] | ['id', 'Premium Amount'] |
| **缺失值與重複值** | 缺失: 0，重複: 0 | 缺失: 1203748，重複: 0 |
| **Gate 1 審查結果** | **PASS** | **PASS** |

---

## 2. 商業問題與決策目標 (Business Understanding)

### 主題 A：洪水機率預測 (Flood Prediction)
- **商業背景**：洪澇災害對城市民生與經濟造成嚴重衝擊。傳統水利物理數值模擬運算耗時，難以即時在短延遲內估計大區域綜合風險。
- **目標變數**：`FloodProbability` (0.0 ~ 1.0)。
- **決策價值**：主管機關在強降雨預警發布時，輸入氣象指標、集水區治理評分與河道清淤狀況，即時推估洪水發生機率，精確調配抽水機與防汛資源。
- **成功指標**：RMSE、MAE 優於 DummyRegressor (平均值常數基準)，R² 顯著提升，並產出 95% 預測區間供決策風險容許度評估。

### 主題 B：保險金額預測 (Insurance Premium Prediction)
- **商業背景**：保險業者需精準評估投保人風險屬性，提供合宜之保費定價，避免逆選擇與保費失真。
- **目標變數**：`Premium Amount` (保險費總額)。
- **決策價值**：依據投保人健康分數、信用等級、過往理賠次數、車齡與年收入等 19 個維度，自動化建議基礎保費，提升核保效率與定價公信力。
- **成功指標**：建立可解釋的多元線性回歸定價公式，RMSE 與 R² 具備實際商業參考價值。

---

## 3. 原始特徵清單詳表 (符合 >13 規格證明)

### 主題 A (20 個特徵)：
[
  "MonsoonIntensity",
  "TopographyDrainage",
  "RiverManagement",
  "Deforestation",
  "Urbanization",
  "ClimateChange",
  "DamsQuality",
  "Siltation",
  "AgriculturalPractices",
  "Encroachments",
  "IneffectiveDisasterPreparedness",
  "DrainageSystems",
  "CoastalVulnerability",
  "Landslides",
  "Watersheds",
  "DeterioratingInfrastructure",
  "PopulationScore",
  "WetlandLoss",
  "InadequatePlanning",
  "PoliticalFactors"
]

### 主題 B (19 個特徵)：
- **數值特徵 (8 個)**：[
  "Age",
  "Annual Income",
  "Number of Dependents",
  "Health Score",
  "Previous Claims",
  "Vehicle Age",
  "Credit Score",
  "Insurance Duration"
]
- **類別特徵 (11 個)**：[
  "Gender",
  "Marital Status",
  "Education Level",
  "Occupation",
  "Location",
  "Policy Type",
  "Policy Start Date",
  "Customer Feedback",
  "Smoking Status",
  "Exercise Frequency",
  "Property Type"
]

---

## 4. 驗收核可結論

兩個資料集均嚴格滿足：
1. Kaggle 官方公開比賽資料來源
2. 原始特徵數分別為 **20** 與 **19**，完全符合作業規定的 **14–39 (且 >13)**
3. 預測目標均為連續數值（非分類）
4. 排除純識別碼 (`id`) 與目標變數本身，杜絕資料外洩

**Gate 1 判定：ALL PASS ✅**
