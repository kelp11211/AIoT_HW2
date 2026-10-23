# HW2 執行計畫：CRISP-DM 多元線性回歸

> 目標：以 Kaggle 公開資料集完成多元線性回歸、特徵選擇、模型評估、預測區間視覺化，並以 GitHub + 網頁部署為主要成果。

## 0. 作業規格與驗收條件

- [ ] 選用 Kaggle 公開資料集，附名稱、網址、授權／使用條件與資料字典。
- [ ] **原始輸入特徵數 > 13**；建議選 **14–39 個原始特徵**（不含 Target），保留檢查紀錄。編碼後特徵數另行記錄。
- [ ] 預測目標為連續數值，使用 `sklearn.linear_model.LinearRegression` 建模。
- [ ] 完成特徵選擇，避免測試資料洩漏。
- [ ] 提供 MAE、MSE、RMSE、R²（可加 Adjusted R²）。
- [ ] 預測結果圖包含 **95% 信賴區間或預測區間**，標註計算方法及適用條件。
- [ ] 說明 CRISP-DM 六階段。
- [ ] 報告標示「GPT 輔助內容」與「NotebookLM 摘要」，後者至少 100 個中文字。
- [ ] 主要繳交 GitHub repository 與可開啟的 Web 網址。
- [ ] 留存 ChatGPT 對話 PDF；依教師說明可作為補充附件。

## 1. 確認專案與環境

1. 參考 DIC2 專案：<https://github.com/kelp11211/AIoT_DIC2_MultipleLR>。
2. 決定沿用現有 repository 或建立 HW2 專用 repository；建議分支或獨立資料夾，避免覆蓋 DIC2 成果。
3. 建立 Python 虛擬環境，安裝 `pandas`、`numpy`、`scikit-learn`、`statsmodels`、`matplotlib`、`seaborn`、`streamlit`、`pytest` 等依賴，將版本寫入 `requirements.txt`。
4. 設定固定隨機種子（例如 42），並記錄 Python、套件與資料版本。

**驗收：** 能在乾淨環境安裝依賴、啟動 Web，README 有執行方式。

## 2. 選題與資料集確認（Business Understanding）

**建議主題 A：Predict the Flood**。優先確認 Kaggle 官方競賽／資料集頁面、下載方式、Target 名稱、樣本數、是否確實有 20 個輸入特徵、使用條款與評估限制；未核實前不可直接宣稱已符合規格。

**主題 B：** 選另一個 Kaggle 上具 14–39 個原始輸入特徵、連續 Target 的回歸資料集，作為加分下拉選單的第二題。需記錄確切 Kaggle URL，避免選到分類資料。

**加分用外部 CSV 範例：** <https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv>。此 CSV 僅作為「任意資料集輸入」功能測試；其原始特徵數不符合 HW2 主資料集的 >13 條件，不能拿來取代主題 A/B。

**驗收：** 每個正式主題都有資料來源、原始特徵數、Target、業務問題及成功指標。

## 3. 資料理解（Data Understanding）

- [ ] 載入 CSV，檢查筆數、欄位、型別、Target 分布。
- [ ] 確認特徵數：排除 Target、ID、資料洩漏欄位；記錄剔除原因。
- [ ] 分析缺失值、重複列、異常值、相關係數與多重共線性（例如 VIF）。
- [ ] 產出缺失值圖、Target 分布圖、相關性熱圖。

**驗收：** `data_profile` 報表及可重現的 EDA 圖表。

## 4. 資料準備（Data Preparation）

1. 先切分訓練集／測試集（例如 80/20），保留測試集到最終評估。
2. 用 `Pipeline` + `ColumnTransformer` 在訓練資料內擬合缺失值補齊、類別編碼與必要的標準化。
3. 處理未知類別、非數值 Target、全空欄位與極端值；避免將 ID 當成可預測特徵。
4. 交叉驗證、特徵選擇與調參全部只在訓練資料進行；最後才評估測試集。

**驗收：** 前處理可重複套用於新資料，且不會洩漏測試資訊。

## 5. 特徵選擇與建模（Modeling）

1. 建立基準模型：使用全部合格特徵的 `LinearRegression`。
2. 使用至少一種正式特徵選擇法，例如 `SelectKBest(f_regression)`、`RFE` 或 `SequentialFeatureSelector`；可加 Lasso 比較。
3. 用交叉驗證選擇特徵數 `k`，將選擇器放進 Pipeline；不可在全資料先挑特徵。
4. 比較基準模型與選後模型，記錄選入特徵、係數、交叉驗證結果。
5. 可選擇增加 Ridge／Lasso 作為對照，但主要模型必須包含 LinearRegression。

**驗收：** 有基準／特徵選擇後的模型比較表及選取依據。

## 6. 模型評估與預測區間（Evaluation）

- [ ] 在保留測試集上計算 MAE、MSE、RMSE、R²；Adjusted R² 僅在樣本數與參數數量允許時呈現。
- [ ] 產出實際值 vs 預測值散佈圖、殘差 vs 預測值圖、殘差 Q-Q 圖及殘差分布圖。
- [ ] 檢查線性關係、誤差獨立性、同方差性、殘差常態性、多重共線性；說明檢驗的侷限。
- [ ] **產出含 95% 預測區間的預測圖**：建議採訓練資料內的校準集進行 split conformal prediction，使用校準殘差絕對值分位數建立區間，測試集僅做最終展示與覆蓋率評估。標示交換性假設與邊際覆蓋率意義；不得將一般迴歸線的平均值信賴區間誤稱為個別觀測的預測區間。
- [ ] 報告測試集預測區間覆蓋率與平均寬度。

**驗收：** 至少一張清楚標示「實際值、預測值、95% 預測區間」的圖，並附區間算法與評估結果。

## 7. 網頁實作與部署（Deployment）

建議以 **Streamlit** 實作，並以 GitHub + Streamlit Community Cloud 或其他可公開訪問的部署服務展示。

**基本功能：** 主題介紹、資料來源、EDA、特徵選擇、模型指標、預測圖（含區間）、單筆或批次預測、CRISP-DM 說明。

**加分 1 分：** 下拉選單提供兩個正式回歸 Problem。切換時應同步切換資料、Target、特徵、模型與圖表，避免混用。

**加分 2 分：** 支援使用者輸入 CSV URL 或上傳 CSV，預覽資料、選擇連續 Target、選取欄位、訓練並評估模型。處理無效 URL、非 CSV、資料過小、欄位缺失、分類 Target、無法讀取等錯誤；限制下載大小與請求逾時，避免存取內部／私人網路位址（SSRF）。公開服務應對不可信外部 URL 設定安全限制。

**驗收：** 在無本機檔案的部署環境仍能操作；兩個 Problem 可切換；外部 CSV 範例可成功分析或顯示合理錯誤。

## 8. NotebookLM 研究與 AI 使用揭露

1. 搜集與選定主題、特徵選擇、線性回歸假設及預測區間有關的網路文章／論文／教學，保存來源網址與存取日期。
2. 將來源加入 NotebookLM，提出比較方法、優缺點、實務限制等研究問題。
3. 自行核對 NotebookLM 摘要與原始來源，寫 **至少 100 個中文字**的「NotebookLM 摘要」，附來源及使用方式；不得將未實際執行的 NotebookLM 研究寫成已完成。
4. 在報告另設「GPT 輔助內容」：列出 GPT 協助範圍（規劃、程式、除錯、圖表、文字）及自行驗證內容。
5. 匯出本作業相關 ChatGPT 對話為 PDF（可使用列印為 PDF 或合適的匯出工具），檢查文字可讀、頁數完整；避免將金鑰、私人資料公開。

**驗收：** 研究摘要字數達標、來源可追溯、AI 協助明確揭露、對話 PDF 已保存。

## 9. 建議 GitHub 結構

```text
HW2_Multiple_Linear_Regression/
├── README.md
├── requirements.txt
├── app.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── intervals.py
├── notebooks/
│   └── hw2_crisp_dm.ipynb
├── data/
│   └── README.md
├── reports/
│   ├── HW2_REPORT.md
│   ├── notebooklm_summary.md
│   └── gpt_assistance.md
├── figures/
├── tests/
│   └── test_pipeline.py
└── .gitignore
```

大型資料、授權受限資料、模型快取與敏感資訊不要直接提交；提供下載與重現指引。

## 10. README 與報告內容

README 建議包含：專案目的、Kaggle 來源、資料與特徵數、兩個問題說明、環境安裝、執行指令、CRISP-DM 六階段摘要、特徵選擇方法、評估結果表、95% 預測區間圖、線上 Demo、NotebookLM 摘要連結、GPT 輔助內容及限制。

報告 `HW2_REPORT.md` 建議依序撰寫：摘要、Business Understanding、Data Understanding、Data Preparation、Modeling、Evaluation、Deployment、GPT 輔助內容、NotebookLM 摘要（100 字以上）、參考資料。

## 11. 建議執行順序（Gate 驗收）

| Gate | 工作 | PASS 條件 |
|---|---|---|
| G0 | 專案初始化 | Repo、環境、README 骨架完成 |
| G1 | 選定兩個 Kaggle Problem | 各有 URL、Target、原始特徵數 >13 |
| G2 | EDA 與資料前處理 | 可重現、無資料洩漏 |
| G3 | 基準模型與特徵選擇 | 交叉驗證完成、模型比較表完成 |
| G4 | 評估與預測區間 | 指標、假設檢查、95% 區間圖完成 |
| G5 | Web 與加分功能 | 兩題下拉選單、外部 CSV 載入可用 |
| G6 | 研究與文件 | NotebookLM 100 字摘要、GPT 標示、對話 PDF |
| G7 | GitHub 與部署驗收 | GitHub 與公開 Web 網址可訪問、README 完整 |

## 12. 最終繳交檢查

- [ ] GitHub repository URL（主要繳交）
- [ ] 公開 Web Demo URL（主要繳交）
- [ ] Kaggle 資料集連結與 >13 原始特徵證明
- [ ] CRISP-DM 六階段說明
- [ ] Feature Selection 與模型評估結果
- [ ] 95% 信賴／預測區間圖及計算說明
- [ ] 兩種 Problem 下拉選單（加分）
- [ ] 任意 CSV URL／上傳功能（加分）
- [ ] NotebookLM 研究摘要（至少 100 個中文字）
- [ ] 「GPT 輔助內容」標示
- [ ] ChatGPT 對話 PDF 已備妥
- [ ] 可選：資訊圖表、完整 PDF 報告

> **注意：** 本文件是執行計畫，並非已完成實驗的報告；資料集細節、模型分數、NotebookLM 研究成果、部署網址均須在實際執行後填入。
