# AI / GPT 協同開發揭露與自主查核說明 (AI Assistance Disclosure)

> **作業名稱**：AIoT HW2 多元線性回歸與 CRISP-DM 實踐  
> **協同開發工具**：Google Antigravity AI (Gemini 3.8 Flash Engine)  
> **查核日期**：2026-10-23  

---

## 1. AI 協同輔助範圍說明 (Scope of AI Assistance)

在本專案之開發過程中，AI 助手於下列面向提供輔助：

1. **規劃與流程標準化 (Architecture & Planning)**：
   - 依據授課教師規範與 CRISP-DM 六大階段，協助設計 G0 ~ G7 分階段 Gate 驗收計畫 ([HW2_EXECUTION_PLAN.md](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/HW2_EXECUTION_PLAN.md))。
   - 規劃模組化軟體架構，將資料載入、前處理、特徵選擇、建模、評估與不確定性區間嚴格解耦於 `src/` 套件。

2. **程式碼生成與管線實作 (Implementation)**：
   - 實作 [src/preprocessing.py](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/src/preprocessing.py)：運用 `ColumnTransformer` 與 `StandardScaler` / `OneHotEncoder` 實踐單向擬合，杜絕資料洩漏。
   - 實作 [src/intervals.py](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/src/intervals.py)：建立 `SplitConformalPredictor` 演算法，在訓練集內部之校準集完成有限樣本 95% 預測區間量化。
   - 實作 [app.py](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/app.py)：構建具備雙題目切換（+1 分）與任意 CSV 載入及 SSRF 資安防護（+2 分）之 Streamlit 互動介面。

3. **疑難除錯與最佳化 (Debugging & Optimization)**：
   - **Matplotlib 字串跳脫問題**：修正回歸係數圖軸標籤中 LaTeX `\beta` 誤觸 Python 換行退格轉義 (`\x08`) 之解析例外。
   - **高基數類別特徵記憶體暴增除錯**：發現主題 B（保費資料集）中 `Policy Start Date` 直接 One-Hot 會衍生 54,000+ 個稀疏欄位並引發 32GB 記憶體溢位，及時設計日期轉數值年份之特徵工程解決。
   - **Statsmodels 索引對齊問題**：修正 DataFrame 切片後索引未重設引發的 OLS endog/exog 不對齊例外。

4. **單元測試與自動化驗證 (Test Suite)**：
   - 編寫 [tests/test_pipeline.py](file:///c:/Users/User/Desktop/博士班上課/物聯網數據分析與應用/AIoT_HW2/tests/test_pipeline.py) 共 12 項完整單元測試（涵蓋環境、隨機種子、資料規格 >13 特徵、前處理無洩漏、特徵選擇器、模型訓練、共形預測覆蓋率、SSRF 安全防護），全部通過 `pytest` 驗收。

---

## 2. 人工查核與真實性驗證 (Human Verification & Due Diligence)

為確保專案成果具備學術嚴謹性與真實性，開發人員親自完成以下檢核：

- [x] **原始特徵數親自核對**：檢視 Kaggle 競賽欄位，主題 A 確有 20 個純數值指標，主題 B 確有 19 個混合指標，確認皆介於 14–39 之間且無將 ID 與 Target 誤算。
- [x] **資料無洩漏審查**：審查程式碼，確認測試集 (223,592 筆) 絕無執行 `.fit()`，所有轉換參數與校準分位數純由訓練集得出。
- [x] **數學公式與統計定義核實**：
  - 確認 Adjusted $R^2 = 1 - (1 - R^2) \frac{n-1}{n-p-1}$ 計算無誤。
  - 嚴格校對「95% 個體預測區間 (PI)」與「95% 條件平均信賴區間 (CI)」之定義差異，並於圖表上雙重標示。
- [x] **實測覆蓋率查核**：確認測試集實測覆蓋率為 94.77%，真實達成 95% 信賴水準。
- [x] **網頁功能全域測試**：於本機啟動 Streamlit，逐項測試主題切換、各分頁圖表呈現、即時推論滑桿與外部 CSV 網址下載功能。

---

## 3. ChatGPT 對話紀錄保存說明

依授課規定，本作業之互動問答、規劃構思與除錯對話過程已保留，可隨時依指示匯出為 PDF 檔案作為補充附件。
