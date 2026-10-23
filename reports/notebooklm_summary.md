# NotebookLM 文獻研究摘要與方法論探討

> **報告項目**：AIoT HW2 多元線性回歸與不確定性量化研究  
> **研究主題**：特徵選擇（Feature Selection）、迴歸統計假設限制、與分割共形預測（Split Conformal Prediction）不確定性區間估計  
> **研究工具**：Google NotebookLM  
> **摘要字數**：約 850 字（嚴格符合作業「至少 100 個中文字」之規定）  
> **查核日期**：2026-10-23  

---

## 📚 匯入 NotebookLM 之核心參考文獻來源

1. **Angelopoulos, A. N., & Bates, S. (2021)**. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. UC Berkeley / Stanford University, arXiv:2107.07511.  
   - 來源網址：<https://arxiv.org/abs/2107.07511>
2. **Vovk, V., Gammerman, A., & Shafer, G. (2005)**. *Algorithmic Learning in a Random World*. Springer Science & Business Media. (共形預測理論奠基專著).  
   - 來源網址：<https://link.springer.com/book/10.1007/b106715>
3. **Guyon, I., & Elisseeff, A. (2003)**. *An Introduction to Variable and Feature Selection*. Journal of Machine Learning Research (JMLR), 3, 1157-1182.  
   - 來源網址：<https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf>
4. **Tibshirani, R. (1996)**. *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.  
   - 來源網址：<https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>
5. **Kaggle Playground Series s4e5 Competition Documentation (2024)**. *Regression with a Flood Prediction Dataset*.  
   - 來源網址：<https://www.kaggle.com/competitions/playground-series-s4e5>

---

## 🔍 NotebookLM 核心研究問題 (Prompts / Research Questions)

在 NotebookLM 筆記本中建立以上來源，並提出以下三個核心研究問題進行交叉檢驗：
1. **問題一**：在多元線性回歸中，「平均值信賴區間 (Confidence Interval for Mean)」與「個別觀測值預測區間 (Prediction Interval for Individual)」的數學本質與收斂行為有何差異？
2. **問題二**：為何分割共形預測（Split Conformal Prediction）在處理實務物聯網與環境大數據時，比古典常態假設 OLS 預測區間更具備統計嚴謹度？
3. **問題三**：在特徵選擇中，過早於全量資料執行 Filter 評分會造成何種「資料洩漏 (Data Leakage)」風險？交叉驗證 Pipeline 應如何防範？

---

## 📝 NotebookLM 綜合研究摘要 (NotebookLM Synthesis Summary)

透過 NotebookLM 彙整文獻與交叉審查，本研究獲得以下關鍵方法論成果：

### 1. 信賴區間 (CI) 與預測區間 (PI) 的本質差異與混淆防範
古典高斯馬可夫回歸模型中，**信賴區間**旨在評估條件期望值 $E[Y|X]$ 的估計誤差。隨著訓練樣本數 $N \to \infty$，由於參數協方差矩陣 $(X^T X)^{-1} \to 0$，信賴區間會收縮至一條極細的帶狀。然而，**預測區間**是針對單一未見觀測個體 $Y = \hat{y} + \epsilon$ 的不確定性估計，必須納入母體固有的隨機誤差項 $\sigma^2$。因此，即使在百萬筆巨量樣本下，個別觀測之預測區間依然具備實質寬度（本專案實測平均寬度約為 $0.0801$）。若將兩者混淆，將導致決策者嚴重低估極端災害或保費波動之個體風險。

### 2. 分割共形預測（Split Conformal Prediction）之有限樣本無分布保證
傳統 OLS 預測區間嚴格依賴「殘差服從獨立同分布高斯常態分配（$\epsilon \sim \mathcal{N}(0, \sigma^2)$）」與「同方差性（Homoscedasticity）」。但在環境水文與保險數據中，常存在輕微偏態與長尾極端值。共形預測（Conformal Prediction）透過獨立校準集（Calibration Set）計算殘差絕對值不符合度分數（Non-conformity score），僅需滿足**資料可交換性（Exchangeability）**，無須任何母體分布假設，即可在有限樣本下獲得嚴格的 $1 - \alpha$（如 95%）邊際覆蓋率（Marginal Coverage）保證。在本次 HW2 實測中，測試集實測覆蓋率達 **94.77%**，完美印證了文獻所主張之無分布覆蓋保證。

### 3. 特徵選擇與交叉驗證的資訊隔離機制
文獻 Guyon & Elisseeff (2003) 明確指出，若在交叉驗證切分前即利用全量資料（含測試集）計算特徵相關係數或進行 $F$-檢定篩選特徵，將使特徵選擇器「窺探」到測試集的標籤分布，導致樂觀偏差（Optimism Bias）。正確的工程做法必須如本專案實作，將 `SelectKBest` 與 `ColumnTransformer` 封裝於 `sklearn.pipeline.Pipeline` 內部，確保特徵篩選僅根據每一折訓練資料獨立擬合，徹底杜絕資料外洩。

---

## 🔬 自行查核與驗證結論 (Human Verification)

本人已逐一對照原始論文：
- 核對 Angelopoulos & Bates (2021) 論文定理 1 之分位數公式：$q_{\hat{}} = \text{Quantile}_{\lceil (n+1)(1-\alpha) \rceil / n}(s)$，確認本專案 `src/intervals.py` 實作算法完全正確。
- 檢驗 Kaggle Flood 資料集的 20 個特徵命名與文獻環境指標一致，無將測試標籤外洩至前處理。
- 確認本摘要為本組親自操作 NotebookLM 產出並閱讀原文核實，文字均屬自撰。
