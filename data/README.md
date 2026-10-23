# Dataset Information & Reproduction Guide

This directory holds the raw and processed data for HW2: Multiple Linear Regression with CRISP-DM methodology.

> **Note**: Due to GitHub file size limits (>100MB) and dataset license terms, raw CSV files are excluded from Git tracking via `.gitignore`. Please follow the instructions below to download and place the data files.

---

## 1. Problem A: Flood Prediction (Main Required Topic)
- **Kaggle Competition**: [Playground Series - Season 4, Episode 5 (Regression with a Flood Prediction Dataset)](https://www.kaggle.com/competitions/playground-series-s4e5/data)
- **Target Variable**: `FloodProbability` (Continuous float: `0.285` - `0.725`)
- **Original Features**: 20 numerical features (all continuous / integer metrics of environmental and governmental factors).
- **Target File Path**: `data/flood/train.csv` (1,117,957 rows, 22 columns including `id` and `FloodProbability`).

### Feature Dictionary (20 Features):
1. `MonsoonIntensity`: Frequency and intensity of monsoons
2. `TopographyDrainage`: Effectiveness of geographical and regional drainage
3. `RiverManagement`: River maintenance and infrastructure quality
4. `Deforestation`: Rate of local deforestation
5. `Urbanization`: Urban expansion and impermeable surface density
6. `ClimateChange`: Impact metric of climate shift
7. `DamsQuality`: Structural health and management of dams
8. `Siltation`: Riverbed sedimentation and silt buildup
9. `AgriculturalPractices`: Sustainable or vulnerable agricultural methods
10. `Encroachments`: Illegal or unplanned structures on riverbanks
11. `IneffectiveDisasterPreparedness`: Emergency readiness score
12. `DrainageSystems`: Urban and rural drainage adequacy
13. `CoastalVulnerability`: Sea-level and storm-surge exposure
14. `Landslides`: Geological slope instability
15. `Watersheds`: Condition and coverage of catchment areas
16. `DeterioratingInfrastructure`: Age and decay of infrastructure
17. `PopulationScore`: Population density in risk areas
18. `WetlandLoss`: Destruction of natural water buffer wetlands
19. `InadequatePlanning`: Municipal zoning and development deficiencies
20. `PoliticalFactors`: Policy and governance responsiveness

---

## 2. Problem B: Insurance Premium Prediction (Bonus Topic 1)
- **Kaggle Competition**: [Playground Series - Season 4, Episode 12 (Regression with an Insurance Dataset)](https://www.kaggle.com/competitions/playground-series-s4e12/data)
- **Target Variable**: `Premium Amount` (Continuous float: insurance premium)
- **Original Features**: 19 mixed features (8 numerical, 11 categorical; excluding `id` and target).
- **Target File Path**: `data/insurance/train.csv` (1,200,000 rows, 21 columns).

### Feature Breakdown:
- **Numerical Features (8)**: `Age`, `Annual Income`, `Number of Dependents`, `Health Score`, `Previous Claims`, `Vehicle Age`, `Credit Score`, `Insurance Duration`.
- **Categorical Features (11)**: `Gender`, `Marital Status`, `Education Level`, `Occupation`, `Location`, `Policy Type`, `Customer Feedback`, `Smoking Status`, `Exercise Frequency`, `Property Type`, `Policy Start Date`.

---

## 3. External CSV for Dynamic Loading (Bonus Topic 2 Demo)
- **Source**: `50_Startups.csv`
- **URL**: `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`
- **Purpose**: Live demonstration of uploading custom CSV or loading via remote URL.
