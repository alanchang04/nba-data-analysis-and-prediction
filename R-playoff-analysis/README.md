# 📈 Playoff Impact Factor Analysis — R + Random Forest

This module analyzes which team statistics contribute most significantly to whether a team makes the NBA playoffs.  
Using R and a Random Forest classifier, we evaluated both **basic** and **advanced** team metrics to uncover key predictive features.

---

## 🎯 Goal

> Predict whether a team will reach the playoffs, and identify the most influential features that impact this outcome.

---

## 🧪 Dataset & Features

The dataset contains team-level stats over multiple NBA seasons.  
The target label is `Playoffs` (1 = Made Playoffs, 0 = Missed Playoffs)

**Selected Features:**

- `Age`: Team average age  
- `W`, `L`: Wins and losses  
- `MOV`: Margin of victory  
- `SOS`: Strength of schedule  
- `ORtg`, `DRtg`: Offensive / Defensive ratings  
- `TS`: True Shooting %  
- `OeFG`, `DeFG`: Effective FG% (offensive & defensive)

---

## 🛠 Tools Used

- **Language**: R
- **Packages**: `randomForest`, `caret`, `dplyr`, `ggplot2`

---

## ⚙️ Modeling Pipeline

1. Feature selection & preprocessing with `dplyr`
2. Train/test split using `caret`
3. Train a `randomForest` model with:
   - `ntree = 500`
   - `mtry = sqrt(# of features)`
4. Generate confusion matrix and accuracy scores
5. Visualize feature importance

---


### 🔍 Feature Importance (Top 5)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `Net Rating / MOV` | Margin of Victory is the strongest indicator of playoff success |
| 2 | `True Shooting % (TS)` | Efficient scoring is crucial |
| 3 | `DRtg` | Defensive efficiency matters |
| 4 | `ORtg` | Offensive efficiency |
| 5 | `W` | Total wins (obviously important, but also captured indirectly by ratings)

### 📉 Importance Plot

![Feature Importance](feature_importance_plot.png)

---

## 🧠 Bonus: Confidence & Error Analysis

- Computed prediction **confidence** per sample
- Identified **high-confidence misclassifications**
- Showcased examples of **false positives/negatives**

---

## 📁 Files

| File | Description |
|------|-------------|
| `nba_rf_playoff.R` | Full R script: preprocessing, modeling, evaluation |
| `feature_importance_plot.png` | Plot showing variable importance |
| `data.csv` (optional) | Cleaned dataset (if shareable) |

---

## 🧩 Connection to Python Model

The features ranked here were used as **input variables** for deeper machine learning prediction in the Python-based match outcome module.

