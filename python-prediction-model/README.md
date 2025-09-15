# 🤖 NBA Match Outcome Prediction — Python

This module builds machine learning models to predict the outcome of NBA matchups based on team statistics and game context.  
We compare the performance of a traditional **Random Forest** classifier and a **Transformer-based deep learning model**.

---

## 🎯 Goal

> Predict whether Team A will win or lose against Team B based on both teams' statistical profiles during the regular season and playoffs.

---

## 📊 Dataset

The dataset includes historical NBA matchup data across several seasons, each sample containing:

- Team A & Team B features (basic + advanced stats)
- Home/Away status
- Whether it's a regular season or playoff game
- Target: `1` = Team A wins, `0` = Team A loses

---

## 🔍 Feature Engineering

- Computed differences between Team A and Team B stats
- Normalized numerical features
- Encoded categorical variables (season type, home court)
- Removed high-correlation noise

---

## ⚙️ Models Built

| Model | Description |
|-------|-------------|
| `RandomForestClassifier` | Fast, interpretable baseline |
| `Transformer (HuggingFace)` | Sequence-based model trained on feature vectors |

---

## 📈 Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest | ~59% | High precision on regular season games |
| Transformer   | ~61% | Better generalization on playoff data |


---

## 🛠 Tools Used

- **Language**: Python 3
- **Packages**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `transformers`, `torch`, `seaborn`

---

## 📁 Files

| File/Folder | Description |
|-------------|-------------|
| `data/nba_match_data.csv` | Cleaned and labeled data used for training |
| `notebooks/1_feature_engineering.ipynb` | Feature generation and EDA |
| `notebooks/2_random_forest_model.ipynb` | Baseline model |
| `notebooks/3_transformer_model.ipynb` | Transformer architecture & training |
| `models/` | Saved models (optional) |
| `utils.py` | Helper functions for data processing |

---

## 🧠 Insights

- Offensive/Defensive Rating difference, recent win streaks, and net rating gap are strong predictors
- Transformers work surprisingly well on structured stat vectors with enough training

---

## 📌 Next Steps

- Integrate real-time team stats for live game prediction
- Deploy via Flask + Streamlit interface
- Add player-level inputs for even more accurate predictions

---

## 👨‍💻 Author
主要coding:張耀仁、方敬棠、吳長恩

contact:張耀仁 | zy84946@gmail.com  


