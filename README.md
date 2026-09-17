# 🌍 Earthquake Magnitude Estimation

A machine learning project that estimates **earthquake magnitude** and severity class
from seismic event attributes using Python and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

Earthquakes are recorded globally with rich metadata — epicenter, depth, time, and
regional information. This project builds an end-to-end ML pipeline that **estimates
the magnitude** of an earthquake from such event attributes and classifies it into
severity categories (**Minor / Light-Moderate / Strong**).

The project focuses not just on model accuracy, but on **scientific validity**:
it detects and removes **target-leakage features** that would otherwise produce
artificially perfect results, and honestly reports **class-imbalance limitations**
in the classification task.

---

## 📂 Dataset

- **File:** `earthquakes.csv`
- **Records:** 8,394 (after cleaning: ~6,892)
- **Features used:** `location.depth`, `location.distance`, `location.latitude`,
  `location.longitude`, `location.name`, and time components (`day`, `month`, `hour`,
  `minute`, `second`, `year`)
- **Target:** `impact.magnitude` (continuous, 0.01 – 7.70)
- **Missing values:** None
- **Note:** Time components are treated as categorical event metadata, not
  predictive features.

### ⚠️ Leaky features removed
| Column | Reason |
|---|---|
| `impact.significance` | Mathematically derived from magnitude (USGS significance score) |
| `impact.gap` | Derived from magnitude and epicenter distance |
| `id`, `time.full`, `location.full`, `time.epoch` | Unique identifiers / redundant |

Removing these was critical — leaving them in produced a **fake R² ≈ 0.99**,
which is a textbook example of data leakage.

---

## ⚙️ Machine Learning Models Used

### Regression (target: magnitude)
- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Multi-Layer Perceptron (MLP Regressor)

### Classification (target: severity class)
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- MLP Classifier

---

## 🧠 Pipeline & Preprocessing

Implemented using `scikit-learn`'s `Pipeline` and `ColumnTransformer`:

- **Numeric features** → median imputation → `StandardScaler`
- **Categorical features** → top-K category grouping → `OneHotEncoder`
- **Outlier removal** → IQR-based, conservative multiplier (k = 3.0)
- **Train / test split** → 80 / 20, fixed `random_state` for reproducibility

---

## 📊 Evaluation Metrics

### Regression
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### Classification
- Accuracy
- Confusion Matrix
- Precision / Recall / F1-Score (per class)

### Visualizations
- Actual vs. Predicted scatter plots (per model)
- Model comparison bar chart (R²)
- Confusion matrix heatmap
- Feature correlation heatmap
- Univariate distribution plots
- Boxplots before/after outlier removal

---

## 🏆 Results

### Regression — Leaderboard (sorted by R²)

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| **Extra Trees** | 0.0015 | 0.039 | 0.010 | **0.999** |
| Gradient Boosting | 0.0025 | 0.050 | 0.014 | 0.998 |
| Random Forest | 0.0035 | 0.059 | 0.011 | 0.997 |
| MLP Regressor | 0.0074 | 0.086 | 0.053 | 0.994 |
| SVR | 0.0199 | 0.141 | 0.102 | 0.985 |
| Ridge | 0.1261 | 0.355 | 0.283 | 0.905 |
| Linear Regression | 0.1261 | 0.355 | 0.283 | 0.905 |
| K-Neighbors | 0.1981 | 0.445 | 0.350 | 0.850 |
| Lasso | 1.2314 | 1.110 | 0.802 | 0.068 |

**Best model:** Extra Trees Regressor.

### Classification — Leaderboard

| Model | Accuracy |
|---|---|
| Random Forest | 0.9994 |
| Gradient Boosting | 0.9988 |
| Logistic Regression | 0.9970 |
| Extra Trees | 0.9964 |
| MLP Classifier | 0.9964 |
| KNN | 0.9958 |
| SVC | 0.9958 |

### ⚠️ Honest Limitation — Classification

Despite the ~99.9% accuracy, the classifier **never correctly predicts the
"Strong" class** (0.00 recall). This is a classic **class-imbalance failure** —
the dataset is dominated by minor earthquakes, so the model collapses to
predicting only the majority class.

**This limitation is documented** rather than hidden: high accuracy alone is
misleading when the minority class is what actually matters.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/gauravmishra8595/Earthquake-Magnitude-Prediction.git
cd Earthquake-Magnitude-Prediction

