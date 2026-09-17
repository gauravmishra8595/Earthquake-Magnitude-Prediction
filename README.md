# Earthquake Magnitude Prediction using Machine Learning

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)

## 📖 Table of Contents
- [Earthquake Magnitude Prediction using Machine Learning](#earthquake-magnitude-prediction-using-machine-learning)
  - [📖 Table of Contents](#-table-of-contents)
  - [🎯 Project Overview](#-project-overview)
  - [📈 Business Objective](#-business-objective)
  - [📊 Dataset](#-dataset)
  - [🛠 Methodology](#-methodology)
      - [1. Exploratory Data Analysis (EDA) \& Preprocessing](#1-exploratory-data-analysis-eda--preprocessing)
      - [2. Model Training \& Evaluation](#2-model-training--evaluation)
  - [📊 Results](#-results)
    - [Model Performance Comparison](#model-performance-comparison)
    - [Best Model: Gradient Boosting Regressor](#best-model-gradient-boosting-regressor)
    - [Feature Importance (Top Features)](#feature-importance-top-features)
    - [Interpretation](#interpretation)
  - [💡 Key Findings](#-key-findings)
  - [💻 Tools \& Libraries](#-tools--libraries)
  - [🚀 How to Use This Repository](#-how-to-use-this-repository)
  - [🔮 Future Work \& Potential Improvements](#-future-work--potential-improvements)

## 🎯 Project Overview
This project explores the application of various machine learning regression models to predict the magnitude of earthquakes based on a set of seismic and geographical features. The goal is to analyze the provided data, perform comprehensive feature engineering and selection, and build a robust predictive model. This notebook serves as a complete case study in data analysis, feature preprocessing, model selection, and performance evaluation.

## 📈 Business Objective
The primary objective is to develop a predictive model that can estimate an earthquake's magnitude given other recorded properties like depth, location, and time. While predicting the exact magnitude is a significant scientific challenge, a successful model can provide valuable insights into the factors most correlated with earthquake magnitude. Such a tool could be a foundational step in developing more sophisticated early warning or risk assessment systems.

## 📊 Dataset
The dataset used in this project is `earthquakes.csv`, which contains 8,394 records of earthquakes.

**Features Include:**
- **Target Variable:** `impact.magnitude` (The magnitude of the earthquake, which we aim to predict).
- **Numerical Features:** `location.depth`, `location.latitude`, `location.longitude`, `impact.gap`, `impact.significance`, `location.distance`, and components of the timestamp (`time.year`, `time.month`, `time.day`, `time.hour`, `time.minute`, `time.second`, `time.epoch`).
- **Categorical/Text Features:** `id`, `location.full`, `location.name`.

*Note: The features `impact.gap` and `impact.significance` were identified as data leakage and removed as they are derived from the magnitude itself. Other columns like `id` were removed as they provide no predictive value.*

## 🛠 Methodology

#### 1. Exploratory Data Analysis (EDA) & Preprocessing
The project starts with a rigorous data exploration and preparation phase to ensure data quality and model reliability.

- **Data Inspection:** Initial analysis of the dataset's shape, data types, and summary statistics.
- **Missing Values:** Confirmed that the dataset is clean with no missing values.
- **Feature Drops:** Key columns that could lead to data leakage (`impact.gap`, `impact.significance`) or were irrelevant (`id`, `time.full`, `location.full`, `time.epoch`) were dropped.
- **Outlier Handling:** Outliers were detected using the IQR method (visualized with boxplots) and a conservative removal strategy (IQR × 3) was applied, removing 17.9% of the data to reduce noise and improve model stability.
- **Correlation Analysis:** A correlation heatmap was used to identify multicollinearity and understand the linear relationships between features and the target variable. The analysis revealed that `location.depth` and `location.distance` have the highest positive correlation with magnitude.

#### 2. Model Training & Evaluation
A systematic approach was taken to find the most effective model for this regression task.

- **Data Splitting:** The data was split into training and testing sets to allow for robust model evaluation.
- **Preprocessing Pipeline:** A `ColumnTransformer` was used to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features, creating a reproducible preprocessing pipeline.
- **Model Selection:** Multiple regression algorithms were chosen for comparison, including:
    - **Baseline:** `DummyRegressor`
    - **Linear Models:** `LinearRegression`, `Ridge`, `Lasso`
    - **Ensemble Models:** `RandomForestRegressor`, `GradientBoostingRegressor`
    - **Other Models:** `SVR`, `MLPRegressor`
- **Performance Evaluation:** Models were evaluated using key regression metrics: **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R² Score**.

## 📊 Results

### Model Performance Comparison

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Dummy Regressor (Baseline) | ~0.00 | ~1.15 | ~0.92 |
| Linear Regression | ~0.15 | ~1.06 | ~0.85 |
| Ridge Regression | ~0.15 | ~1.06 | ~0.85 |
| Lasso Regression | ~0.14 | ~1.07 | ~0.86 |
| Random Forest Regressor | ~0.16 | ~1.05 | ~0.84 |
| **Gradient Boosting Regressor** | **~0.18** | **~1.04** | **~0.83** |
| SVR | ~0.12 | ~1.08 | ~0.87 |
| MLP Regressor | ~0.10 | ~1.10 | ~0.88 |

### Best Model: Gradient Boosting Regressor
- **R² Score:** 0.18
- **RMSE:** 1.04
- **MAE:** 0.83

### Feature Importance (Top Features)
1. `location.distance` - Highest importance
2. `location.depth` - Second highest importance
3. `location.latitude` - Moderate importance
4. `location.longitude` - Moderate importance
5. `time.hour` - Low importance

### Interpretation
- The **R² score of 0.18** indicates that only about 18% of the variance in earthquake magnitude can be explained by the provided features.
- The **baseline DummyRegressor** achieved an R² of ~0.00, confirming that the models are learning some signal from the data.
- The **GradientBoostingRegressor** outperformed all other models, demonstrating the effectiveness of ensemble methods for this type of problem.
- The low R² score suggests that earthquake magnitude prediction requires additional features beyond what is available in this dataset (e.g., tectonic plate data, historical seismic patterns, geological surveys).

## 💡 Key Findings
- The exploratory analysis revealed that the provided features have a low-to-moderate linear correlation with earthquake magnitude.
- Despite trying a variety of models, the predictive performance was modest. The **best-performing model was `GradientBoostingRegressor`**, which achieved an **R² score of approximately 0.18**.
- This indicates that while there is some predictive signal, the magnitude of an earthquake is not easily predictable from the given features alone.
- The most important features for the best model were `location.distance` and `location.depth`, confirming the findings from the correlation analysis.

## 💻 Tools & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Environment:** Jupyter Notebook

## 🚀 How to Use This Repository
To run this analysis locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your-username]/earthquake-magnitude-prediction.git
    ```
2.  **Navigate to the directory:**
    ```bash
    cd earthquake-magnitude-prediction
    ```
3.  **Install the necessary libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook Earthquake_Prediction.ipynb
    ```

## 🔮 Future Work & Potential Improvements
- **External Data:** The most significant improvement would come from integrating external datasets, such as tectonic plate boundary information, historical seismic activity in the region, and geological surveys.
- **Advanced Feature Engineering:** Create more sophisticated features from the existing data, such as the distance to the nearest fault line or the energy released in preceding events.
- **Time-Series Analysis:** Treat the problem as a time-series forecasting task to capture temporal dependencies between earthquakes.
- **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to perform a more exhaustive search for optimal model hyperparameters.
