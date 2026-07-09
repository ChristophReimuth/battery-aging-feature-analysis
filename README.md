# Battery State of Health Prediction using Machine Learning

## Overview

This project investigates how machine learning can be used to predict the State of Health (SOH) of lithium-ion batteries from operational cycling data.

Instead of relying on direct capacity measurements, which require complete charge/discharge cycles, this project estimates SOH using electrical signals that are continuously available during battery operation, such as voltage, current, and time-derived features.

The project demonstrates an end-to-end data science workflow, including feature engineering, model development, hyperparameter optimization, and model interpretation.



## Project Goal

The objective of this project is to predict the State of Health (SOH) from operational battery data collected during charge/discharge cycles.

The target variable is

SOH = Remaining Capacity / Initial Capacity

where the battery capacity is calculated from the transferred charge (current integrated over time) during a cycle.

The models are trained using measurements such as

- Voltage
- Current
- Time
- Engineered statistical features

The ultimate goal is to estimate battery health without requiring direct capacity measurements.

## Features

- Battery SOH prediction using Random Forest regression
- Feature engineering from battery cycling data
- Evaluation of different feature groups
- Hyperparameter optimization with Grid Search
- Model evaluation using MAE and R²
- Feature importance analysis using Permutation Importance

## Dataset

This project uses the preprocessed **BatteryLife** dataset provided by the **BatteryML** project.

The following subsets are used:

- HUST
- Tongji (TJU)

The original datasets contain battery cycling measurements including

- Voltage
- Current
- Time

This project starts from the preprocessed data supplied by BatteryML. The focus is therefore on feature engineering, feature selection, machine learning, and model interpretation rather than raw data cleaning.

The engineered feature tables generated during preprocessing are stored in the `engineered_data/` directory.

## Requirements

- Python 3.13
- pip

## Installation

Clone the repository

```bash
git clone https://github.com/ChristophReimuth/battery-soh-prediction.git
cd battery-soh-prediction
```

Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```


## Usage

The project workflow is organized into three main notebooks:

1. **battery_soh_data_preprocessing.ipynb**
   - Load and preprocess the battery dataset
   - Perform data cleaning and preparation
   - Generate engineered features from battery measurement data
   - Export processed feature tables for model training

2. **baseline_model.ipynb**
   - Load the processed feature dataset
   - Train baseline regression models for SOH prediction
   - Evaluate model performance using metrics such as Mean Absolute Error (MAE) and $R^2$
   - Visualize predicted versus measured SOH values

3. **feature_importance.ipynb**
   - Train Random Forest regression models
   - Compare different feature groups and model configurations
   - Evaluate model performance
   - Analyze the contribution of individual features using Permutation Feature Importance

### Recommended workflow

Run the notebooks in the following order:

1. `battery_soh_data_preprocessing.ipynb`  
   → prepares the datasets and generates the required input features

2. `baseline_model.ipynb`  
   → establishes baseline model performance

3. `feature_importance.ipynb`  
   → performs advanced model comparison and feature analysis

## Project Workflow

```text
Battery Cycling Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Feature Group Definition
        │
        ▼
Random Forest Training
        │
        ▼
Grid Search Hyperparameter Optimization
        │
        ▼
Performance Evaluation (MAE, R²)
        │
        ▼
Permutation Feature Importance
        │
        ▼
Feature Interpretation
```


## Project Structure

```
.
├── engineered_data/
│   ├── processed_battery_features_HUST.pkl
│   └── processed_battery_features_Tongji.pkl
│
├── figures/
│   ├── feature_group_comparison_HUST.png
│   ├── permutation_report_vtime_full_HUST.png
│   └── permutation_top3_HUST.png
│
├── notebooks/
│   ├── battery_soh_data_preprocessing.ipynb
│   ├── baseline_model.ipynb
│   └── feature_importance.ipynb
│
└── src/
    └── battery_aging/
        ├── data_preprocessing.py
        ├── feature_engineering.py
        ├── model_functions.py
        └── __init__.py
```

## Machine Learning Pipeline


The machine learning pipeline consists of

- Feature engineering from battery cycling data
- Definition of multiple feature groups
- Random Forest regression
- Hyperparameter optimization using Grid Search
- Performance evaluation using MAE and R²
- Comparison of feature groups
- Interpretation using Permutation Feature Importance

The central research question is:

> Which engineered battery features provide the most accurate prediction of battery State of Health?

## Feature Groups

Several feature groups are evaluated to quantify the predictive value of different battery measurements.

The investigated groups include

- Voltage statistics
- Voltage curve similarity metrics (DTW, RMSE, Correlation, Area Difference, etc.)
- Current statistics
- Time-based features
- Combined feature sets

Each feature group is used to train an independently optimized Random Forest model.

## Results

Model performance is evaluated using

- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

The best-performing feature groups are further analyzed using Permutation Feature Importance to identify the most influential battery features.

The repository includes visualizations of

- Feature group performance comparison
- Permutation feature importance for the best-performing models

## References

The battery data used in this project originates from the BatteryLife dataset developed by the BatteryML project.

If you use this repository for research, please also cite the original dataset and the corresponding publications.