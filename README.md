# Battery SOH Prediction

A machine learning project for State of Health (SOH) prediction using public lithium-ion battery datasets.

## Features

- Data preprocessing
- Feature extraction
- Exploratory Data Analysis
- Machine Learning models
- Deep Learning models
- Model evaluation

## Datasets

- HUST
- Tongji (TJU)
- MIT / Severson

## Requirements

- Python 3.13
- pip

## Installation

```bash
git clone https://github.com/ChristophReimuth/battery-aging-feature-analysis.git
cd battery-aging-feature-analysis
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

## Project Structure

```
Battery-SOH/
│
├── data/
├── notebooks/
├── src/battery_aging
├── figures/
├── pyproject.toml
└── README.md
```

## Results


![Workflow](figures/permutation_report_vtime_full_HUST.png)

Example:

| Model | RMSE | MAE |
|-------|------|-----|
| XGBoost | 1.9 | 1.4 |
| LSTM | 1.7 | 1.3 |

## License

MIT License