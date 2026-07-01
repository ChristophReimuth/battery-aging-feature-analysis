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

## Installation

```bash
git clone https://github.com/username/Battery-SOH.git
cd Battery-SOH
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
├── src/
├── models/
├── figures/
├── README.md
└── requirements.txt
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