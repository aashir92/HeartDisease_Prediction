# Heart Disease Risk Prediction

This project trains a logistic regression model to predict heart disease risk using a UCI-style heart disease dataset.

## Overview

The script `heart_disease_risk_model.py` performs:
- dataset loading with fallback filename handling
- missing value imputation
- exploratory data analysis
- categorical encoding and numeric scaling
- logistic regression training
- evaluation with accuracy, classification report, confusion matrix, and ROC curve
- saving visualizations as PNG files

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

Place the dataset in `task3` or `task3/archive`. The script searches for:

- `heart.csv`
- `heart_disease.csv`
- `heart_disease_uci.csv`
- `archive/heart_disease_uci.csv`

Run:

```powershell
python heart_disease_risk_model.py
```

## Output

The script generates:

- `correlation_heatmap.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

It also prints dataset shape, sample rows, accuracy, and a classification report to the console.

## Notes

- The dataset used in this folder is `archive/heart_disease_uci.csv`.
- If the dataset is not found, the script exits with an informative error message.
