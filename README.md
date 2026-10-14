# Financial Fraud Detection Using XGBoost

## Project Overview

This project uses machine learning to detect fraudulent credit card transactions. Since fraud cases are rare compared to genuine transactions, a cost-sensitive XGBoost model is used to handle class imbalance.

## Dataset

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains 284,807 transactions, including 492 fraudulent transactions.

## Methodology

- Performed basic data exploration and checked for missing values.
- Split the dataset chronologically into training, validation, and testing sets.
- Trained an XGBoost classifier with class weighting to handle imbalanced data.
- Evaluated the model using classification metrics and top-100 fraud detection.

## Model Performance

| Metric | Score |
|---|---|
| Average Precision | 0.7692 |
| ROC-AUC | 0.9804 |
| Precision | 82.61% |
| Recall | 73.08% |
| F1-score | 77.55% |

**Top-100 Analysis:** The model identified 40 out of 52 fraudulent transactions among the 100 highest-risk predictions.

## How to Run

1. Open `FinAuto_Final.ipynb` in Google Colab.
2. Install the required libraries from `requirements.txt`.
3. Download the dataset from Kaggle and load it into the notebook.
4. Run all cells to train, evaluate, and save the model.

## Tools Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, Google Colab.
