# Financial Fraud Detection Using XGBoost

A reproducible machine-learning project for detecting fraudulent credit-card transactions in a highly imbalanced dataset. The final notebook trains a cost-sensitive XGBoost classifier and evaluates both classification performance and a fixed-budget fraud investigation workflow.

## Dataset

[Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions, including 492 labeled frauds
- 30 input columns: `Time`, `Amount`, and anonymized features `V1`–`V28`
- Target: `Class` (0 = genuine, 1 = fraud)
- No missing values in the complete source dataset

Download the dataset separately; `creditcard.csv` is not included in this repository.

## Method

1. Load the complete dataset and inspect its class distribution.
2. Sort by `Time`, then split sequentially: 70% training, 15% validation, 15% test.
3. Train an XGBoost binary classifier using a fraud-class weight equal to half the training genuine-to-fraud ratio.
4. Evaluate the held-out test partition using Average Precision, ROC-AUC, precision, recall, F1, and a confusion matrix at a 0.5 threshold.
5. Rank test transactions by predicted fraud probability and assess the top 100 highest-risk alerts.
6. Save the fitted model as `finauto_xgboost.pkl`.

The validation partition is reserved in the notebook but is not used to tune this simplified implementation. The final notebook uses the original input features, without SMOTE or additional engineered features.

## XGBoost configuration

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `min_child_weight` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | `0.5 * (genuine_train / fraud_train)` |
| `random_state` | 42 |
| `eval_metric` | `aucpr` |

## Results

Results recorded in `FinAuto_Final.ipynb` on its held-out chronological test partition (42,722 transactions, 52 frauds):

| Metric | Value |
|---|---:|
| Average Precision | 0.7692 |
| ROC-AUC | 0.9804 |
| Precision (threshold 0.5) | 0.8261 |
| Recall (threshold 0.5) | 0.7308 |
| F1 (threshold 0.5) | 0.7755 |

Confusion matrix (rows = actual genuine/fraud; columns = predicted genuine/fraud):

```text
[[42662,     8],
 [   14,    38]]
```

### Top-100 investigation budget

- Actual frauds in test data: **52**
- Frauds detected among top 100 alerts: **40**
- False alarms among top 100 alerts: **60**
- Recall@100: **0.7692**
- Precision@100: **0.4000**

The top-100 alert workflow and the 0.5 classification threshold are distinct operating policies; their precision and recall should not be conflated.

## Run in Google Colab

1. Download `creditcard.csv` from the Kaggle dataset link above.
2. Open `FinAuto_Final.ipynb` in Colab.
3. Install the packages listed in `requirements.txt` if needed.
4. Upload `creditcard.csv` into the notebook's working directory (or change the `pd.read_csv` path to its location).
5. Run the cells from top to bottom. The last cell saves `finauto_xgboost.pkl` to the Colab runtime.

Note: files saved to Colab's temporary runtime do not automatically persist to your computer or Google Drive.

## Limitations and responsible interpretation

- Fraud is rare, so Average Precision and alert-budget metrics are more informative than accuracy alone.
- The anonymized PCA features limit domain-level interpretation.
- The dataset is historical; performance may differ on future transactions or under distribution shift.
- The reported metrics reflect one chronological split, not repeated cross-validation or production deployment.
- The original research process also explored alternative algorithms and engineered features; this repository presents the final **simplified XGBoost implementation**, not a deployed hybrid anomaly-detection system.

## Files

- `FinAuto_Final.ipynb`: model training, evaluation, top-100 ranking, and saving
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes raw dataset and generated model artifacts

## Dataset attribution

Dataset: *Credit Card Fraud Detection*, provided by the Machine Learning Group, Université Libre de Bruxelles, via Kaggle. Please consult the dataset page for citation and usage information.
