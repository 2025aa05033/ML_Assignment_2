Untitled (1).md
# Machine Learning Assignment 2 - Classification and Streamlit Deployment

## a) Problem Statement
Build an end-to-end machine learning classification project on one public dataset and implement 6 required models on the same train/test split:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Then create an interactive Streamlit app with CSV upload, model selection, metric display, and confusion matrix / classification report.

## b) Dataset Description
- **Dataset:** UCI Adult Income dataset (fetched via OpenML: `adult`, version `2`)
- **Task type:** Binary classification (`<=50K` vs `>50K`)
- **Instances:** 48,842
- **Features:** 14 total (mixed numeric + categorical), satisfying the assignment minimum of 12 features
- **Target column:** `class` (mapped to `0` for `<=50K`, `1` for `>50K`)
- **Split strategy:** Stratified train/test split (`80/20`, `random_state=42`)

## c) Models Used and Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8524 | 0.9042 | 0.7414 | 0.5885 | 0.6562 | 0.5699 |
| Decision Tree | 0.8169 | 0.7490 | 0.6171 | 0.6189 | 0.6180 | 0.4976 |
| kNN | 0.8384 | 0.8699 | 0.6827 | 0.6065 | 0.6424 | 0.5400 |
| Naive Bayes (Gaussian) | 0.6204 | 0.8287 | 0.3794 | 0.9213 | 0.5374 | 0.3866 |
| Random Forest (Ensemble) | 0.8644 | 0.9163 | 0.7953 | 0.5834 | 0.6731 | 0.6013 |
| XGBoost (Ensemble) | 0.8726 | 0.9272 | 0.7943 | 0.6309 | 0.7032 | 0.6301 |

### Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with balanced performance; lower recall than ensembles but stable and interpretable. |
| Decision Tree | Captures nonlinearity but overfits relative to ensembles, giving lower AUC and MCC. |
| kNN | Better than single tree on this dataset, but performance is sensitive to distance behavior in high-dimensional encoded space. |
| Naive Bayes (Gaussian) | Very high recall but poor precision, producing many false positives and weaker overall balance. |
| Random Forest (Ensemble) | Improves robustness and ranking quality over single tree with better F1 and MCC. |
| XGBoost (Ensemble) | Best overall model across all key metrics, especially AUC, F1, and MCC. |

## Repository Structure

```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- train_models.py
│   └-- saved_models/
│       │-- logistic_regression.joblib
│       │-- decision_tree.joblib
│       │-- knn.joblib
│       │-- naive_bayes.joblib
│       │-- random_forest.joblib
│       └-- xgboost.joblib
│-- artifacts/
│   │-- metrics.csv
│   │-- reports.json
│   └-- schema.json
└-- data/
    │-- holdout_test_with_labels.csv
    └-- sample_upload_test_data.csv
```

## How To Run Locally

1. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train models and generate artifacts:
   ```bash
   python model/train_models.py
   ```
4. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment Size Notes
- Model artifacts are saved only in `model/saved_models/` to avoid duplication.
- Models are serialized with compression for easier GitHub upload and faster Streamlit deployment.
- Current compressed model directory size is approximately **6 MB**.
- If you retrain with different hyperparameters, run `python model/train_models.py` again before deploy.

## Streamlit App Features Implemented
- CSV upload option for test data
- Model selection dropdown
- Display of required evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report display
- Fallback evaluation using saved holdout set when CSV is not uploaded

## Screenshots

### BITS Virtual Lab Execution - Code
![BITS Virtual Lab Execution Code](./screenshots/code.png)

### Local Streamlit App
![BITS Virtual Lab Local Streamlit](./screenshots/local_streamlit.png)

### Hosted Streamlit App
![BITS Virtual Lab Hosted Streamlit](./screenshots/hosted_streamlit.png)

## Deployment Steps (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Go to `https://streamlit.io/cloud`.
3. Sign in with GitHub and click **New App**.
4. Select repository, branch (`main`), and file (`app.py`).
5. Click **Deploy** and copy the live app link.

## Final Submission Checklist
- [ ] GitHub repo link works
- [ ] Streamlit app link opens correctly
- [ ] App loads without errors
- [ ] All required features are implemented
- [ ] README content included in submitted PDF
- [ ] BITS Virtual Lab execution screenshot captured and attached