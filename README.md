# Loan Approval Predictor

This project predicts whether a loan application will be approved or rejected using the Kaggle loan approval dataset by Archit Sharma. It includes data analysis notebooks, trained model artifacts, and a Flask web app ready for deployment on Render.

## Features

- Loan approval prediction from applicant and asset details
- Flask web interface with input guidance and dataset-based value ranges
- Saved preprocessing pipeline and trained model for deployment
- Jupyter notebooks for EDA and dataset comparison
- Render-ready project structure

## Dataset Used

Final deployed model uses:

- `Dataset/LoanApprovalSynthetic/loan_approval_dataset.csv`

This is the Kaggle dataset from:

- `architsharma01/loan-approval-prediction-dataset`

The deployed model uses these inputs:

- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`

Target:

- `loan_status`

## Model Results

Saved comparison file:

- `artifacts/synthetic_model_results.csv`

Best model:

- `Random Forest`

Latest evaluation:

- Accuracy: `0.9813`
- Precision: `0.9831`
- Recall: `0.9868`
- F1 Score: `0.9850`

## Project Structure

```text
loan-approval-predictor/
├── application.py
├── requirements.txt
├── render.yaml
├── Procfile
├── startup.txt
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── synthetic_model_results.csv
│   └── LoanApprovalSynthetic/
├── Dataset/
│   ├── LoanApprovalSynthetic/
│   └── LoanApprovalKaggle/
├── Notebook/
│   ├── loan_approval_synthetic_analysis.ipynb
│   ├── loan_approval_kaggle_analysis.ipynb
│   └── dataset_comparison_and_selection.ipynb
├── src/
├── static/
└── templates/
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python application.py
```

4. Open:

```text
http://127.0.0.1:5000
```

## Deployment

This repo is prepared for Render.

Important files:

- `render.yaml`
- `application.py`
- `requirements.txt`
- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`

### Render Steps

1. Push the repo to GitHub.
2. Sign in to Render.
3. Click `New +`.
4. Choose `Blueprint`.
5. Select this repository.
6. Deploy.

If Blueprint is not used, create a manual Python web service with:

- Build Command:

```bash
pip install -r requirements.txt
```

- Start Command:

```bash
gunicorn --bind 0.0.0.0:$PORT application:app
```

## Notes

- The notebooks include both the synthetic dataset analysis and the earlier Kaggle train/test style dataset analysis.
- The deployed app currently uses the Archit Sharma loan approval dataset schema.
- The form placeholders and help text are based on the observed dataset ranges.
