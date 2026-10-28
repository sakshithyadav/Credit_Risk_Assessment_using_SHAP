# Credit Ledger — Loan Default Risk Assessment

A web application that estimates the probability that a loan applicant will default and returns a **High Risk / Low Risk** verdict. A calibrated XGBoost model is served through a FastAPI backend, with a single-page frontend for entering applications and viewing results.

## Problem Statement

Lenders need to judge how likely an applicant is to default before approving a loan. This project predicts the probability of default from applicant details, loan terms, and credit bureau history, then converts that probability into a risk decision using a decision threshold.

## Key Features

- Predicts default probability and a High Risk / Low Risk verdict for a loan application
- REST API (`POST /predict`) with request validation through Pydantic
- Responsive frontend with an animated probability gauge and verdict badge
- Loan-to-income ratio is calculated automatically in the form (and can be overridden)
- Frontend shows whether the API service is reachable
- Model and threshold are loaded once at application startup
- Notebook covers model comparison, hyperparameter tuning, probability calibration, and SHAP explanations

## ML Approach

The full workflow is in the notebook ("Credit_Risk__1_.ipynb").

1. **Data:** 32,581 loan records with 11 input features and a binary target (`loan_status`: 1 = default, 0 = repaid). Roughly 22% of records are defaults.
2. **Cleaning:** removed duplicates, ages outside 18–100, employment length above 60 years or greater than age, and loans with non-positive amounts.
3. **Split:** 80/20 stratified train/test split.
4. **Preprocessing:** `ColumnTransformer` with median imputation for numeric features (plus scaling for Logistic Regression) and constant imputation with one-hot encoding for categorical features.
5. **Class imbalance:** `class_weight="balanced"` for Logistic Regression and `scale_pos_weight` for XGBoost.
6. **Models:** Logistic Regression baseline compared against XGBoost using 5-fold stratified cross-validation.
7. **Tuning:** `RandomizedSearchCV` (150 candidates, 5-fold CV, scored on average precision).
8. **Calibration:** the tuned XGBoost pipeline is wrapped in sigmoid `CalibratedClassifierCV`; this calibrated model is the one saved and served.
9. **Threshold:** a decision threshold is stored in `best_threshold.pkl` and applied to the predicted probability.
10. **Explainability and error analysis:** SHAP (global summary and local waterfall plots) and a false positive / false negative review.

**Notebook results on the held-out test set (6,305 rows):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.82 | 0.56 | 0.79 | 0.65 |
| XGBoost (tuned, before calibration) | 0.92 | 0.85 | 0.80 | 0.82 |

## Technologies Used

- **Language:** Python 3.11.9
- **Backend:** FastAPI, Uvicorn, Pydantic
- **ML / Data:** scikit-learn, XGBoost, pandas, joblib
- **Notebook analysis:** NumPy, Matplotlib, Seaborn, SciPy, SHAP
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Deployment:** Render

## Project Structure


.
├── main.py                    # FastAPI app: /predict endpoint + static file serving
├── credit_risk_model.pkl      # Calibrated XGBoost pipeline
├── best_threshold.pkl         # Decision threshold
├── Credit_Risk__1_.ipynb      # EDA, training, tuning, calibration, SHAP
├── requirements.txt
├── runtime.txt                # Python version
├── render.yaml                # Render deployment config
└── static/                    # Frontend served at "/"
    ├── index.html
    ├── style.css
    └── script.js


The training dataset ('credit_risk_dataset.csv') is not included in this repository.

## Installation and Usage


# 1. Clone the repository and enter the folder
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app (run from the project root so the .pkl files are found)
uvicorn main:app --reload


Open `http://127.0.0.1:8000` for the web interface. FastAPI's interactive API docs are available at `http://127.0.0.1:8000/docs`.

## API

### `POST /predict`

**Request body (JSON):**

| Field | Type | Values used by the UI |
|---|---|---|
| person_age | integer | |
| person_income | float | |
| person_home_ownership | string | RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| person_emp_length | float | years |
| loan_intent | string | PERSONAL, EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| loan_grade | string | A to G |
| loan_amnt | float | |
| loan_int_rate | float | percent |
| loan_percent_income | float | loan amount / income |
| cb_person_default_on_file | string | `Y`, `N` |
| cb_person_cred_hist_length | integer | years |

**Response fields:** default_probability, default_prediction (1 if probability ≥ threshold), threshold`, `Result` (`"High Risk"` or `"Low Risk"`).

### Example


curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6
  }'


Example response (values rounded):

json
{
  "default_probability": 0.0226,
  "default_prediction": 0,
  "threshold": 0.9996,
  "Result": "Low Risk"
}

## Deployment

The project includes a 'render.yaml' for [Render](https://render.com): a Python web service named 'credit-ledger' on the free plan with auto-deploy enabled.

- **Build command:** 'pip install -r requirements.txt'
- **Start command:** 'uvicorn main:app --host 0.0.0.0 --port $PORT'

## Future Improvements

- Evaluate the calibrated model at the chosen decision threshold and document the results
- Show SHAP-based explanations for each prediction in the web interface
- Restrict categorical API inputs to the allowed values and add range validation
- Move data cleaning and training from the notebook into a reproducible script
- Align dependency versions with the training environment
- Add automated tests for the API

## Disclaimer

Predictions are model estimates and are not a lending decision.
