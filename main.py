from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Store ML model and threshold
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and threshold when application starts
    ml_model["model"] = joblib.load("credit_risk_model.pkl")
    ml_model["threshold"] = joblib.load("best_threshold.pkl")

    yield

    # Clear resources when application shuts down
    ml_model.clear()


app = FastAPI(lifespan=lifespan)


# Pydantic model for user input
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post("/predict")
def predict(data: LoanApplication):

    # Convert Pydantic input into DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Get probability from ML model
    # Convert NumPy float32 → Python float
    probability = float(
        ml_model["model"].predict_proba(input_df)[:, 1][0]
    )

    # Convert threshold to Python float
    threshold = float(ml_model["threshold"])

    # Make prediction
    prediction = int(probability >= threshold)

    # Return JSON-compatible response
    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }


# Serve frontend files
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)