from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.pkl"


class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        if not MODEL_PATH.exists():
            raise CustomException(
                "Missing deployment artifact: artifacts/model.pkl. Train and save the final Kaggle loan model first."
            )

        if not PREPROCESSOR_PATH.exists():
            raise CustomException(
                "Missing deployment artifact: artifacts/preprocessor.pkl. Save the preprocessing pipeline before deployment."
            )

        model = load_object(MODEL_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)
        data_scaled = preprocessor.transform(features)
        preds = model.predict(data_scaled)
        return preds


@dataclass
class CustomData:
    gender: str
    married: str
    dependents: str
    education: str
    self_employed: str
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float
    credit_history: float
    property_area: str

    def get_data_as_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Gender": self.gender,
                    "Married": self.married,
                    "Dependents": self.dependents,
                    "Education": self.education,
                    "Self_Employed": self.self_employed,
                    "ApplicantIncome": self.applicant_income,
                    "CoapplicantIncome": self.coapplicant_income,
                    "LoanAmount": self.loan_amount,
                    "Loan_Amount_Term": self.loan_amount_term,
                    "Credit_History": self.credit_history,
                    "Property_Area": self.property_area,
                }
            ]
        )
