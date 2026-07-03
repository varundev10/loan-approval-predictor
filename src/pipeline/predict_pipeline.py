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
                "Missing deployment artifact: artifacts/model.pkl."
            )

        if not PREPROCESSOR_PATH.exists():
            raise CustomException(
                "Missing deployment artifact: artifacts/preprocessor.pkl."
            )

        model = load_object(MODEL_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)
        data_scaled = preprocessor.transform(features)
        preds = model.predict(data_scaled)
        return preds


@dataclass
class CustomData:
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

    def get_data_as_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "no_of_dependents": self.no_of_dependents,
                    "education": self.education,
                    "self_employed": self.self_employed,
                    "income_annum": self.income_annum,
                    "loan_amount": self.loan_amount,
                    "loan_term": self.loan_term,
                    "cibil_score": self.cibil_score,
                    "residential_assets_value": self.residential_assets_value,
                    "commercial_assets_value": self.commercial_assets_value,
                    "luxury_assets_value": self.luxury_assets_value,
                    "bank_asset_value": self.bank_asset_value,
                }
            ]
        )
