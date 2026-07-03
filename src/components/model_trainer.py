from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: Path = Path("artifacts/model.pkl")
