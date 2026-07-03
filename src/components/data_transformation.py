from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path("artifacts/preprocessor.pkl")
