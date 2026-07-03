from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path("Dataset/LoanApprovalKaggle/train_u6lujuX_CVtuZ9i.csv")
    test_data_path: Path = Path("Dataset/LoanApprovalKaggle/test_Y3wMUE5_7gLdaTN.csv")
