from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "Dataset" / "LoanApprovalSynthetic" / "loan_approval_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
SYNTHETIC_ARTIFACTS_DIR = ARTIFACTS_DIR / "LoanApprovalSynthetic"

ARTIFACTS_DIR.mkdir(exist_ok=True)
SYNTHETIC_ARTIFACTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

for column in df.select_dtypes(include=["object", "string"]).columns:
    df[column] = df[column].str.strip()

df = df.drop_duplicates().drop(columns=["loan_id"]).copy()

categorical_features = ["education", "self_employed"]
numerical_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

X = df[categorical_features + numerical_features]
y = df["loan_status"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_split=10,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_split=4,
        min_samples_leaf=2,
    ),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_test_processed)
    fitted_models[name] = model
    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, pos_label="Approved"),
            "Recall": recall_score(y_test, predictions, pos_label="Approved"),
            "F1 Score": f1_score(y_test, predictions, pos_label="Approved"),
        }
    )

results_df = pd.DataFrame(results).sort_values(
    by=["F1 Score", "Accuracy"],
    ascending=False,
).reset_index(drop=True)

best_model_name = results_df.loc[0, "Model"]
best_model = fitted_models[best_model_name]

results_path = ARTIFACTS_DIR / "synthetic_model_results.csv"
preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"
model_path = ARTIFACTS_DIR / "model.pkl"
synthetic_model_path = SYNTHETIC_ARTIFACTS_DIR / "best_loan_model.pkl"

results_df.to_csv(results_path, index=False)
joblib.dump(preprocessor, preprocessor_path)
joblib.dump(best_model, model_path)
joblib.dump(best_model, synthetic_model_path)

print(results_df.to_string(index=False))
print(f"BEST_MODEL={best_model_name}")
print(f"DATASET={DATASET_PATH}")
print(f"PREPROCESSOR={preprocessor_path}")
print(f"MODEL={model_path}")
