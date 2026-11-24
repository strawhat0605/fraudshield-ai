"""Dataset loading and validation utilities."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET_COLUMN = "Class"


def load_transactions(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and return numeric features and a validated binary target."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate demo data with "
            "`python scripts/generate_demo_data.py` or provide a CSV."
        )

    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"CSV must contain a '{TARGET_COLUMN}' target column.")
    if frame.empty:
        raise ValueError("Dataset is empty.")

    y = frame[TARGET_COLUMN]
    if y.isna().any():
        raise ValueError("Target column contains missing values.")
    observed = set(pd.Series(y).unique().tolist())
    if not observed.issubset({0, 1, 0.0, 1.0}) or len(observed) != 2:
        raise ValueError("Target must contain both binary classes: 0 and 1.")

    X = frame.drop(columns=[TARGET_COLUMN])
    X = X.select_dtypes(include="number")
    if X.empty:
        raise ValueError("No numeric feature columns were found.")
    if X.isna().all().any():
        empty_cols = X.columns[X.isna().all()].tolist()
        raise ValueError(f"Feature columns contain only missing values: {empty_cols}")

    return X, y.astype(int)


def validate_feature_frame(features: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Align input features to the model's expected column order."""
    missing = sorted(set(expected_columns) - set(features.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    aligned = features.loc[:, expected_columns].copy()
    for column in aligned.columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    return aligned
