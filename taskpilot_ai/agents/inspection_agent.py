from typing import Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np

class InspectionResult(BaseModel):
    missing_values: Dict[str, int]
    outliers: Dict[str, int]
    feature_types: Dict[str, str]
    class_imbalance: Optional[Dict[str, Any]] = None
    correlation: Optional[Dict[str, float]] = None
    shape: tuple
    basic_statistics: Optional[Dict[str, Dict[str, float]]] = None

class InspectionAgent:
    """
    Agent to inspect the dataset for structure, missing values, outliers, feature types, class imbalance, and correlation.
    """
    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None, verbose: bool = False) -> InspectionResult:
        if df.empty:
            raise ValueError("The provided DataFrame is empty.")

        # Missing values
        missing = df.isnull().sum().to_dict()
        if verbose:
            print("Missing values detected.")

        # Outliers detection using Z-score
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()
            if col_data.std() == 0:
                outliers[col] = 0
                continue
            z = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-8))
            outliers[col] = int((z > 3).sum())
        if verbose:
            print("Outliers detected.")

        # Feature types
        feature_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Class imbalance (if applicable)
        class_imbalance = None
        if target_column and target_column in df.columns:
            value_counts = df[target_column].value_counts(dropna=False).to_dict()
            total = sum(value_counts.values())
            class_imbalance = {
                str(k): {"count": int(v), "percent": float(v) / total * 100}
                for k, v in value_counts.items()
            }

        # Correlation (flattened)
        corr = None
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = df.corr(numeric_only=True)
            corr = {
                f"{i}__{j}": float(corr_matrix.loc[i, j])
                for i in corr_matrix.columns for j in corr_matrix.columns
                if i < j  # Only upper triangle
            }

        # Basic statistics
        basic_statistics = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats = df[col].describe()
            basic_statistics[col] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"])
            }

        # Shape
        shape = df.shape

        return InspectionResult(
            missing_values=missing,
            outliers=outliers,
            feature_types=feature_types,
            class_imbalance=class_imbalance,
            correlation=corr,
            shape=shape,
            basic_statistics=basic_statistics
        )
