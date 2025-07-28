from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    f1_score
)
import pandas as pd
import numpy as np
import os


class ModelResult:
    def __init__(self, model, metrics: Dict[str, Any], model_path: Optional[str] = None):
        self.model = model
        self.metrics = metrics
        self.model_path = model_path


class ModelTrainerAgent:
    """
    Automatically detects task type (classification or regression), selects, trains, and evaluates the best model.
    """

    def detect_task_type(self, y: pd.Series) -> str:
        """Infer if task is regression or classification based on target."""
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.nunique()
            if unique_vals <= 10:
                return "classification"
            return "regression"
        else:
            return "classification"

    def train(self, df: pd.DataFrame, task_type: str, target_column: str, model_dir: str = "reports/models") -> ModelResult:
        # Get target variable
        y = df[target_column]
        
        # Convert task_type to lowercase and validate
        if task_type is None or task_type.strip() == "":
            # If no task_type provided, infer it
            task_type = self.detect_task_type(y)
            print(f"No task type provided, inferred: {task_type}")
        else:
            task_type = task_type.lower().strip()
        
        # Validate task_type
        if task_type not in ["regression", "classification"]:
            raise ValueError(f"Invalid task_type: '{task_type}'. Must be 'regression' or 'classification'")

        print(f"Training model for task type: {task_type}")

        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        all_metrics = {}
        best_model = None
        best_score = float('-inf')

        if task_type == "regression":
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mse)

                all_metrics[name] = {
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2
                }

                if r2 > best_score:
                    best_score = r2
                    best_model = model

        elif task_type == "classification":
            # Convert to categorical if not already
            if pd.api.types.is_numeric_dtype(y):
                y = y.astype(int)
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForestClassifier": RandomForestClassifier(random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')

                all_metrics[name] = {
                    "accuracy": acc,
                    "f1_score": f1
                }

                if f1 > best_score:
                    best_score = f1
                    best_model = model

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        return ModelResult(
            model=best_model,
            metrics={
                "model": best_model.__class__.__name__,
                "all_model_scores": all_metrics,
                "best_score": best_score,
                "task_type_used": task_type
            }
        )