import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Optional


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps outliers using IQR."""
    def fit(self, X, y=None):
        self.caps = {}
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=np.number).columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.caps[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col, (lower, upper) in self.caps.items():
                X[col] = X[col].clip(lower, upper)
        return X


class DataPreprocessingAgent:
    """
    Fully autonomous preprocessing agent.
    Handles: missing values, outliers, scaling, encoding, feature selection, target analysis.
    """

    def __init__(self, output_dir: str = "reports/preprocessing"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipeline = None
        self.original_columns = []
        self.target_column = None
        self.final_feature_names = []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.problem_type = None

    def infer_problem_type(self, y: pd.Series) -> str:
        """Infers whether the problem is regression or classification based on y."""
        if y.dtype == 'object' or y.nunique() <= 10:
            return "classification"
        elif pd.api.types.is_numeric_dtype(y):
            return "regression"
        return "classification"

    def preprocess(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
        self.original_columns = df.columns.tolist()
        self.target_column = target_column

        if target_column:
            y = df[target_column].reset_index(drop=True)
            df = df.drop(columns=[target_column])
            self.problem_type = self.infer_problem_type(y)
        else:
            y = None

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('outlier_capper', OutlierCapper()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', VarianceThreshold(threshold=0.0))
        ])

        processed_array = self.pipeline.fit_transform(df)

        # Get final feature names
        cat_feature_names = []
        if cat_cols:
            encoder = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()

        all_features = num_cols + cat_feature_names
        selector_mask = self.pipeline.named_steps['selector'].get_support()
        self.final_feature_names = [f for f, keep in zip(all_features, selector_mask) if keep]

        X_processed = pd.DataFrame(processed_array, columns=self.final_feature_names)

        if y is not None:
            if self.problem_type == 'classification':
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y.values
        else:
            y_processed = None

        # Save preprocessing report
        with open(os.path.join(self.output_dir, "summary.txt"), "w") as f:
            f.write("Original Columns:\n")
            f.write(str(self.original_columns) + "\n\n")
            f.write("Numerical Columns:\n")
            f.write(str(num_cols) + "\n\n")
            f.write("Categorical Columns:\n")
            f.write(str(cat_cols) + "\n\n")
            f.write(f"Problem Type: {self.problem_type}\n")
            f.write(f"Final Processed Shape: {X_processed.shape}\n")
            f.write(f"Final Feature Names: {self.final_feature_names}\n")

        return X_processed.values, y_processed, self.problem_type, self.final_feature_names

    def run(self, df: pd.DataFrame, target_column: str = None):
        """Wrapper for simple execution."""
        return self.preprocess(df, target_column)
