from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class CleanResult:
    def __init__(self, df: pd.DataFrame, actions: Dict[str, Any]):
        self.df = df
        self.actions = actions

class CleanerAgent:
    """
    Agent to clean and preprocess the dataset based on inspection results.
    Handles missing values, encodes categorical variables, normalizes/standardizes numeric features, and handles outliers.
    """
    def clean(self, df: pd.DataFrame, inspection_results: Optional[Dict[str, Any]] = None, strategy: str = 'mean', scale: bool = True, encode: bool = True, handle_outliers: bool = True, verbose: bool = False) -> CleanResult:
        actions = {}
        df_clean = df.copy()
        # 1. Handle missing values
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        if num_cols:
            imputer = SimpleImputer(strategy=strategy)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
            actions['missing_numeric'] = f"Imputed with {strategy}"
        if cat_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            df_clean[cat_cols] = imputer.fit_transform(df_clean[cat_cols])
            actions['missing_categorical'] = "Imputed with most_frequent"
        # 2. Encode categorical variables
        if encode and cat_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df_clean[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_clean.index)
            df_clean = pd.concat([df_clean.drop(columns=cat_cols), df_encoded], axis=1)
            actions['categorical_encoding'] = f"OneHot encoded: {cat_cols}"
        # 3. Normalize/standardize numeric features
        if scale and num_cols:
            scaler = StandardScaler()
            df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
            actions['scaling'] = f"StandardScaler applied to: {num_cols}"
        # 4. Handle outliers (clip to 3 std)
        if handle_outliers and num_cols:
            for col in num_cols:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                df_clean[col] = df_clean[col].clip(mean - 3*std, mean + 3*std)
            actions['outlier_handling'] = "Clipped to 3 std for numeric columns"
        if verbose:
            print("Cleaning actions:", actions)
        return CleanResult(df_clean, actions) 