# app/agents/eda_agent.py

import os
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

class EDAResult:
    def __init__(self, plots: List[str], summary: Dict[str, Any], llm_explanation: str = None):
        self.plots = plots
        self.summary = summary
        self.llm_explanation = llm_explanation

class EDAAgent:
    def __init__(self, output_dir: str = "reports/eda"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _clear_old_plots(self):
        """Delete all old plot files to prevent confusion and clutter"""
        try:
            if os.path.exists(self.output_dir):
                for file in os.listdir(self.output_dir):
                    if file.endswith(".png"):
                        file_path = os.path.join(self.output_dir, file)
                        os.remove(file_path)
                        print(f"Removed old plot: {file_path}")
        except Exception as e:
            print(f"Error clearing old plots: {e}")

    def _make_json_serializable(self, obj):
        """Convert numpy/pandas objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif pd.isna(obj) or obj is None:
            return None
        elif isinstance(obj, (str, bool)):
            return obj
        else:
            # For any other type, convert to string as fallback
            return str(obj)

    def analyze(self, df: pd.DataFrame, target_column: str = None, use_llm: bool = False) -> EDAResult:
        """
        Perform EDA analysis and generate plots
        """
        print(f"Starting EDA analysis. Output directory: {self.output_dir}")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Clear old plots
        self._clear_old_plots()

        plots = []
        summary = {}

        # Set matplotlib to use non-interactive backend to avoid display issues
        plt.switch_backend('Agg')
        
        # Configure plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # Generate plots for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        print(f"Found {len(numeric_columns)} numeric columns: {list(numeric_columns)}")

        for col in numeric_columns:
            try:
                # Skip columns with all NaN values
                if df[col].isna().all():
                    print(f"Skipping column {col} - all values are NaN")
                    continue

                # Histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].dropna().hist(bins=30, alpha=0.7, ax=ax)
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                
                hist_filename = f"hist_{col}_{uuid.uuid4().hex[:8]}.png"
                hist_path = os.path.join(self.output_dir, hist_filename)
                plt.savefig(hist_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Return relative path for download endpoint
                plots.append(f"eda/{hist_filename}")
                print(f"Generated histogram: {hist_path}")

                # Boxplot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x=col, ax=ax)
                ax.set_title(f"Boxplot of {col}")
                
                box_filename = f"box_{col}_{uuid.uuid4().hex[:8]}.png"
                box_path = os.path.join(self.output_dir, box_filename)
                plt.savefig(box_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                plots.append(f"eda/{box_filename}")
                print(f"Generated boxplot: {box_path}")

            except Exception as e:
                print(f"Error generating plots for column {col}: {e}")
                continue

        # Correlation Heatmap
        try:
            if len(numeric_columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                corr = df[numeric_columns].corr()
                
                # Create mask for upper triangle to make heatmap cleaner
                mask = np.triu(np.ones_like(corr, dtype=bool))
                
                sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                           center=0, square=True, fmt='.2f', ax=ax)
                ax.set_title("Correlation Heatmap")
                
                heatmap_filename = f"correlation_heatmap_{uuid.uuid4().hex[:8]}.png"
                heatmap_path = os.path.join(self.output_dir, heatmap_filename)
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                plots.append(f"eda/{heatmap_filename}")
                # Convert correlation matrix to JSON-serializable format
                summary['correlation'] = self._make_json_serializable(corr.to_dict())
                print(f"Generated correlation heatmap: {heatmap_path}")
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")

        # Target distribution (for both numeric and categorical targets)
        if target_column and target_column in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 20:
                    # Categorical or discrete target
                    df[target_column].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f"Distribution of {target_column}")
                    ax.set_xlabel(target_column)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                else:
                    # Continuous target
                    df[target_column].dropna().hist(bins=30, alpha=0.7, ax=ax)
                    ax.set_title(f"Distribution of {target_column}")
                    ax.set_xlabel(target_column)
                    ax.set_ylabel("Frequency")
                
                dist_filename = f"target_dist_{target_column}_{uuid.uuid4().hex[:8]}.png"
                dist_path = os.path.join(self.output_dir, dist_filename)
                plt.savefig(dist_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                plots.append(f"eda/{dist_filename}")
                # Convert target distribution to JSON-serializable format
                summary['target_distribution'] = self._make_json_serializable(
                    df[target_column].value_counts().to_dict()
                )
                print(f"Generated target distribution: {dist_path}")
                
            except Exception as e:
                print(f"Error generating target distribution plot: {e}")

        # Generate basic statistics summary (JSON-serializable)
        try:
            # Convert describe() output to JSON-serializable format
            describe_dict = {}
            describe_df = df.describe(include='all')
            for col in describe_df.columns:
                describe_dict[col] = {}
                for stat, value in describe_df[col].items():
                    describe_dict[col][stat] = self._make_json_serializable(value)
            
            summary['describe'] = describe_dict
            summary['shape'] = list(df.shape)  # Convert tuple to list
            
            # Convert dtypes to JSON-serializable format
            dtypes_dict = {}
            for col, dtype in df.dtypes.items():
                dtypes_dict[col] = str(dtype)
            summary['dtypes'] = dtypes_dict
            
            # Convert missing values to JSON-serializable format
            missing_dict = {}
            for col, missing_count in df.isnull().sum().items():
                missing_dict[col] = int(missing_count)
            summary['missing_values'] = missing_dict
            
            summary['total_plots_generated'] = len(plots)
            
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
            # Provide minimal summary if there's an error
            summary['describe'] = {}
            summary['shape'] = list(df.shape)
            summary['dtypes'] = {}
            summary['missing_values'] = {}
            summary['total_plots_generated'] = len(plots)

        # LLM explanation (stub for now)
        llm_explanation = "(LLM explanation would go here.)" if use_llm else None

        print(f"EDA analysis complete. Generated {len(plots)} plots.")
        return EDAResult(plots, summary, llm_explanation)