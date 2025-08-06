import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import json
import joblib
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold, TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    model_name: str
    model_object: Any
    train_score: float
    validation_score: float
    test_score: float
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_interpretation: Dict[str, Any]

@dataclass
class ModelingResult:
    best_model: ModelResult
    all_models: List[ModelResult]
    ensemble_model: Optional[ModelResult]
    model_comparison: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    model_recommendations: str
    deployment_guide: str
    model_files: List[str]

class ModelMaestroAgent:
    """
    🎭 Model Maestro Agent - The Virtuoso of Machine Learning
    
    This agent performs sophisticated model development and optimization:
    1. Intelligent model selection based on data characteristics
    2. Advanced hyperparameter optimization using Optuna
    3. Ensemble learning and model stacking
    4. AutoML capabilities with multiple algorithms
    5. Model interpretation and explainability
    6. Cross-validation and robust evaluation
    7. Production-ready model deployment preparation
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "reports/models"):
        self.gemini_api_key = gemini_api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
        self.modeling_history = []
        self.best_models = {}
        
    def orchestrate_modeling(self, df: pd.DataFrame, target_column: str,
                           task_type: str = "classification",
                           time_budget: int = 300,  # 5 minutes default
                           optimization_metric: str = None,
                           cross_validation_folds: int = 5) -> ModelingResult:
        """
        🎭 Orchestrate comprehensive machine learning modeling
        """
        logger.info("🎭 Model Maestro: Beginning advanced modeling orchestration...")
        
        # Step 1: Prepare data and setup
        X, y, data_info = self._prepare_modeling_data(df, target_column, task_type)
        
        # Step 2: Choose optimization metric
        if optimization_metric is None:
            optimization_metric = self._select_optimization_metric(task_type, y)
        
        # Step 3: Smart model selection based on data characteristics
        candidate_models = self._select_candidate_models(data_info, task_type, time_budget)
        
        # Step 4: Automated hyperparameter optimization
        optimized_models = self._optimize_models_with_optuna(
            X, y, candidate_models, task_type, optimization_metric, 
            time_budget, cross_validation_folds
        )
        
        # Step 5: Ensemble learning
        ensemble_model = self._create_ensemble_model(
            optimized_models, X, y, task_type, optimization_metric
        )
        
        # Step 6: Model interpretation and explainability
        interpreted_models = self._interpret_models(optimized_models, X, y)
        
        # Step 7: Final evaluation and comparison
        model_comparison = self._comprehensive_model_evaluation(
            interpreted_models, ensemble_model, X, y, task_type
        )
        
        # Step 8: Generate AI-powered recommendations
        recommendations = self._generate_model_recommendations(
            interpreted_models, ensemble_model, data_info, task_type
        )
        
        # Step 9: Create deployment guide
        deployment_guide = self._create_deployment_guide(
            interpreted_models[0] if interpreted_models else ensemble_model,
            data_info, task_type
        )
        
        # Step 10: Save models and artifacts
        model_files = self._save_modeling_artifacts(
            interpreted_models, ensemble_model, X.columns.tolist()
        )
        
        result = ModelingResult(
            best_model=interpreted_models[0] if interpreted_models else ensemble_model,
            all_models=interpreted_models,
            ensemble_model=ensemble_model,
            model_comparison=model_comparison,
            optimization_history=self.modeling_history,
            model_recommendations=recommendations,
            deployment_guide=deployment_guide,
            model_files=model_files
        )
        
        logger.info("✅ Model Maestro: Advanced modeling orchestration complete!")
        return result
    
    def _prepare_modeling_data(self, df: pd.DataFrame, target_column: str, 
                              task_type: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Prepare and analyze data for modeling"""
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Handle target variable for classification
        target_encoder = None
        if task_type == "classification" and y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y), index=y.index)
        
        # Analyze data characteristics
        data_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_numeric_features': len(numeric_cols),
            'n_categorical_features': len(categorical_cols),
            'class_balance': y.value_counts().to_dict() if task_type == "classification" else None,
            'feature_types': {col: 'numeric' if col in numeric_cols else 'categorical' for col in X.columns},
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'missing_data_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'data_complexity': self._assess_data_complexity(X, y, task_type)
        }
        
        return X, y, data_info
    
    def _select_optimization_metric(self, task_type: str, y: pd.Series) -> str:
        """Select appropriate optimization metric"""
        if task_type == "classification":
            n_classes = y.nunique()
            if n_classes == 2:
                return "roc_auc"
            else:
                return "f1_macro"
        else:  # regression
            return "neg_mean_squared_error"
    
    def _select_candidate_models(self, data_info: Dict[str, Any], task_type: str, 
                                time_budget: int) -> List[Dict[str, Any]]:
        """Intelligently select candidate models based on data characteristics"""
        
        n_samples = data_info['n_samples']
        n_features = data_info['n_features']
        complexity = data_info['data_complexity']
        
        candidates = []
        
        if task_type == "classification":
            # Always include these robust models
            candidates.extend([
                {
                    'name': 'RandomForest',
                    'model_class': RandomForestClassifier,
                    'search_space': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'priority': 'high'
                },
                {
                    'name': 'XGBoost',
                    'model_class': xgb.XGBClassifier,
                    'search_space': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 10],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    },
                    'priority': 'high'
                },
                {
                    'name': 'LightGBM',
                    'model_class': lgb.LGBMClassifier,
                    'search_space': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 10],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 100]
                    },
                    'priority': 'high'
                }
            ])
            
            # Add models based on data size and complexity
            if n_samples < 10000:  # Smaller datasets
                candidates.append({
                    'name': 'SVM',
                    'model_class': SVC,
                    'search_space': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    },
                    'priority': 'medium'
                })
                
            if n_features < 50:  # Lower dimensional data
                candidates.append({
                    'name': 'LogisticRegression',
                    'model_class': LogisticRegression,
                    'search_space': {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'lbfgs']
                    },
                    'priority': 'medium'
                })
                
            if time_budget > 600:  # More time available
                candidates.append({
                    'name': 'CatBoost',
                    'model_class': CatBoostClassifier,
                    'search_space': {
                        'iterations': [100, 200],
                        'depth': [4, 6, 8],
                        'learning_rate': [0.01, 0.1],
                        'l2_leaf_reg': [1, 3, 5]
                    },
                    'priority': 'medium'
                })
        
        else:  # regression
            candidates.extend([
                {
                    'name': 'RandomForest',
                    'model_class': RandomForestRegressor,
                    'search_space': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'priority': 'high'
                },
                {
                    'name': 'XGBoost',
                    'model_class': xgb.XGBRegressor,
                    'search_space': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 10],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    },
                    'priority': 'high'
                },
                {
                    'name': 'LightGBM',
                    'model_class': lgb.LGBMRegressor,
                    'search_space': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 10],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 100]
                    },
                    'priority': 'high'
                }
            ])
            
            # Add regression-specific models
            if n_features < 100:
                candidates.extend([
                    {
                        'name': 'Ridge',
                        'model_class': Ridge,
                        'search_space': {
                            'alpha': [0.1, 1, 10, 100]
                        },
                        'priority': 'medium'
                    },
                    {
                        'name': 'Lasso',
                        'model_class': Lasso,
                        'search_space': {
                            'alpha': [0.01, 0.1, 1, 10]
                        },
                        'priority': 'medium'
                    }
                ])
        
        # Sort by priority and limit based on time budget
        candidates.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
        
        # Limit number of models based on time budget
        if time_budget < 300:  # 5 minutes
            return candidates[:3]
        elif time_budget < 600:  # 10 minutes
            return candidates[:5]
        else:
            return candidates
    
    def _optimize_models_with_optuna(self, X: pd.DataFrame, y: pd.Series,
                                   candidate_models: List[Dict[str, Any]],
                                   task_type: str, optimization_metric: str,
                                   time_budget: int, cv_folds: int) -> List[ModelResult]:
        """Optimize models using Optuna for hyperparameter tuning"""
        
        optimized_models = []
        time_per_model = time_budget // len(candidate_models)
        
        # Setup cross-validation
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if task_type == "classification" else None
        )
        
        for model_config in candidate_models:
            logger.info(f"Optimizing {model_config['name']}...")
            
            try:
                # Create Optuna study
                study = optuna.create_study(
                    direction='maximize' if 'roc_auc' in optimization_metric or 'f1' in optimization_metric or 'r2' in optimization_metric else 'minimize',
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
                
                # Define objective function
                def objective(trial):
                    # Sample hyperparameters
                    params = {}
                    for param, values in model_config['search_space'].items():
                        if isinstance(values[0], int):
                            params[param] = trial.suggest_int(param, min(values), max(values))
                        elif isinstance(values[0], float):
                            params[param] = trial.suggest_float(param, min(values), max(values))
                        else:
                            params[param] = trial.suggest_categorical(param, values)
                    
                    # Handle special parameters
                    if model_config['name'] in ['XGBoost', 'LightGBM', 'CatBoost']:
                        params['random_state'] = 42
                        if model_config['name'] == 'CatBoost':
                            params['verbose'] = False
                    
                    # Create and evaluate model
                    model = model_config['model_class'](**params)
                    
                    try:
                        scores = cross_val_score(model, X_train, y_train, 
                                               cv=cv, scoring=optimization_metric, n_jobs=-1)
                        return np.mean(scores)
                    except Exception as e:
                        logger.warning(f"Trial failed for {model_config['name']}: {e}")
                        return float('-inf') if 'roc_auc' in optimization_metric or 'f1' in optimization_metric or 'r2' in optimization_metric else float('inf')
                
                # Optimize
                study.optimize(objective, timeout=time_per_model, n_jobs=1)
                
                # Train best model
                best_params = study.best_params
                if model_config['name'] in ['XGBoost', 'LightGBM', 'CatBoost']:
                    best_params['random_state'] = 42
                    if model_config['name'] == 'CatBoost':
                        best_params['verbose'] = False
                
                best_model = model_config['model_class'](**best_params)
                best_model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = self._calculate_score(best_model, X_train, y_train, optimization_metric)
                val_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=optimization_metric)
                val_score = np.mean(val_scores)
                test_score = self._calculate_score(best_model, X_test, y_test, optimization_metric)
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(
                    best_model, X_test, y_test, task_type
                )
                
                # Feature importance
                feature_importance = self._extract_feature_importance(best_model, X.columns)
                
                model_result = ModelResult(
                    model_name=model_config['name'],
                    model_object=best_model,
                    train_score=train_score,
                    validation_score=val_score,
                    test_score=test_score,
                    metrics=metrics,
                    hyperparameters=best_params,
                    feature_importance=feature_importance,
                    model_interpretation={}
                )
                
                optimized_models.append(model_result)
                
                # Log optimization history
                self.modeling_history.append({
                    'model_name': model_config['name'],
                    'best_score': study.best_value,
                    'best_params': best_params,
                    'n_trials': len(study.trials)
                })
                
                logger.info(f"✅ {model_config['name']} optimized - Score: {val_score:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Optimization failed for {model_config['name']}: {e}")
                continue
        
        # Sort by validation score
        optimized_models.sort(key=lambda x: x.validation_score, reverse=True)
        return optimized_models
    
    def _create_ensemble_model(self, models: List[ModelResult], X: pd.DataFrame, 
                              y: pd.Series, task_type: str, optimization_metric: str) -> Optional[ModelResult]:
        """Create ensemble model from top performing models"""
        
        if len(models) < 2:
            return None
        
        try:
            # Take top 3 models for ensemble
            top_models = models[:min(3, len(models))]
            
            # Create ensemble
            if task_type == "classification":
                ensemble = VotingClassifier(
                    estimators=[(f"model_{i}", model.model_object) for i, model in enumerate(top_models)],
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=[(f"model_{i}", model.model_object) for i, model in enumerate(top_models)]
                )
            
            # Split data for ensemble evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None
            )
            
            # Fit ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            train_score = self._calculate_score(ensemble, X_train, y_train, optimization_metric)
            test_score = self._calculate_score(ensemble, X_test, y_test, optimization_metric)
            
            # Cross-validation score
            if task_type == "classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
            val_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring=optimization_metric)
            val_score = np.mean(val_scores)
            
            # Comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(ensemble, X_test, y_test, task_type)
            
            ensemble_result = ModelResult(
                model_name="Ensemble",
                model_object=ensemble,
                train_score=train_score,
                validation_score=val_score,
                test_score=test_score,
                metrics=metrics,
                hyperparameters={'ensemble_type': 'voting', 'base_models': [m.model_name for m in top_models]},
                feature_importance={},
                model_interpretation={'ensemble_composition': [m.model_name for m in top_models]}
            )
            
            logger.info(f"✅ Ensemble model created - Score: {val_score:.4f}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"❌ Ensemble creation failed: {e}")
            return None
    
    def _interpret_models(self, models: List[ModelResult], X: pd.DataFrame, 
                         y: pd.Series) -> List[ModelResult]:
        """Add model interpretation and explainability"""
        
        interpreted_models = []
        
        for model_result in models:
            try:
                interpretation = {}
                
                # Model complexity analysis
                interpretation['model_complexity'] = self._assess_model_complexity(model_result.model_object)
                
                # Feature importance analysis
                if hasattr(model_result.model_object, 'feature_importances_'):
                    interpretation['feature_importance_analysis'] = self._analyze_feature_importance(
                        model_result.feature_importance, X.columns
                    )
                
                # Model stability analysis
                interpretation['stability_analysis'] = self._analyze_model_stability(
                    model_result.model_object, X, y
                )
                
                # Performance analysis
                interpretation['performance_analysis'] = {
                    'overfitting_score': model_result.train_score - model_result.test_score,
                    'generalization_gap': model_result.validation_score - model_result.test_score,
                    'is_overfitting': (model_result.train_score - model_result.test_score) > 0.1,
                    'is_underfitting': model_result.train_score < 0.7
                }
                
                # Update model result with interpretation
                model_result.model_interpretation = interpretation
                interpreted_models.append(model_result)
                
            except Exception as e:
                logger.warning(f"Model interpretation failed for {model_result.model_name}: {e}")
                interpreted_models.append(model_result)
        
        return interpreted_models
    
    def _comprehensive_model_evaluation(self, models: List[ModelResult], 
                                       ensemble_model: Optional[ModelResult],
                                       X: pd.DataFrame, y: pd.Series, 
                                       task_type: str) -> Dict[str, Any]:
        """Comprehensive evaluation and comparison of all models"""
        
        comparison = {
            'model_rankings': [],
            'performance_summary': {},
            'best_model_recommendation': '',
            'performance_matrix': {},
            'stability_comparison': {},
            'complexity_comparison': {}
        }
        
        all_models = models.copy()
        if ensemble_model:
            all_models.append(ensemble_model)
        
        # Create performance matrix
        metrics_matrix = {}
        for model in all_models:
            metrics_matrix[model.model_name] = {
                'validation_score': model.validation_score,
                'test_score': model.test_score,
                'train_score': model.train_score,
                **model.metrics
            }
        
        comparison['performance_matrix'] = metrics_matrix
        
        # Rank models
        model_rankings = sorted(all_models, key=lambda x: x.validation_score, reverse=True)
        comparison['model_rankings'] = [
            {
                'rank': i + 1,
                'model_name': model.model_name,
                'validation_score': model.validation_score,
                'test_score': model.test_score
            }
            for i, model in enumerate(model_rankings)
        ]
        
        # Best model recommendation
        if model_rankings:
            best_model = model_rankings[0]
            comparison['best_model_recommendation'] = f"{best_model.model_name} with validation score: {best_model.validation_score:.4f}"
        
        # Performance summary
        comparison['performance_summary'] = {
            'best_validation_score': max(model.validation_score for model in all_models),
            'best_test_score': max(model.test_score for model in all_models),
            'score_std': np.std([model.validation_score for model in all_models]),
            'total_models_evaluated': len(all_models)
        }
        
        return comparison
    
    def _generate_model_recommendations(self, models: List[ModelResult], 
                                       ensemble_model: Optional[ModelResult],
                                       data_info: Dict[str, Any], 
                                       task_type: str) -> str:
        """Generate AI-powered model recommendations"""
        
        try:
            # Prepare model summary for AI
            model_summary = []
            for model in models[:5]:  # Top 5 models
                model_summary.append({
                    'name': model.model_name,
                    'validation_score': model.validation_score,
                    'test_score': model.test_score,
                    'overfitting': model.train_score - model.test_score,
                    'hyperparameters': model.hyperparameters,
                    'complexity': model.model_interpretation.get('model_complexity', {})
                })
            
            prompt = f"""
            As a senior ML engineer, analyze these modeling results and provide recommendations:
            
            Task Type: {task_type}
            Data Characteristics: {data_info}
            
            Model Results: {json.dumps(model_summary, default=str, indent=2)}
            
            Ensemble Available: {ensemble_model is not None}
            
            Please provide:
            1. Best model recommendation and why
            2. Production deployment considerations
            3. Potential improvements and next steps
            4. Risk assessment and monitoring recommendations
            5. Business impact and ROI considerations
            
            Keep recommendations practical and business-focused.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"AI model recommendations failed: {e}")
            return f"Modeling completed successfully. Best model: {models[0].model_name if models else 'None'} with validation score: {models[0].validation_score:.4f if models else 0}"
    
    def _create_deployment_guide(self, best_model: ModelResult, 
                                data_info: Dict[str, Any], task_type: str) -> str:
        """Create comprehensive deployment guide"""
        
        guide = f"""
# Model Deployment Guide

## Model Information
- **Model Type**: {best_model.model_name}
- **Task Type**: {task_type}
- **Validation Score**: {best_model.validation_score:.4f}
- **Test Score**: {best_model.test_score:.4f}

## Performance Metrics
"""
        
        for metric, value in best_model.metrics.items():
            guide += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
        
        guide += f"""

## Data Requirements
- **Features Required**: {data_info['n_features']}
- **Numeric Features**: {data_info['n_numeric_features']}
- **Categorical Features**: {data_info['n_categorical_features']}
- **Missing Data Handling**: Required (median for numeric, mode for categorical)

## Preprocessing Pipeline
1. Handle missing values (median for numeric, mode for categorical)
2. Encode categorical variables using saved label encoders
3. Feature scaling (if required by model)

## Model Files
- Model object: `{best_model.model_name.lower()}_model.joblib`
- Label encoders: `label_encoders.joblib`
- Feature names: `feature_names.json`

## Production Code Example
```python
import joblib
import pandas as pd
import numpy as np

# Load model and preprocessors
model = joblib.load('{best_model.model_name.lower()}_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

def predict(input_data):
    # Preprocess input data
    processed_data = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return prediction

def preprocess_input(data):
    # Handle missing values
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())
    
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
        if col in label_encoders:
            data[col] = label_encoders[col].transform(data[col].astype(str))
    
    return data
```

## Monitoring Recommendations
1. Track prediction accuracy over time
2. Monitor for data drift in input features
3. Set up alerts for unusual prediction patterns
4. Retrain model when performance degrades

## Performance Expectations
- Expected accuracy: {best_model.test_score:.1%}
- Overfitting risk: {'High' if (best_model.train_score - best_model.test_score) > 0.1 else 'Low'}
- Model complexity: {best_model.model_interpretation.get('model_complexity', {}).get('complexity_level', 'Medium')}
"""
        
        return guide
    
    def _save_modeling_artifacts(self, models: List[ModelResult], 
                                ensemble_model: Optional[ModelResult],
                                feature_names: List[str]) -> List[str]:
        """Save all modeling artifacts"""
        
        saved_files = []
        
        try:
            # Save best model
            if models:
                best_model = models[0]
                model_path = os.path.join(self.output_dir, f"{best_model.model_name.lower()}_model.joblib")
                joblib.dump(best_model.model_object, model_path)
                saved_files.append(model_path)
                
                # Save model metadata
                metadata = {
                    'model_name': best_model.model_name,
                    'hyperparameters': best_model.hyperparameters,
                    'metrics': best_model.metrics,
                    'feature_importance': best_model.feature_importance,
                    'feature_names': feature_names
                }
                
                metadata_path = os.path.join(self.output_dir, f"{best_model.model_name.lower()}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                saved_files.append(metadata_path)
            
            # Save ensemble model if available
            if ensemble_model:
                ensemble_path = os.path.join(self.output_dir, "ensemble_model.joblib")
                joblib.dump(ensemble_model.model_object, ensemble_path)
                saved_files.append(ensemble_path)
            
            # Save feature names
            feature_names_path = os.path.join(self.output_dir, "feature_names.json")
            with open(feature_names_path, 'w') as f:
                json.dump(feature_names, f, indent=2)
            saved_files.append(feature_names_path)
            
            # Save comprehensive modeling report
            report = {
                'models_evaluated': [
                    {
                        'name': model.model_name,
                        'validation_score': model.validation_score,
                        'test_score': model.test_score,
                        'hyperparameters': model.hyperparameters,
                        'metrics': model.metrics
                    }
                    for model in models
                ],
                'best_model': models[0].model_name if models else None,
                'ensemble_available': ensemble_model is not None,
                'optimization_history': self.modeling_history
            }
            
            report_path = os.path.join(self.output_dir, "modeling_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            saved_files.append(report_path)
            
            # Create model comparison visualization
            self._create_model_comparison_plot(models, ensemble_model)
            plot_path = os.path.join(self.output_dir, "model_comparison.png")
            saved_files.append(plot_path)
            
        except Exception as e:
            logger.error(f"Error saving modeling artifacts: {e}")
        
        return saved_files
    
    # Helper methods
    def _assess_data_complexity(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> str:
        """Assess data complexity level"""
        n_samples, n_features = X.shape
        
        # Calculate complexity score
        complexity_score = 0
        
        # Size complexity
        if n_samples < 1000:
            complexity_score += 1
        elif n_samples > 100000:
            complexity_score += 1
        
        # Dimensionality complexity
        if n_features > 100:
            complexity_score += 2
        elif n_features > 50:
            complexity_score += 1
        
        # Class imbalance complexity (for classification)
        if task_type == "classification":
            class_counts = y.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 10:
                complexity_score += 2
            elif imbalance_ratio > 5:
                complexity_score += 1
        
        # Feature correlation complexity
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            high_corr_pairs = (corr_matrix > 0.8).sum().sum() - len(corr_matrix)
            if high_corr_pairs > len(corr_matrix) * 0.5:
                complexity_score += 1
        
        # Return complexity level
        if complexity_score <= 2:
            return "low"
        elif complexity_score <= 4:
            return "medium"
        else:
            return "high"
    
    def _calculate_score(self, model, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
        """Calculate score for given metric"""
        y_pred = model.predict(X)
        
        if metric == "roc_auc":
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X)[:, 1]
                return roc_auc_score(y, y_pred_proba)
            else:
                return 0.5
        elif metric == "f1_macro":
            return f1_score(y, y_pred, average='macro')
        elif metric == "neg_mean_squared_error":
            return -mean_squared_error(y, y_pred)
        elif metric == "r2":
            return r2_score(y, y_pred)
        else:
            return accuracy_score(y, y_pred)
    
    def _calculate_comprehensive_metrics(self, model, X_test: pd.DataFrame, 
                                        y_test: pd.Series, task_type: str) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        y_pred = model.predict(X_test)
        metrics = {}
        
        if task_type == "classification":
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    metrics['roc_auc'] = 0.5
        else:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
        
        return metrics
    
    def _extract_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(importance)
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                for i, importance in enumerate(coef):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(abs(importance))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return importance_dict
    
    def _assess_model_complexity(self, model) -> Dict[str, Any]:
        """Assess model complexity"""
        complexity = {
            'model_type': type(model).__name__,
            'interpretability': 'high',
            'complexity_level': 'medium'
        }
        
        # Assess interpretability
        if type(model).__name__ in ['LinearRegression', 'LogisticRegression', 'DecisionTreeClassifier', 'DecisionTreeRegressor']:
            complexity['interpretability'] = 'high'
            complexity['complexity_level'] = 'low'
        elif type(model).__name__ in ['RandomForestClassifier', 'RandomForestRegressor']:
            complexity['interpretability'] = 'medium'
            complexity['complexity_level'] = 'medium'
        elif type(model).__name__ in ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor']:
            complexity['interpretability'] = 'medium'
            complexity['complexity_level'] = 'high'
        else:
            complexity['interpretability'] = 'low'
            complexity['complexity_level'] = 'high'
        
        # Add parameter count if available
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                complexity['parameter_count'] = len(params)
        except:
            pass
        
        return complexity
    
    def _analyze_feature_importance(self, feature_importance: Dict[str, float], 
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance distribution"""
        if not feature_importance:
            return {}
        
        importances = list(feature_importance.values())
        
        analysis = {
            'top_5_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5],
            'importance_concentration': np.sum(sorted(importances, reverse=True)[:5]) / np.sum(importances),
            'feature_importance_std': np.std(importances),
            'dominant_feature_ratio': max(importances) / np.mean(importances) if importances else 0
        }
        
        return analysis
    
    def _analyze_model_stability(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze model stability across different data splits"""
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            
            # Determine CV strategy
            if y.nunique() <= 10:  # Classification
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:  # Regression
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scoring = 'r2'
            
            # Calculate cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            stability = {
                'cv_scores_mean': float(np.mean(cv_scores)),
                'cv_scores_std': float(np.std(cv_scores)),
                'stability_score': float(1 - (np.std(cv_scores) / np.mean(cv_scores))),
                'is_stable': np.std(cv_scores) < 0.05
            }
            
            return stability
            
        except Exception as e:
            logger.warning(f"Stability analysis failed: {e}")
            return {'error': str(e)}
    
    def _create_model_comparison_plot(self, models: List[ModelResult], 
                                     ensemble_model: Optional[ModelResult]):
        """Create model comparison visualization"""
        try:
            all_models = models.copy()
            if ensemble_model:
                all_models.append(ensemble_model)
            
            # Prepare data for plotting
            model_names = [model.model_name for model in all_models]
            validation_scores = [model.validation_score for model in all_models]
            test_scores = [model.test_score for model in all_models]
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Validation vs Test scores
            x = np.arange(len(model_names))
            width = 0.35
            
            ax1.bar(x - width/2, validation_scores, width, label='Validation Score', alpha=0.8)
            ax1.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Overfitting analysis
            overfitting_scores = [model.train_score - model.test_score for model in all_models]
            colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'green' for score in overfitting_scores]
            
            ax2.bar(model_names, overfitting_scores, color=colors, alpha=0.7)
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Overfitting Score (Train - Test)')
            ax2.set_title('Overfitting Analysis')
            ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High Overfitting')
            ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate Overfitting')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Model comparison plot creation failed: {e}")
    
    def get_production_model_code(self, best_model: ModelResult, 
                                 data_info: Dict[str, Any]) -> str:
        """Generate production-ready model code"""
        
        code = f'''
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ProductionModel:
    """
    Production-ready model for {best_model.model_name}
    Generated by TaskPilot AI Model Maestro
    """
    
    def __init__(self, model_path: str, encoders_path: str = None):
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path) if encoders_path else {{}}
        self.feature_names = {list(data_info.get('feature_types', {}).keys())}
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        processed_data = data.copy()
        
        # Handle missing values
        numeric_columns = {[col for col, dtype in data_info.get('feature_types', {}).items() if dtype == 'numeric']}
        categorical_columns = {[col for col, dtype in data_info.get('feature_types', {}).items() if dtype == 'categorical']}
        
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        for col in categorical_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0] if not processed_data[col].mode().empty else 'unknown')
                if col in self.encoders:
                    processed_data[col] = self.encoders[col].transform(processed_data[col].astype(str))
        
        return processed_data
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        processed_data = self.preprocess(data)
        return self.model.predict(processed_data)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (for classification)"""
        if hasattr(self.model, 'predict_proba'):
            processed_data = self.preprocess(data)
            return self.model.predict_proba(processed_data)
        else:
            raise AttributeError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {{}}

# Usage example:
# model = ProductionModel('model.joblib', 'encoders.joblib')
# predictions = model.predict(new_data)
'''
        
        return code