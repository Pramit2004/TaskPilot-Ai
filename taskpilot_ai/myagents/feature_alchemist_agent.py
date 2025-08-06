import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import feature_engine as fe
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineering:
    feature_name: str
    feature_type: str  # "created", "transformed", "selected"
    importance_score: float
    creation_method: str
    description: str
    correlation_with_target: float

@dataclass
class FeatureAlchemyResult:
    original_features: List[str]
    engineered_features: List[FeatureEngineering]
    feature_importance_ranking: Dict[str, float]
    transformation_pipeline: Dict[str, Any]
    recommended_features: List[str]
    feature_interactions: List[Dict[str, Any]]
    dimensionality_reduction: Dict[str, Any]
    feature_creation_summary: str
    performance_impact: Dict[str, float]

class FeatureAlchemistAgent:
    """
    ⚗️ Feature Alchemist Agent - The Master of Feature Engineering
    
    This agent performs intelligent feature engineering across all data types:
    1. Automated feature creation and transformation
    2. Advanced feature selection using multiple techniques
    3. Feature interaction discovery
    4. Domain-specific feature engineering
    5. Dimensionality reduction strategies
    6. Feature importance analysis and ranking
    7. AI-powered feature recommendations
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "reports/features"):
        self.gemini_api_key = gemini_api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
        self.feature_history = []
        self.transformation_registry = {}
        
    def engineer_features(self, df: pd.DataFrame, target_column: str = None,
                         task_type: str = "classification", 
                         domain_context: str = "",
                         max_features: int = None) -> FeatureAlchemyResult:
        """
        ⚗️ Perform comprehensive feature engineering alchemy
        """
        logger.info("⚗️ Feature Alchemist: Beginning feature engineering...")
        
        original_features = df.columns.tolist()
        if target_column and target_column in original_features:
            original_features.remove(target_column)
        
        # Step 1: Automated feature creation
        df_enhanced = self._create_automated_features(df, target_column, task_type, domain_context)
        
        # Step 2: Advanced transformations
        df_transformed = self._apply_advanced_transformations(df_enhanced, target_column, task_type)
        
        # Step 3: Feature interactions
        df_interactions = self._discover_feature_interactions(df_transformed, target_column, task_type)
        
        # Step 4: Feature selection
        selected_features, importance_scores = self._intelligent_feature_selection(
            df_interactions, target_column, task_type, max_features
        )
        
        # Step 5: Dimensionality reduction analysis
        dimensionality_analysis = self._analyze_dimensionality_reduction(
            df_interactions[selected_features], target_column
        )
        
        # Step 6: AI-powered feature recommendations
        ai_recommendations = self._get_ai_feature_recommendations(
            df_interactions, target_column, domain_context, importance_scores
        )
        
        # Step 7: Create feature engineering summary
        engineered_features = self._document_engineered_features(
            original_features, df_interactions.columns.tolist(), importance_scores
        )
        
        # Step 8: Assess performance impact
        performance_impact = self._assess_feature_performance_impact(
            df[original_features], df_interactions[selected_features], 
            df[target_column] if target_column else None, task_type
        )
        
        result = FeatureAlchemyResult(
            original_features=original_features,
            engineered_features=engineered_features,
            feature_importance_ranking=importance_scores,
            transformation_pipeline=self.transformation_registry,
            recommended_features=selected_features,
            feature_interactions=self._extract_interaction_details(df_interactions, selected_features),
            dimensionality_reduction=dimensionality_analysis,
            feature_creation_summary=ai_recommendations,
            performance_impact=performance_impact
        )
        
        # Save feature engineering report
        self._save_feature_report(result, df_interactions[selected_features])
        
        logger.info(f"✅ Feature Alchemist: Created {len(selected_features)} optimal features!")
        return result
    
    def _create_automated_features(self, df: pd.DataFrame, target_column: str = None,
                                  task_type: str = "classification", 
                                  domain_context: str = "") -> pd.DataFrame:
        """Create automated features using multiple strategies"""
        df_enhanced = df.copy()
        
        # Numeric feature engineering
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
            
        # 1. Mathematical transformations
        for col in numeric_cols:
            if df_enhanced[col].min() > 0:  # Avoid log of negative numbers
                df_enhanced[f'{col}_log'] = np.log1p(df_enhanced[col])
                self.transformation_registry[f'{col}_log'] = f"log1p({col})"
                
            if df_enhanced[col].std() > 0:  # Avoid division by zero
                df_enhanced[f'{col}_squared'] = df_enhanced[col] ** 2
                df_enhanced[f'{col}_sqrt'] = np.sqrt(np.abs(df_enhanced[col]))
                self.transformation_registry[f'{col}_squared'] = f"{col}^2"
                self.transformation_registry[f'{col}_sqrt'] = f"sqrt(abs({col}))"
        
        # 2. Statistical features
        if len(numeric_cols) > 1:
            # Rolling statistics for potential time series
            for col in numeric_cols[:5]:  # Limit to prevent explosion
                df_enhanced[f'{col}_rolling_mean_3'] = df_enhanced[col].rolling(window=3, min_periods=1).mean()
                df_enhanced[f'{col}_rolling_std_3'] = df_enhanced[col].rolling(window=3, min_periods=1).std()
                self.transformation_registry[f'{col}_rolling_mean_3'] = f"rolling_mean({col}, 3)"
                self.transformation_registry[f'{col}_rolling_std_3'] = f"rolling_std({col}, 3)"
        
        # 3. Categorical feature engineering
        categorical_cols = df_enhanced.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            # Frequency encoding
            freq_map = df_enhanced[col].value_counts().to_dict()
            df_enhanced[f'{col}_frequency'] = df_enhanced[col].map(freq_map)
            self.transformation_registry[f'{col}_frequency'] = f"frequency_encoding({col})"
            
            # Length features for text-like categories
            df_enhanced[f'{col}_length'] = df_enhanced[col].astype(str).str.len()
            self.transformation_registry[f'{col}_length'] = f"length({col})"
            
            # Target encoding (if target is available)
            if target_column and target_column in df_enhanced.columns:
                target_mean = df_enhanced.groupby(col)[target_column].mean()
                df_enhanced[f'{col}_target_encoded'] = df_enhanced[col].map(target_mean)
                self.transformation_registry[f'{col}_target_encoded'] = f"target_encoding({col})"
        
        # 4. Date/time features (if detected)
        for col in df_enhanced.columns:
            if df_enhanced[col].dtype == 'object':
                try:
                    dt_series = pd.to_datetime(df_enhanced[col], errors='coerce')
                    if dt_series.notna().sum() > len(df_enhanced) * 0.7:  # If >70% valid dates
                        df_enhanced[f'{col}_year'] = dt_series.dt.year
                        df_enhanced[f'{col}_month'] = dt_series.dt.month
                        df_enhanced[f'{col}_day'] = dt_series.dt.day
                        df_enhanced[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                        df_enhanced[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
                        
                        self.transformation_registry.update({
                            f'{col}_year': f"year({col})",
                            f'{col}_month': f"month({col})",
                            f'{col}_day': f"day({col})",
                            f'{col}_dayofweek': f"dayofweek({col})",
                            f'{col}_is_weekend': f"is_weekend({col})"
                        })
                except:
                    continue
        
        # 5. Domain-specific features
        if domain_context:
            df_enhanced = self._create_domain_specific_features(df_enhanced, domain_context, target_column)
        
        logger.info(f"Created {len(df_enhanced.columns) - len(df.columns)} automated features")
        return df_enhanced
    
    def _apply_advanced_transformations(self, df: pd.DataFrame, target_column: str = None,
                                      task_type: str = "classification") -> pd.DataFrame:
        """Apply advanced feature transformations"""
        df_transformed = df.copy()
        
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # 1. Outlier-robust transformations
        for col in numeric_cols:
            if df_transformed[col].std() > 0:
                # Robust scaling
                robust_scaler = RobustScaler()
                df_transformed[f'{col}_robust_scaled'] = robust_scaler.fit_transform(
                    df_transformed[[col]]
                ).flatten()
                self.transformation_registry[f'{col}_robust_scaled'] = f"robust_scale({col})"
                
                # Quantile transformation
                try:
                    from sklearn.preprocessing import QuantileTransformer
                    qt = QuantileTransformer(output_distribution='normal', random_state=42)
                    df_transformed[f'{col}_quantile_normal'] = qt.fit_transform(
                        df_transformed[[col]]
                    ).flatten()
                    self.transformation_registry[f'{col}_quantile_normal'] = f"quantile_normal({col})"
                except:
                    pass
        
        # 2. Polynomial features for top correlated features
        if target_column and len(numeric_cols) > 0:
            # Find top correlated features
            correlations = []
            for col in numeric_cols:
                try:
                    corr = abs(df_transformed[col].corr(df_transformed[target_column]))
                    correlations.append((col, corr))
                except:
                    continue
            
            # Sort by correlation and take top 3
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [col for col, _ in correlations[:3]]
            
            # Create polynomial features
            from sklearn.preprocessing import PolynomialFeatures
            if len(top_features) >= 2:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_features = poly.fit_transform(df_transformed[top_features])
                poly_feature_names = poly.get_feature_names_out(top_features)
                
                for i, name in enumerate(poly_feature_names):
                    if name not in top_features:  # Skip original features
                        df_transformed[f'poly_{name}'] = poly_features[:, i]
                        self.transformation_registry[f'poly_{name}'] = f"polynomial({name})"
        
        # 3. Binning for high-cardinality features
        for col in numeric_cols:
            if df_transformed[col].nunique() > 50:  # High cardinality
                df_transformed[f'{col}_binned'] = pd.cut(
                    df_transformed[col], bins=10, labels=False
                )
                self.transformation_registry[f'{col}_binned'] = f"binning({col}, 10)"
        
        logger.info(f"Applied advanced transformations, total features: {len(df_transformed.columns)}")
        return df_transformed
    
    def _discover_feature_interactions(self, df: pd.DataFrame, target_column: str = None,
                                     task_type: str = "classification") -> pd.DataFrame:
        """Discover and create meaningful feature interactions"""
        df_interactions = df.copy()
        
        numeric_cols = df_interactions.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Limit interactions to prevent combinatorial explosion
        max_interactions = min(20, len(numeric_cols))
        
        if len(numeric_cols) >= 2 and target_column:
            # Find most promising feature pairs based on individual correlation
            correlations = {}
            for col in numeric_cols:
                try:
                    corr = abs(df_interactions[col].corr(df_interactions[target_column]))
                    correlations[col] = corr if not np.isnan(corr) else 0
                except:
                    correlations[col] = 0
            
            # Sort by correlation and take top features
            top_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)
            top_features = top_features[:min(8, len(top_features))]  # Limit to top 8
            
            # Create interactions between top features
            interaction_count = 0
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    if interaction_count >= max_interactions:
                        break
                    
                    # Multiplication
                    df_interactions[f'{col1}_x_{col2}'] = df_interactions[col1] * df_interactions[col2]
                    self.transformation_registry[f'{col1}_x_{col2}'] = f"{col1} * {col2}"
                    
                    # Division (avoid division by zero)
                    if (df_interactions[col2] != 0).all():
                        df_interactions[f'{col1}_div_{col2}'] = df_interactions[col1] / df_interactions[col2]
                        self.transformation_registry[f'{col1}_div_{col2}'] = f"{col1} / {col2}"
                    
                    # Addition
                    df_interactions[f'{col1}_plus_{col2}'] = df_interactions[col1] + df_interactions[col2]
                    self.transformation_registry[f'{col1}_plus_{col2}'] = f"{col1} + {col2}"
                    
                    # Difference
                    df_interactions[f'{col1}_minus_{col2}'] = df_interactions[col1] - df_interactions[col2]
                    self.transformation_registry[f'{col1}_minus_{col2}'] = f"{col1} - {col2}"
                    
                    interaction_count += 4
                
                if interaction_count >= max_interactions:
                    break
        
        # Categorical interactions
        categorical_cols = df_interactions.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) >= 2:
            # Create combined categorical features
            for i, col1 in enumerate(categorical_cols[:3]):  # Limit to avoid explosion
                for col2 in categorical_cols[i+1:3]:
                    df_interactions[f'{col1}_combined_{col2}'] = (
                        df_interactions[col1].astype(str) + '_' + df_interactions[col2].astype(str)
                    )
                    self.transformation_registry[f'{col1}_combined_{col2}'] = f"combine({col1}, {col2})"
        
        logger.info(f"Created feature interactions, total features: {len(df_interactions.columns)}")
        return df_interactions
    
    def _intelligent_feature_selection(self, df: pd.DataFrame, target_column: str = None,
                                     task_type: str = "classification", 
                                     max_features: int = None) -> Tuple[List[str], Dict[str, float]]:
        """Intelligent feature selection using multiple techniques"""
        
        if target_column is None or target_column not in df.columns:
            # Return all features if no target
            feature_cols = [col for col in df.columns if col != target_column]
            return feature_cols, {col: 1.0 for col in feature_cols}
        
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        # Handle missing values for feature selection
        X_clean = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        
        # Convert categorical variables for feature selection
        categorical_cols = X_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
        
        importance_scores = {}
        
        # 1. Statistical tests
        try:
            if task_type == "classification":
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
            
            selector.fit(X_clean, y)
            
            for i, col in enumerate(feature_cols):
                importance_scores[col] = selector.scores_[i] if not np.isnan(selector.scores_[i]) else 0
        except:
            logger.warning("Statistical feature selection failed")
        
        # 2. Tree-based importance
        try:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            
            model.fit(X_clean, y)
            
            for i, col in enumerate(feature_cols):
                tree_importance = model.feature_importances_[i]
                if col in importance_scores:
                    importance_scores[col] = (importance_scores[col] + tree_importance) / 2
                else:
                    importance_scores[col] = tree_importance
        except Exception as e:
            logger.warning(f"Tree-based feature selection failed: {e}")
        
        # 3. Correlation with target
        try:
            for col in feature_cols:
                if X_clean[col].dtype in [np.number, 'int64', 'float64']:
                    corr = abs(X_clean[col].corr(y))
                    if not np.isnan(corr):
                        if col in importance_scores:
                            importance_scores[col] = (importance_scores[col] + corr) / 2
                        else:
                            importance_scores[col] = corr
        except:
            logger.warning("Correlation-based feature selection failed")
        
        # 4. Variance threshold
        try:
            var_selector = VarianceThreshold(threshold=0.01)
            var_selector.fit(X_clean)
            
            for i, col in enumerate(feature_cols):
                if not var_selector.get_support()[i]:
                    importance_scores[col] = importance_scores.get(col, 0) * 0.1  # Penalize low variance
        except:
            logger.warning("Variance threshold selection failed")
        
        # Normalize importance scores
        if importance_scores:
            max_score = max(importance_scores.values())
            if max_score > 0:
                importance_scores = {k: v / max_score for k, v in importance_scores.items()}
        
        # Select top features
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        if max_features:
            selected_features = [col for col, _ in sorted_features[:max_features]]
        else:
            # Select features with importance > threshold
            threshold = 0.1
            selected_features = [col for col, score in sorted_features if score > threshold]
            
            # Ensure minimum number of features
            if len(selected_features) < 5 and len(sorted_features) > 0:
                selected_features = [col for col, _ in sorted_features[:min(20, len(sorted_features))]]
        
        logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")
        return selected_features, importance_scores
    
    def _analyze_dimensionality_reduction(self, df: pd.DataFrame, 
                                        target_column: str = None) -> Dict[str, Any]:
        """Analyze dimensionality reduction options"""
        analysis = {}
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {"message": "Insufficient numeric features for dimensionality reduction"}
        
        # Fill missing values
        numeric_df_clean = numeric_df.fillna(numeric_df.median())
        
        try:
            # PCA Analysis
            pca = PCA()
            pca.fit(numeric_df_clean)
            
            # Find optimal number of components for 95% variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            analysis['pca'] = {
                'total_components': len(pca.explained_variance_ratio_),
                'components_for_95_variance': int(n_components_95),
                'explained_variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
                'recommended': n_components_95 < len(numeric_df.columns) * 0.8
            }
            
        except Exception as e:
            analysis['pca'] = {'error': str(e)}
        
        try:
            # t-SNE recommendation
            analysis['tsne'] = {
                'recommended': len(numeric_df.columns) > 50,
                'suggested_components': 2 if len(numeric_df.columns) > 10 else None
            }
            
        except Exception as e:
            analysis['tsne'] = {'error': str(e)}
        
        return analysis
    
    def _get_ai_feature_recommendations(self, df: pd.DataFrame, target_column: str = None,
                                      domain_context: str = "", 
                                      importance_scores: Dict[str, float] = None) -> str:
        """Get AI-powered feature engineering recommendations"""
        try:
            # Prepare data summary for AI
            data_summary = {
                'total_features': len(df.columns),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'top_features': list(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]) if importance_scores else [],
                'feature_types': [col for col in df.columns if any(keyword in col for keyword in ['_x_', '_div_', '_plus_', 'log', 'sqrt'])]
            }
            
            prompt = f"""
            As a senior feature engineering expert, analyze this feature engineering result and provide recommendations:
            
            Domain Context: {domain_context}
            Target Column: {target_column}
            Data Summary: {json.dumps(data_summary, default=str)}
            
            Please provide:
            1. Assessment of current feature engineering quality
            2. Recommendations for additional features to create
            3. Potential domain-specific features that might be valuable
            4. Feature selection strategy recommendations
            5. Risk areas to watch out for
            
            Keep recommendations practical and actionable.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"AI feature recommendations failed: {e}")
            return "Feature engineering completed successfully. Review feature importance scores for optimization."
    
    def _create_domain_specific_features(self, df: pd.DataFrame, domain_context: str, 
                                       target_column: str = None) -> pd.DataFrame:
        """Create domain-specific features based on context"""
        df_domain = df.copy()
        
        domain_lower = domain_context.lower()
        
        # Financial domain features
        if any(word in domain_lower for word in ['finance', 'financial', 'bank', 'trading', 'investment']):
            numeric_cols = df_domain.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            for col in numeric_cols[:5]:  # Limit to prevent explosion
                if 'price' in col.lower() or 'amount' in col.lower() or 'value' in col.lower():
                    # Moving averages
                    df_domain[f'{col}_ma_5'] = df_domain[col].rolling(window=5, min_periods=1).mean()
                    df_domain[f'{col}_ma_10'] = df_domain[col].rolling(window=10, min_periods=1).mean()
                    
                    # Volatility
                    df_domain[f'{col}_volatility'] = df_domain[col].rolling(window=5, min_periods=1).std()
                    
                    # Returns
                    df_domain[f'{col}_returns'] = df_domain[col].pct_change()
                    
                    self.transformation_registry.update({
                        f'{col}_ma_5': f"moving_average({col}, 5)",
                        f'{col}_ma_10': f"moving_average({col}, 10)",
                        f'{col}_volatility': f"volatility({col})",
                        f'{col}_returns': f"returns({col})"
                    })
        
        # Healthcare domain features
        elif any(word in domain_lower for word in ['health', 'medical', 'patient', 'clinical']):
            numeric_cols = df_domain.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            # Age-related features
            age_cols = [col for col in df_domain.columns if 'age' in col.lower()]
            for col in age_cols:
                df_domain[f'{col}_group'] = pd.cut(df_domain[col], bins=[0, 18, 35, 50, 65, 100], labels=['child', 'young', 'middle', 'senior', 'elderly'])
                self.transformation_registry[f'{col}_group'] = f"age_group({col})"
        
        # Retail/E-commerce domain features
        elif any(word in domain_lower for word in ['retail', 'sales', 'customer', 'purchase', 'order']):
            # Recency, Frequency, Monetary features
            date_cols = [col for col in df_domain.columns if 'date' in col.lower() or 'time' in col.lower()]
            amount_cols = [col for col in df_domain.columns if 'amount' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
            
            for col in amount_cols[:3]:
                # Cumulative features
                df_domain[f'{col}_cumsum'] = df_domain[col].cumsum()
                df_domain[f'{col}_cummax'] = df_domain[col].cummax()
                
                self.transformation_registry.update({
                    f'{col}_cumsum': f"cumulative_sum({col})",
                    f'{col}_cummax': f"cumulative_max({col})"
                })
        
        return df_domain
    
    def _document_engineered_features(self, original_features: List[str], 
                                    final_features: List[str], 
                                    importance_scores: Dict[str, float]) -> List[FeatureEngineering]:
        """Document all engineered features"""
        engineered_features = []
        
        for feature in final_features:
            if feature not in original_features:
                # Determine feature type
                if any(keyword in feature for keyword in ['_x_', '_div_', '_plus_', '_minus_']):
                    feature_type = "interaction"
                elif any(keyword in feature for keyword in ['log', 'sqrt', 'squared', 'scaled']):
                    feature_type = "transformation"
                elif any(keyword in feature for keyword in ['frequency', 'target_encoded', 'binned']):
                    feature_type = "encoding"
                else:
                    feature_type = "created"
                
                engineered_features.append(FeatureEngineering(
                    feature_name=feature,
                    feature_type=feature_type,
                    importance_score=importance_scores.get(feature, 0.0),
                    creation_method=self.transformation_registry.get(feature, "unknown"),
                    description=f"Engineered feature: {feature}",
                    correlation_with_target=0.0  # Would need target to calculate
                ))
        
        return engineered_features
    
    def _extract_interaction_details(self, df: pd.DataFrame, 
                                   selected_features: List[str]) -> List[Dict[str, Any]]:
        """Extract details about feature interactions"""
        interactions = []
        
        for feature in selected_features:
            if '_x_' in feature or '_div_' in feature or '_plus_' in feature or '_minus_' in feature:
                parts = feature.replace('_x_', '|').replace('_div_', '|').replace('_plus_', '|').replace('_minus_', '|').split('|')
                if len(parts) == 2:
                    interactions.append({
                        'interaction_feature': feature,
                        'base_features': parts,
                        'interaction_type': 'mathematical',
                        'description': self.transformation_registry.get(feature, "Unknown interaction")
                    })
        
        return interactions
    
    def _assess_feature_performance_impact(self, original_df: pd.DataFrame, 
                                         engineered_df: pd.DataFrame,
                                         target: pd.Series = None, 
                                         task_type: str = "classification") -> Dict[str, float]:
        """Assess the performance impact of feature engineering"""
        impact = {}
        
        if target is None:
            return {"message": "Cannot assess performance impact without target variable"}
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data
            original_clean = original_df.fillna(original_df.median() if original_df.select_dtypes(include=[np.number]).shape[1] > 0 else original_df.mode().iloc[0])
            engineered_clean = engineered_df.fillna(engineered_df.median() if engineered_df.select_dtypes(include=[np.number]).shape[1] > 0 else engineered_df.mode().iloc[0])
            
            # Handle categorical variables
            for df in [original_clean, engineered_clean]:
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            
            # Choose model
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
                scoring = 'r2'
            
            # Cross-validation scores
            original_scores = cross_val_score(model, original_clean, target, cv=3, scoring=scoring)
            engineered_scores = cross_val_score(model, engineered_clean, target, cv=3, scoring=scoring)
            
            impact = {
                'original_performance': float(np.mean(original_scores)),
                'engineered_performance': float(np.mean(engineered_scores)),
                'performance_improvement': float(np.mean(engineered_scores) - np.mean(original_scores)),
                'improvement_percentage': float((np.mean(engineered_scores) - np.mean(original_scores)) / np.mean(original_scores) * 100)
            }
            
        except Exception as e:
            logger.warning(f"Performance impact assessment failed: {e}")
            impact = {"error": str(e)}
        
        return impact
    
    def _save_feature_report(self, result: FeatureAlchemyResult, final_df: pd.DataFrame):
        """Save comprehensive feature engineering report"""
        
        # Save feature engineering summary
        report_path = os.path.join(self.output_dir, "feature_engineering_report.json")
        
        report_data = {
            'original_feature_count': len(result.original_features),
            'final_feature_count': len(result.recommended_features),
            'engineered_features': [
                {
                    'name': feat.feature_name,
                    'type': feat.feature_type,
                    'importance': feat.importance_score,
                    'method': feat.creation_method
                }
                for feat in result.engineered_features
            ],
            'feature_importance_ranking': result.feature_importance_ranking,
            'transformation_pipeline': result.transformation_pipeline,
            'feature_interactions': result.feature_interactions,
            'dimensionality_reduction': result.dimensionality_reduction,
            'ai_recommendations': result.feature_creation_summary,
            'performance_impact': result.performance_impact
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save processed dataset
        dataset_path = os.path.join(self.output_dir, "engineered_dataset.csv")
        final_df.to_csv(dataset_path, index=False)
        
        # Save feature importance plot
        try:
            import matplotlib.pyplot as plt
            
            # Plot top 20 features
            top_features = sorted(result.feature_importance_ranking.items(), 
                                key=lambda x: x[1], reverse=True)[:20]
            
            if top_features:
                features, scores = zip(*top_features)
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(features)), scores)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Importance Score')
                plt.title('Top 20 Feature Importance Rankings')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plot_path = os.path.join(self.output_dir, "feature_importance.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Feature importance plot creation failed: {e}")
        
        logger.info(f"Feature engineering report saved to: {report_path}")
    
    def get_feature_pipeline(self) -> Dict[str, Any]:
        """Get the feature engineering pipeline for reuse"""
        return {
            'transformation_registry': self.transformation_registry,
            'feature_creation_methods': [
                'mathematical_transformations',
                'statistical_features', 
                'categorical_encoding',
                'datetime_features',
                'domain_specific_features',
                'feature_interactions',
                'polynomial_features',
                'binning'
            ]
        }
    
    def apply_feature_pipeline(self, df: pd.DataFrame, pipeline: Dict[str, Any]) -> pd.DataFrame:
        """Apply saved feature engineering pipeline to new data"""
        df_transformed = df.copy()
        
        # Apply transformations from registry
        for feature_name, transformation in pipeline.get('transformation_registry', {}).items():
            try:
                # Parse and apply transformation
                # This is a simplified version - in production, you'd save actual transformation objects
                if 'log1p(' in transformation:
                    base_col = transformation.replace('log1p(', '').replace(')', '')
                    if base_col in df_transformed.columns and df_transformed[base_col].min() > 0:
                        df_transformed[feature_name] = np.log1p(df_transformed[base_col])
                
                elif '_squared' in feature_name and '^2' in transformation:
                    base_col = transformation.replace('^2', '').strip()
                    if base_col in df_transformed.columns:
                        df_transformed[feature_name] = df_transformed[base_col] ** 2
                
                # Add more transformation applications as needed
                
            except Exception as e:
                logger.warning(f"Failed to apply transformation {transformation}: {e}")
        
        return df_transformed