import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    title: str
    content: str
    visualizations: List[str]
    importance: str  # "high", "medium", "low"
    section_type: str  # "executive", "technical", "business", "appendix"

@dataclass
class ComprehensiveReport:
    executive_summary: str
    technical_analysis: str
    business_insights: str
    recommendations: str
    methodology: str
    appendices: List[str]
    visualizations: List[str]
    report_files: List[str]
    interactive_dashboard: Optional[str]

class ReportArtisanAgent:
    """
    📊 Report Artisan Agent - The Master of Data Storytelling
    
    This agent creates beautiful, comprehensive reports from all analysis results:
    1. Executive summaries for business stakeholders
    2. Technical documentation for data scientists
    3. Interactive dashboards and visualizations
    4. Business insights and actionable recommendations
    5. Model deployment and monitoring guides
    6. PDF reports with professional formatting
    7. Automated insight generation and storytelling
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "reports/final"):
        self.gemini_api_key = gemini_api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
        self.report_sections = []
        self.visualizations = []
        
    def create_comprehensive_report(self, 
                                   data_detective_results: Dict[str, Any],
                                   feature_alchemy_results: Dict[str, Any], 
                                   modeling_results: Dict[str, Any],
                                   user_query: str = "",
                                   business_context: str = "") -> ComprehensiveReport:
        """
        📊 Create a comprehensive, multi-stakeholder report
        """
        logger.info("📊 Report Artisan: Creating comprehensive report...")
        
        # Step 1: Generate executive summary
        executive_summary = self._create_executive_summary(
            data_detective_results, feature_alchemy_results, modeling_results, 
            user_query, business_context
        )
        
        # Step 2: Create technical analysis
        technical_analysis = self._create_technical_analysis(
            data_detective_results, feature_alchemy_results, modeling_results
        )
        
        # Step 3: Generate business insights
        business_insights = self._create_business_insights(
            data_detective_results, modeling_results, business_context
        )
        
        # Step 4: Create actionable recommendations
        recommendations = self._create_recommendations(
            data_detective_results, feature_alchemy_results, modeling_results, 
            business_context
        )
        
        # Step 5: Document methodology
        methodology = self._document_methodology(
            data_detective_results, feature_alchemy_results, modeling_results
        )
        
        # Step 6: Create comprehensive visualizations
        visualizations = self._create_comprehensive_visualizations(
            data_detective_results, feature_alchemy_results, modeling_results
        )
        
        # Step 7: Generate interactive dashboard
        interactive_dashboard = self._create_interactive_dashboard(
            data_detective_results, feature_alchemy_results, modeling_results
        )
        
        # Step 8: Create appendices
        appendices = self._create_appendices(
            data_detective_results, feature_alchemy_results, modeling_results
        )
        
        # Step 9: Generate report files
        report_files = self._generate_report_files(
            executive_summary, technical_analysis, business_insights, 
            recommendations, methodology, visualizations
        )
        
        result = ComprehensiveReport(
            executive_summary=executive_summary,
            technical_analysis=technical_analysis,
            business_insights=business_insights,
            recommendations=recommendations,
            methodology=methodology,
            appendices=appendices,
            visualizations=visualizations,
            report_files=report_files,
            interactive_dashboard=interactive_dashboard
        )
        
        logger.info("✅ Report Artisan: Comprehensive report created!")
        return result
    
    def _create_executive_summary(self, data_results: Dict[str, Any], 
                                 feature_results: Dict[str, Any],
                                 model_results: Dict[str, Any],
                                 user_query: str, business_context: str) -> str:
        """Create executive summary for business stakeholders"""
        
        try:
            # Extract key metrics
            best_model = model_results.get('best_model', {})
            data_quality = data_results.get('quality_report', {}).get('overall_score', 0)
            feature_count = len(feature_results.get('recommended_features', []))
            
            prompt = f"""
            Create a compelling executive summary for business stakeholders based on this AI analysis:
            
            Original Request: {user_query}
            Business Context: {business_context}
            
            Key Results:
            - Data Quality Score: {data_quality:.1%}
            - Best Model: {best_model.get('model_name', 'Unknown')}
            - Model Performance: {best_model.get('validation_score', 0):.1%}
            - Features Used: {feature_count}
            - Data Insights: {len(data_results.get('insights', []))} key insights discovered
            
            Write a 4-paragraph executive summary that covers:
            1. Project overview and objectives
            2. Key findings and model performance
            3. Business impact and opportunities
            4. Next steps and recommendations
            
            Make it compelling for C-level executives who need to understand ROI and business impact.
            Use clear, non-technical language.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            return self._create_fallback_executive_summary(data_results, feature_results, model_results)
    
    def _create_technical_analysis(self, data_results: Dict[str, Any],
                                  feature_results: Dict[str, Any], 
                                  model_results: Dict[str, Any]) -> str:
        """Create detailed technical analysis for data scientists"""
        
        analysis = "# Technical Analysis Report\n\n"
        
        # Data Analysis Section
        analysis += "## Data Analysis\n\n"
        data_profile = data_results.get('data_profile', {})
        quality_report = data_results.get('quality_report', {})
        
        analysis += f"**Dataset Characteristics:**\n"
        analysis += f"- Shape: {data_profile.get('shape', 'Unknown')}\n"
        analysis += f"- Data Type: {data_profile.get('data_type', 'Unknown')}\n"
        analysis += f"- Quality Score: {quality_report.get('overall_score', 0):.2f}/1.0\n"
        analysis += f"- Missing Data: {quality_report.get('data_completeness', 1):.1%} complete\n\n"
        
        # Quality Issues
        quality_issues = quality_report.get('quality_issues', [])
        if quality_issues:
            analysis += f"**Data Quality Issues:**\n"
            for issue in quality_issues:
                analysis += f"- {issue}\n"
            analysis += "\n"
        
        # Key Insights
        insights = data_results.get('insights', [])
        if insights:
            analysis += f"**Key Data Insights ({len(insights)} found):**\n"
            for i, insight in enumerate(insights[:5], 1):
                analysis += f"{i}. **{insight.get('insight_type', 'General').title()}**: {insight.get('description', '')}\n"
                analysis += f"   - *Recommendation*: {insight.get('actionable_recommendation', '')}\n"
            analysis += "\n"
        
        # Feature Engineering Section
        analysis += "## Feature Engineering\n\n"
        analysis += f"**Feature Transformation Summary:**\n"
        analysis += f"- Original Features: {len(feature_results.get('original_features', []))}\n"
        analysis += f"- Final Features: {len(feature_results.get('recommended_features', []))}\n"
        analysis += f"- Engineered Features: {len(feature_results.get('engineered_features', []))}\n\n"
        
        # Feature Importance
        feature_importance = feature_results.get('feature_importance_ranking', {})
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis += f"**Top 10 Most Important Features:**\n"
            for i, (feature, importance) in enumerate(top_features, 1):
                analysis += f"{i}. {feature}: {importance:.3f}\n"
            analysis += "\n"
        
        # Performance Impact
        performance_impact = feature_results.get('performance_impact', {})
        if performance_impact and 'improvement_percentage' in performance_impact:
            analysis += f"**Feature Engineering Impact:**\n"
            analysis += f"- Performance Improvement: {performance_impact['improvement_percentage']:.1f}%\n"
            analysis += f"- Original Performance: {performance_impact.get('original_performance', 0):.3f}\n"
            analysis += f"- Enhanced Performance: {performance_impact.get('engineered_performance', 0):.3f}\n\n"
        
        # Model Analysis Section
        analysis += "## Model Analysis\n\n"
        best_model = model_results.get('best_model', {})
        all_models = model_results.get('all_models', [])
        
        analysis += f"**Best Model: {best_model.get('model_name', 'Unknown')}**\n"
        analysis += f"- Validation Score: {best_model.get('validation_score', 0):.4f}\n"
        analysis += f"- Test Score: {best_model.get('test_score', 0):.4f}\n"
        analysis += f"- Overfitting Score: {best_model.get('train_score', 0) - best_model.get('test_score', 0):.4f}\n\n"
        
        # Model Metrics
        metrics = best_model.get('metrics', {})
        if metrics:
            analysis += f"**Detailed Metrics:**\n"
            for metric, value in metrics.items():
                analysis += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
            analysis += "\n"
        
        # Model Comparison
        if len(all_models) > 1:
            analysis += f"**Model Comparison ({len(all_models)} models evaluated):**\n"
            for i, model in enumerate(all_models[:5], 1):
                analysis += f"{i}. {model.get('model_name', 'Unknown')}: {model.get('validation_score', 0):.4f}\n"
            analysis += "\n"
        
        # Hyperparameters
        hyperparams = best_model.get('hyperparameters', {})
        if hyperparams:
            analysis += f"**Optimal Hyperparameters:**\n"
            for param, value in hyperparams.items():
                analysis += f"- {param}: {value}\n"
            analysis += "\n"
        
        return analysis
    
    def _create_business_insights(self, data_results: Dict[str, Any],
                                 model_results: Dict[str, Any],
                                 business_context: str) -> str:
        """Generate business-focused insights and opportunities"""
        
        try:
            # Prepare business-relevant data
            best_model = model_results.get('best_model', {})
            model_performance = best_model.get('validation_score', 0)
            data_quality = data_results.get('quality_report', {}).get('overall_score', 0)
            insights = data_results.get('insights', [])
            
            # Extract business-relevant insights
            business_relevant_insights = [
                insight for insight in insights 
                if insight.get('importance', 'low') in ['high', 'medium']
            ]
            
            prompt = f"""
            Generate business insights and opportunities based on this AI analysis:
            
            Business Context: {business_context}
            Model Performance: {model_performance:.1%}
            Data Quality: {data_quality:.1%}
            Key Technical Insights: {[insight.get('description', '') for insight in business_relevant_insights[:5]]}
            
            Provide business insights covering:
            1. Revenue opportunities and cost savings
            2. Operational improvements possible
            3. Risk mitigation strategies
            4. Competitive advantages gained
            5. ROI estimation and business case
            
            Focus on quantifiable business benefits and actionable opportunities.
            Use business language, not technical jargon.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"Business insights generation failed: {e}")
            return self._create_fallback_business_insights(data_results, model_results)
    
    def _create_recommendations(self, data_results: Dict[str, Any],
                               feature_results: Dict[str, Any],
                               model_results: Dict[str, Any],
                               business_context: str) -> str:
        """Create actionable recommendations"""
        
        recommendations = "# Actionable Recommendations\n\n"
        
        # Immediate Actions
        recommendations += "## Immediate Actions (Next 30 Days)\n\n"
        
        # Data quality recommendations
        quality_issues = data_results.get('quality_report', {}).get('quality_issues', [])
        if quality_issues:
            recommendations += "**Data Quality Improvements:**\n"
            for issue in quality_issues[:3]:
                recommendations += f"- Address: {issue}\n"
            recommendations += "\n"
        
        # Model deployment recommendations
        best_model = model_results.get('best_model', {})
        if best_model:
            recommendations += "**Model Deployment:**\n"
            recommendations += f"- Deploy {best_model.get('model_name', 'best')} model with {best_model.get('validation_score', 0):.1%} accuracy\n"
            recommendations += "- Set up monitoring dashboard for model performance\n"
            recommendations += "- Establish data pipeline for real-time predictions\n\n"
        
        # Short-term Improvements (Next 90 Days)
        recommendations += "## Short-term Improvements (Next 90 Days)\n\n"
        
        # Feature engineering recommendations
        feature_ai_recs = feature_results.get('feature_creation_summary', '')
        if feature_ai_recs and len(feature_ai_recs) > 100:
            recommendations += "**Feature Engineering Enhancements:**\n"
            recommendations += "- Implement advanced feature engineering pipeline\n"
            recommendations += "- Explore domain-specific feature creation\n"
            recommendations += "- Set up automated feature selection process\n\n"
        
        # Model improvements
        model_comparison = model_results.get('model_comparison', {})
        if model_comparison:
            recommendations += "**Model Optimization:**\n"
            recommendations += "- Implement ensemble methods for improved accuracy\n"
            recommendations += "- Set up automated hyperparameter tuning\n"
            recommendations += "- Establish model retraining schedule\n\n"
        
        # Long-term Strategy (Next 6-12 Months)
        recommendations += "## Long-term Strategy (Next 6-12 Months)\n\n"
        
        recommendations += "**Advanced Analytics:**\n"
        recommendations += "- Implement real-time machine learning pipeline\n"
        recommendations += "- Explore deep learning approaches for complex patterns\n"
        recommendations += "- Develop automated model validation and testing\n\n"
        
        recommendations += "**Business Integration:**\n"
        recommendations += "- Integrate predictions into business decision workflows\n"
        recommendations += "- Develop business intelligence dashboards\n"
        recommendations += "- Train staff on model interpretation and usage\n\n"
        
        # Risk Mitigation
        recommendations += "## Risk Mitigation\n\n"
        
        overfitting_risk = best_model.get('train_score', 0) - best_model.get('test_score', 0)
        if overfitting_risk > 0.1:
            recommendations += "**Model Risk:**\n"
            recommendations += "- High overfitting detected - implement regularization\n"
            recommendations += "- Increase cross-validation rigor\n"
            recommendations += "- Collect more diverse training data\n\n"
        
        data_quality_score = data_results.get('quality_report', {}).get('overall_score', 1)
        if data_quality_score < 0.8:
            recommendations += "**Data Risk:**\n"
            recommendations += "- Implement data quality monitoring\n"
            recommendations += "- Establish data governance policies\n"
            recommendations += "- Set up automated data validation\n\n"
        
        return recommendations
    
    def _document_methodology(self, data_results: Dict[str, Any],
                             feature_results: Dict[str, Any],
                             model_results: Dict[str, Any]) -> str:
        """Document the complete methodology used"""
        
        methodology = "# Methodology Documentation\n\n"
        
        # Data Analysis Methodology
        methodology += "## Data Analysis Methodology\n\n"
        methodology += "**Data Detective Agent Process:**\n"
        methodology += "1. **Holistic Data Profiling**: Comprehensive analysis of data structure, types, and characteristics\n"
        methodology += "2. **Quality Assessment**: Multi-dimensional data quality evaluation including completeness, consistency, and validity\n"
        methodology += "3. **Pattern Discovery**: Statistical and AI-powered insight generation\n"
        methodology += "4. **Anomaly Detection**: Identification of outliers and unusual patterns\n"
        methodology += "5. **Domain Analysis**: Context-aware interpretation of data patterns\n\n"
        
        # Feature Engineering Methodology
        methodology += "## Feature Engineering Methodology\n\n"
        methodology += "**Feature Alchemist Agent Process:**\n"
        methodology += "1. **Automated Feature Creation**: Mathematical transformations, statistical features, and domain-specific features\n"
        methodology += "2. **Advanced Transformations**: Polynomial features, interaction terms, and binning strategies\n"
        methodology += "3. **Intelligent Feature Selection**: Multi-criteria selection using statistical tests, tree-based importance, and correlation analysis\n"
        methodology += "4. **Feature Interaction Discovery**: Systematic exploration of feature combinations\n"
        methodology += "5. **Performance Impact Assessment**: Quantitative evaluation of feature engineering benefits\n\n"
        
        # Modeling Methodology
        methodology += "## Modeling Methodology\n\n"
        methodology += "**Model Maestro Agent Process:**\n"
        methodology += "1. **Smart Model Selection**: Data-driven choice of candidate algorithms based on problem characteristics\n"
        methodology += "2. **Hyperparameter Optimization**: Optuna-based Bayesian optimization for each model\n"
        methodology += "3. **Cross-Validation**: Stratified k-fold validation for robust performance estimation\n"
        methodology += "4. **Ensemble Learning**: Voting classifier/regressor construction from top-performing models\n"
        methodology += "5. **Model Interpretation**: Feature importance analysis and explainability assessment\n\n"
        
        # Technical Details
        methodology += "## Technical Implementation Details\n\n"
        
        # Models evaluated
        all_models = model_results.get('all_models', [])
        if all_models:
            methodology += f"**Models Evaluated ({len(all_models)}):**\n"
            for model in all_models:
                methodology += f"- {model.get('model_name', 'Unknown')}: {model.get('validation_score', 0):.4f}\n"
            methodology += "\n"
        
        # Cross-validation strategy
        methodology += "**Cross-Validation Strategy:**\n"
        methodology += "- 5-fold stratified cross-validation for classification\n"
        methodology += "- 5-fold cross-validation for regression\n"
        methodology += "- 80/20 train-test split for final evaluation\n\n"
        
        # Optimization details
        optimization_history = model_results.get('optimization_history', [])
        if optimization_history:
            methodology += "**Hyperparameter Optimization:**\n"
            methodology += "- Bayesian optimization using Optuna framework\n"
            methodology += "- Tree-structured Parzen Estimator (TPE) sampling\n"
            methodology += f"- Total optimization trials: {sum(h.get('n_trials', 0) for h in optimization_history)}\n\n"
        
        return methodology
    
    def _create_comprehensive_visualizations(self, data_results: Dict[str, Any],
                                           feature_results: Dict[str, Any],
                                           model_results: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualizations for the report"""
        
        viz_files = []
        
        try:
            # 1. Data Quality Dashboard
            viz_files.append(self._create_data_quality_dashboard(data_results))
            
            # 2. Feature Importance Chart
            viz_files.append(self._create_feature_importance_chart(feature_results))
            
            # 3. Model Performance Comparison
            viz_files.append(self._create_model_performance_chart(model_results))
            
            # 4. Business Impact Visualization
            viz_files.append(self._create_business_impact_chart(model_results))
            
            # 5. Executive Summary Dashboard
            viz_files.append(self._create_executive_dashboard(
                data_results, feature_results, model_results
            ))
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
        
        return [f for f in viz_files if f]  # Filter out None values
    
    def _create_data_quality_dashboard(self, data_results: Dict[str, Any]) -> Optional[str]:
        """Create data quality visualization dashboard"""
        try:
            quality_report = data_results.get('quality_report', {})
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Overall Quality Score', 'Quality Dimensions', 
                               'Data Completeness', 'Quality Issues'],
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "table"}]]
            )
            
            # Overall Quality Score
            overall_score = quality_report.get('overall_score', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Quality Score (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Quality Dimensions
            dimensions = ['Completeness', 'Consistency', 'Uniqueness', 'Validity']
            scores = [
                quality_report.get('data_completeness', 0) * 100,
                quality_report.get('consistency_score', 0) * 100,
                quality_report.get('uniqueness_score', 0) * 100,
                quality_report.get('validity_score', 0) * 100
            ]
            
            fig.add_trace(
                go.Bar(x=dimensions, y=scores, name="Quality Dimensions",
                      marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
                row=1, col=2
            )
            
            # Data Completeness Pie Chart
            completeness = quality_report.get('data_completeness', 1)
            fig.add_trace(
                go.Pie(labels=['Complete', 'Missing'], 
                      values=[completeness * 100, (1 - completeness) * 100],
                      name="Data Completeness"),
                row=2, col=1
            )
            
            # Quality Issues Table
            quality_issues = quality_report.get('quality_issues', [])
            if quality_issues:
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Quality Issues Detected'],
                                   fill_color='paleturquoise',
                                   align='left'),
                        cells=dict(values=[quality_issues[:5]],  # Top 5 issues
                                  fill_color='lavender',
                                  align='left')
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="Data Quality Assessment Dashboard",
                title_x=0.5,
                height=800,
                showlegend=False
            )
            
            output_path = os.path.join(self.output_dir, "data_quality_dashboard.html")
            fig.write_html(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Data quality dashboard creation failed: {e}")
            return None
    
    def _create_feature_importance_chart(self, feature_results: Dict[str, Any]) -> Optional[str]:
        """Create feature importance visualization"""
        try:
            feature_importance = feature_results.get('feature_importance_ranking', {})
            if not feature_importance:
                return None
            
            # Get top 20 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            features, importance_scores = zip(*top_features)
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=importance_scores,
                y=features,
                orientation='h',
                marker_color='skyblue',
                text=[f'{score:.3f}' for score in importance_scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Top 20 Feature Importance Rankings',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            output_path = os.path.join(self.output_dir, "feature_importance_chart.html")
            fig.write_html(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Feature importance chart creation failed: {e}")
            return None
    
    def _create_model_performance_chart(self, model_results: Dict[str, Any]) -> Optional[str]:
        """Create model performance comparison chart"""
        try:
            all_models = model_results.get('all_models', [])
            if not all_models:
                return None
            
            model_names = [model.get('model_name', 'Unknown') for model in all_models]
            validation_scores = [model.get('validation_score', 0) for model in all_models]
            test_scores = [model.get('test_score', 0) for model in all_models]
            
            fig = go.Figure()
            
            # Add validation scores
            fig.add_trace(go.Bar(
                name='Validation Score',
                x=model_names,
                y=validation_scores,
                marker_color='lightblue'
            ))
            
            # Add test scores
            fig.add_trace(go.Bar(
                name='Test Score',
                x=model_names,
                y=test_scores,
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                barmode='group',
                height=500
            )
            
            output_path = os.path.join(self.output_dir, "model_performance_chart.html")
            fig.write_html(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Model performance chart creation failed: {e}")
            return None
    
    def _create_business_impact_chart(self, model_results: Dict[str, Any]) -> Optional[str]:
        """Create business impact visualization"""
        try:
            best_model = model_results.get('best_model', {})
            if not best_model:
                return None
            
            # Create business impact metrics
            accuracy = best_model.get('validation_score', 0) * 100
            confidence = 100 - (best_model.get('train_score', 0) - best_model.get('test_score', 0)) * 100
            roi_estimate = accuracy * 1.2  # Simple ROI estimation
            
            metrics = ['Model Accuracy', 'Confidence Level', 'Est. ROI Impact']
            values = [accuracy, max(confidence, 0), min(roi_estimate, 100)]
            
            fig = go.Figure(go.Bar(
                x=metrics,
                y=values,
                marker_color=['green', 'orange', 'blue'],
                text=[f'{val:.1f}%' for val in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Business Impact Assessment',
                yaxis_title='Impact Score (%)',
                height=400
            )
            
            output_path = os.path.join(self.output_dir, "business_impact_chart.html")
            fig.write_html(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Business impact chart creation failed: {e}")
            return None
    
    def _create_executive_dashboard(self, data_results: Dict[str, Any],
                                   feature_results: Dict[str, Any],
                                   model_results: Dict[str, Any]) -> Optional[str]:
        """Create executive summary dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=['Data Quality', 'Model Performance', 'Feature Engineering',
                               'Business Readiness', 'Risk Assessment', 'ROI Potential'],
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Extract key metrics
            data_quality = data_results.get('quality_report', {}).get('overall_score', 0) * 100
            model_performance = model_results.get('best_model', {}).get('validation_score', 0) * 100
            feature_improvement = feature_results.get('performance_impact', {}).get('improvement_percentage', 0)
            
            # Risk assessment
            overfitting_risk = model_results.get('best_model', {}).get('train_score', 0) - model_results.get('best_model', {}).get('test_score', 0)
            risk_score = max(0, 100 - overfitting_risk * 500)  # Convert to 0-100 scale
            
            # Business readiness
            readiness_score = (data_quality + model_performance) / 2
            
            # ROI potential
            roi_potential = min(100, model_performance * 1.2)
            
            # Add indicators
            indicators = [
                (data_quality, "Data Quality", [0, 60, 80, 100], 1, 1),
                (model_performance, "Model Accuracy", [0, 70, 85, 100], 1, 2),
                (max(0, feature_improvement), "Feature Boost", [0, 5, 15, 50], 1, 3),
                (readiness_score, "Business Ready", [0, 60, 80, 100], 2, 1),
                (risk_score, "Risk Level", [0, 60, 80, 100], 2, 2),
                (roi_potential, "ROI Potential", [0, 60, 80, 100], 2, 3)
            ]
            
            for value, title, ranges, row, col in indicators:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': title},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, ranges[1]], 'color': "red"},
                                {'range': [ranges[1], ranges[2]], 'color': "yellow"},
                                {'range': [ranges[2], 100], 'color': "green"}
                            ]
                        }
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title_text="Executive Dashboard - AI Project Status",
                title_x=0.5,
                height=600
            )
            
            output_path = os.path.join(self.output_dir, "executive_dashboard.html")
            fig.write_html(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Executive dashboard creation failed: {e}")
            return None
    
    def _create_interactive_dashboard(self, data_results: Dict[str, Any],
                                     feature_results: Dict[str, Any],
                                     model_results: Dict[str, Any]) -> Optional[str]:
        """Create comprehensive interactive dashboard"""
        try:
            # This would create a more complex interactive dashboard
            # For now, return the executive dashboard
            return self._create_executive_dashboard(data_results, feature_results, model_results)
            
        except Exception as e:
            logger.error(f"Interactive dashboard creation failed: {e}")
            return None
    
    def _create_appendices(self, data_results: Dict[str, Any],
                          feature_results: Dict[str, Any],
                          model_results: Dict[str, Any]) -> List[str]:
        """Create detailed appendices"""
        appendices = []
        
