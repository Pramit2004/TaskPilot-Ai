import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from PIL import Image, ImageStat
import cv2
import librosa
import nltk
from textblob import TextBlob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataInsight:
    insight_type: str
    description: str
    importance: str  # "high", "medium", "low"
    actionable_recommendation: str
    evidence: Dict[str, Any]

@dataclass
class DataQualityReport:
    overall_score: float
    quality_issues: List[str]
    data_completeness: float
    consistency_score: float
    uniqueness_score: float
    validity_score: float
    recommendations: List[str]

@dataclass
class DataUnderstandingResult:
    data_profile: Dict[str, Any]
    quality_report: DataQualityReport
    insights: List[DataInsight]
    domain_patterns: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    data_story: str
    visualizations: List[str]

class DataDetectiveAgent:
    """
    🔍 Data Detective Agent - The Sherlock Holmes of Data
    
    This agent performs deep investigation and understanding of any data type:
    1. Multi-modal data profiling and analysis
    2. Advanced data quality assessment
    3. Pattern and anomaly detection
    4. Domain-specific insight generation
    5. Automated data storytelling
    6. Cross-modal relationship discovery
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "reports/detective"):
        self.gemini_api_key = gemini_api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass
            
        self.investigation_history = []
        
    def investigate_data(self, data_path: str, target_column: str = None, 
                        additional_files: List[str] = None, 
                        user_context: str = "") -> DataUnderstandingResult:
        """
        🔍 Conduct comprehensive data investigation across all modalities
        """
        logger.info("🔍 Data Detective: Beginning comprehensive investigation...")
        
        # Step 1: Profile the data
        data_profile = self._profile_data_comprehensively(data_path, additional_files)
        
        # Step 2: Assess data quality
        quality_report = self._assess_data_quality_advanced(data_path, data_profile)
        
        # Step 3: Generate deep insights
        insights = self._generate_deep_insights(data_path, data_profile, target_column, user_context)
        
        # Step 4: Detect domain patterns
        domain_patterns = self._detect_domain_patterns(data_path, data_profile, user_context)
        
        # Step 5: Find anomalies
        anomalies = self._detect_anomalies_advanced(data_path, data_profile)
        
        # Step 6: Calculate feature importance
        feature_importance = self._calculate_feature_importance(data_path, target_column, data_profile)
        
        # Step 7: Create data story
        data_story = self._create_data_story(data_profile, insights, quality_report, user_context)
        
        # Step 8: Generate visualizations
        visualizations = self._create_comprehensive_visualizations(data_path, data_profile, target_column)
        
        result = DataUnderstandingResult(
            data_profile=data_profile,
            quality_report=quality_report,
            insights=insights,
            domain_patterns=domain_patterns,
            anomalies=anomalies,
            feature_importance=feature_importance,
            data_story=data_story,
            visualizations=visualizations
        )
        
        # Save investigation report
        self._save_investigation_report(result)
        
        logger.info("✅ Data Detective: Investigation complete!")
        return result
    
    def _profile_data_comprehensively(self, data_path: str, additional_files: List[str] = None) -> Dict[str, Any]:
        """Comprehensive data profiling across all modalities"""
        profile = {}
        
        file_extension = os.path.splitext(data_path)[1].lower()
        
        if file_extension == '.csv':
            profile.update(self._profile_tabular_data(data_path))
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            profile.update(self._profile_image_data(data_path))
        elif file_extension in ['.wav', '.mp3', '.flac']:
            profile.update(self._profile_audio_data(data_path))
        elif file_extension in ['.txt', '.md']:
            profile.update(self._profile_text_data(data_path))
        
        # Handle additional files for multimodal analysis
        if additional_files:
            profile['multimodal_files'] = []
            for file_path in additional_files:
                file_profile = self._profile_single_file(file_path)
                profile['multimodal_files'].append(file_profile)
                
        profile['investigation_timestamp'] = pd.Timestamp.now()
        return profile
    
    def _profile_tabular_data(self, data_path: str) -> Dict[str, Any]:
        """Deep profiling of tabular data"""
        try:
            df = pd.read_csv(data_path)
            
            profile = {
                'data_type': 'tabular',
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'column_types': df.dtypes.to_dict(),
                'missing_data': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'unique_values': {col: df[col].nunique() for col in df.columns},
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
            }
            
            # Advanced statistics for numeric columns
            numeric_stats = {}
            for col in profile['numeric_columns']:
                stats = df[col].describe()
                numeric_stats[col] = {
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'outliers_iqr': self._count_outliers_iqr(df[col]),
                    'outliers_zscore': self._count_outliers_zscore(df[col]),
                    'distribution_type': self._detect_distribution_type(df[col])
                }
            profile['numeric_statistics'] = numeric_stats
            
            # Categorical analysis
            categorical_stats = {}
            for col in profile['categorical_columns']:
                categorical_stats[col] = {
                    'most_common': df[col].value_counts().head(5).to_dict(),
                    'entropy': self._calculate_entropy(df[col]),
                    'cardinality': df[col].nunique(),
                    'cardinality_ratio': df[col].nunique() / len(df)
                }
            profile['categorical_statistics'] = categorical_stats
            
            # Correlation analysis
            if len(profile['numeric_columns']) > 1:
                corr_matrix = df[profile['numeric_columns']].corr()
                profile['correlation_matrix'] = corr_matrix.to_dict()
                profile['high_correlations'] = self._find_high_correlations(corr_matrix)
                
            # Time series detection
            profile['is_time_series'] = self._detect_time_series_patterns(df)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling tabular data: {e}")
            return {'data_type': 'tabular', 'error': str(e)}
    
    def _profile_image_data(self, image_path: str) -> Dict[str, Any]:
        """Deep profiling of image data"""
        try:
            # PIL analysis
            with Image.open(image_path) as img:
                profile = {
                    'data_type': 'image',
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'aspect_ratio': img.size[0] / img.size[1],
                }
                
                # Color analysis
                if img.mode in ['RGB', 'RGBA']:
                    stat = ImageStat.Stat(img)
                    profile['color_statistics'] = {
                        'mean_rgb': stat.mean,
                        'std_rgb': stat.stddev,
                        'extrema': stat.extrema
                    }
                    
                    # Dominant colors
                    profile['dominant_colors'] = self._extract_dominant_colors(img)
                
            # OpenCV analysis
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                profile['opencv_analysis'] = {
                    'channels': cv_img.shape[2] if len(cv_img.shape) == 3 else 1,
                    'brightness': np.mean(cv_img),
                    'contrast': np.std(cv_img),
                    'sharpness': self._calculate_image_sharpness(cv_img)
                }
                
                # Edge detection
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                profile['edge_density'] = np.sum(edges > 0) / edges.size
                
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling image data: {e}")
            return {'data_type': 'image', 'error': str(e)}
    
    def _profile_audio_data(self, audio_path: str) -> Dict[str, Any]:
        """Deep profiling of audio data"""
        try:
            y, sr = librosa.load(audio_path)
            
            profile = {
                'data_type': 'audio',
                'sample_rate': sr,
                'duration': len(y) / sr,
                'channels': 1,  # librosa loads as mono by default
                'samples': len(y),
            }
            
            # Audio features
            profile['audio_features'] = {
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'mfcc_mean': librosa.feature.mfcc(y=y, sr=sr).mean(axis=1).tolist(),
                'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
            }
            
            # Silence detection
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)
            profile['silence_ratio'] = silence_ratio
            
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling audio data: {e}")
            return {'data_type': 'audio', 'error': str(e)}
    
    def _profile_text_data(self, text_path: str) -> Dict[str, Any]:
        """Deep profiling of text data"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Basic statistics
            words = text.split()
            sentences = text.split('.')
            
            profile = {
                'data_type': 'text',
                'character_count': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'paragraph_count': len(text.split('\n\n')),
                'avg_word_length': np.mean([len(word) for word in words]),
                'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences if sent.strip()]),
            }
            
            # Language detection and sentiment
            blob = TextBlob(text)
            profile['sentiment'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # Vocabulary analysis
            unique_words = set(word.lower() for word in words if word.isalpha())
            profile['vocabulary'] = {
                'unique_words': len(unique_words),
                'vocabulary_richness': len(unique_words) / len(words) if words else 0
            }
            
            # Most common words
            from collections import Counter
            word_freq = Counter(word.lower() for word in words if word.isalpha())
            profile['most_common_words'] = dict(word_freq.most_common(20))
            
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling text data: {e}")
            return {'data_type': 'text', 'error': str(e)}
    
    def _assess_data_quality_advanced(self, data_path: str, data_profile: Dict[str, Any]) -> DataQualityReport:
        """Advanced data quality assessment"""
        quality_issues = []
        recommendations = []
        
        data_type = data_profile.get('data_type', 'unknown')
        
        if data_type == 'tabular':
            df = pd.read_csv(data_path)
            
            # Completeness
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells)
            
            # Consistency
            consistency_score = self._assess_consistency(df)
            
            # Uniqueness
            uniqueness_score = self._assess_uniqueness(df)
            
            # Validity
            validity_score = self._assess_validity(df)
            
            # Identify quality issues
            if completeness < 0.95:
                quality_issues.append(f"Missing data: {missing_cells} cells ({100*(1-completeness):.1f}%)")
                recommendations.append("Consider imputation strategies or data collection improvements")
            
            if consistency_score < 0.8:
                quality_issues.append("Data inconsistencies detected")
                recommendations.append("Standardize data formats and validation rules")
            
            if uniqueness_score < 0.9:
                quality_issues.append("Duplicate records detected")
                recommendations.append("Implement deduplication process")
                
            overall_score = np.mean([completeness, consistency_score, uniqueness_score, validity_score])
            
        else:
            # For non-tabular data, use simplified quality assessment
            completeness = 1.0 if 'error' not in data_profile else 0.0
            consistency_score = 0.8
            uniqueness_score = 1.0
            validity_score = 0.8
            overall_score = np.mean([completeness, consistency_score, uniqueness_score, validity_score])
        
        return DataQualityReport(
            overall_score=overall_score,
            quality_issues=quality_issues,
            data_completeness=completeness,
            consistency_score=consistency_score,
            uniqueness_score=uniqueness_score,
            validity_score=validity_score,
            recommendations=recommendations
        )
    
    def _generate_deep_insights(self, data_path: str, data_profile: Dict[str, Any], 
                               target_column: str = None, user_context: str = "") -> List[DataInsight]:
        """Generate deep insights using AI and statistical analysis"""
        insights = []
        
        # Statistical insights
        insights.extend(self._generate_statistical_insights(data_profile, target_column))
        
        # Pattern insights
        insights.extend(self._generate_pattern_insights(data_path, data_profile))
        
        # AI-powered insights
        ai_insights = self._generate_ai_insights(data_profile, user_context)
        insights.extend(ai_insights)
        
        # Business insights
        insights.extend(self._generate_business_insights(data_profile, user_context))
        
        return insights
    
    def _generate_statistical_insights(self, data_profile: Dict[str, Any], target_column: str = None) -> List[DataInsight]:
        """Generate insights from statistical analysis"""
        insights = []
        
        if data_profile.get('data_type') == 'tabular':
            # Distribution insights
            numeric_stats = data_profile.get('numeric_statistics', {})
            for col, stats in numeric_stats.items():
                if stats['skewness'] > 1:
                    insights.append(DataInsight(
                        insight_type="distribution",
                        description=f"Column '{col}' is highly right-skewed (skewness: {stats['skewness']:.2f})",
                        importance="medium",
                        actionable_recommendation="Consider log transformation or outlier removal",
                        evidence={"skewness": stats['skewness'], "column": col}
                    ))
                
                if stats['outliers_iqr'] > len(data_profile.get('numeric_columns', [])) * 0.05:
                    insights.append(DataInsight(
                        insight_type="outliers",
                        description=f"Column '{col}' has {stats['outliers_iqr']} outliers",
                        importance="high",
                        actionable_recommendation="Investigate outliers - may indicate data quality issues or interesting patterns",
                        evidence={"outlier_count": stats['outliers_iqr'], "column": col}
                    ))
            
            # Correlation insights
            high_corrs = data_profile.get('high_correlations', [])
            if high_corrs:
                insights.append(DataInsight(
                    insight_type="correlation",
                    description=f"Found {len(high_corrs)} highly correlated feature pairs",
                    importance="high",
                    actionable_recommendation="Consider feature selection to reduce multicollinearity",
                    evidence={"high_correlations": high_corrs}
                ))
        
        return insights
    
    def _generate_ai_insights(self, data_profile: Dict[str, Any], user_context: str) -> List[DataInsight]:
        """Generate AI-powered insights"""
        try:
            prompt = f"""
            As a senior data scientist, analyze this data profile and provide 3-5 key insights:
            
            Data Profile: {json.dumps(data_profile, default=str, indent=2)}
            User Context: {user_context}
            
            For each insight, provide:
            1. Type (e.g., "quality", "pattern", "opportunity", "risk")
            2. Description (what you observed)
            3. Importance (high/medium/low)
            4. Actionable recommendation
            
            Focus on insights that would help a data scientist make better modeling decisions.
            Format as JSON list with keys: type, description, importance, recommendation
            """
            
            response = self.llm.invoke(prompt)
            
            # Parse AI response
            insights = []
            try:
                ai_insights = json.loads(response.content)
                for insight_data in ai_insights:
                    insights.append(DataInsight(
                        insight_type=insight_data.get("type", "general"),
                        description=insight_data.get("description", ""),
                        importance=insight_data.get("importance", "medium"),
                        actionable_recommendation=insight_data.get("recommendation", ""),
                        evidence={"source": "ai_analysis"}
                    ))
            except json.JSONDecodeError:
                # Fallback: parse text response
                insights.append(DataInsight(
                    insight_type="ai_analysis",
                    description="AI analysis completed",
                    importance="medium",
                    actionable_recommendation="Review AI insights in the full report",
                    evidence={"ai_response": response.content[:500]}
                ))
                
            return insights
            
        except Exception as e:
            logger.warning(f"AI insight generation failed: {e}")
            return []
    
    def _create_data_story(self, data_profile: Dict[str, Any], insights: List[DataInsight], 
                          quality_report: DataQualityReport, user_context: str) -> str:
        """Create compelling data story using AI"""
        try:
            prompt = f"""
            Create a compelling data story (executive summary) based on this analysis:
            
            Data Profile: {data_profile.get('data_type', 'unknown')} data with {data_profile.get('shape', 'unknown')} shape
            Quality Score: {quality_report.overall_score:.2f}/1.0
            Key Insights: {[insight.description for insight in insights[:5]]}
            User Context: {user_context}
            
            Write a 3-paragraph executive summary that:
            1. Describes what the data represents and its characteristics
            2. Highlights the most important findings and opportunities
            3. Provides clear next steps and recommendations
            
            Make it engaging and business-focused.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"Data story generation failed: {e}")
            return "Data investigation completed. See detailed insights in the report."
    
    def _create_comprehensive_visualizations(self, data_path: str, data_profile: Dict[str, Any], 
                                           target_column: str = None) -> List[str]:
        """Create comprehensive visualizations"""
        viz_files = []
        
        try:
            if data_profile.get('data_type') == 'tabular':
                viz_files.extend(self._create_tabular_visualizations(data_path, target_column))
            elif data_profile.get('data_type') == 'image':
                viz_files.extend(self._create_image_visualizations(data_path, data_profile))
            elif data_profile.get('data_type') == 'audio':
                viz_files.extend(self._create_audio_visualizations(data_path, data_profile))
                
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            
        return viz_files
    
    def _create_tabular_visualizations(self, data_path: str, target_column: str = None) -> List[str]:
        """Create visualizations for tabular data"""
        viz_files = []
        df = pd.read_csv(data_path)
        
        # 1. Data Overview Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Missing Data Heatmap', 'Correlation Matrix', 
                          'Data Types Distribution', 'Target Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Missing data heatmap
        missing_data = df.isnull().astype(int)
        if not missing_data.empty:
            fig.add_trace(
                go.Heatmap(z=missing_data.values, colorscale='Reds'),
                row=1, col=1
            )
        
        # Correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, 
                          colorscale='RdBu', zmid=0),
                row=1, col=2
            )
        
        fig.update_layout(height=800, title="Data Overview Dashboard")
        
        viz_file = os.path.join(self.output_dir, "data_overview_dashboard.html")
        fig.write_html(viz_file)
        viz_files.append(viz_file)
        
        return viz_files
    
    # Helper methods
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
        return len(outliers)
    
    def _count_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> int:
        """Count outliers using Z-score method"""
        z_scores = np.abs((series - series.mean()) / series.std())
        return len(series[z_scores > threshold])
    
    def _detect_distribution_type(self, series: pd.Series) -> str:
        """Detect the distribution type of a numeric series"""
        from scipy import stats
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 50:
            return "insufficient_data"
        
        # Test for normality
        _, p_normal = stats.normaltest(clean_series)
        if p_normal > 0.05:
            return "normal"
        
        # Check skewness
        skewness = clean_series.skew()
        if skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        
        # Check for uniform distribution
        _, p_uniform = stats.kstest(clean_series, 'uniform')
        if p_uniform > 0.05:
            return "uniform"
        
        return "unknown"
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a categorical series"""
        value_counts = series.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
        return entropy
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find highly correlated feature pairs"""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return high_corr_pairs
    
    def _detect_time_series_patterns(self, df: pd.DataFrame) -> bool:
        """Detect if data has time series characteristics"""
        # Look for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            return True
        
        # Look for date-like string columns
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().head(10)
            try:
                pd.to_datetime(sample_values)
                return True
            except:
                continue
        
        return False
    
    def _save_investigation_report(self, result: DataUnderstandingResult):
        """Save comprehensive investigation report"""
        report_path = os.path.join(self.output_dir, "investigation_report.json")
        
        # Convert result to JSON-serializable format
        report_data = {
            'data_profile': result.data_profile,
            'quality_report': {
                'overall_score': result.quality_report.overall_score,
                'quality_issues': result.quality_report.quality_issues,
                'recommendations': result.quality_report.recommendations
            },
            'insights': [
                {
                    'type': insight.insight_type,
                    'description': insight.description,
                    'importance': insight.importance,
                    'recommendation': insight.actionable_recommendation
                }
                for insight in result.insights
            ],
            'data_story': result.data_story,
            'visualizations': result.visualizations
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Investigation report saved to: {report_path}")