import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import io
import base64
from PIL import Image
import mimetypes

logger = logging.getLogger(__name__)

class DataType(Enum):
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"
    MIXED = "mixed"

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH_GRADE = "research_grade"

@dataclass
class DataProfile:
    data_type: DataType
    shape: Tuple[int, int]
    file_types: List[str]
    complexity: TaskComplexity
    domain: str
    business_context: str
    data_quality_score: float
    unique_challenges: List[str]
    recommended_approaches: List[str]

@dataclass
class AnalysisStrategy:
    primary_approach: str
    agents_to_deploy: List[str]
    analysis_pipeline: List[str]
    expected_outcomes: List[str]
    success_metrics: List[str]
    estimated_complexity: TaskComplexity
    risk_factors: List[str]
    alternative_strategies: List[str]

class MasterStrategistAgent:
    """
    🧠 The Master Strategist Agent - The Brain of TaskPilot AI
    
    This agent orchestrates the entire data science workflow by:
    1. Analyzing data holistically across all modalities
    2. Understanding business context and domain requirements
    3. Designing optimal analysis strategies
    4. Coordinating specialized agents
    5. Making high-level strategic decisions
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "reports/strategy"):
        self.gemini_api_key = gemini_api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
        # Domain knowledge base
        self.domain_expertise = {
            "finance": ["risk_analysis", "fraud_detection", "algorithmic_trading", "credit_scoring"],
            "healthcare": ["diagnostic_prediction", "treatment_optimization", "epidemiology", "drug_discovery"],
            "retail": ["customer_segmentation", "demand_forecasting", "recommendation_systems", "price_optimization"],
            "manufacturing": ["predictive_maintenance", "quality_control", "supply_chain", "process_optimization"],
            "marketing": ["customer_lifetime_value", "churn_prediction", "campaign_optimization", "attribution_modeling"],
            "hr": ["talent_acquisition", "performance_prediction", "retention_analysis", "workforce_planning"],
            "transportation": ["route_optimization", "demand_prediction", "safety_analysis", "autonomous_systems"],
            "energy": ["demand_forecasting", "grid_optimization", "renewable_integration", "efficiency_analysis"]
        }
        
        self.analysis_history = []
        
    def analyze_data_holistically(self, data_path: str, user_query: str, 
                                 additional_files: List[str] = None) -> DataProfile:
        """
        🔍 Perform holistic data analysis to understand the complete data landscape
        """
        logger.info("🧠 Master Strategist: Beginning holistic data analysis...")
        
        # Detect data types and structure
        data_insights = self._detect_data_types(data_path, additional_files)
        
        # Understand business context
        business_context = self._analyze_business_context(user_query)
        
        # Assess data quality and challenges
        quality_assessment = self._assess_data_quality(data_path)
        
        # Determine domain and complexity
        domain = self._infer_domain(user_query, data_insights)
        complexity = self._assess_complexity(data_insights, business_context)
        
        profile = DataProfile(
            data_type=data_insights["primary_type"],
            shape=data_insights["shape"],
            file_types=data_insights["file_types"],
            complexity=complexity,
            domain=domain,
            business_context=business_context,
            data_quality_score=quality_assessment["score"],
            unique_challenges=quality_assessment["challenges"],
            recommended_approaches=self._get_domain_approaches(domain)
        )
        
        logger.info(f"📊 Data Profile Created: {profile.data_type.value} | {profile.domain} | {profile.complexity.value}")
        return profile
    
    def design_analysis_strategy(self, data_profile: DataProfile, user_query: str) -> AnalysisStrategy:
        """
        🎯 Design comprehensive analysis strategy based on data profile and user needs
        """
        logger.info("🎯 Master Strategist: Designing analysis strategy...")
        
        # Create strategy prompt
        strategy_prompt = self._create_strategy_prompt(data_profile, user_query)
        
        # Get AI-powered strategy recommendations
        ai_strategy = self._get_ai_strategy(strategy_prompt)
        
        # Combine AI insights with domain expertise
        final_strategy = self._synthesize_strategy(ai_strategy, data_profile)
        
        # Validate and refine strategy
        refined_strategy = self._refine_strategy(final_strategy, data_profile)
        
        logger.info(f"✅ Strategy Designed: {refined_strategy.primary_approach}")
        return refined_strategy
    
    def coordinate_agents(self, strategy: AnalysisStrategy) -> Dict[str, Any]:
        """
        🎭 Coordinate specialized agents based on the analysis strategy
        """
        logger.info("🎭 Master Strategist: Coordinating agent deployment...")
        
        agent_assignments = {}
        
        for agent_name in strategy.agents_to_deploy:
            agent_config = self._get_agent_configuration(agent_name, strategy)
            agent_assignments[agent_name] = agent_config
            
        coordination_plan = {
            "execution_order": self._determine_execution_order(strategy.agents_to_deploy),
            "agent_assignments": agent_assignments,
            "coordination_checkpoints": self._define_checkpoints(strategy),
            "success_criteria": strategy.success_metrics,
            "fallback_plans": self._create_fallback_plans(strategy)
        }
        
        return coordination_plan
    
    def _detect_data_types(self, data_path: str, additional_files: List[str] = None) -> Dict[str, Any]:
        """Detect and analyze data types across all provided files"""
        file_types = []
        total_size = 0
        primary_type = DataType.TABULAR
        
        # Analyze main file
        file_extension = os.path.splitext(data_path)[1].lower()
        file_types.append(file_extension)
        total_size += os.path.getsize(data_path) if os.path.exists(data_path) else 0
        
        # Determine primary data type
        if file_extension in ['.csv', '.xlsx', '.parquet', '.json']:
            primary_type = DataType.TABULAR
            try:
                df = pd.read_csv(data_path) if file_extension == '.csv' else pd.read_excel(data_path)
                shape = df.shape
                
                # Check for time series patterns
                date_columns = df.select_dtypes(include=['datetime', 'object']).columns
                for col in date_columns:
                    try:
                        pd.to_datetime(df[col])
                        primary_type = DataType.TIME_SERIES
                        break
                    except:
                        continue
                        
                # Check for text-heavy data
                text_ratio = sum(df.select_dtypes(include=['object']).count()) / len(df.columns)
                if text_ratio > 0.7:
                    primary_type = DataType.TEXT
                    
            except Exception as e:
                shape = (0, 0)
                logger.warning(f"Could not read file: {e}")
                
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            primary_type = DataType.IMAGE
            shape = self._get_image_dimensions(data_path)
            
        elif file_extension in ['.wav', '.mp3', '.flac', '.aac']:
            primary_type = DataType.AUDIO
            shape = self._get_audio_dimensions(data_path)
            
        elif file_extension in ['.txt', '.md', '.docx']:
            primary_type = DataType.TEXT
            shape = self._get_text_dimensions(data_path)
            
        # Check for multimodal data
        if additional_files:
            additional_types = set()
            for file_path in additional_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    additional_types.add(DataType.IMAGE)
                elif ext in ['.wav', '.mp3']:
                    additional_types.add(DataType.AUDIO)
                elif ext in ['.txt', '.md']:
                    additional_types.add(DataType.TEXT)
                    
            if len(additional_types) > 0:
                primary_type = DataType.MULTIMODAL
                
        return {
            "primary_type": primary_type,
            "file_types": file_types,
            "shape": shape,
            "total_size": total_size,
            "multimodal": len(set(file_types)) > 1
        }
    
    def _analyze_business_context(self, user_query: str) -> str:
        """Analyze user query to understand business context and objectives"""
        context_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Analyze this user query to understand the business context and objectives:
            
            Query: {query}
            
            Please identify:
            1. Business domain/industry
            2. Primary business objective
            3. Key stakeholders
            4. Success criteria
            5. Constraints and requirements
            
            Provide a concise business context summary.
            """
        )
        
        context_chain = LLMChain(llm=self.llm, prompt=context_prompt)
        return context_chain.run(query=user_query)
    
    def _assess_data_quality(self, data_path: str) -> Dict[str, Any]:
        """Assess data quality and identify potential challenges"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                
                # Calculate quality metrics
                missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                duplicate_ratio = df.duplicated().sum() / len(df)
                
                # Identify challenges
                challenges = []
                if missing_ratio > 0.1:
                    challenges.append("high_missing_data")
                if duplicate_ratio > 0.05:
                    challenges.append("duplicate_records")
                if df.shape[1] > 100:
                    challenges.append("high_dimensionality")
                if df.shape[0] < 1000:
                    challenges.append("small_sample_size")
                    
                # Calculate overall quality score
                quality_score = 1.0 - (missing_ratio * 0.4 + duplicate_ratio * 0.3)
                quality_score = max(0.1, min(1.0, quality_score))
                
                return {
                    "score": quality_score,
                    "challenges": challenges,
                    "missing_ratio": missing_ratio,
                    "duplicate_ratio": duplicate_ratio
                }
                
        except Exception as e:
            logger.warning(f"Could not assess data quality: {e}")
            
        return {"score": 0.5, "challenges": ["unknown_quality"], "missing_ratio": 0, "duplicate_ratio": 0}
    
    def _infer_domain(self, user_query: str, data_insights: Dict[str, Any]) -> str:
        """Infer business domain from query and data characteristics"""
        domain_keywords = {
            "finance": ["price", "stock", "financial", "investment", "trading", "credit", "loan", "bank"],
            "healthcare": ["patient", "medical", "diagnosis", "treatment", "health", "disease", "clinical"],
            "retail": ["customer", "sales", "product", "purchase", "order", "revenue", "commerce"],
            "manufacturing": ["production", "quality", "defect", "machine", "process", "assembly"],
            "marketing": ["campaign", "conversion", "engagement", "audience", "brand", "advertising"],
            "hr": ["employee", "performance", "recruitment", "talent", "workforce", "hiring"],
            "transportation": ["route", "delivery", "vehicle", "logistics", "shipping", "traffic"],
            "energy": ["power", "electricity", "consumption", "grid", "renewable", "efficiency"]
        }
        
        query_lower = user_query.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
            
        # Return domain with highest score, default to "general"
        best_domain = max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "general"
        return best_domain
    
    def _assess_complexity(self, data_insights: Dict[str, Any], business_context: str) -> TaskComplexity:
        """Assess overall task complexity"""
        complexity_score = 0
        
        # Data complexity factors
        if data_insights["primary_type"] == DataType.MULTIMODAL:
            complexity_score += 3
        elif data_insights["primary_type"] in [DataType.IMAGE, DataType.AUDIO]:
            complexity_score += 2
        elif data_insights["primary_type"] == DataType.TEXT:
            complexity_score += 1
            
        # Size complexity
        if data_insights["total_size"] > 1_000_000:  # > 1MB
            complexity_score += 1
            
        # Business complexity
        if any(word in business_context.lower() for word in ["predict", "forecast", "optimize", "recommend"]):
            complexity_score += 1
        if any(word in business_context.lower() for word in ["causal", "explainable", "fair", "bias"]):
            complexity_score += 2
            
        # Map score to complexity level
        if complexity_score <= 2:
            return TaskComplexity.SIMPLE
        elif complexity_score <= 4:
            return TaskComplexity.MODERATE
        elif complexity_score <= 6:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.RESEARCH_GRADE
    
    def _get_domain_approaches(self, domain: str) -> List[str]:
        """Get recommended approaches for specific domain"""
        return self.domain_expertise.get(domain, ["general_ml_pipeline"])
    
    def _create_strategy_prompt(self, data_profile: DataProfile, user_query: str) -> str:
        """Create comprehensive strategy prompt for AI"""
        return f"""
        As a world-class data science strategist, design an optimal analysis strategy for this project:
        
        DATA PROFILE:
        - Data Type: {data_profile.data_type.value}
        - Shape: {data_profile.shape}
        - Domain: {data_profile.domain}
        - Complexity: {data_profile.complexity.value}
        - Quality Score: {data_profile.data_quality_score:.2f}
        - Challenges: {', '.join(data_profile.unique_challenges)}
        
        USER QUERY: {user_query}
        
        BUSINESS CONTEXT: {data_profile.business_context}
        
        Please provide a comprehensive analysis strategy including:
        1. Primary analytical approach
        2. Specific agents/tools needed
        3. Step-by-step pipeline
        4. Expected outcomes
        5. Success metrics
        6. Risk factors
        7. Alternative strategies
        
        Consider advanced techniques like:
        - AutoML and hyperparameter optimization
        - Explainable AI and model interpretation
        - Bias detection and fairness analysis
        - Causal inference when appropriate
        - Ensemble methods and model stacking
        - Domain-specific best practices
        
        Format response as JSON with keys: primary_approach, agents_needed, pipeline_steps, expected_outcomes, success_metrics, risk_factors, alternatives
        """
    
    def _get_ai_strategy(self, prompt: str) -> Dict[str, Any]:
        """Get AI-powered strategy recommendations"""
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            strategy_text = response.content
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', strategy_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._parse_strategy_response(strategy_text)
                
        except Exception as e:
            logger.warning(f"AI strategy generation failed: {e}")
            return self._get_fallback_strategy()
    
    def _synthesize_strategy(self, ai_strategy: Dict[str, Any], data_profile: DataProfile) -> AnalysisStrategy:
        """Synthesize AI recommendations with domain expertise"""
        
        agents_mapping = {
            "data_detective": "DataDetectiveAgent",
            "feature_alchemist": "FeatureAlchemistAgent", 
            "model_maestro": "ModelMaestroAgent",
            "insight_oracle": "InsightOracleAgent",
            "report_artisan": "ReportArtisanAgent"
        }
        
        # Map AI recommendations to actual agents
        agents_to_deploy = []
        for agent_name in ai_strategy.get("agents_needed", []):
            mapped_agent = agents_mapping.get(agent_name.lower().replace(" ", "_"), agent_name)
            agents_to_deploy.append(mapped_agent)
        
        # Ensure essential agents are included
        essential_agents = ["DataDetectiveAgent", "ReportArtisanAgent"]
        for agent in essential_agents:
            if agent not in agents_to_deploy:
                agents_to_deploy.append(agent)
        
        return AnalysisStrategy(
            primary_approach=ai_strategy.get("primary_approach", "comprehensive_analysis"),
            agents_to_deploy=agents_to_deploy,
            analysis_pipeline=ai_strategy.get("pipeline_steps", []),
            expected_outcomes=ai_strategy.get("expected_outcomes", []),
            success_metrics=ai_strategy.get("success_metrics", []),
            estimated_complexity=data_profile.complexity,
            risk_factors=ai_strategy.get("risk_factors", []),
            alternative_strategies=ai_strategy.get("alternatives", [])
        )
    
    def _refine_strategy(self, strategy: AnalysisStrategy, data_profile: DataProfile) -> AnalysisStrategy:
        """Refine strategy based on constraints and best practices"""
        # Add data-type specific refinements
        if data_profile.data_type == DataType.IMAGE:
            if "computer_vision_pipeline" not in strategy.analysis_pipeline:
                strategy.analysis_pipeline.insert(0, "computer_vision_pipeline")
                
        elif data_profile.data_type == DataType.TEXT:
            if "nlp_preprocessing" not in strategy.analysis_pipeline:
                strategy.analysis_pipeline.insert(0, "nlp_preprocessing")
                
        elif data_profile.data_type == DataType.TIME_SERIES:
            if "time_series_analysis" not in strategy.analysis_pipeline:
                strategy.analysis_pipeline.insert(0, "time_series_analysis")
        
        return strategy
    
    def _get_agent_configuration(self, agent_name: str, strategy: AnalysisStrategy) -> Dict[str, Any]:
        """Get configuration for specific agent"""
        base_config = {
            "agent_name": agent_name,
            "priority": "high",
            "timeout": 300,
            "retry_attempts": 3
        }
        
        # Agent-specific configurations
        if agent_name == "ModelMaestroAgent":
            base_config.update({
                "model_types": self._get_recommended_models(strategy),
                "optimization_strategy": "bayesian",
                "cross_validation": "stratified_kfold"
            })
        elif agent_name == "FeatureAlchemistAgent":
            base_config.update({
                "feature_selection": True,
                "automated_engineering": True,
                "interaction_terms": True
            })
            
        return base_config
    
    def _determine_execution_order(self, agents: List[str]) -> List[str]:
        """Determine optimal execution order for agents"""
        order_priority = {
            "DataDetectiveAgent": 1,
            "FeatureAlchemistAgent": 2,
            "ModelMaestroAgent": 3,
            "InsightOracleAgent": 4,
            "ReportArtisanAgent": 5
        }
        
        return sorted(agents, key=lambda x: order_priority.get(x, 999))
    
    def _define_checkpoints(self, strategy: AnalysisStrategy) -> List[Dict[str, Any]]:
        """Define coordination checkpoints"""
        return [
            {"phase": "data_understanding", "criteria": ["data_quality_assessed", "features_identified"]},
            {"phase": "modeling", "criteria": ["baseline_established", "best_model_selected"]},
            {"phase": "validation", "criteria": ["performance_validated", "insights_generated"]},
            {"phase": "reporting", "criteria": ["report_generated", "recommendations_provided"]}
        ]
    
    def _create_fallback_plans(self, strategy: AnalysisStrategy) -> List[Dict[str, Any]]:
        """Create fallback plans for potential failures"""
        return [
            {
                "scenario": "model_training_failure",
                "action": "switch_to_simpler_models",
                "fallback_agents": ["ModelMaestroAgent"]
            },
            {
                "scenario": "data_quality_issues",
                "action": "enhanced_preprocessing",
                "fallback_agents": ["DataDetectiveAgent", "FeatureAlchemistAgent"]
            }
        ]
    
    def _get_recommended_models(self, strategy: AnalysisStrategy) -> List[str]:
        """Get recommended models based on strategy"""
        if "classification" in strategy.primary_approach.lower():
            return ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression"]
        elif "regression" in strategy.primary_approach.lower():
            return ["RandomForest", "XGBoost", "LightGBM", "LinearRegression"]
        else:
            return ["AutoML", "EnsembleVoting", "NeuralNetwork"]
    
    def _parse_strategy_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback parser for strategy response"""
        return {
            "primary_approach": "comprehensive_ml_analysis",
            "agents_needed": ["data_detective", "feature_alchemist", "model_maestro", "insight_oracle"],
            "pipeline_steps": ["data_exploration", "preprocessing", "feature_engineering", "modeling", "evaluation"],
            "expected_outcomes": ["trained_model", "performance_metrics", "insights"],
            "success_metrics": ["accuracy", "precision", "recall"],
            "risk_factors": ["data_quality", "overfitting"],
            "alternatives": ["ensemble_approach", "deep_learning"]
        }
    
    def _get_fallback_strategy(self) -> Dict[str, Any]:
        """Get fallback strategy when AI fails"""
        return self._parse_strategy_response("")
    
    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except:
            return (0, 0)
    
    def _get_audio_dimensions(self, audio_path: str) -> Tuple[int, int]:
        """Get audio file information"""
        try:
            import librosa
            y, sr = librosa.load(audio_path)
            return (len(y), sr)
        except:
            return (0, 0)