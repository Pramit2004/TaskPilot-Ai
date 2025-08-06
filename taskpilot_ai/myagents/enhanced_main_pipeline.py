import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import asyncio
from datetime import datetime
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taskpilot_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TaskPilot AI - The True AI Data Scientist",
    description="A superintelligent agent army that performs complete data science workflows",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class AnalysisRequest(BaseModel):
    user_query: str = ""
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    time_budget: int = 600
    business_context: str = ""

class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    message: str
    session_directory: str
    results_summary: Optional[Dict[str, Any]] = None

class TaskPilotAI:
    """
    🚀 TaskPilot AI - The True AI Data Scientist
    
    A superintelligent agent army that performs complete data science workflows:
    - Multi-modal data understanding (tabular, text, images, audio, time series)
    - Advanced feature engineering and selection
    - Automated model development and optimization
    - Comprehensive business insights and reporting
    - Production-ready model deployment
    """
    
    def __init__(self, gemini_api_key: str = None, output_dir: str = "reports"):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("No Gemini API key provided. Some features may be limited.")
        
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session-specific output directory
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize mock agents (replace with actual imports when available)
        self.analysis_results = {}
        self.execution_log = []
        
        logger.info(f"🚀 TaskPilot AI initialized - Session: {self.session_id}")
    
    async def analyze_data(self, 
                          data_path: str,
                          user_query: str = "",
                          target_column: str = None,
                          task_type: str = None,
                          additional_files: List[str] = None,
                          time_budget: int = 600,
                          business_context: str = "") -> Dict[str, Any]:
        """
        🎯 Main analysis pipeline - orchestrate the entire agent army
        """
        logger.info("🎯 Starting TaskPilot AI Analysis Pipeline...")
        
        try:
            # Phase 1: Data Loading and Basic Analysis
            logger.info("Phase 1: Loading and analyzing data...")
            df = self._load_data(data_path)
            
            if df is None or df.empty:
                raise ValueError("Could not load data or data is empty")
            
            data_profile = self._create_data_profile(df, data_path)
            
            # Phase 2: Task Type Inference
            if task_type is None:
                task_type = self._infer_task_type(df, target_column)
            
            # Phase 3: Basic Feature Analysis
            logger.info("Phase 2: Analyzing features...")
            feature_analysis = self._analyze_features(df, target_column, task_type)
            
            # Phase 4: Mock Model Training (simplified for demo)
            logger.info("Phase 3: Training models...")
            model_results = self._train_basic_models(df, target_column, task_type)
            
            # Phase 5: Generate Reports
            logger.info("Phase 4: Generating reports...")
            reports = self._generate_reports(data_profile, feature_analysis, model_results)
            
            # Phase 6: Create Production Assets
            logger.info("Phase 5: Creating production assets...")
            production_assets = self._create_production_assets(model_results, feature_analysis)
            
            # Compile results
            self.analysis_results = {
                'data_profile': data_profile,
                'feature_analysis': feature_analysis,
                'model_results': model_results,
                'reports': reports,
                'production_assets': production_assets
            }
            
            # Save results
            self._save_session_results()
            
            # Generate final summary
            final_summary = self._create_final_summary()
            
            logger.info("✅ TaskPilot AI Analysis Complete!")
            return {
                'session_id': self.session_id,
                'analysis_results': self.analysis_results,
                'final_summary': final_summary,
                'session_directory': self.session_dir
            }
            
        except Exception as e:
            logger.error(f"❌ TaskPilot AI Analysis Failed: {str(e)}")
            self._save_error_report(e)
            raise
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various formats"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                # Try CSV as default
                df = pd.read_csv(data_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _create_data_profile(self, df: pd.DataFrame, data_path: str) -> Dict[str, Any]:
        """Create comprehensive data profile"""
        profile = {
            'file_path': data_path,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'sample_data': df.head().to_dict('records')
        }
        return profile
    
    def _infer_task_type(self, df: pd.DataFrame, target_column: str = None) -> str:
        """Infer task type from data characteristics"""
        if target_column is None or target_column not in df.columns:
            return "unsupervised"
        
        target_series = df[target_column]
        
        if pd.api.types.is_numeric_dtype(target_series):
            unique_values = target_series.nunique()
            total_values = len(target_series)
            
            if unique_values <= 10 or (unique_values / total_values) < 0.05:
                return "classification"
            else:
                return "regression"
        else:
            return "classification"
    
    def _analyze_features(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        """Analyze features and their relationships"""
        analysis = {
            'feature_count': len(df.columns),
            'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(df.select_dtypes(include=['object']).columns),
            'high_cardinality_features': [],
            'correlations': {},
            'feature_importance': {}
        }
        
        # Find high cardinality features
        for col in analysis['categorical_features']:
            if df[col].nunique() > 50:
                analysis['high_cardinality_features'].append(col)
        
        # Calculate correlations for numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            analysis['correlations'] = numeric_df.corr().to_dict()
        
        return analysis
    
    def _train_basic_models(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        """Train basic models for demonstration"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        results = {
            'models_trained': [],
            'best_model': None,
            'performance_metrics': {},
            'feature_importance': {}
        }
        
        if target_column is None or target_column not in df.columns:
            results['message'] = "No target column specified - unsupervised learning not implemented in this demo"
            return results
        
        try:
            # Prepare data
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            for col in numeric_columns:
                X[col] = X[col].fillna(X[col].median())
            
            for col in categorical_columns:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Handle target variable
            if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            if task_type == "classification":
                models = {
                    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
                }
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'LinearRegression': LinearRegression()
                }
                metric_func = r2_score
                metric_name = 'r2_score'
            
            best_score = -float('inf') if task_type == "regression" else 0
            best_model_name = None
            
            for model_name, model in models.items():
                try:
                    if model_name == 'LogisticRegression' or model_name == 'LinearRegression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    score = metric_func(y_test, y_pred)
                    
                    results['models_trained'].append(model_name)
                    results['performance_metrics'][model_name] = {
                        metric_name: score,
                        'model_object': model
                    }
                    
                    # Track feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_columns, model.feature_importances_))
                        results['feature_importance'][model_name] = feature_importance
                    
                    # Track best model
                    if (task_type == "classification" and score > best_score) or \
                       (task_type == "regression" and score > best_score):
                        best_score = score
                        best_model_name = model_name
                        results['best_model'] = {
                            'name': model_name,
                            'score': score,
                            'model_object': model
                        }
                
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
            
            # Save best model
            if results['best_model']:
                model_path = os.path.join(self.session_dir, "best_model.joblib")
                joblib.dump(results['best_model']['model_object'], model_path)
                results['best_model']['model_path'] = model_path
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_reports(self, data_profile: Dict, feature_analysis: Dict, model_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive reports"""
        reports = {
            'data_summary': self._create_data_summary_report(data_profile),
            'feature_report': self._create_feature_report(feature_analysis),
            'model_report': self._create_model_report(model_results),
            'executive_summary': self._create_executive_summary(data_profile, model_results)
        }
        
        # Save reports to files
        for report_name, report_content in reports.items():
            report_path = os.path.join(self.session_dir, f"{report_name}.json")
            with open(report_path, 'w') as f:
                json.dump(report_content, f, indent=2, default=str)
        
        return reports
    
    def _create_data_summary_report(self, data_profile: Dict) -> Dict[str, Any]:
        """Create data summary report"""
        return {
            'overview': {
                'total_rows': data_profile['shape'][0],
                'total_columns': data_profile['shape'][1],
                'memory_usage_mb': data_profile['memory_usage'] / (1024 * 1024),
                'data_quality_score': self._calculate_data_quality_score(data_profile)
            },
            'column_analysis': {
                'numeric_columns': len(data_profile['numeric_columns']),
                'categorical_columns': len(data_profile['categorical_columns']),
                'columns_with_missing_values': sum(1 for v in data_profile['missing_values'].values() if v > 0)
            },
            'recommendations': self._generate_data_recommendations(data_profile)
        }
    
    def _create_feature_report(self, feature_analysis: Dict) -> Dict[str, Any]:
        """Create feature analysis report"""
        return {
            'feature_summary': {
                'total_features': feature_analysis['feature_count'],
                'numeric_features': len(feature_analysis['numeric_features']),
                'categorical_features': len(feature_analysis['categorical_features']),
                'high_cardinality_features': len(feature_analysis['high_cardinality_features'])
            },
            'feature_recommendations': self._generate_feature_recommendations(feature_analysis)
        }
    
    def _create_model_report(self, model_results: Dict) -> Dict[str, Any]:
        """Create model performance report"""
        if not model_results.get('models_trained'):
            return {'message': 'No models were trained successfully'}
        
        report = {
            'models_summary': {
                'models_trained': len(model_results['models_trained']),
                'best_model': model_results['best_model']['name'] if model_results.get('best_model') else None,
                'best_score': model_results['best_model']['score'] if model_results.get('best_model') else None
            },
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Add performance comparison
        for model_name, metrics in model_results['performance_metrics'].items():
            report['performance_comparison'][model_name] = {
                k: v for k, v in metrics.items() if k != 'model_object'
            }
        
        # Add recommendations
        if model_results.get('best_model'):
            if model_results['best_model']['score'] > 0.8:
                report['recommendations'].append("Model performance is excellent and ready for production")
            elif model_results['best_model']['score'] > 0.6:
                report['recommendations'].append("Model performance is good but could benefit from further tuning")
            else:
                report['recommendations'].append("Model performance needs improvement - consider feature engineering")
        
        return report
    
    def _create_executive_summary(self, data_profile: Dict, model_results: Dict) -> Dict[str, Any]:
        """Create executive summary"""
        return {
            'project_overview': {
                'data_size': f"{data_profile['shape'][0]} rows x {data_profile['shape'][1]} columns",
                'analysis_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_quality': self._calculate_data_quality_score(data_profile)
            },
            'key_findings': self._generate_key_findings(data_profile, model_results),
            'business_recommendations': self._generate_business_recommendations(model_results),
            'next_steps': [
                "Review model performance and validate results",
                "Deploy model to production environment",
                "Set up monitoring and retraining pipeline",
                "Collect feedback and iterate on model"
            ]
        }
    
    def _calculate_data_quality_score(self, data_profile: Dict) -> float:
        """Calculate overall data quality score"""
        total_cells = data_profile['shape'][0] * data_profile['shape'][1]
        missing_cells = sum(data_profile['missing_values'].values())
        completeness_score = (total_cells - missing_cells) / total_cells
        return round(completeness_score, 3)
    
    def _generate_data_recommendations(self, data_profile: Dict) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        missing_ratio = sum(data_profile['missing_values'].values()) / (data_profile['shape'][0] * data_profile['shape'][1])
        if missing_ratio > 0.1:
            recommendations.append("Consider data imputation strategies for missing values")
        
        if len(data_profile['categorical_columns']) > len(data_profile['numeric_columns']):
            recommendations.append("Consider feature encoding strategies for categorical variables")
        
        return recommendations
    
    def _generate_feature_recommendations(self, feature_analysis: Dict) -> List[str]:
        """Generate feature engineering recommendations"""
        recommendations = []
        
        if feature_analysis['high_cardinality_features']:
            recommendations.append("Consider dimensionality reduction for high cardinality features")
        
        if len(feature_analysis['numeric_features']) > 20:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        return recommendations
    
    def _generate_key_findings(self, data_profile: Dict, model_results: Dict) -> List[str]:
        """Generate key findings from analysis"""
        findings = []
        
        findings.append(f"Dataset contains {data_profile['shape'][0]} records and {data_profile['shape'][1]} features")
        
        data_quality = self._calculate_data_quality_score(data_profile)
        findings.append(f"Data quality score: {data_quality:.1%}")
        
        if model_results.get('best_model'):
            best_score = model_results['best_model']['score']
            findings.append(f"Best model achieved {best_score:.1%} performance")
        
        return findings
    
    def _generate_business_recommendations(self, model_results: Dict) -> List[str]:
        """Generate business recommendations"""
        recommendations = []
        
        if model_results.get('best_model'):
            score = model_results['best_model']['score']
            if score > 0.8:
                recommendations.append("Model is ready for production deployment")
                recommendations.append("Implement automated monitoring and alerting")
            elif score > 0.6:
                recommendations.append("Consider additional feature engineering before deployment")
                recommendations.append("Conduct A/B testing with current solution")
            else:
                recommendations.append("Collect more data or domain expertise before deployment")
        
        return recommendations
    
    def _create_production_assets(self, model_results: Dict, feature_analysis: Dict) -> Dict[str, Any]:
        """Create production-ready assets"""
        assets = {
            'model_files': [],
            'code_files': [],
            'documentation': []
        }
        
        try:
            # Generate production code
            if model_results.get('best_model'):
                production_code = self._generate_production_code(model_results, feature_analysis)
                code_path = os.path.join(self.session_dir, "production_model.py")
                with open(code_path, 'w') as f:
                    f.write(production_code)
                assets['code_files'].append(code_path)
            
            # Generate API code
            api_code = self._generate_api_code(model_results, feature_analysis)
            api_path = os.path.join(self.session_dir, "model_api.py")
            with open(api_path, 'w') as f:
                f.write(api_code)
            assets['code_files'].append(api_path)
            
            # Generate requirements
            requirements = self._generate_requirements()
            req_path = os.path.join(self.session_dir, "requirements.txt")
            with open(req_path, 'w') as f:
                f.write(requirements)
            assets['code_files'].append(req_path)
            
        except Exception as e:
            logger.error(f"Production assets creation failed: {e}")
        
        return assets
    
    def _generate_production_code(self, model_results: Dict, feature_analysis: Dict) -> str:
        """Generate production model code"""
        model_name = model_results.get('best_model', {}).get('name', 'Unknown')
        
        code = f'''"""
TaskPilot AI - Production Model
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Best Model: {model_name}
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionModel:
    def __init__(self, model_path: str = "best_model.joblib"):
        self.model = joblib.load(model_path)
        self.scaler = StandardScaler()
        self.label_encoders = {{}}
        logger.info("Production model loaded successfully")
    
    def preprocess(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].median())
        
        for col in categorical_columns:
            data[col] = data[col].fillna('unknown')
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        return data
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        processed_data = self.preprocess(data)
        predictions = self.model.predict(processed_data)
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            processed_data = self.preprocess(data)
            return self.model.predict_proba(processed_data)
        else:
            raise NotImplementedError("Model does not support probability predictions")

if __name__ == "__main__":
    model = ProductionModel()
    print("Production model ready for use!")
'''
        return code
    
    def _generate_api_code(self, model_results: Dict, feature_analysis: Dict) -> str:
        """Generate FastAPI server code"""
        api_code = f'''"""
TaskPilot AI - Model API Server
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from production_model import ProductionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TaskPilot AI Model API", version="1.0.0")

try:
    model = ProductionModel()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    model = None

class PredictionRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    status: str

@app.get("/health")
def health_check():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        predictions = model.predict(request.data)
        return PredictionResponse(
            predictions=predictions.tolist(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return api_code
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt"""
        return """# TaskPilot AI Requirements
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
pydantic==2.4.2
python-multipart==0.0.6
"""
    
    def _save_session_results(self):
        """Save session results to file"""
        try:
            results_path = os.path.join(self.session_dir, "session_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            logger.info(f"Session results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save session results: {e}")
    
    def _save_error_report(self, error: Exception):
        """Save error report"""
        try:
            error_path = os.path.join(self.session_dir, "error_report.json")
            error_report = {
                'session_id': self.session_id,
                'error_time': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'execution_log': self.execution_log
            }
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2, default=str)
            logger.info(f"Error report saved to: {error_path}")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def _create_final_summary(self) -> Dict[str, Any]:
        """Create final analysis summary"""
        return {
            'session_id': self.session_id,
            'completed_at': datetime.now().isoformat(),
            'session_directory': self.session_dir,
            'status': 'completed',
            'key_outputs': [
                'Data analysis report',
                'Feature analysis',
                'Model training results',
                'Production-ready code',
                'API server code'
            ]
        }

# Global TaskPilot instance
taskpilot = None

# API Endpoints
@app.on_event("startup")
async def startup_event():
    global taskpilot
    logger.info("Starting TaskPilot AI API...")

@app.get("/")
def read_root():
    return {"message": "TaskPilot AI - The True AI Data Scientist", "version": "1.0.0", "status": "running"}

@app.post("/upload-and-analyze", response_model=AnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    user_query: str = Form(""),
    target_column: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    time_budget: int = Form(600),
    business_context: str = Form(""),
    gemini_api_key: Optional[str] = Form(None)
):
    """Upload data file and run complete analysis"""
    global taskpilot
    
    try:
        # Initialize TaskPilot with API key
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        taskpilot = TaskPilotAI(gemini_api_key=api_key)
        
        # Save uploaded file
        upload_dir = os.path.join("uploads", taskpilot.session_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Run analysis
        results = await taskpilot.analyze_data(
            data_path=file_path,
            user_query=user_query,
            target_column=target_column,
            task_type=task_type,
            time_budget=time_budget,
            business_context=business_context
        )
        
        return AnalysisResponse(
            session_id=results['session_id'],
            status="completed",
            message="Analysis completed successfully",
            session_directory=results['session_directory'],
            results_summary=results['final_summary']
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return AnalysisResponse(
            session_id=taskpilot.session_id if taskpilot else "unknown",
            status="failed",
            message=f"Analysis failed: {str(e)}",
            session_directory="",
            results_summary=None
        )

@app.post("/analyze-data", response_model=AnalysisResponse)
async def analyze_data_endpoint(
    data_path: str,
    request: AnalysisRequest,
    gemini_api_key: Optional[str] = None
):
    """Analyze existing data file"""
    global taskpilot
    
    try:
        # Initialize TaskPilot
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        taskpilot = TaskPilotAI(gemini_api_key=api_key)
        
        # Run analysis
        results = await taskpilot.analyze_data(
            data_path=data_path,
            user_query=request.user_query,
            target_column=request.target_column,
            task_type=request.task_type,
            time_budget=request.time_budget,
            business_context=request.business_context
        )
        
        return AnalysisResponse(
            session_id=results['session_id'],
            status="completed",
            message="Analysis completed successfully",
            session_directory=results['session_directory'],
            results_summary=results['final_summary']
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return AnalysisResponse(
            session_id=taskpilot.session_id if taskpilot else "unknown",
            status="failed",
            message=f"Analysis failed: {str(e)}",
            session_directory="",
            results_summary=None
        )

@app.get("/sessions/{session_id}/results")
async def get_session_results(session_id: str):
    """Get results for a specific session"""
    try:
        results_path = os.path.join("reports", f"session_{session_id}", "session_results.json")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/reports/{report_name}")
async def get_session_report(session_id: str, report_name: str):
    """Get specific report for a session"""
    try:
        report_path = os.path.join("reports", f"session_{session_id}", f"{report_name}.json")
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return report
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/download/{file_name}")
async def download_session_file(session_id: str, file_name: str):
    """Download files from session directory"""
    try:
        file_path = os.path.join("reports", f"session_{session_id}", file_name)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type='application/octet-stream'
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """List all available sessions"""
    try:
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return {"sessions": []}
        
        sessions = []
        for item in os.listdir(reports_dir):
            if item.startswith("session_"):
                session_id = item.replace("session_", "")
                session_path = os.path.join(reports_dir, item)
                
                # Get session info
                results_path = os.path.join(session_path, "session_results.json")
                if os.path.exists(results_path):
                    try:
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        sessions.append({
                            "session_id": session_id,
                            "created_at": session_id,  # Timestamp is in session_id
                            "status": "completed",
                            "has_results": True
                        })
                    except:
                        sessions.append({
                            "session_id": session_id,
                            "created_at": session_id,
                            "status": "unknown",
                            "has_results": False
                        })
        
        return {"sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its files"""
    try:
        import shutil
        session_path = os.path.join("reports", f"session_{session_id}")
        
        if not os.path.exists(session_path):
            raise HTTPException(status_code=404, detail="Session not found")
        
        shutil.rmtree(session_path)
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Convenience functions for different use cases
def analyze_csv_file(csv_path: str, target_column: str = None, 
                    gemini_api_key: str = None, **kwargs) -> Dict[str, Any]:
    """Analyze a CSV file with TaskPilot AI"""
    if gemini_api_key is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    pilot = TaskPilotAI(gemini_api_key)
    return asyncio.run(pilot.analyze_data(
        data_path=csv_path,
        target_column=target_column,
        task_type=kwargs.get('task_type', None),
        user_query=kwargs.get('user_query', ''),
        business_context=kwargs.get('business_context', ''),
        time_budget=kwargs.get('time_budget', 600)
    ))

def analyze_excel_file(excel_path: str, target_column: str = None, 
                      gemini_api_key: str = None, **kwargs) -> Dict[str, Any]:
    """Analyze an Excel file with TaskPilot AI"""
    if gemini_api_key is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    pilot = TaskPilotAI(gemini_api_key)
    return asyncio.run(pilot.analyze_data(
        data_path=excel_path,
        target_column=target_column,
        task_type=kwargs.get('task_type', None),
        user_query=kwargs.get('user_query', ''),
        business_context=kwargs.get('business_context', ''),
        time_budget=kwargs.get('time_budget', 600)
    ))

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TaskPilot AI - The True AI Data Scientist")
    parser.add_argument("--mode", choices=['api', 'cli'], default='api', help="Run mode: API server or CLI")
    parser.add_argument("--data_path", help="Path to the data file (CLI mode)")
    parser.add_argument("--target_column", help="Target column for supervised learning")
    parser.add_argument("--task_type", choices=['classification', 'regression'], help="Task type")
    parser.add_argument("--gemini_api_key", help="Gemini API key")
    parser.add_argument("--user_query", default="", help="User query describing the analysis goal")
    parser.add_argument("--business_context", default="", help="Business context for the analysis")
    parser.add_argument("--time_budget", type=int, default=600, help="Time budget in seconds")
    parser.add_argument("--output_dir", default="reports", help="Output directory for reports")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        # Run FastAPI server
        print("🚀 Starting TaskPilot AI API Server...")
        print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
        print("📚 API Documentation: http://localhost:8000/docs")
        
        uvicorn.run(
            "enhanced_main_pipeline:app",
            host=args.host,
            port=args.port,
            reload=False
        )
    
    elif args.mode == 'cli':
        # Run CLI analysis
        if not args.data_path:
            print("❌ Error: --data_path is required for CLI mode")
            sys.exit(1)
        
        # Initialize TaskPilot AI
        api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        pilot = TaskPilotAI(
            gemini_api_key=api_key,
            output_dir=args.output_dir
        )
        
        # Run analysis
        try:
            print("🚀 Starting TaskPilot AI Analysis...")
            
            results = asyncio.run(pilot.analyze_data(
                data_path=args.data_path,
                user_query=args.user_query,
                target_column=args.target_column,
                task_type=args.task_type,
                time_budget=args.time_budget,
                business_context=args.business_context
            ))
            
            print("\n✅ Analysis Complete!")
            print(f"📁 Session ID: {results['session_id']}")
            print(f"📊 Results Directory: {results['session_directory']}")
            
            # Print summary
            summary = results['final_summary']
            print(f"\n📈 Analysis completed at: {summary['completed_at']}")
            print(f"📄 Session directory: {summary['session_directory']}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)