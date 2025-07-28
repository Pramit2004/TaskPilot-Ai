from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
import pandas as pd
from db import init_db, log_query, log_model_result
from agents.model_trainer_agent import ModelTrainerAgent
from agents.eda_agent import EDAAgent
from agents.data_preprocessing_agent import DataPreprocessorAgent
from sklearn.model_selection import train_test_split
import uuid
import logging

# Add logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_datasets"
EDA_PLOT_DIR = "reports/eda"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EDA_PLOT_DIR, exist_ok=True)

init_db()

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Handle initial file upload and return file_id"""
    try:
        file_id = str(uuid.uuid4())
        file_location = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"file_id": file_id, "original_filename": file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"File upload failed: {str(e)}"})

@app.post("/run_pipeline/")
async def run_pipeline(
    file_id: str = Form(...),
    query: str = Form(...),
    task_type: str = Form(...),
    target_column: str = Form(...)
):
    """Run the analysis pipeline on uploaded data"""
    try:
        # Enhanced debugging
        logger.debug(f"=== PIPELINE DEBUG ===")
        logger.debug(f"Raw received - file_id: {file_id}")
        logger.debug(f"Raw received - query: '{query}'")
        logger.debug(f"Raw received - task_type: '{task_type}' (type: {type(task_type)})")
        logger.debug(f"Raw received - target_column: '{target_column}'")
        logger.debug(f"Task type after strip/lower: '{task_type.strip().lower()}'")

        file_location = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if not os.path.exists(file_location):
            raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")

        try:
            df = pd.read_csv(file_location)
            df.columns = df.columns.str.strip()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Error reading CSV file: {str(e)}")

        if target_column not in df.columns:
            raise HTTPException(
                status_code=422, 
                detail=f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}"
            )

        query_id = log_query(query, os.path.basename(file_location), file_location)

        # Clear previous EDA plots
        for f in os.listdir(EDA_PLOT_DIR):
            os.remove(os.path.join(EDA_PLOT_DIR, f))

        # Run EDA
        eda_agent = EDAAgent(output_dir=EDA_PLOT_DIR)
        eda_result = eda_agent.analyze(df, target_column=target_column, use_llm=False)

        # Train model with explicit task_type logging
        logger.debug(f"Passing task_type to ModelTrainerAgent: '{task_type}'")
        trainer = ModelTrainerAgent()
        result = trainer.train(df, task_type=task_type, target_column=target_column)

        # Log results
        log_model_result(query_id, result.metrics["model"], result.metrics)

        return {
            "message": "Pipeline executed successfully",
            "query_id": query_id,
            "model": result.metrics["model"],
            "metrics": result.metrics,
            "eda_summary": eda_result.summary,
            "eda_plots": eda_result.plots,
            "debug_info": {
                "received_task_type": task_type,
                "actual_task_type_used": result.metrics.get("task_type_used", "unknown")
            }
        }

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {str(e)}"})

@app.post("/run_pipeline_direct/")
async def run_pipeline_direct(
    data: dict,
    query: str = Form(...),
    task_type: str = Form(None),
    target_column: str = Form(...)
):
    """Run the analysis pipeline on provided data"""
    try:
        logger.debug(f"Received direct data request - query: {query}, task_type: {task_type}, target_column: {target_column}")

        df = pd.DataFrame(data)

        if target_column not in df.columns:
            raise HTTPException(
                status_code=422, 
                detail=f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}"
            )

        query_id = log_query(query, "direct_input", "N/A")

        eda_agent = EDAAgent(output_dir=EDA_PLOT_DIR)
        eda_result = eda_agent.analyze(df, target_column=target_column, use_llm=False)

        trainer = ModelTrainerAgent()
        result = trainer.train(df, task_type=task_type, target_column=target_column)

        log_model_result(query_id, result.metrics["model"], result.metrics)

        return {
            "message": "Pipeline executed successfully on direct input",
            "query_id": query_id,
            "model": result.metrics["model"],
            "metrics": result.metrics,
            "eda_summary": eda_result.summary,
            "eda_plots": eda_result.plots
        }

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        logger.error(f"Pipeline failed on direct input: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {str(e)}"})

def run_pipeline_func(X, y, problem_type=None):
    preprocessor = DataPreprocessorAgent()
    trainer = ModelTrainerAgent()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_processed, y_train_processed = preprocessor.preprocess(X_train, y_train, problem_type)
    X_test_processed, y_test_processed = preprocessor.preprocess(X_test, y_test, problem_type)

    trainer.train(X_train_processed, y_train_processed)
    metrics = trainer.evaluate(X_test_processed, y_test_processed)

    return trainer, metrics