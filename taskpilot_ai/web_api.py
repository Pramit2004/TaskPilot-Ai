from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import uuid
import logging

# Agents
from taskpilot_ai.agents.parser_agent import ParserAgent
from taskpilot_ai.agents.inspection_agent import InspectionAgent
from taskpilot_ai.agents.cleaner_agent import CleanerAgent
from taskpilot_ai.agents.eda_agent import EDAAgent
from taskpilot_ai.agents.model_trainer_agent import ModelTrainerAgent
from taskpilot_ai.agents.insight_agent import InsightAgent
from taskpilot_ai.agents.data_preprocessing_agent import DataPreprocessingAgent

# DB & Email
from taskpilot_ai.db import init_db, log_query, log_model_result, log_retry
from taskpilot_ai.email_utils import send_report_email

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(title="TaskPilot AI Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev — restrict for prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
REPORTS_DIR = "reports"
EDA_DIR = "reports/eda"

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
init_db()

# ✅ Mount static files for serving EDA plots and reports
app.mount("/static/reports", StaticFiles(directory="reports"), name="reports")

# Serve static frontend (if exists)
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Handle initial file upload and return file_id"""
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists
        
        with open(file_path, "wb+") as file_object:
            content = await file.read()
            file_object.write(content)
        
        return JSONResponse(
            status_code=200,
            content={
                "file_id": file_id,
                "message": "File uploaded successfully"
            }
        )
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to upload file: {str(e)}"}
        )

@app.post("/run_pipeline/")
async def run_pipeline(
    file_id: str = Form(...),
    query: str = Form(...),
    task_type: str = Form(...),  # ✅ We get this from frontend
    target_column: str = Form(...),  # ✅ We get this from frontend
    email: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found."})

    try:
        # Enhanced debugging
        logger.info(f"=== PIPELINE DEBUG ===")
        logger.info(f"Received task_type from frontend: '{task_type}'")
        logger.info(f"Received target_column from frontend: '{target_column}'")
        logger.info(f"Query: '{query}'")

        # 1. Read data
        df = pd.read_csv(file_path)

        # 2. Parse natural language query (but DON'T override user's explicit choices)
        parser = ParserAgent()
        parse_result = parser.parse(query)
        
        # ✅ FIX: Use frontend values, only fall back to parser if not provided
        final_target_column = target_column if target_column and target_column.strip() else parse_result.target_column
        final_task_type = task_type if task_type and task_type.strip() else parse_result.task_type
        
        logger.info(f"Final task_type to use: '{final_task_type}'")
        logger.info(f"Final target_column to use: '{final_target_column}'")
        
        if not final_target_column:
            return JSONResponse(status_code=400, content={"error": "Target column not found in query and not provided explicitly."})
        
        if not final_task_type:
            return JSONResponse(status_code=400, content={"error": "Task type not found in query and not provided explicitly."})

        # Validate target column exists in dataframe
        if final_target_column not in df.columns:
            return JSONResponse(
                status_code=400, 
                content={
                    "error": f"Target column '{final_target_column}' not found in dataset. Available columns: {list(df.columns)}"
                }
            )

        # 3. Log query
        user_query_id = log_query(query, dataset_name=os.path.basename(file_path), dataset_path=file_path)

        # 4. Inspect dataset
        inspector = InspectionAgent()
        inspection_results = inspector.inspect(df, target_column=final_target_column)

        # 5. Clean data
        cleaner = CleanerAgent()
        clean_result = cleaner.clean(df, inspection_results=inspection_results.model_dump())

        # 6. EDA
        eda = EDAAgent(output_dir=EDA_DIR)
        eda_result = eda.analyze(clean_result.df, target_column=final_target_column)

        # Convert EDA plot paths to accessible URLs
        eda_plot_urls = []
        for plot_path in eda_result.plots:
            # Convert "eda/filename.png" to "/static/reports/eda/filename.png"
            if plot_path.startswith("eda/"):
                plot_url = f"/static/reports/{plot_path}"
                eda_plot_urls.append(plot_url)
            else:
                eda_plot_urls.append(plot_path)

        # 7. Train model (✅ Using final_task_type from frontend, not parser)
        logger.info(f"Training model with task_type: '{final_task_type}'")
        trainer = ModelTrainerAgent()
        model_result = trainer.train(clean_result.df, final_task_type, final_target_column)

        # 8. Log model results
        log_model_result(user_query_id, model_result.metrics.get('model'), model_result.metrics)

        # 9. Generate insights
        insight_agent = InsightAgent()
        insights = insight_agent.generate(eda_result.summary, model_result.metrics)

        # 10. Send email report (if email provided)
        if email:
            background_tasks.add_task(
                send_report_email,
                email,
                "Your TaskPilot AI Report",
                "The model training has completed. (Report generation disabled in this version)",
                None
            )

        return {
            "parse_result": {
                **parse_result.model_dump(),
                "final_task_type_used": final_task_type,
                "final_target_column_used": final_target_column
            },
            "inspection_results": inspection_results.model_dump(),
            "cleaning_actions": clean_result.actions,
            "eda_summary": eda_result.summary,
            "eda_plots": eda_plot_urls,  # ✅ Updated URLs
            "model_metrics": model_result.metrics,
            "model_path": model_result.model_path,
            "report_path": None,
            "email_sent": bool(email),
            "debug_info": {
                "frontend_task_type": task_type,
                "parser_suggested_task_type": parse_result.task_type,
                "final_task_type_used": final_task_type
            }
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze")
async def analyze_data(file: UploadFile, target_column: str, task_type: str):
    try:
        df = pd.read_csv(file.file)
        preprocessor = DataPreprocessingAgent()
        trainer = ModelTrainerAgent()
        clean_df = preprocessor.run(df, target_column=target_column)
        result = trainer.train(clean_df, task_type, target_column)
        return {"status": "success", "metrics": result.metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/download/{file_type}/{file_name}")
def download_file(file_type: str, file_name: str):
    """Download endpoint for files"""
    if file_type == "eda":
        file_path = os.path.join("reports/eda", file_name)
    elif file_type == "model":
        file_path = os.path.join("reports/models", file_name)
    elif file_type == "report":
        file_path = os.path.join("reports", file_name)
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid file type."})

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found."})

    return FileResponse(file_path)