from agents.model_trainer_agent import ModelTrainerAgent
from agents.data_preprocessing_agent import DataPreprocessingAgent
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(data_path: str, target_column: str, task_type: str):
    """
    Run the complete ML pipeline
    
    Args:
        data_path: Path to the CSV file
        target_column: Name of the target column
        task_type: Either 'classification' or 'regression'
    """
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Initialize agents
        preprocessor = DataPreprocessingAgent()
        trainer = ModelTrainerAgent()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_df = preprocessor.run(df, target_column=target_column)
        
        # Train model
        logger.info(f"Training {task_type} model...")
        result = trainer.train(processed_df, task_type, target_column)
        
        logger.info("Pipeline completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    DATA_PATH = "california_housing_train.csv"
    TARGET_COLUMN = "median_house_value"
    TASK_TYPE = "regression"
    
    result = run_pipeline(
        data_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        task_type=TASK_TYPE
    )
    
    print("\nPipeline Results:")
    print(f"Best Model: {result.metrics['model']}")
    print(f"Model Path: {result.model_path}")
    print("\nMetrics:")
    for model, scores in result.metrics['all_model_scores'].items():
        print(f"\n{model}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")