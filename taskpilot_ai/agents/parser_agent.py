from pydantic import BaseModel
from typing import Optional
import re

# LLM imports are commented out for now to avoid ModuleNotFoundError
# from langchain.llms import OpenAI, GoogleGemini
# from langchain.prompts import PromptTemplate

class ParseResult(BaseModel):
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    dataset_name: Optional[str] = None
    action_plan: Optional[str] = None
    raw_response: Optional[str] = None

class ParserAgent:
    """
    Advanced agent to parse user natural language queries using LLMs (LangChain, Gemini/OpenAI) or heuristics.
    Extracts task type, target column, dataset name, and generates an action plan.
    """
    def __init__(self, use_llm: bool = False, llm_type: str = "openai"):
        self.use_llm = use_llm
        self.llm_type = llm_type
        # Uncomment and configure when ready
        # if use_llm:
        #     if llm_type == "openai":
        #         self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #     elif llm_type == "gemini":
        #         self.llm = GoogleGemini(api_key=os.getenv("GEMINI_API_KEY"))

    def parse(self, query: str) -> ParseResult:
        """
        Parse the user query to extract task type, target column, dataset name, and action plan.
        Uses LLM if enabled, otherwise falls back to heuristics.
        """
        if self.use_llm:
            # Placeholder for LLM-based parsing (LangChain prompt)
            # prompt = PromptTemplate(...)
            # response = self.llm(prompt.format(query=query))
            # Parse response for fields
            # return ParseResult(...)
            pass  # To be implemented with LLM
        # Heuristic fallback
        task_type = None
        if any(word in query.lower() for word in ["classify", "classification", "churn", "predict", "label"]):
            task_type = "classification"
        elif any(word in query.lower() for word in ["regress", "regression", "forecast", "estimate"]):
            task_type = "regression"
        elif any(word in query.lower() for word in ["time-series", "sequence", "trend"]):
            task_type = "time-series"
        # Extract target column
        match = re.search(r"target(?: column)? ([\w_]+)", query, re.IGNORECASE)
        target_column = match.group(1) if match else None
        # Extract dataset name (e.g., 'from this CSV', 'on file churn.csv')
        dataset_match = re.search(r"(?:file|csv|dataset) ([\w\-.]+)", query, re.IGNORECASE)
        dataset_name = dataset_match.group(1) if dataset_match else None
        action_plan = f"Task: {task_type}, Target: {target_column}, Dataset: {dataset_name}" if task_type else "Unknown task"
        return ParseResult(task_type=task_type, target_column=target_column, dataset_name=dataset_name, action_plan=action_plan, raw_response=None)
