from typing import Dict, Any

class InsightAgent:
    """
    Agent to convert metrics and EDA into natural language insights.
    Optionally uses LLMs for explanations.
    """
    def generate(self, eda_results: Dict[str, Any], model_results: Dict[str, Any], logs: Dict[str, Any] = None, use_llm: bool = False) -> str:
        # Placeholder for LLM-based insight generation
        if use_llm:
            return "(LLM-generated insight would go here.)"
        # Template-based summary
        summary = []
        if 'target_distribution' in eda_results:
            summary.append(f"Target distribution: {eda_results['target_distribution']}")
        if 'correlation' in eda_results:
            summary.append(f"Feature correlations: {eda_results['correlation']}")
        if 'model' in model_results:
            summary.append(f"Best model: {model_results['model']}")
        if 'accuracy' in model_results:
            summary.append(f"Accuracy: {model_results['accuracy']:.2f}")
        if 'f1' in model_results:
            summary.append(f"F1 Score: {model_results['f1']:.2f}")
        if 'rmse' in model_results:
            summary.append(f"RMSE: {model_results['rmse']:.2f}")
        if logs:
            summary.append(f"Logs: {logs}")
        return '\n'.join(summary) 