from typing import Dict, Any, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class ReflectiveAgent:
    """
    Agent to evaluate results, retry with improvements, and log actions.
    For now, retries with a different model if accuracy is below threshold.
    """
    def reflect(self, X_train, X_test, y_train, y_test, initial_model, initial_metrics: Dict[str, Any], threshold: float = 0.85) -> Dict[str, Any]:
        logs = []
        # Check if accuracy is below threshold
        acc = initial_metrics.get('accuracy', 0)
        if acc >= threshold:
            logs.append(f"Initial model met threshold: {acc:.2f} >= {threshold}")
            return {'final_model': initial_model, 'final_metrics': initial_metrics, 'logs': logs}
        # Retry with a different model
        logs.append(f"Initial model below threshold: {acc:.2f} < {threshold}. Retrying with RandomForest.")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, preds)
        f1_rf = f1_score(y_test, preds, average='weighted')
        logs.append(f"RandomForest accuracy: {acc_rf:.2f}, F1: {f1_rf:.2f}")
        if acc_rf > acc:
            logs.append("RandomForest outperformed initial model.")
            return {'final_model': rf, 'final_metrics': {'model': 'RandomForest', 'accuracy': acc_rf, 'f1': f1_rf}, 'logs': logs}
        else:
            logs.append("Initial model retained.")
            return {'final_model': initial_model, 'final_metrics': initial_metrics, 'logs': logs} 