import pandas as pd
from taskpilot_ai.agents.inspection_agent import InspectionAgent

# Sample DataFrame for testing
data = {
    'age': [25, 30, 22, 40, 28, 30, 120, None],
    'salary': [50000, 60000, 52000, 80000, 58000, 60000, 1000000, 59000],
    'gender': ['M', 'F', 'F', 'M', 'F', 'F', 'M', 'F'],
    'churn': [0, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

agent = InspectionAgent()
results = agent.inspect(df, target_column='churn', verbose=True)

print("\nInspection Results:")
print(results.model_dump_json(indent=2)) 