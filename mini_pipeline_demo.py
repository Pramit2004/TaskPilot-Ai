import pandas as pd
from taskpilot_ai.agents.parser_agent import ParserAgent
from taskpilot_ai.agents.inspection_agent import InspectionAgent
from taskpilot_ai.agents.cleaner_agent import CleanerAgent

# 1. User query
query = "Predict customer churn from file churn.csv with target Churn"

# 2. Parse query
parser = ParserAgent()
parse_result = parser.parse(query)
print("\nParsed Query:")
print(parse_result.model_dump_json(indent=2))

target_column = parse_result.target_column or 'churn'

# 3. Load sample DataFrame
data = {
    'age': [25, 30, 22, 40, 28, 30, 120, None],
    'salary': [50000, 60000, 52000, 80000, 58000, 60000, 1000000, 59000],
    'gender': ['M', 'F', 'F', 'M', 'F', 'F', 'M', 'F'],
    'Churn': [0, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 4. Inspection
inspector = InspectionAgent()
inspection_results = inspector.inspect(df, target_column=target_column, verbose=True)
print("\nInspection Results:")
print(inspection_results.model_dump_json(indent=2))

# 5. Cleaning
cleaner = CleanerAgent()
clean_result = cleaner.clean(df, inspection_results=inspection_results.model_dump(), verbose=True)
print("\nCleaning Actions:")
print(clean_result.actions)
print("\nCleaned DataFrame:")
print(clean_result.df) 