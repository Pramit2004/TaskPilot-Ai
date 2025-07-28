try:
    import fastapi
    import langchain
    import sklearn
    import pandas
    import numpy
    import matplotlib
    import seaborn
    import joblib
    import fpdf
    import pptx
    import jinja2
    import typer
    import dotenv
    print('✅ All core dependencies imported successfully!')
except Exception as e:
    print('❌ Import failed:', e) 