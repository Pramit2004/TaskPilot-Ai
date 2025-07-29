import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration for Render cloud
POSTGRES_USER = os.getenv('POSTGRES_USER', 'root')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'plhZeHp84YD3HUvzjUgiM51LgznixfsN')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'dpg-d2457ljuibrs73a8b730-a.render.com')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'taskpilot_ai')

# Alternative: Use DATABASE_URL directly (recommended for Render)
DATABASE_URL = os.getenv('DATABASE_URL') or f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Handle Render's internal DATABASE_URL format
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg2://', 1)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class UserQuery(Base):
    __tablename__ = 'user_queries'
    id = Column(Integer, primary_key=True)
    query = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    dataset = relationship('Dataset', back_populates='queries')
    model_results = relationship('ModelResult', back_populates='user_query')
    retry_logs = relationship('RetryLog', back_populates='user_query')

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    path = Column(String(255))
    queries = relationship('UserQuery', back_populates='dataset')

class ModelResult(Base):
    __tablename__ = 'model_results'
    id = Column(Integer, primary_key=True)
    user_query_id = Column(Integer, ForeignKey('user_queries.id'))
    model_name = Column(String(255))
    accuracy = Column(Float)
    f1 = Column(Float)
    rmse = Column(Float)
    metrics = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    user_query = relationship('UserQuery', back_populates='model_results')

class RetryLog(Base):
    __tablename__ = 'retry_logs'
    id = Column(Integer, primary_key=True)
    user_query_id = Column(Integer, ForeignKey('user_queries.id'))
    action = Column(Text)
    result = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    user_query = relationship('UserQuery', back_populates='retry_logs')

def init_db():
    Base.metadata.create_all(bind=engine)

def log_query(query: str, dataset_name: str, dataset_path: str) -> int:
    with SessionLocal() as session:
        dataset = session.query(Dataset).filter_by(name=dataset_name).first()
        if not dataset:
            dataset = Dataset(name=dataset_name, path=dataset_path)
            session.add(dataset)
            session.commit()
        user_query = UserQuery(query=query, dataset=dataset)
        session.add(user_query)
        session.commit()
        return user_query.id

def log_model_result(user_query_id: int, model_name: str, metrics: dict):
    with SessionLocal() as session:
        result = ModelResult(
            user_query_id=user_query_id,
            model_name=model_name,
            accuracy=metrics.get('accuracy'),
            f1=metrics.get('f1'),
            rmse=metrics.get('rmse'),
            metrics=str(metrics),
        )
        session.add(result)
        session.commit()

def log_retry(user_query_id: int, action: str, result: str):
    with SessionLocal() as session:
        retry = RetryLog(
            user_query_id=user_query_id,
            action=action,
            result=result,
        )
        session.add(retry)
        session.commit()