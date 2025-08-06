import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

# Use the external DATABASE_URL directly (recommended for Render)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://root:plhZeHp84YD3HUvzjUgiM51LgznixfsN@dpg-d2457ljuibrs73a8b730-a.oregon-postgres.render.com/taskpilot_ai')

# Convert postgres:// to postgresql+psycopg2:// if needed
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg2://', 1)
elif DATABASE_URL.startswith('postgresql://') and 'psycopg2' not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg2://', 1)

# Add connection pool settings for better stability
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    pool_timeout=20,     # Wait up to 20 seconds for connection
    max_overflow=0       # Don't allow overflow connections
)

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
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise

def log_query(query: str, dataset_name: str, dataset_path: str) -> int:
    """Log a user query and return the query ID"""
    try:
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
    except Exception as e:
        print(f"Error logging query: {e}")
        raise

def log_model_result(user_query_id: int, model_name: str, metrics: dict):
    """Log model results for a query"""
    try:
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
    except Exception as e:
        print(f"Error logging model result: {e}")
        raise

def log_retry(user_query_id: int, action: str, result: str):
    """Log retry attempts"""
    try:
        with SessionLocal() as session:
            retry = RetryLog(
                user_query_id=user_query_id,
                action=action,
                result=result,
            )
            session.add(retry)
            session.commit()
    except Exception as e:
        print(f"Error logging retry: {e}")
        raise

def test_connection():
    """Test database connection"""
    try:
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Test connection when module is imported
if __name__ == "__main__":
    test_connection()