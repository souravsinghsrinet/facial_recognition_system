from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from typing import Optional

# Create the declarative base instance at module level
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    face_encoding = Column(String, nullable=False)  # Stored as string representation of numpy array
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_access = Column(DateTime)

class AccessLog(Base):
    __tablename__ = 'access_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    access_time = Column(DateTime, default=datetime.utcnow)
    access_type = Column(String(20), nullable=False)  # 'entry' or 'exit'
    confidence = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)  # 'success' or 'failed'

class Database:
    def __init__(self, db_url: str = None):
        if db_url is None:
            # Get PostgreSQL configuration from environment variables
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'security_system')
            db_user = os.getenv('POSTGRES_USER', 'postgres')
            db_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
            
            db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Only create tables automatically if SKIP_MIGRATIONS environment variable is set
        if os.getenv('SKIP_MIGRATIONS', 'false').lower() == 'true':
            self.init_db()

    def init_db(self):
        """Initialize the database by creating all tables."""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    def add_user(self, name: str, face_encoding: str) -> Optional[int]:
        """Add a new user to the database."""
        session = self.get_session()
        try:
            user = User(
                name=name,
                face_encoding=face_encoding
            )
            session.add(user)
            session.commit()
            return user.id
        finally:
            session.close()

    def get_user(self, user_id: int) -> Optional[User]:
        """Retrieve a user by ID."""
        session = self.get_session()
        try:
            return session.query(User).filter_by(id=user_id).first()
        finally:
            session.close()

    def log_access(self, user_id: int, access_type: str, confidence: float, status: str):
        """Log an access attempt."""
        session = self.get_session()
        try:
            log = AccessLog(
                user_id=user_id,
                access_type=access_type,
                confidence=confidence,
                status=status
            )
            session.add(log)
            
            # Update user's last access time
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.last_access = datetime.utcnow()
            
            session.commit()
        finally:
            session.close()

    def get_access_logs(self, limit: int = 100):
        """Retrieve recent access logs."""
        session = self.get_session()
        try:
            return session.query(AccessLog).order_by(AccessLog.access_time.desc()).limit(limit).all()
        finally:
            session.close()

# Create a default database instance
db = Database() 