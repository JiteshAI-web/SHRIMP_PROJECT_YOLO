"""
Database Models for Shrimp Detection System
SQLAlchemy ORM Models
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, LargeBinary,
    Float, ForeignKey, Text, Enum, Index, text, create_engine, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import db_config


Base = declarative_base()


class ImageType(str, PyEnum):
    """Types of images stored in the database"""
    TRAIN = "train"           # Training images
    VALIDATION = "validation" # Validation images
    TEST = "test"             # Test images
    PREDICT = "predict"       # Prediction input images
    DETECT = "detect"         # Detection output images
    COUNT_LENGTH = "count_length"  # Count & length measurement images


class RealImage(Base):
    """
    Table for storing real/original images
    - Train images
    - Validation images
    - Test images
    - Predict input images
    """
    __tablename__ = "real_images"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String(255), nullable=False)
    image_type = Column(Enum(ImageType), nullable=False)
    image_data = Column(LargeBinary, nullable=False)  # Binary image data
    file_format = Column(String(10), nullable=False)  # jpg, png, etc.
    file_size_bytes = Column(Integer, nullable=False)
    
    # Metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Source information
    source_path = Column(Text, nullable=True)  # Original file path
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    ai_images = relationship("AIImage", back_populates="real_image", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('image_name', 'image_type', name='uix_real_image_name_type'),
        Index('ix_real_images_image_type', 'image_type'),
        Index('ix_real_images_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<RealImage(id={self.id}, name='{self.image_name}', type='{self.image_type}')>"


class AIImage(Base):
    """
    Table for storing AI-processed images
    - Detection results
    - Count & length measurement images
    """
    __tablename__ = "ai_images"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to real image
    real_image_id = Column(Integer, ForeignKey("real_images.id", ondelete="CASCADE"), nullable=False)
    
    image_name = Column(String(255), nullable=False)
    image_type = Column(Enum(ImageType), nullable=False)  # count_length, detect, predict
    image_data = Column(LargeBinary, nullable=False)  # Binary image data
    file_format = Column(String(10), nullable=False)  # jpg, png, etc.
    file_size_bytes = Column(Integer, nullable=False)
    
    # Image dimensions
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # AI Processing Results
    shrimp_count = Column(Integer, nullable=True)  # Number of shrimps detected
    avg_shrimp_length_cm = Column(Float, nullable=True)  # Average length in cm
    avg_shrimp_weight_g = Column(Float, nullable=True)  # Average weight in grams (calculated from length)
    
    # Store individual measurements as JSON string
    # Format: [{"length_cm": 12.5, "confidence": 0.95}, ...]
    measurements_json = Column(Text, nullable=True)
    
    # Pond Metadata
    pond_id = Column(String(50), nullable=True)  # e.g., "pond1", "pond2"
    cycle_no = Column(String(50), nullable=True)  # e.g., "cycle1", "cycle2"
    device_id = Column(String(50), nullable=True)  # e.g., "D1", "D2"
    water_depth_m = Column(Float, nullable=True)  # Water depth in meters
    
    # Processing metadata
    model_used = Column(String(100), nullable=True)  # e.g., "yolov8n.pt"
    confidence_threshold = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)  # Processing time in milliseconds
    
    # Source information
    source_path = Column(Text, nullable=True)  # Original file path
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    real_image = relationship("RealImage", back_populates="ai_images")
    
    __table_args__ = (
        UniqueConstraint('image_name', 'image_type', name='uix_ai_image_name_type'),
        Index('ix_ai_images_real_image_id', 'real_image_id'),
        Index('ix_ai_images_type_created_at', 'image_type', 'created_at'),
        Index(
            'ix_ai_images_count_created_at',
            'created_at',
            postgresql_where=text('shrimp_count IS NOT NULL'),
        ),
    )
    
    def __repr__(self):
        return f"<AIImage(id={self.id}, name='{self.image_name}', type='{self.image_type}', count={self.shrimp_count})>"


# Database Engine and Session
def get_engine():
    """Create and return database engine with connection pooling"""
    return create_engine(
        db_config.connection_string,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def get_session_maker(engine=None):
    """Get session maker bound to engine"""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def create_tables(engine=None):
    """Create all tables in the database"""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully!")


def drop_tables(engine=None):
    """Drop all tables from the database"""
    if engine is None:
        engine = get_engine()
    Base.metadata.drop_all(engine)
    print("⚠️ Database tables dropped!")


if __name__ == "__main__":
    # Test database connection and create tables
    print("Testing database connection...")
    try:
        engine = get_engine()
        # Test connection
        with engine.connect() as conn:
            print("✅ Database connection successful!")
        
        # Create tables
        create_tables(engine)
        print("\n📋 Tables created:")
        print("  - real_images")
        print("  - ai_images")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
