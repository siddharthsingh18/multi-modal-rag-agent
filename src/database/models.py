"""SQLAlchemy models for metadata storage."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DocumentMetadata(Base):
    """Document metadata model."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(255), unique=True, index=True, nullable=False)
    source = Column(String(500))
    type = Column(String(50))
    content_hash = Column(String(64))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryLog(Base):
    """Query logging model."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(255), index=True)
    query = Column(Text)
    answer = Column(Text)
    num_documents = Column(Integer)
    reflection_score = Column(Integer)
    processing_time = Column(Integer)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class IngestionLog(Base):
    """Ingestion logging model."""

    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(255), index=True)
    source = Column(String(500))
    num_documents = Column(Integer)
    num_chunks = Column(Integer)
    success = Column(Integer)
    error_message = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
