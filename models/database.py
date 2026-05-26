"""
Database models for the RAG Research Assistant.
Tables: Paper, Chunk, ChatSession, ChatMessage
"""

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime,
    ForeignKey, Boolean, JSON, create_engine
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.sql import func
import uuid

from core.config import get_settings

Base = declarative_base()
settings = get_settings()


def generate_uuid():
    return str(uuid.uuid4())


# ─── Models ──────────────────────────────────────────────────────────────────

class Paper(Base):
    """Represents an uploaded research paper."""
    __tablename__ = "papers"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    title = Column(String, nullable=True)          # extracted from PDF
    abstract = Column(Text, nullable=True)
    authors = Column(JSON, nullable=True)          # list of author names
    year = Column(Integer, nullable=True)
    file_path = Column(String, nullable=False)     # path on disk
    file_size_bytes = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    is_processed = Column(Boolean, default=False)  # embedding done?
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="paper")

    def __repr__(self):
        return f"<Paper id={self.id} title={self.title!r}>"


class Chunk(Base):
    """A text chunk from a paper, with its FAISS vector index reference."""
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # order within paper
    text = Column(Text, nullable=False)
    section = Column(String, nullable=True)        # e.g. "Introduction"
    page_number = Column(Integer, nullable=True)
    faiss_id = Column(Integer, nullable=True)      # index in FAISS
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper = relationship("Paper", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk id={self.id} paper_id={self.paper_id} idx={self.chunk_index}>"


class ChatSession(Base):
    """A conversation session, tied to one or more papers."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=True)  # primary paper
    title = Column(String, nullable=True)          # auto-generated from first message
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    paper = relationship("Paper", back_populates="sessions")
    messages = relationship(
        "ChatMessage", back_populates="session",
        cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )

    def __repr__(self):
        return f"<ChatSession id={self.id}>"


class ChatMessage(Base):
    """A single message in a chat session."""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)          # "user" | "assistant"
    content = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON, nullable=True) # chunk IDs used for this answer
    citations = Column(JSON, nullable=True)        # {section, page, text_snippet}
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage id={self.id} role={self.role}>"


# ─── Database Engine & Session ────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}  # needed for SQLite
    if "sqlite" in settings.database_url else {},
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()