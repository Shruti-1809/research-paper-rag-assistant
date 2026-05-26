"""
Chat Routes
POST /chat/                        — Send a message, get RAG answer
GET  /chat/sessions/               — List all sessions
GET  /chat/sessions/{id}           — Get session + full history
DELETE /chat/sessions/{id}         — Delete a session
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import ChatSession, ChatMessage, Paper, get_db
from models.schemas import (
    ChatRequest, ChatResponse, ChatSessionResponse,
    ChatHistoryResponse, ChatMessageResponse
)
from services.rag_pipeline import run_rag_pipeline

router = APIRouter(prefix="/chat", tags=["Chat"])


def _build_history(messages: List[ChatMessage]) -> List[dict]:
    """Convert DB messages to simple dicts for the RAG pipeline."""
    return [{"role": m.role, "content": m.content} for m in messages]


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint.
    - If session_id is None, creates a new session.
    - Retrieves history, runs RAG, saves messages, returns answer.
    """

    # ── Validate paper exists and is processed ──
    paper = db.query(Paper).filter(Paper.id == request.paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if not paper.is_processed:
        raise HTTPException(
            status_code=400,
            detail="Paper is still being processed. Please wait and retry."
        )

    # ── Get or create session ──
    if request.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == request.session_id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        session = ChatSession(
            id=str(uuid.uuid4()),
            paper_id=request.paper_id,
            title=request.message[:60],  # first message as title
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # ── Load chat history ──
    history_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    chat_history = _build_history(history_messages)

    # ── Run RAG pipeline ──
    result = await run_rag_pipeline(
        query=request.message,
        paper_id=request.paper_id,
        db=db,
        chat_history=chat_history,
    )

    # ── Save user message ──
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)

    # ── Save assistant message ──
    citations_json = [c.dict() for c in result["citations"]]
    assistant_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        retrieved_chunks=result["chunk_ids"],
        citations=citations_json,
        confidence_score=result["confidence_score"],
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(
        session_id=session.id,
        message_id=assistant_msg.id,
        answer=result["answer"],
        citations=result["citations"],
        confidence_score=result["confidence_score"],
        retrieved_chunk_ids=result["chunk_ids"],
    )


@router.get("/sessions/", response_model=List[ChatSessionResponse])
def list_sessions(paper_id: Optional[str] = None, db: Session = Depends(get_db)):
    """List sessions, optionally filtered by paper."""
    query = db.query(ChatSession).order_by(ChatSession.created_at.desc())
    if paper_id:
        query = query.filter(ChatSession.paper_id == paper_id)
    sessions = query.all()
    result = []
    for s in sessions:
        result.append(ChatSessionResponse(
            id=s.id,
            paper_id=s.paper_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(s.messages),
        ))
    return result


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get a session with its full message history."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = []
    for m in session.messages:
        citations = [c for c in (m.citations or [])]
        messages.append(ChatMessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            citations=citations,
            confidence_score=m.confidence_score,
            created_at=m.created_at,
        ))

    return ChatHistoryResponse(
        session=ChatSessionResponse(
            id=session.id,
            paper_id=session.paper_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(messages),
        ),
        messages=messages,
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session and all its messages."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    db.delete(session)
    db.commit()