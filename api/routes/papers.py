"""
Paper Routes
POST /papers/upload        — Upload a PDF
POST /papers/{id}/process  — Embed & index a paper
GET  /papers/              — List all papers
GET  /papers/{id}          — Get paper details
DELETE /papers/{id}        — Delete paper
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from core.config import get_settings
from models.database import Paper, Chunk, get_db
from models.schemas import PaperUploadResponse, PaperListResponse, PaperProcessResponse
from services.pdf_processor import process_pdf
from services.embedding_service import get_embedding_service, get_vector_store

router = APIRouter(prefix="/papers", tags=["Papers"])
settings = get_settings()


# ─── Background Task: Process & Embed ────────────────────────────────────────

def _process_and_embed_paper(paper_id: str, file_path: str, db_url: str):
    """
    Runs in background after upload:
    1. Extract text + metadata from PDF
    2. Chunk text
    3. Generate embeddings
    4. Store in FAISS
    5. Save chunks to DB
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Need a fresh DB session for background task
    engine = create_engine(db_url, connect_args={"check_same_thread": False}
                           if "sqlite" in db_url else {})
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        paper = db.query(Paper).filter(Paper.id == paper_id).first()
        if not paper:
            return

        # ── PDF Processing ──
        processed = process_pdf(file_path)

        # Update paper metadata
        paper.title = processed.metadata.title
        paper.abstract = processed.metadata.abstract
        paper.authors = processed.metadata.authors
        paper.year = processed.metadata.year
        paper.page_count = processed.metadata.page_count
        db.commit()

        # ── Embedding ──
        embedder = get_embedding_service()
        texts = [chunk.text for chunk in processed.chunks]
        embeddings = embedder.embed_texts(texts)

        # ── Save Chunks to DB ──
        db_chunks = []
        for chunk in processed.chunks:
            db_chunk = Chunk(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                section=chunk.section,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
            )
            db.add(db_chunk)
            db_chunks.append(db_chunk)

        db.commit()

        # ── Store in FAISS ──
        vector_store = get_vector_store()
        chunk_db_ids = [c.id for c in db_chunks]
        faiss_ids = vector_store.add_paper_embeddings(
            paper_id=paper_id,
            embeddings=embeddings,
            chunk_db_ids=chunk_db_ids,
            dimension=embedder.dimension,
        )

        # Update FAISS IDs in DB
        for db_chunk, faiss_id in zip(db_chunks, faiss_ids):
            db_chunk.faiss_id = faiss_id

        # Mark as processed
        paper.is_processed = True
        paper.chunk_count = len(db_chunks)
        db.commit()

        print(f"[Processor] Paper {paper_id} processed: {len(db_chunks)} chunks")

    except Exception as e:
        print(f"[Processor] Error processing paper {paper_id}: {e}")
        paper = db.query(Paper).filter(Paper.id == paper_id).first()
        if paper:
            paper.is_processed = False
            db.commit()
        raise
    finally:
        db.close()


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=PaperUploadResponse, status_code=201)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a research paper PDF.
    Processing (embedding) happens in the background.
    Poll GET /papers/{id} to check is_processed status.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    paper_id = str(uuid.uuid4())
    file_path = upload_dir / f"{paper_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(contents)

    # Create DB record
    paper = Paper(
        id=paper_id,
        filename=file.filename,
        file_path=str(file_path),
        file_size_bytes=len(contents),
        is_processed=False,
        chunk_count=0,
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)

    # Trigger background processing
    background_tasks.add_task(
        _process_and_embed_paper,
        paper_id=paper_id,
        file_path=str(file_path),
        db_url=settings.database_url,
    )

    return paper


@router.get("/", response_model=PaperListResponse)
def list_papers(db: Session = Depends(get_db)):
    """List all uploaded papers."""
    papers = db.query(Paper).order_by(Paper.created_at.desc()).all()
    return PaperListResponse(papers=papers, total=len(papers))


@router.get("/{paper_id}", response_model=PaperUploadResponse)
def get_paper(paper_id: str, db: Session = Depends(get_db)):
    """Get a single paper by ID. Use is_processed to check if ready."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@router.delete("/{paper_id}", status_code=204)
def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    """Delete a paper, its chunks, FAISS index, and file."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Remove file
    if os.path.exists(paper.file_path):
        os.remove(paper.file_path)

    # Remove FAISS index
    vector_store = get_vector_store()
    vector_store.delete_paper(paper_id)

    # Remove from DB (cascades to chunks and sessions)
    db.delete(paper)
    db.commit()