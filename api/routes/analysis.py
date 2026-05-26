"""
Analysis Routes
POST /analysis/summarize    — Summarize a paper
POST /analysis/compare      — Compare multiple papers
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict

from models.database import Paper, Chunk, get_db
from models.schemas import SummaryRequest, SummaryResponse, ComparisonRequest, ComparisonResponse
from services.rag_pipeline import call_llm

router = APIRouter(prefix="/analysis", tags=["Analysis"])


SUMMARY_SYSTEM = """You are an expert academic research assistant.
Summarize research papers clearly and concisely.
Use structured output with labeled sections.
Be precise, technical, and faithful to the paper content."""


async def _get_paper_chunks_text(paper_id: str, db: Session, max_chunks: int = 20) -> str:
    """Get representative chunks from a paper for summarization."""
    chunks = (
        db.query(Chunk)
        .filter(Chunk.paper_id == paper_id)
        .order_by(Chunk.chunk_index)
        .limit(max_chunks)
        .all()
    )
    return "\n\n".join(f"[{c.section or 'Body'}] {c.text}" for c in chunks)


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_paper(request: SummaryRequest, db: Session = Depends(get_db)):
    """Generate a structured summary of a research paper."""
    paper = db.query(Paper).filter(Paper.id == request.paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if not paper.is_processed:
        raise HTTPException(status_code=400, detail="Paper not yet processed.")

    context = await _get_paper_chunks_text(request.paper_id, db)

    sections_requested = []
    if request.include_abstract:
        sections_requested.append("Abstract/Overview")
    if request.include_methodology:
        sections_requested.append("Methodology")
    if request.include_contributions:
        sections_requested.append("Key Contributions")
    if request.include_results:
        sections_requested.append("Results & Findings")

    sections_str = "\n".join(f"- {s}" for s in sections_requested)

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"""Summarize this research paper.

PAPER CONTEXT:
{context}

Generate summaries for these sections:
{sections_str}

Format your response as:
## [Section Name]
[Summary paragraph]

Be concise but thorough. Use academic language."""},
    ]

    summary_text = await call_llm(messages)

    # Parse sections from response
    sections: Dict[str, str] = {}
    current_section = None
    current_lines = []
    for line in summary_text.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return SummaryResponse(
        paper_id=request.paper_id,
        paper_title=paper.title,
        summary=summary_text,
        sections=sections,
    )


@router.post("/compare", response_model=ComparisonResponse)
async def compare_papers(request: ComparisonRequest, db: Session = Depends(get_db)):
    """Compare multiple papers on a specific aspect."""
    papers = db.query(Paper).filter(Paper.id.in_(request.paper_ids)).all()
    if len(papers) < 2:
        raise HTTPException(status_code=404, detail="Need at least 2 valid papers.")

    unprocessed = [p.id for p in papers if not p.is_processed]
    if unprocessed:
        raise HTTPException(
            status_code=400,
            detail=f"Papers not yet processed: {unprocessed}"
        )

    # Get context for each paper
    per_paper_context = {}
    per_paper_titles = {}
    for paper in papers:
        context = await _get_paper_chunks_text(paper.id, db, max_chunks=10)
        per_paper_context[paper.id] = context
        per_paper_titles[paper.id] = paper.title or paper.filename

    # Build comparison prompt
    papers_str = "\n\n".join(
        f"=== PAPER {i+1}: {per_paper_titles[pid]} ===\n{per_paper_context[pid]}"
        for i, pid in enumerate(request.paper_ids)
        if pid in per_paper_context
    )

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"""Compare these research papers on the aspect of: {request.aspect.upper()}

{papers_str}

Provide:
1. A comparative analysis paragraph
2. Per-paper summary of their {request.aspect}
3. Key differences and similarities

Format:
## Comparative Analysis
[paragraph]

## Per-Paper Breakdown
### Paper 1: [title]
[summary]
### Paper 2: [title]
[summary]"""},
    ]

    comparison_text = await call_llm(messages)

    # Build per-paper summaries (simple extraction)
    per_paper: Dict[str, str] = {}
    for pid in request.paper_ids:
        title = per_paper_titles.get(pid, pid)
        per_paper[pid] = f"See full comparison for details on: {title}"

    return ComparisonResponse(
        paper_ids=request.paper_ids,
        aspect=request.aspect,
        comparison=comparison_text,
        per_paper=per_paper,
    )