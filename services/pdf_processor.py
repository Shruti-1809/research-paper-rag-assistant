"""
PDF Processing Service
Handles: text extraction, metadata parsing, and semantic chunking.
Library: PyMuPDF (fitz) for speed.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.config import get_settings

settings = get_settings()


# --- Data Structures ----------------------------------------------------------

@dataclass
class ExtractedChunk:
    """A single text chunk ready for embedding."""
    chunk_index: int
    text: str
    section: Optional[str]
    page_number: Optional[int]
    token_count: int = 0

    def __post_init__(self):
        # Rough token estimate: words * 1.3
        self.token_count = int(len(self.text.split()) * 1.3)


@dataclass
class PaperMetadata:
    """Metadata extracted from a research paper PDF."""
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    page_count: int = 0
    sections: List[str] = field(default_factory=list)


@dataclass
class ProcessedPaper:
    metadata: PaperMetadata
    chunks: List[ExtractedChunk]
    full_text: str


# --- Section Header Detection -------------------------------------------------

SECTION_PATTERNS = [
    r"^abstract$",
    r"^1\.?\s+introduction",
    r"^2\.?\s+related work",
    r"^3\.?\s+methodology|method|approach|proposed",
    r"^4\.?\s+experiments?|evaluation|results?",
    r"^5\.?\s+discussion",
    r"^6\.?\s+conclusion",
    r"^references?$",
    r"^\d+\.?\s+\w+",
]

SECTION_REGEX = re.compile(
    "|".join(SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE
)


def detect_section(text: str) -> Optional[str]:
    """Try to detect if a line is a section header."""
    line = text.strip()
    if len(line) > 80:
        return None
    if SECTION_REGEX.match(line):
        return line.title()
    return None


# --- PDF Text Extraction ------------------------------------------------------

def extract_with_pymupdf(pdf_path: str) -> Tuple[str, int, Dict[int, str]]:
    """
    Extract full text and per-page text using PyMuPDF.
    Returns: (full_text, page_count, {page_num: page_text})
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages: Dict[int, str] = {}
    all_text_parts = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        pages[page_num] = page_text
        all_text_parts.append(page_text)

    doc.close()
    full_text = "\n".join(all_text_parts)
    return full_text, page_count, pages


def extract_metadata(full_text: str, page_count: int) -> PaperMetadata:
    """
    Heuristically extract title, abstract, authors, year from raw text.
    Works for most IEEE/ACM/Springer papers.
    """
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    metadata = PaperMetadata(page_count=page_count)

    # Title: usually the longest line in the first 15 lines
    first_lines = lines[:15]
    if first_lines:
        candidate_lines = [
            l for l in first_lines
            if "@" not in l and len(l) > 10
        ]
        if candidate_lines:
            metadata.title = max(candidate_lines, key=len)

    # Abstract
    abstract_match = re.search(
        r"(?:abstract|Abstract)[:\s-]*(.*?)(?=\n(?:keywords?|introduction|1\.|I\.))",
        full_text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        abstract_text = re.sub(r"\s+", " ", abstract_text)
        metadata.abstract = abstract_text[:2000]

    # Year: 4-digit year between 1990-2030
    year_match = re.search(r"\b(19[9]\d|20[0-2]\d)\b", full_text[:500])
    if year_match:
        metadata.year = int(year_match.group(1))

    # Authors
    author_section = full_text[:1000]
    author_pattern = re.compile(
        r"^([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+(?:, [A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)*)",
        re.MULTILINE,
    )
    author_matches = author_pattern.findall(author_section)
    if author_matches:
        best = max(author_matches, key=len)
        metadata.authors = [a.strip() for a in best.split(",")]

    return metadata


# --- Semantic Text Chunking ---------------------------------------------------

def chunk_text_by_section(
    pages: Dict[int, str],
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[ExtractedChunk]:
    """
    Chunk paper text with awareness of:
    1. Section boundaries
    2. Paragraph boundaries
    3. Token size limits with overlap
    """
    chunks: List[ExtractedChunk] = []
    chunk_index = 0
    current_section = "Preamble"

    for page_num, page_text in pages.items():
        paragraphs = re.split(r"\n{2,}", page_text)

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue

            section = detect_section(para)
            if section:
                current_section = section
                continue

            words = para.split()
            if len(words) <= chunk_size:
                chunks.append(ExtractedChunk(
                    chunk_index=chunk_index,
                    text=para,
                    section=current_section,
                    page_number=page_num,
                ))
                chunk_index += 1
            else:
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append(ExtractedChunk(
                        chunk_index=chunk_index,
                        text=chunk_text,
                        section=current_section,
                        page_number=page_num,
                    ))
                    chunk_index += 1
                    start += chunk_size - overlap

    return chunks


# --- Main Entry Point ---------------------------------------------------------

def process_pdf(pdf_path: str) -> ProcessedPaper:
    """
    Full pipeline: PDF -> metadata + chunks.
    Called after a paper is uploaded.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text, page_count, pages = extract_with_pymupdf(pdf_path)
    metadata = extract_metadata(full_text, page_count)
    chunks = chunk_text_by_section(
        pages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    return ProcessedPaper(
        metadata=metadata,
        chunks=chunks,
        full_text=full_text,
    )


# --- Text Cleaning ------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove PDF artifacts, normalize whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    return text.strip()