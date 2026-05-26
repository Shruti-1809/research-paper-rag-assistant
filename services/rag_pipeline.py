"""
RAG Pipeline Service
Steps: Query -> Embed -> Retrieve -> Rerank -> Prompt -> LLM -> Response
Supports: OpenAI, Anthropic, Gemini, Groq
"""

from typing import List, Optional, Tuple, Dict
from sqlalchemy.orm import Session

from core.config import get_settings
from models.database import Chunk, Paper
from models.schemas import Citation
from services.embedding_service import (
    get_embedding_service, get_vector_store, get_reranker
)

settings = get_settings()


# --- Query Rewriter -----------------------------------------------------------

def rewrite_query(query: str, chat_history: List[Dict]) -> str:
    """
    Rewrites ambiguous queries using recent chat context.
    Example: "What about its limitations?" ->
             "In the context of '...': What about its limitations?"
    """
    ambiguous_pronouns = ["it", "its", "they", "their", "this", "that", "these", "those"]
    words = query.lower().split()
    is_ambiguous = any(w in ambiguous_pronouns for w in words[:4])

    if not is_ambiguous or not chat_history:
        return query

    recent_topics = []
    for msg in chat_history[-4:]:
        if msg["role"] == "user":
            recent_topics.append(msg["content"])

    context_str = " | ".join(recent_topics[-2:])
    return f"In the context of '{context_str}': {query}"


# --- Prompt Builder -----------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a Research Paper Assistant specializing in academic literature.

Your role is to help researchers understand research papers by:
1. Answering questions based ONLY on the provided paper context
2. Citing specific sections when possible (e.g., "According to Section 3...")
3. Acknowledging when information is not in the provided context
4. Being precise and technical when the question demands it

Rules:
- Ground every answer in the retrieved context
- Never hallucinate facts not in the context
- Use academic language appropriate for research
- If the context is insufficient, say so clearly
"""


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Tuple[str, str, Optional[str], Optional[int]]],
    chat_history: List[Dict],
) -> List[Dict]:
    """Build the full message list for the LLM."""

    context_parts = []
    for i, (chunk_id, text, section, page) in enumerate(retrieved_chunks, 1):
        location = ""
        if section:
            location += f"[{section}]"
        if page:
            location += f" [Page {page}]"
        context_parts.append(f"[Context {i}]{location}\n{text}")

    context_str = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"""Based on the following context from the research paper, answer the question.

RETRIEVED CONTEXT:
{context_str}

QUESTION:
{query}

Provide a comprehensive answer citing specific sections where relevant."""

    messages.append({"role": "user", "content": user_message})
    return messages


# --- LLM Client --------------------------------------------------------------

async def call_llm(messages: List[Dict]) -> str:
    """Call the configured LLM provider."""

    if settings.llm_provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    elif settings.llm_provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        conv_messages = [m for m in messages if m["role"] != "system"]
        response = await client.messages.create(
            model=settings.llm_model,
            system=system_msg,
            messages=conv_messages,
            max_tokens=1500,
        )
        return response.content[0].text

    elif settings.llm_provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        history_parts = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                history_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                history_parts.append(f"Assistant: {msg['content']}")
        full_prompt = f"{system_msg}\n\n" + "\n\n".join(history_parts)
        model = genai.GenerativeModel(settings.llm_model)
        response = await model.generate_content_async(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500,
            ),
        )
        return response.text

    elif settings.llm_provider == "groq":
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,  # Groq supports system messages natively
            temperature=0.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


# --- Citation Extractor -------------------------------------------------------

def extract_citations(
    retrieved_chunks: List[Tuple[str, str, Optional[str], Optional[int]]],
    scores: List[float],
    paper_id: str,
    paper_title: Optional[str],
) -> List[Citation]:
    """Convert retrieved chunks into Citation objects."""
    citations = []
    for (chunk_id, text, section, page), score in zip(retrieved_chunks, scores):
        snippet = text[:200] + "..." if len(text) > 200 else text
        citations.append(Citation(
            chunk_id=chunk_id,
            paper_id=paper_id,
            paper_title=paper_title,
            section=section,
            page_number=page,
            text_snippet=snippet,
            relevance_score=round(score, 4),
        ))
    return citations


# --- Main RAG Pipeline --------------------------------------------------------

async def run_rag_pipeline(
    query: str,
    paper_id: str,
    db: Session,
    chat_history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Full RAG pipeline:
    query -> rewrite -> embed -> retrieve -> rerank -> LLM -> response
    """
    chat_history = chat_history or []

    # Step 1: Query Rewriting
    rewritten_query = rewrite_query(query, chat_history)

    # Step 2: Embed Query
    embedder = get_embedding_service()
    query_embedding = embedder.embed_query(rewritten_query)

    # Step 3: Retrieve Top-K from FAISS
    vector_store = get_vector_store()
    retrieved = vector_store.search(
        paper_id=paper_id,
        query_embedding=query_embedding,
        top_k=settings.top_k_retrieval,
    )

    if not retrieved:
        return {
            "answer": "I could not find relevant information in the paper for your question.",
            "citations": [],
            "chunk_ids": [],
            "confidence_score": 0.0,
        }

    # Step 4: Fetch Chunk Texts from DB
    chunk_ids = [r[0] for r in retrieved]
    retrieval_scores = {r[0]: r[1] for r in retrieved}

    chunks_db = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
    chunk_map = {c.id: c for c in chunks_db}

    # Step 5: Rerank
    reranker = get_reranker()
    chunks_for_rerank = [
        (cid, chunk_map[cid].text, retrieval_scores[cid])
        for cid in chunk_ids
        if cid in chunk_map
    ]

    top_chunks = reranker.rerank(
        query=rewritten_query,
        chunks=chunks_for_rerank,
        top_n=settings.top_k_rerank,
    )

    # Step 6: Build Context for LLM
    final_chunk_data = []
    final_scores = []
    for chunk_id, score in top_chunks:
        if chunk_id not in chunk_map:
            continue
        c = chunk_map[chunk_id]
        final_chunk_data.append((c.id, c.text, c.section, c.page_number))
        final_scores.append(score)

    # Step 7: Get Paper Title
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    paper_title = paper.title if paper else None

    # Step 8: Call LLM
    messages = build_rag_prompt(
        query=rewritten_query,
        retrieved_chunks=final_chunk_data,
        chat_history=chat_history,
    )
    answer = await call_llm(messages)

    # Step 9: Build Citations
    citations = extract_citations(
        retrieved_chunks=final_chunk_data,
        scores=final_scores,
        paper_id=paper_id,
        paper_title=paper_title,
    )

    # Reranker scores can be negative logits — normalize to [0, 1]
    raw_confidence = sum(final_scores) / len(final_scores) if final_scores else 0.0
    confidence = max(0.0, min(1.0, abs(raw_confidence) / 10.0))

    return {
        "answer": answer,
        "citations": citations,
        "chunk_ids": [c[0] for c in final_chunk_data],
        "confidence_score": round(confidence, 3),
    }