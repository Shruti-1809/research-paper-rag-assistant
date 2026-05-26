"""
RAG Research Assistant — FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import get_settings
from models.database import init_db
from models.schemas import HealthResponse
from api.routes import papers, chat, analysis

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown logic."""
    print("🚀 Starting RAG Research Assistant...")
    init_db()
    print("✅ Database initialized")
    # Pre-load embedding model on startup (avoids cold start on first request)
    from services.embedding_service import get_embedding_service
    get_embedding_service()
    print("✅ Embedding model loaded")
    yield
    print("🛑 Shutting down...")


app = FastAPI(
    title="RAG Research Assistant API",
    description="Conversational AI system for research paper understanding using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(papers.router)
app.include_router(chat.router)
app.include_router(analysis.router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        vector_store="FAISS",
    )


@app.get("/", tags=["System"])
def root():
    return {
        "message": "RAG Research Assistant API",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )