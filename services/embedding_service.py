"""
Embedding Service
- Generates embeddings using sentence-transformers (BGE / SPECTER2 / E5)
- Stores & retrieves vectors using FAISS
- Maps FAISS IDs -> Chunk DB records
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer

from core.config import get_settings

settings = get_settings()


# --- Embedding Model ----------------------------------------------------------

class EmbeddingService:
    """
    Wraps a SentenceTransformer model.
    Supports: BGE-small, BGE-large, SPECTER2, E5, etc.
    """

    def __init__(self):
        print(f"[EmbeddingService] Loading model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingService] Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of strings.
        Returns: float32 numpy array of shape (N, dimension)
        """
        if "bge" in settings.embedding_model.lower():
            texts = [f"Represent this sentence: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query.
        BGE uses a different prefix for queries vs documents.
        """
        if "bge" in settings.embedding_model.lower():
            query = f"Represent this question for searching relevant passages: {query}"
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype("float32")


# --- FAISS Vector Store -------------------------------------------------------

class FAISSVectorStore:
    """
    Manages a FAISS index per paper (paper_id -> index).
    Uses IndexFlatIP (inner product = cosine similarity on normalized vecs).
    Persists to disk.
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._indices: Dict[str, faiss.Index] = {}
        self._id_maps: Dict[str, List[str]] = {}

    def _index_path(self, paper_id: str) -> Path:
        return self.index_dir / f"{paper_id}.faiss"

    def _idmap_path(self, paper_id: str) -> Path:
        return self.index_dir / f"{paper_id}.pkl"

    def _load_index(self, paper_id: str) -> Optional[faiss.Index]:
        """Load index from disk if exists."""
        path = self._index_path(paper_id)
        if path.exists():
            index = faiss.read_index(str(path))
            self._indices[paper_id] = index
            with open(self._idmap_path(paper_id), "rb") as f:
                self._id_maps[paper_id] = pickle.load(f)
            return index
        return None

    def add_paper_embeddings(
        self,
        paper_id: str,
        embeddings: np.ndarray,
        chunk_db_ids: List[str],
        dimension: int,
    ) -> List[int]:
        """
        Add all embeddings for a paper.
        Returns list of FAISS integer IDs assigned.
        """
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self._indices[paper_id] = index
        self._id_maps[paper_id] = chunk_db_ids
        faiss.write_index(index, str(self._index_path(paper_id)))
        with open(self._idmap_path(paper_id), "wb") as f:
            pickle.dump(chunk_db_ids, f)

        return list(range(len(chunk_db_ids)))

    def search(
        self,
        paper_id: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        Returns: list of (chunk_db_id, score) sorted by score desc.
        """
        if paper_id not in self._indices:
            index = self._load_index(paper_id)
            if index is None:
                raise ValueError(
                    f"No FAISS index found for paper {paper_id}. "
                    f"Has the paper been processed?"
                )

        index = self._indices[paper_id]
        id_map = self._id_maps[paper_id]

        scores, faiss_ids = index.search(query_embedding, min(top_k, index.ntotal))

        results = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid == -1:
                continue
            chunk_db_id = id_map[fid]
            results.append((chunk_db_id, float(score)))

        return results

    def delete_paper(self, paper_id: str):
        """Remove a paper's index from disk and memory."""
        self._indices.pop(paper_id, None)
        self._id_maps.pop(paper_id, None)
        self._index_path(paper_id).unlink(missing_ok=True)
        self._idmap_path(paper_id).unlink(missing_ok=True)

    def paper_exists(self, paper_id: str) -> bool:
        return self._index_path(paper_id).exists()


# --- Reranker -----------------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross-encoder reranker using BGE-reranker.
    Falls back gracefully if FlagEmbedding is unavailable.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(model_name, use_fp16=False)  # fp16 off for CPU Windows
            self.available = True
            print(f"[Reranker] Loaded: {model_name}")
        except Exception as e:
            print(f"[Reranker] Disabled — falling back to retrieval scores. Reason: {e}")
            self.available = False

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, str, float]],  # (chunk_id, text, initial_score)
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Rerank chunks using cross-encoder.
        Returns: top_n (chunk_id, rerank_score) sorted by score desc.
        """
        if not self.available or not chunks:
            sorted_chunks = sorted(chunks, key=lambda x: x[2], reverse=True)
            return [(c[0], c[2]) for c in sorted_chunks[:top_n]]

        pairs = [[query, chunk_text] for _, chunk_text, _ in chunks]
        scores = self.reranker.compute_score(pairs)

        reranked = sorted(
            zip([c[0] for c in chunks], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked[:top_n]


# --- Singletons ---------------------------------------------------------------

_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[FAISSVectorStore] = None
_reranker: Optional[CrossEncoderReranker] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store() -> FAISSVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore(settings.faiss_index_path)
    return _vector_store


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker