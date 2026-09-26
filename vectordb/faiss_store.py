"""
FAISS Vector Store — persistent index with metadata support.
Stores embeddings + metadata (chunk text, source, section, topic, year).
"""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class FAISSStore:
    """
    FAISS-based vector store with parallel metadata storage.
    Uses IndexFlatIP (Inner Product) with L2-normalized vectors = cosine similarity.
    """

    def __init__(self, embedding_dim: int = 384, index_path: str = "vectordb/faiss_index"):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []

    def build(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        """Build FAISS index from embedded chunks."""
        print(f"[FAISS] Building index from {len(embedded_chunks)} chunks...", flush=True)

        embeddings = np.array(
            [c["embedding"] for c in embedded_chunks], dtype=np.float32
        )

        # Ensure L2 normalized
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        self.texts = [c.get("chunk_text", "") for c in embedded_chunks]
        self.metadata = [c.get("metadata", {}) for c in embedded_chunks]

        print(f"[FAISS] Index built. Total vectors: {self.index.ntotal}", flush=True)

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

        with open(self.index_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        with open(self.index_path / "texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

        print(f"[FAISS] Saved index → {self.index_path}", flush=True)

    def load(self) -> None:
        """Load persisted FAISS index and metadata from disk."""
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. "
                "Run: python scripts/build_index.py"
            )

        self.index = faiss.read_index(str(index_file))

        with open(self.index_path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(self.index_path / "texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)

        print(f"[FAISS] Loaded index. Total vectors: {self.index.ntotal}", flush=True)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter_section: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense retrieval with optional metadata filtering.

        Returns list of result dicts:
        [{"text": str, "metadata": dict, "score": float, "rank": int}]
        """
        if self.index is None:
            self.load()

        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)

        # Retrieve more than top_k to allow for post-filtering
        k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            meta = self.metadata[idx]

            # Apply filters
            if filter_section and meta.get("section", "") != filter_section:
                continue
            if filter_source and filter_source.lower() not in meta.get("source", "").lower():
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": meta,
                "score": float(score),
                "index": int(idx),
            })

            if len(results) >= top_k:
                break

        return results

    def is_loaded(self) -> bool:
        return self.index is not None
