"""
BGE Reranker -- cross-encoder reranking using sentence-transformers CrossEncoder.

Fallback chain:
  1. BAAI/bge-reranker-base  (best quality, requires download)
  2. cross-encoder/ms-marco-MiniLM-L-6-v2  (lighter, requires download)
  3. TF-IDF cosine similarity  (zero dependencies, always works offline)
"""
import sys
sys.modules['tensorflow'] = None

from typing import List, Dict, Any


class BGEReranker:
    """
    Cross-encoder reranker with a 3-tier fallback strategy.
    Falls back to TF-IDF cosine similarity if no model can be loaded.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model_name = model_name
        self._reranker = None
        self._available = False
        self._mode = "tfidf"  # default until a real model loads
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """
        Attempt to load CrossEncoder models in order.
        Falls back to TF-IDF cosine similarity if all fail.
        """
        models_to_try = [
            model_name,
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ]

        for m in models_to_try:
            try:
                # pyrefly: ignore [missing-import]
                from sentence_transformers import CrossEncoder
                print(f"[Reranker] Loading cross-encoder: {m}", flush=True)

                # Explicitly pass device and trust_remote_code to avoid NoneType errors
                reranker = CrossEncoder(
                    m,
                    max_length=512,
                    device="cpu",
                )
                # Verify it works by checking it has a predict method
                if not hasattr(reranker, "predict"):
                    raise RuntimeError("CrossEncoder loaded but has no predict method")

                self._reranker = reranker
                self._available = True
                self._model_name = m
                self._mode = "crossencoder"
                print(f"[Reranker] Cross-encoder loaded: {m}", flush=True)
                return

            except Exception as e:
                print(f"[Reranker] Warning: Could not load '{m}': {e}", flush=True)
                continue

        # Final fallback: TF-IDF cosine similarity (zero external dependencies)
        print("[Reranker] Using TF-IDF cosine similarity fallback (offline, always works).", flush=True)
        self._mode = "tfidf"
        self._available = True  # TF-IDF is always available

    def _tfidf_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank using TF-IDF + cosine similarity.
        sklearn is already installed (required by ragas and many other deps).
        Falls back to score-based slicing if sklearn is missing.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            texts = [c.get("text", "") for c in candidates]
            if not any(texts):
                return candidates[:top_k]

            # Fit TF-IDF on all documents + query together
            corpus = [query] + texts
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Query is index 0; documents are indices 1..N
            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]

            scores = cosine_similarity(query_vec, doc_vecs)[0]

            scored = [
                {**c, "rerank_score": float(scores[i])}
                for i, c in enumerate(candidates)
            ]
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:top_k]

        except ImportError:
            print("[Reranker] sklearn not available. Using score-based slicing.")
            scored = [
                {**c, "rerank_score": float(c.get("score", c.get("rrf_score", 0.0)))}
                for c in candidates
            ]
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:top_k]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents for a given query.

        Args:
            query:      The user's query string
            candidates: List of retrieval results (each with 'text' key)
            top_k:      Number of final results to return
        """
        if not candidates:
            return []

        # TF-IDF fallback mode
        if self._mode == "tfidf":
            return self._tfidf_rerank(query, candidates, top_k)

        # CrossEncoder mode
        if self._mode == "crossencoder" and self._reranker is not None:
            try:
                texts = [c.get("text", "") for c in candidates]
                pairs = [[query, t] for t in texts]
                scores = self._reranker.predict(pairs)

                if not hasattr(scores, "__len__"):
                    scores = [float(scores)]

                scored = [
                    {**c, "rerank_score": float(s)}
                    for c, s in zip(candidates, scores)
                ]
                scored.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored[:top_k]

            except Exception as e:
                print(f"[Reranker] CrossEncoder error: {e}. Falling back to TF-IDF.")
                return self._tfidf_rerank(query, candidates, top_k)

        # Final safety net
        return self._tfidf_rerank(query, candidates, top_k)
