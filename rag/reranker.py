"""
BGE Reranker — cross-encoder reranking using BAAI/bge-reranker-base.
Takes top-20 retrieved candidates, returns top-5 reranked results.
"""
from typing import List, Dict, Any
from FlagEmbedding import FlagReranker


class BGEReranker:
    """
    Cross-encoder reranker. Unlike bi-encoders (FAISS/BM25),
    cross-encoders score query-document pairs jointly for higher accuracy.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"[Reranker] Loading cross-encoder: {model_name}")
        self.reranker = FlagReranker(model_name, use_fp16=False)
        self.model_name = model_name
        print("[Reranker] Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents for a given query.

        Args:
            query: The user's query string
            candidates: List of retrieval results (each with 'text' key)
            top_k: Number of final results to return

        Returns:
            Top-K reranked results with 'rerank_score' added
        """
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]
        pairs = [[query, t] for t in texts]

        scores = self.reranker.compute_score(pairs, normalize=True)

        # Handle single-item list edge case
        if isinstance(scores, float):
            scores = [scores]

        # Attach scores and sort
        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]
