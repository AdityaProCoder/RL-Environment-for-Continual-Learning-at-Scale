"""
Embedding-based Router using MiniLM / fallback embeddings and incremental clustering.
"""

from typing import Dict, List, Optional
import numpy as np
from open_continual_env.routing.router import Router
from open_continual_env.memory.faiss_memory import ST_AVAILABLE, SentenceTransformer, SimpleEmbedder


class EmbeddingRouter(Router):
    """
    Semantic router mapping prompts to cluster IDs via embedding distance & heuristic fallback.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.dimension = dimension
        if ST_AVAILABLE and os.getenv("ENABLE_HF_EMBEDDINGS") == "1":
            try:
                self.embedder = SentenceTransformer(model_name, local_files_only=True)
                self.dimension = self.embedder.get_sentence_embedding_dimension()
            except Exception:
                self.embedder = SimpleEmbedder(dim=self.dimension)
        else:
            self.embedder = SimpleEmbedder(dim=self.dimension)


        self.cluster_centroids: Dict[str, np.ndarray] = {}
        self.cluster_counts: Dict[str, int] = {
            "cluster_algorithms": 0,
            "cluster_math": 0,
            "cluster_text": 0,
            "cluster_general": 0,
        }
        self.adapters: Dict[str, str] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        if hasattr(self.embedder, "encode"):
            emb = self.embedder.encode(text, normalize_embeddings=True)
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)
            return emb.astype(np.float32)
        return SimpleEmbedder(dim=self.dimension).encode(text)

    def _keyword_heuristic(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ["sort", "array", "search", "tree", "graph", "algorithm", "binary", "dp"]):
            return "cluster_algorithms"
        if any(k in p for k in ["math", "calculate", "number", "sum", "multiply", "prime", "equation"]):
            return "cluster_math"
        if any(k in p for k in ["string", "text", "regex", "format", "word", "char", "parse"]):
            return "cluster_text"
        return "cluster_general"

    def get_cluster_id(self, prompt: str) -> str:
        emb = self._get_embedding(prompt)

        # If centroids exist, find nearest centroid
        best_cluster = None
        best_sim = -1.0

        for c_id, centroid in self.cluster_centroids.items():
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_cluster = c_id

        # If high similarity match (>0.6), return centroid cluster
        if best_cluster is not None and best_sim >= 0.6:
            cluster_id = best_cluster
        else:
            cluster_id = self._keyword_heuristic(prompt)

        # Update centroid incrementally
        if cluster_id not in self.cluster_centroids:
            self.cluster_centroids[cluster_id] = emb
        else:
            c_cnt = self.cluster_counts.get(cluster_id, 0) + 1
            curr_c = self.cluster_centroids[cluster_id]
            updated_c = (curr_c * c_cnt + emb) / (c_cnt + 1)
            norm = np.linalg.norm(updated_c)
            if norm > 0:
                updated_c = updated_c / norm
            self.cluster_centroids[cluster_id] = updated_c

        self.cluster_counts[cluster_id] = self.cluster_counts.get(cluster_id, 0) + 1
        return cluster_id

    def get_adapter(self, cluster_id: str) -> Optional[str]:
        return self.adapters.get(cluster_id)

    def register_adapter(self, cluster_id: str, adapter_path: str) -> None:
        self.adapters[cluster_id] = adapter_path

    def list_clusters(self) -> Dict[str, int]:
        return dict(self.cluster_counts)
