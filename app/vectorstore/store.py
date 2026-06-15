import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class VectorStore:
    """Thin wrapper around a persisted FAISS index and its chunk metadata."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load index + metadata from disk. Returns False if not found."""
        if not settings.INDEX_PATH.exists() or not settings.METADATA_PATH.exists():
            return False

        self.index = faiss.read_index(str(settings.INDEX_PATH))

        with open(settings.METADATA_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)

        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int | None = None) -> list[dict]:
        """Return the top-k most similar chunks with an added 'score' field."""
        if self.index is None or self._model is None:
            return []

        k = k or settings.TOP_K

        query_vec: np.ndarray = self._model.encode(
            [query], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)

        return results
