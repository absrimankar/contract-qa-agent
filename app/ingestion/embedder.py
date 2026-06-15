import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


def build_index(chunks: list[dict]) -> None:
    """
    Embed all chunks with sentence-transformers and persist a FAISS index
    plus the chunk metadata to disk.

    Uses cosine similarity via L2-normalised inner-product search.
    """
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    embeddings: np.ndarray = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    # Normalise so inner-product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(settings.INDEX_PATH))

    with open(settings.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
