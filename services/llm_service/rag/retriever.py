from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from services.llm_service.rag.store import (
    load_index,
    EMBEDDING_MODEL_NAME,
)

_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

_index, _meta = load_index()


def search(
    query_text: str,
    top_k: int = 10,
) -> List[Tuple[float, dict]]:
    query_embedding = _embedding_model.encode(
        [query_text],
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = _index.search(query_embedding, top_k)

    results: List[Tuple[float, dict]] = []

    for score, idx in zip(scores[0], indices[0]):
        q = _meta[idx]
        results.append((float(score), q))

    return results
