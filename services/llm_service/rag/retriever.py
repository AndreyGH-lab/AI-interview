from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from services.llm_service.rag.store import (
    load_index,
    EMBEDDING_MODEL_NAME,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # импорт внутри функции: sentence_transformers тянет torch (~14 с)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_index():
    return load_index()


def search(
    query_text: str,
    top_k: int = 10,
) -> List[Tuple[float, dict]]:
    index, meta = _get_index()

    query_embedding = _get_model().encode(
        [query_text],
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results: List[Tuple[float, dict]] = []

    for score, idx in zip(scores[0], indices[0]):
        q = meta[idx]
        results.append((float(score), q))

    return results
