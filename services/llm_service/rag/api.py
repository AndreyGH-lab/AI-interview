from typing import List, Optional

from services.llm_service.rag.retriever import search
from services.llm_service.rag.schema import RetrievedQuestion


def retrieve_questions(
    answer_text: str,
    *,
    domain: str,
    difficulty: str,
    exclude_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[RetrievedQuestion]:
    """
    HIGH-LEVEL Retrieval API.
    Здесь вся логика интервью.
    """

    exclude_ids = set(exclude_ids or [])

    # Берём больше, чтобы отфильтровать
    raw_results = search(answer_text, top_k=top_k * 3)

    results: List[RetrievedQuestion] = []

    for score, q in raw_results:
        if q["id"] in exclude_ids:
            continue

        if q["domain"] != domain:
            continue

        if not _difficulty_allowed(q["difficulty"], difficulty):
            continue

        results.append(
            RetrievedQuestion(
                id=q["id"],
                domain=q["domain"],
                topic=q["topic"],
                difficulty=q["difficulty"],
                question=q["question"],
                rubric=q["rubric"],
                tags=q["tags"],
                score=score,
            )
        )

        if len(results) >= top_k:
            break

    return results


def _difficulty_allowed(q_diff: str, current_diff: str) -> bool:
    """
    Разрешаем вопросы текущего уровня и ниже.
    """
    order = {"junior": 0, "middle": 1, "senior": 2}
    return order[q_diff] <= order[current_diff]
