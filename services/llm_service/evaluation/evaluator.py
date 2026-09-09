from functools import lru_cache
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from services.llm_service.agents.text_utils import safe_json_parse
from shared.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    MODEL_NAME,
    PROMPT_SUFFIX,
)


@lru_cache(maxsize=1)
def get_evaluator_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=MODEL_NAME,
        temperature=0,
        max_tokens=512,
    )

SYSTEM_PROMPT = """
Ты — технический интервьюер-оценщик.

Твоя задача — оценить ответ кандидата на технический вопрос.

Оценивай строго по ожидаемым ключевым аспектам (rubric).
Не додумывай за кандидата.
Если аспект упомянут частично — считай его покрытым.
"score" — число от 0.0 до 1.0, где 0.0 означает, что ни один аспект rubric
не раскрыт, а 1.0 — что раскрыты все аспекты.
"comment" должен быть на русском языке.
Верни результат СТРОГО в формате JSON без пояснений и без markdown.
""" + PROMPT_SUFFIX  # промпт уже заканчивается переводом строки

def _clamp_score(value: Any) -> float:
    """
    Приводит score к диапазону [0.0, 1.0]; некорректное значение — 0.0.
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def evaluate_answer(
    *,
    question: str,
    answer: str,
    rubric: List[str],
) -> Dict[str, Any]:
    """
    Оценивает ответ кандидата по rubric.
    """

    prompt = f"""
Вопрос:
{question}

Ответ кандидата:
{answer}

Ожидаемые ключевые аспекты:
{rubric}

Верни JSON следующего вида (score — число от 0.0 до 1.0):
{{
  "covered": [],
  "missed": [],
  "score": 0.0,
  "comment": ""
}}
"""

    response = get_evaluator_llm().invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    raw_text = response.content

    try:
        data = safe_json_parse(raw_text)

        return {
            "covered": data.get("covered", []),
            "missed": data.get("missed", []),
            "score": _clamp_score(data.get("score", 0.0)),
            "comment": data.get("comment", ""),
            "raw": raw_text,
        }

    except Exception as e:
        #Никогда не роняем пайплайн
        if not (raw_text or "").strip():
            comment = "Модель вернула пустой ответ"
        else:
            comment = "Не удалось разобрать JSON в ответе модели"

        return {
            "covered": [],
            "missed": rubric,
            "score": 0.0,
            "comment": comment,
            "raw": raw_text,
            "error": str(e),
        }
