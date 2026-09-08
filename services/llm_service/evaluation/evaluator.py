import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from shared.config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, MODEL_NAME

llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=300,
)

SYSTEM_PROMPT = """
Ты — технический интервьюер-оценщик.

Твоя задача — оценить ответ кандидата на технический вопрос.

Оценивай строго по ожидаемым ключевым аспектам (rubric).
Не додумывай за кандидата.
Если аспект упомянут частично — считай его покрытым.
"comment" должен быть на русском языке.
Верни результат СТРОГО в формате JSON без пояснений и без markdown.
"""

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Парсит JSON даже если он обёрнут в ```json ... ```
    """
    cleaned = text.strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    return json.loads(cleaned)


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

Верни JSON следующего вида:
{{
  "covered": [],
  "missed": [],
  "score": 0.0,
  "comment": ""
}}
"""

    response = llm.invoke(
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
            "score": float(data.get("score", 0.0)),
            "comment": data.get("comment", ""),
            "raw": raw_text,
        }

    except Exception as e:
        #Никогда не роняем пайплайн
        return {
            "covered": [],
            "missed": rubric,
            "score": 0.0,
            "comment": "Ошибка разбора ответа модели",
            "raw": raw_text,
            "error": str(e),
        }
