"""
Чистые функции обработки текста для узлов графа.

Модуль намеренно зависит только от стандартной библиотеки, чтобы его
импорт не тянул langchain и torch.
"""

import json
import re
from typing import Any, Dict, List, Optional


def strip_reasoning(text: str) -> str:
    """
    Убирает блоки рассуждений reasoning-моделей (<think>...</think>).
    """
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )

    # незакрытый <think> означает, что генерация оборвалась внутри
    # рассуждения — полезного текста после него уже не будет
    if re.search(r"<think\b[^>]*>", cleaned, flags=re.IGNORECASE):
        return ""

    return cleaned.strip()


def safe_json_parse(text: str) -> Dict[str, Any]:
    cleaned = strip_reasoning(text)
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    return json.loads(cleaned)


def ensure_question_prefix(text: str) -> str:
    t = strip_reasoning(text).replace("\n", " ").strip()
    if not t.startswith("Вопрос:"):
        t = "Вопрос: " + t
    return t


MIN_FOLLOWUP_LENGTH = 15


def build_followup_fallback(question: str, missed: Optional[List[Any]] = None) -> str:
    """
    Детерминированный уточняющий вопрос на случай, когда модель не выдала
    пригодного текста. Собирается без обращения к LLM.
    """
    q = (question or "").strip()
    text = f"Вопрос: Давай вернёмся к вопросу. {q}" if q else "Вопрос: Давай вернёмся к предыдущему вопросу."

    first_missed = ""
    for item in missed or []:
        first_missed = str(item).strip()
        if first_missed:
            break

    if first_missed:
        text += f" Отдельно остановись на: {first_missed}"

    return text


def sanitize_prefix(prefix: str, original_question: str) -> str:
    p = (prefix or "").strip().replace("\n", " ")
    q = (original_question or "").strip()
    if not p:
        return ""


    if len(p) > 120:
        return ""

    low_p = p.lower()
    low_q = q.lower()

    # мусорные паттерны
    if "..." in p:
        return ""

    banned_starts = (
        "что такое", "из каких", "зачем", "почему", "какие",
        "mlops", "ml-пайплайн", "пайплайн", "это", "— это", "- это"
    )
    if low_p.startswith(banned_starts):
        return ""

    if "?" in p:
        return ""


    # ветка low_q[-20:] недостижима для вопросов, заканчивающихся на "?":
    # префикс с "?" отсеивается проверкой выше
    if len(q) >= 20 and (low_q[:20] in low_p or low_q[-20:] in low_p):
        return ""

    if not p.endswith((".", ":", "—")):
        p += "."

    return p
