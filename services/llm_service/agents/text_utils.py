"""
Чистые функции обработки текста для узлов графа.

Модуль намеренно зависит только от стандартной библиотеки, чтобы его
импорт не тянул langchain и torch.
"""

import json
import re
from typing import Any, Dict


def safe_json_parse(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    return json.loads(cleaned)


def ensure_question_prefix(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t.startswith("Вопрос:"):
        t = "Вопрос: " + t
    return t


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
