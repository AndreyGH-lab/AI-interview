from __future__ import annotations

import json
import re
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from services.llm_service.rag.api import retrieve_questions
from services.llm_service.rag.store import load_index
from services.llm_service.evaluation.evaluator import evaluate_answer

from shared.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    MODEL_NAME,
    DRIFT_SCORE_THRESHOLD,
    ON_TOPIC_OVERRIDE_SCORE,
)



SELECTOR_SYSTEM_PROMPT = """
Ты — ИИ-интервьюер по ML и MLOps.

Тебе передадут:
- последний ответ кандидата
- результаты оценки ответа (может быть пусто)
- список asked_ids (уже заданные вопросы)
- кандидаты следующих вопросов (JSON)

Твоя задача:
- выбрать ОДИН лучший следующий вопрос
- вернуть СТРОГО JSON:

{
  "question_id": "<id>"
}

Правила:
- отвечай ТОЛЬКО JSON (без markdown)
- не добавляй пояснений
- не придумывай вопросы
- используй только предложенные варианты
- не выбирай id из asked_ids
""".strip()

llm_selector = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=96,
)


RENDER_SYSTEM_PROMPT = """
Ты — ИИ-интервьюер.

Тебе передадут:
- selected_question.question (исходный вопрос из банка, его переписывать нельзя)
- last_answer и evaluation

Твоя задача:
- вернуть JSON: {"prefix": "..."} где prefix — КОРОТКАЯ подводка (0-1 предложение)
- prefix НЕ должен повторять вопрос и НЕ должен начинаться словами "Что такое", "Из каких этапов", "Зачем"
- prefix не должен содержать "?" и не должен содержать сам вопрос целиком или частями

Формат СТРОГО JSON без markdown:
{"prefix": "..."}
""".strip()

llm_renderer = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=96,
)

FOLLOWUP_SYSTEM_PROMPT = """
Ты — ИИ-интервьюер.

Тебе передадут:
- question: исходный вопрос
- answer: ответ кандидата
- evaluation: {covered, missed, score, comment, on_topic}

Если on_topic=false ИЛИ missed не пустой — верни кандидата к вопросу.

Твоя задача:
- задать ОДИН уточняющий вопрос по ТОМУ ЖЕ вопросу
- обязательно опирайся на 1–2 пункта из missed (упомяни их явно или перефразируй)
- не меняй тему, не задавай новый вопрос из банка

Формат:
ОДНА строка, начинается строго с "Вопрос: "
""".strip()

llm_followup = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=128,
)

DRIFT_SYSTEM_PROMPT = """
Ты — ассистент, который проверяет, отвечает ли кандидат ПО ТЕМЕ вопроса.

Тебе дадут:
- question: исходный вопрос
- rubric: ожидаемые аспекты
- answer: ответ кандидата

Верни СТРОГО JSON:
{
  "on_topic": true/false
}

Правила (важно):
- true, если ответ хотя бы частично отвечает на вопрос или обсуждает 1+ аспект из rubric
- false, только если кандидат явно ушёл в другую тему и НЕ отвечает на вопрос
- если есть небольшие отвлечения, но ответ всё равно по сути — true
- никакого markdown, только JSON
""".strip()

llm_drift = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=48,
)


class Candidate(TypedDict):
    id: str
    domain: str
    topic: str
    difficulty: str
    question: str
    rubric: List[str]
    score: float


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    domains: List[str]
    difficulty: str
    asked_ids: List[str]

    last_answer: Optional[str]
    last_question_id: Optional[str]
    last_question_text: Optional[str]
    last_rubric: Optional[List[str]]

    last_evaluation: Optional[Dict[str, Any]]
    candidates: Optional[List[Candidate]]
    selected: Optional[Candidate]

    is_first_turn: bool



def _difficulty_allowed(q_diff: str, current_diff: str) -> bool:
    order = {"junior": 0, "middle": 1, "senior": 2}
    return order.get(q_diff, 0) <= order.get(current_diff, 0)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    return json.loads(cleaned)


def _extract_last_user_text(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            txt = (msg.content or "").strip()
            return txt if txt else None
    return None


def _ensure_question_prefix(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t.startswith("Вопрос:"):
        t = "Вопрос: " + t
    return t


def _pick_first_question_from_bank(
    domains: List[str],
    difficulty: str,
    asked_ids: List[str],
) -> Optional[Candidate]:
    _, meta = load_index()
    for q in meta:
        if q["id"] in asked_ids:
            continue
        if q.get("domain") not in domains:
            continue
        if not _difficulty_allowed(q.get("difficulty", "junior"), difficulty):
            continue

        return Candidate(
            id=q["id"],
            domain=q["domain"],
            topic=q["topic"],
            difficulty=q["difficulty"],
            question=q["question"],
            rubric=q.get("rubric", []),
            score=0.0,
        )
    return None



def prepare_state(state: State) -> Dict[str, Any]:
    messages = state.get("messages", [])

    domains = state.get("domains") or ["ml", "mlops"]
    difficulty = state.get("difficulty") or "middle"
    asked_ids = state.get("asked_ids") or []

    last_question_id = state.get("last_question_id")
    is_first_turn = last_question_id is None

    last_answer = _extract_last_user_text(messages)

    return {
        "domains": domains,
        "difficulty": difficulty,
        "asked_ids": asked_ids,
        "last_answer": last_answer,
        "is_first_turn": is_first_turn,
    }


def route_first_turn(state: State) -> str:
    return "init_question" if state.get("is_first_turn", False) else "evaluate"


def init_question(state: State) -> Dict[str, Any]:
    q = _pick_first_question_from_bank(
        domains=state["domains"],
        difficulty=state["difficulty"],
        asked_ids=state["asked_ids"],
    )

    if not q:
        fallback = Candidate(
            id="fallback_intro",
            domain="ml",
            topic="intro",
            difficulty=state["difficulty"],
            question="Расскажи, пожалуйста, про один ML/MLOps проект и твою роль в нём.",
            rubric=[],
            score=0.0,
        )
        asked_ids = state["asked_ids"] + [fallback["id"]]
        return {
            "asked_ids": asked_ids,
            "last_question_id": fallback["id"],
            "last_question_text": fallback["question"],
            "last_rubric": fallback["rubric"],
            "last_evaluation": None,
            "candidates": None,
            "selected": fallback,
        }

    asked_ids = state["asked_ids"] + [q["id"]]
    return {
        "asked_ids": asked_ids,
        "last_question_id": q["id"],
        "last_question_text": q["question"],
        "last_rubric": q["rubric"],
        "last_evaluation": None,
        "candidates": None,
        "selected": q,
    }



def evaluate_node(state: State) -> Dict[str, Any]:
    if not state.get("last_question_text") or state.get("last_rubric") is None:
        return {"last_evaluation": None}

    answer = state.get("last_answer")
    if not answer:
        return {"last_evaluation": None}

    ev = evaluate_answer(
        question=state["last_question_text"],
        answer=answer,
        rubric=state["last_rubric"] or [],
    )

    # drift judge
    drift_payload = {
        "question": state["last_question_text"],
        "rubric": state["last_rubric"] or [],
        "answer": answer,
    }
    on_topic = True
    try:
        resp = llm_drift.invoke(
            [
                SystemMessage(content=DRIFT_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(drift_payload, ensure_ascii=False)),
            ]
        )
        data = _safe_json_parse(resp.content)
        on_topic = bool(data.get("on_topic", True))
    except Exception:
        on_topic = True

    score = 0.0
    missed: List[Any] = []
    try:
        score = float(ev.get("score", 0.0)) if isinstance(ev, dict) else 0.0
    except Exception:
        score = 0.0
    try:
        missed = (ev.get("missed", []) or []) if isinstance(ev, dict) else []
    except Exception:
        missed = []

    if score >= ON_TOPIC_OVERRIDE_SCORE and len(missed) == 0:
        on_topic = True

    if isinstance(ev, dict):
        ev = dict(ev)
        ev["on_topic"] = on_topic

    return {"last_evaluation": ev}


def route_after_evaluate(state: State) -> str:
    ev = state.get("last_evaluation") or {}
    rubric = state.get("last_rubric") or []
    answer = (state.get("last_answer") or "").strip()

    if not answer or not rubric:
        return "retrieve"

    on_topic = bool(ev.get("on_topic", True))

    try:
        score = float(ev.get("score", 0.0))
    except Exception:
        score = 0.0

    covered = ev.get("covered", []) or []
    missed = ev.get("missed", []) or []

    if score >= ON_TOPIC_OVERRIDE_SCORE and len(missed) == 0:
        return "retrieve"


    if (not on_topic) or (score < DRIFT_SCORE_THRESHOLD) or (len(covered) == 0 and len(rubric) > 0) or (
        len(missed) >= max(1, len(rubric) - 1)
    ):
        return "followup"

    return "retrieve"

def followup_node(state: State) -> Dict[str, Any]:
    q = state.get("last_question_text") or ""
    a = state.get("last_answer") or ""
    ev = state.get("last_evaluation") or {}

    payload = {
        "question": q,
        "answer": a,
        "evaluation": {
            "covered": ev.get("covered", []),
            "missed": ev.get("missed", []),
            "score": ev.get("score", 0.0),
            "comment": ev.get("comment", ""),
            "on_topic": ev.get("on_topic", True),
        },
    }

    resp = llm_followup.invoke(
        [
            SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )

    msg_text = _ensure_question_prefix(resp.content)
    return {
        "messages": [AIMessage(content=msg_text)],
        "candidates": None,
        "selected": None,
    }



def retrieve_node(state: State) -> Dict[str, Any]:
    answer = state.get("last_answer") or ""
    asked_ids = state["asked_ids"]
    difficulty = state["difficulty"]
    domains = state["domains"]

    merged: Dict[str, Candidate] = {}

    for dom in domains:
        results = retrieve_questions(
            answer_text=answer,
            domain=dom,
            difficulty=difficulty,
            exclude_ids=asked_ids,
            top_k=3,
        )
        for q in results:
            merged[q.id] = Candidate(
                id=q.id,
                domain=q.domain,
                topic=q.topic,
                difficulty=q.difficulty,
                question=q.question,
                rubric=q.rubric,
                score=float(q.score),
            )

    candidates = list(merged.values())
    candidates.sort(key=lambda x: x["score"], reverse=True)

    return {"candidates": candidates}


def select_node(state: State) -> Dict[str, Any]:
    candidates = state.get("candidates") or []

    if not candidates:
        fallback = Candidate(
            id="fallback_deep",
            domain="ml",
            topic="project",
            difficulty=state["difficulty"],
            question="Расскажи, пожалуйста, о самом сложном ML/MLOps проекте, который ты делал, и какие там были компромиссы.",
            rubric=[],
            score=0.0,
        )
        asked_ids = state["asked_ids"] + [fallback["id"]]
        return {
            "asked_ids": asked_ids,
            "last_question_id": fallback["id"],
            "last_question_text": fallback["question"],
            "last_rubric": fallback["rubric"],
            "selected": fallback,
        }

    payload = {
        "last_answer": state.get("last_answer"),
        "evaluation": state.get("last_evaluation"),
        "asked_ids": state.get("asked_ids"),
        "candidates": [
            {
                "id": c["id"],
                "domain": c["domain"],
                "difficulty": c["difficulty"],
                "question": c["question"],
                "rubric": c["rubric"],
                "score": c["score"],
            }
            for c in candidates[:5]
        ],
    }

    resp = llm_selector.invoke(
        [
            SystemMessage(content=SELECTOR_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )

    chosen_id: Optional[str] = None
    try:
        data = _safe_json_parse(resp.content)
        chosen_id = data.get("question_id")
    except Exception:
        chosen_id = None

    chosen: Optional[Candidate] = None
    if chosen_id and chosen_id not in set(state["asked_ids"]):
        for c in candidates:
            if c["id"] == chosen_id:
                chosen = c
                break

    if not chosen:
        chosen = candidates[0]

    asked_ids = state["asked_ids"] + [chosen["id"]]

    return {
        "asked_ids": asked_ids,
        "last_question_id": chosen["id"],
        "last_question_text": chosen["question"],
        "last_rubric": chosen["rubric"],
        "selected": chosen,
    }

def _sanitize_prefix(prefix: str, original_question: str) -> str:
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


    if len(q) >= 20 and (low_q[:20] in low_p or low_q[-20:] in low_p):
        return ""

    if not p.endswith((".", ":", "—")):
        p += "."

    return p


def render_node(state: State) -> Dict[str, Any]:
    selected = state.get("selected")
    if not selected:
        return {"messages": [AIMessage(content="Вопрос: Расскажи, пожалуйста, про один ML/MLOps проект и твою роль в нём.")]}

    payload = {
        "selected_question": {
            "id": selected["id"],
            "domain": selected["domain"],
            "difficulty": selected["difficulty"],
            "question": selected["question"],
        },
        "last_answer": state.get("last_answer"),
        "evaluation": state.get("last_evaluation"),
    }

    prefix = ""
    try:
        resp = llm_renderer.invoke(
            [
                SystemMessage(content=RENDER_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        data = _safe_json_parse(resp.content)
        raw_prefix = (data.get("prefix") or "")
        prefix = _sanitize_prefix(raw_prefix, selected["question"])
    except Exception:
        prefix = ""

    if prefix:
        text = f"Вопрос: {prefix} {selected['question']}"
    else:
        text = f"Вопрос: {selected['question']}"

    return {
        "messages": [AIMessage(content=text)],
        "candidates": None,
        "selected": None,
    }

graph_builder = StateGraph(State)

graph_builder.add_node("prepare", prepare_state)
graph_builder.add_node("init_question", init_question)
graph_builder.add_node("evaluate", evaluate_node)
graph_builder.add_node("followup", followup_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("select", select_node)
graph_builder.add_node("render", render_node)

graph_builder.add_edge(START, "prepare")

graph_builder.add_conditional_edges(
    "prepare",
    route_first_turn,
    {"init_question": "init_question", "evaluate": "evaluate"},
)


graph_builder.add_edge("init_question", "render")
graph_builder.add_edge("render", END)

graph_builder.add_conditional_edges(
    "evaluate",
    route_after_evaluate,
    {"followup": "followup", "retrieve": "retrieve"},
)

graph_builder.add_edge("followup", END)

graph_builder.add_edge("retrieve", "select")
graph_builder.add_edge("select", "render")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
