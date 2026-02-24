from __future__ import annotations

from uuid import uuid4
from typing import List, Dict, Any, Optional

from services.llm_service.agents.agent import graph
from services.stt_service.stt_client import transcribe_audio


def _ask_choice(prompt: str, choices: List[str], default: str) -> str:
    choices_lower = [c.lower() for c in choices]
    default_lower = default.lower()

    while True:
        raw = input(f"{prompt} ({'/'.join(choices)}), по умолчанию '{default}': ").strip().lower()
        if not raw:
            return default_lower
        if raw in choices_lower:
            return raw
        print(f"Неверный ввод. Доступно: {', '.join(choices)}")


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n), по умолчанию '{d}': ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "д", "да"):
            return True
        if raw in ("n", "no", "н", "нет"):
            return False
        print("Введи y или n")


def _format_open_eval(ev: Dict[str, Any]) -> str:
    # аккуратный принт, без падений
    on_topic = ev.get("on_topic", True)
    score = ev.get("score", 0.0)
    covered = ev.get("covered", []) or []
    missed = ev.get("missed", []) or []
    comment = (ev.get("comment") or "").strip()

    lines = []
    lines.append("— Оценка ответа —")
    lines.append(f"on_topic: {on_topic}")
    lines.append(f"score: {score}")
    if covered:
        lines.append("covered: " + ", ".join(map(str, covered)))
    else:
        lines.append("covered: (пусто)")
    if missed:
        lines.append("missed: " + ", ".join(map(str, missed)))
    else:
        lines.append("missed: (пусто)")
    if comment:
        lines.append("comment: " + comment)
    return "\n".join(lines)


def _get_user_input_text() -> Optional[str]:
    """
    Возвращает текст кандидата из:
    - прямого ввода
    - или аудио-файла через STT
    """
    print("\nВвод кандидата:")
    print("  1) Текст")
    print("  2) Аудиофайл (путь)")
    print("  q) Выход")
    mode = input("Выбор (1/2/q): ").strip().lower()

    if mode in ("q", "quit", "exit"):
        return None

    if mode == "1":
        txt = input("Текст кандидата: ").strip()
        return txt if txt else ""

    if mode == "2":
        path = input("Путь к аудиофайлу: ").strip()
        if not path:
            print("Пустой путь.")
            return ""
        lang = input("Язык STT (ru/en), по умолчанию ru: ").strip().lower() or "ru"
        try:
            txt = transcribe_audio(path, language=lang)
            print(f"\n[STT распознано]: {txt}\n")
            return txt
        except Exception as e:
            print(f"[STT ERROR] Не удалось распознать аудио: {e}")
            return ""

    print("Неверный выбор.")
    return ""


def main():
    print("Техническое собеседование (LLM agent + RAG + Evaluation + STT)\n")

    direction = _ask_choice(
        "Направление",
        choices=["ml", "mlops", "both"],
        default="both",
    )
    difficulty = _ask_choice(
        "Сложность",
        choices=["junior", "middle", "senior"],
        default="middle",
    )
    show_open_eval = _ask_yes_no("Показывать открытую оценку после каждого ответа?", default=True)

    if direction == "ml":
        domains = ["ml"]
    elif direction == "mlops":
        domains = ["mlops"]
    else:
        domains = ["ml", "mlops"]

    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n[session] thread_id = {thread_id}")
    print(f"[session] domains = {domains}, difficulty = {difficulty}\n")

    state: Dict[str, Any] = {
        "messages": [],         # агент хранит историю через memory + add_messages
        "domains": domains,
        "difficulty": difficulty,
        "asked_ids": [],
    }

    # стартовый вопрос
    out = graph.invoke(state, config=config)
    last_msg = out["messages"][-1]
    interviewer_text = getattr(last_msg, "content", str(last_msg))
    print(f"[Интервьюер]: {interviewer_text}\n")


    state = out

    while True:
        user_text = _get_user_input_text()
        if user_text is None:
            break

        state_input = {
            **state,
            "messages": [{"role": "user", "content": user_text}],
        }

        out = graph.invoke(state_input, config=config)

        # печатаем ответ интервьюера
        last_msg = out["messages"][-1]
        interviewer_text = getattr(last_msg, "content", str(last_msg))
        print(f"[Интервьюер]: {interviewer_text}\n")


        if show_open_eval:
            ev = out.get("last_evaluation")
            if isinstance(ev, dict):
                print(_format_open_eval(ev))
                print()

        state = out

    print("\nИнтервью завершено.")
    print("asked_ids:", state.get("asked_ids", []))


if __name__ == "__main__":
    main()
