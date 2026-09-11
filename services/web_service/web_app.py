from __future__ import annotations

import json
import os
import tempfile
from uuid import uuid4
from typing import Any, Dict, List, Optional

import gradio as gr

from services.llm_service.agents.agent import graph
from services.stt_service.stt_client import transcribe_audio
from shared.health import check_model_or_warn


def _domain_choice_to_domains(choice: str) -> List[str]:
    choice = (choice or "both").strip().lower()
    if choice == "ml":
        return ["ml"]
    if choice == "mlops":
        return ["mlops"]
    return ["ml", "mlops"]


def _pretty_eval(ev: Optional[Dict[str, Any]]) -> str:
    if not ev:
        return ""

    covered = ev.get("covered", []) or []
    missed = ev.get("missed", []) or []
    score = ev.get("score", None)
    on_topic = ev.get("on_topic", None)
    comment = ev.get("comment", "") or ""

    def fmt_list(xs: List[str]) -> str:
        if not xs:
            return "—"
        return "\n".join([f"- {x}" for x in xs])

    parts = []
    if on_topic is not None:
        parts.append(f"**on_topic:** `{on_topic}`")
    if score is not None:
        parts.append(f"**score:** `{score}`")
    parts.append(f"**covered:**\n{fmt_list(covered)}")
    parts.append(f"**missed:**\n{fmt_list(missed)}")
    if comment.strip():
        parts.append(f"**comment:** {comment}")

    return "\n\n".join(parts)


def _ensure_msg_text(x: Any) -> str:
    return getattr(x, "content", str(x))


def _new_session_state(thread_id: str, domains: List[str], difficulty: str) -> Dict[str, Any]:
    return {
        "thread_id": thread_id,
        "domains": domains,
        "difficulty": difficulty,
        "started": True,
        "current_question": {"id": None, "text": None, "rubric": None},
        "turns": [],
        "asked_ids": [],
    }


def new_session_ui():
    """
    Разлочить настройки, очистить чат, сбросить session_state.
    """
    thread_id = str(uuid4())
    session_state = {
        "thread_id": thread_id,
        "domains": ["ml", "mlops"],
        "difficulty": "middle",
        "started": False,
        "current_question": {"id": None, "text": None, "rubric": None},
        "turns": [],
        "asked_ids": [],
    }

    chat: List[Dict[str, str]] = []
    recognized = ""
    eval_md = ""
    download_file = None

    return (
        session_state,
        chat,
        recognized,
        eval_md,
        download_file,
        gr.update(interactive=True),   # direction
        gr.update(interactive=True),   # difficulty
    )


def start_interview(direction: str, difficulty: str, session_state: Dict[str, Any], chat: List[Dict[str, str]]):
    """
    Старт: вызываем graph без ответа кандидата -> получаем первый вопрос.
    """
    thread_id = (session_state or {}).get("thread_id") or str(uuid4())
    domains = _domain_choice_to_domains(direction)
    difficulty = (difficulty or "middle").strip().lower()

    config = {"configurable": {"thread_id": thread_id}}

    state_input = {
        "messages": [],
        "domains": domains,
        "difficulty": difficulty,
        "asked_ids": [],
    }

    state_output = graph.invoke(state_input, config=config)
    bot_text = _ensure_msg_text(state_output["messages"][-1])

    ss = _new_session_state(thread_id, domains, difficulty)

    ss["asked_ids"] = state_output.get("asked_ids", []) or []
    ss["current_question"]["id"] = state_output.get("last_question_id")
    ss["current_question"]["text"] = state_output.get("last_question_text")
    ss["current_question"]["rubric"] = state_output.get("last_rubric")

    chat = chat or []
    chat.append({"role": "assistant", "content": bot_text})

    return (
        ss,
        chat,
        gr.update(interactive=False),  # direction lock
        gr.update(interactive=False),  # difficulty lock
    )


def send_answer(
    text_answer: str,
    audio_path: str,
    language: str,
    show_open_eval: bool,
    session_state: Dict[str, Any],
    chat: List[Dict[str, str]],
):
    """
    Отправка ответа кандидата: текст или аудио.
    Далее: graph.invoke + ответ интервьюера + (опционально) открытая оценка.
    """
    if not session_state or not session_state.get("started"):
        return (
            session_state,
            chat,
            "",
            "Сначала нажми **«Начать интервью»**.",
            None,
        )

    chat = chat or []
    recognized = ""
    eval_md = ""
    download_file = None

    candidate_text = (text_answer or "").strip()
    if not candidate_text:
        if not audio_path:
            return (
                session_state,
                chat,
                "",
                "Пришли **текст** или **аудио**.",
                None,
            )
        try:
            candidate_text = transcribe_audio(audio_path, language=language)
            recognized = candidate_text
        except Exception as e:
            return (
                session_state,
                chat,
                "",
                f"Ошибка STT: {e}",
                None,
            )

    # user -> чат (messages-формат)
    chat.append({"role": "user", "content": candidate_text})

    config = {"configurable": {"thread_id": session_state["thread_id"]}}
    state_input = {
        "messages": [{"role": "user", "content": candidate_text}],
    }
    state_output = graph.invoke(state_input, config=config)

    bot_text = _ensure_msg_text(state_output["messages"][-1])

    # assistant -> чат
    chat.append({"role": "assistant", "content": bot_text})

    # evaluation относится к предыдущему current_question
    ev = state_output.get("last_evaluation")

    cur_q = (session_state.get("current_question") or {})
    prev_q_id = cur_q.get("id")
    prev_q_text = cur_q.get("text")

    session_state["asked_ids"] = state_output.get("asked_ids", []) or session_state.get("asked_ids", [])

    if prev_q_id and prev_q_text:
        session_state["turns"].append(
            {
                "question_id": prev_q_id,
                "question": prev_q_text,
                "answer": candidate_text,
                "evaluation": ev,
            }
        )

    # новый текущий вопрос
    session_state["current_question"]["id"] = state_output.get("last_question_id")
    session_state["current_question"]["text"] = state_output.get("last_question_text")
    session_state["current_question"]["rubric"] = state_output.get("last_rubric")

    if show_open_eval:
        eval_md = _pretty_eval(ev)

    return (
        session_state,
        chat,
        recognized,
        eval_md,
        download_file,
    )


def download_session_json(session_state: Dict[str, Any]):
    """
    Формируем JSON файл со всеми turns.
    """
    if not session_state or not session_state.get("started"):
        return None

    payload = {
        "thread_id": session_state.get("thread_id"),
        "domains": session_state.get("domains"),
        "difficulty": session_state.get("difficulty"),
        "asked_ids": session_state.get("asked_ids"),
        "turns": session_state.get("turns", []),
    }

    fd, path = tempfile.mkstemp(prefix="session_", suffix=".json")
    os.close(fd)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


INTRO_MD = """
# Техническое интервью (ML / MLOps)

Как это работает:

1) Выбираешь **направление** и **сложность**.  
2) Жмёшь **«Начать интервью»** — интервьюер задаёт первый вопрос.  
3) Ты отвечаешь **текстом или голосом**.  
4) Интервьюер:
   - оценивает ответ (опционально покажем “открытую оценку”),
   - если ты ушёл от темы — задаёт уточняющий вопрос по тому же пункту,
   - иначе подбирает следующий вопрос из банка через retrieval.

В конце можно скачать **session.json** с вопросами/ответами/оценками.
""".strip()


def main():
    with gr.Blocks() as demo:
        gr.Markdown(INTRO_MD)

        session_state = gr.State(
            {
                "thread_id": str(uuid4()),
                "domains": ["ml", "mlops"],
                "difficulty": "middle",
                "started": False,
                "current_question": {"id": None, "text": None, "rubric": None},
                "turns": [],
                "asked_ids": [],
            }
        )

        chat = gr.Chatbot(label="Диалог", height=420)

        with gr.Row():
            with gr.Column(scale=1):
                direction = gr.Dropdown(
                    choices=["ml", "mlops", "both"],
                    value="both",
                    label="Направление",
                )
                difficulty = gr.Dropdown(
                    choices=["junior", "middle", "senior"],
                    value="middle",
                    label="Сложность",
                )
                language = gr.Dropdown(
                    choices=["ru", "en"],
                    value="ru",
                    label="Язык распознавания (STT)",
                )
                show_open_eval = gr.Checkbox(
                    value=True,
                    label="Показывать открытую оценку после каждого ответа",
                )

                with gr.Row():
                    btn_new = gr.Button("Новая сессия", variant="secondary")
                    btn_start = gr.Button("Начать интервью", variant="primary")

                btn_download = gr.Button("⬇Скачать session.json", variant="secondary")
                download_out = gr.File(label="Файл сессии (JSON)")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Ответ текстом"):
                        text_answer = gr.Textbox(
                            label="Твой ответ (текст)",
                            lines=4,
                            placeholder="Напиши ответ и нажми «Отправить ответ»",
                        )
                    with gr.Tab("Ответ голосом"):
                        audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Аудио кандидата (микрофон или файл)",
                        )
                        recognized = gr.Textbox(
                            label="Распознанный текст (STT)",
                            lines=3,
                        )

                btn_send = gr.Button("Отправить ответ", variant="primary")
                eval_md = gr.Markdown("")

        btn_new.click(
            fn=new_session_ui,
            inputs=[],
            outputs=[session_state, chat, recognized, eval_md, download_out, direction, difficulty],
        )

        btn_start.click(
            fn=start_interview,
            inputs=[direction, difficulty, session_state, chat],
            outputs=[session_state, chat, direction, difficulty],
        )

        btn_send.click(
            fn=send_answer,
            inputs=[text_answer, audio, language, show_open_eval, session_state, chat],
            outputs=[session_state, chat, recognized, eval_md, download_out],
        )

        btn_download.click(
            fn=download_session_json,
            inputs=[session_state],
            outputs=[download_out],
        )

    check_model_or_warn()

    demo.launch()


if __name__ == "__main__":
    main()
