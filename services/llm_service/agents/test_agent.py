# services/llm_service/agents/test_agent.py
from __future__ import annotations

from uuid import uuid4
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage

from services.llm_service.agents.agent import graph


def _last_ai_text(state: Dict[str, Any]) -> str:
    msgs = state.get("messages", []) or []
    if not msgs:
        return ""
    last = msgs[-1]
    return getattr(last, "content", str(last))


def _debug_print(state: Dict[str, Any]) -> None:
    print("\n=== DEBUG ===")
    print("asked_ids:", state.get("asked_ids"))
    print("last_question_id:", state.get("last_question_id"))
    print("last_question_text:", state.get("last_question_text"))

    ev = state.get("last_evaluation")
    print("\n--- evaluation ---")
    if not ev:
        print(None)
        return

    # Печатаем аккуратно основные поля
    print("on_topic:", ev.get("on_topic"))
    print("score:", ev.get("score"))
    print("covered:", ev.get("covered"))
    print("missed:", ev.get("missed"))
    print("comment:", ev.get("comment"))


def _assert(name: str, cond: bool, details: Optional[str] = None) -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}" + (f" :: {details}" if details else ""))


def run_turn(
    *,
    thread_config: Dict[str, Any],
    user_text: str,
    init_domains: Optional[list[str]] = None,
    init_difficulty: Optional[str] = None,
    init_asked_ids: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Один шаг диалога: отправляем HumanMessage в graph.invoke и получаем state.
    ВАЖНО: на первом ходе можно передать domains/difficulty/asked_ids,
    чтобы старт был полностью детерминирован.
    """
    state_input: Dict[str, Any] = {
        "messages": [HumanMessage(content=user_text)],
    }

    # Эти поля удобно задавать на первом ходу (или везде, если хочешь)
    if init_domains is not None:
        state_input["domains"] = init_domains
    if init_difficulty is not None:
        state_input["difficulty"] = init_difficulty
    if init_asked_ids is not None:
        state_input["asked_ids"] = init_asked_ids

    state_output = graph.invoke(state_input, config=thread_config)

    ai_text = _last_ai_text(state_output)
    print(f"\n[Кандидат]: {user_text}")
    print(f"[Интервьюер]: {ai_text}")

    _debug_print(state_output)
    return state_output


def main():
    print("\n=== TEST AGENT (RAG + Evaluation + Drift + Render prefix) ===\n")

    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # ------------------------------------------------------------
    # SCENARIO 1: старт на "Привет"
    # Ожидаем: первый вопрос из банка, asked_ids +1, evaluation None
    # ------------------------------------------------------------
    s1 = run_turn(
        thread_config=config,
        user_text="Привет!",
        init_domains=["ml", "mlops"],
        init_difficulty="middle",
        init_asked_ids=[],
    )
    _assert("S1: asked_ids заполнен", bool(s1.get("asked_ids")))
    _assert("S1: last_question_id задан", s1.get("last_question_id") is not None)
    _assert("S1: evaluation отсутствует", s1.get("last_evaluation") is None)

    first_qid = s1.get("last_question_id")
    first_ai = _last_ai_text(s1)
    _assert("S1: формат ответа 'Вопрос:'", first_ai.strip().startswith("Вопрос:"))

    # ------------------------------------------------------------
    # SCENARIO 2: нормальный ответ по MLOps (по теме, полный)
    # Ожидаем: retrieve -> следующий вопрос (asked_ids +1)
    # ------------------------------------------------------------
    s2 = run_turn(
        thread_config=config,
        user_text=(
            "MLOps — это практики на стыке ML и DevOps для доставки моделей в прод и поддержки. "
            "Важны воспроизводимость, автоматизация (CI/CD), мониторинг, версионирование данных и моделей."
        ),
    )
    _assert("S2: asked_ids увеличился", len(s2.get("asked_ids", [])) >= 2)
    _assert("S2: вопрос сменился", s2.get("last_question_id") != first_qid)

    # ------------------------------------------------------------
    # SCENARIO 3: ответ УШЁЛ НЕ ТУДА (дрейф) — говорим про overfitting,
    # когда вопрос (скорее всего) про пайплайн/этапы.
    # Ожидаем: followup (asked_ids НЕ растёт), last_question_id НЕ меняется
    # ------------------------------------------------------------
    before_qid = s2.get("last_question_id")
    before_len = len(s2.get("asked_ids", []))

    s3 = run_turn(
        thread_config=config,
        user_text=(
            "Если качество на train высокое, а на test падает — это похоже на переобучение. "
            "Я бы добавил регуляризацию, упростил модель и сделал кросс-валидацию."
        ),
    )

    after_len = len(s3.get("asked_ids", []))
    after_qid = s3.get("last_question_id")
    ai3 = _last_ai_text(s3)

    # В followup ветке asked_ids обычно НЕ меняется и last_question_id остаётся прежним
    _assert(
        "S3: asked_ids НЕ вырос (followup ожидается)",
        after_len == before_len,
        details=f"before={before_len}, after={after_len}",
    )
    _assert(
        "S3: last_question_id НЕ сменился (followup по тому же вопросу)",
        after_qid == before_qid,
        details=f"before={before_qid}, after={after_qid}",
    )
    _assert("S3: формат ответа 'Вопрос:'", ai3.strip().startswith("Вопрос:"))

    # ------------------------------------------------------------
    # SCENARIO 4: кандидат исправился (закрыл missed: deploy/monitoring)
    # Ожидаем: теперь идём дальше -> retrieve/select -> новый вопрос,
    # asked_ids +1, last_question_id сменится
    # ------------------------------------------------------------
    before_len_4 = len(s3.get("asked_ids", []))
    before_qid_4 = s3.get("last_question_id")

    s4 = run_turn(
        thread_config=config,
        user_text=(
            "Да, кроме данных и обучения ещё важны деплой (CI/CD), мониторинг качества и дрейфа, "
            "алертинг, логирование, версионирование данных/моделей и возможность отката."
        ),
    )

    _assert(
        "S4: asked_ids вырос (перешли к следующему вопросу)",
        len(s4.get("asked_ids", [])) == before_len_4 + 1,
        details=f"before={before_len_4}, after={len(s4.get('asked_ids', []))}",
    )
    _assert(
        "S4: last_question_id сменился (новый вопрос из банка)",
        s4.get("last_question_id") != before_qid_4,
        details=f"before={before_qid_4}, after={s4.get('last_question_id')}",
    )

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
