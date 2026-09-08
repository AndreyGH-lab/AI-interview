# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI-powered technical interview simulator for ML/MLOps candidates. The system asks questions from a curated bank, evaluates answers using an LLM, and dynamically selects the next question using RAG (FAISS) + a LangGraph agent.

## Commands

This project uses **uv** as the package manager (Python 3.11).

```bash
# Install dependencies
uv sync

# Run CLI interview
uv run python -m services.orchestrator.main

# Run Gradio web UI
uv run python -m services.web_service.web_app

# (Re)build the FAISS index from raw question JSON files
uv run python -m services.llm_service.rag.store

# Run integration tests (require LM Studio running)
uv run python -m services.llm_service.agents.test_agent
uv run python -m services.llm_service.evaluation.test_evaluator
uv run python -m services.llm_service.rag.test_api

# Docker (production)
docker compose up --build
```

## Architecture

### External dependencies
- **LM Studio** must be running locally on `http://127.0.0.1:1234/v1` (OpenAI-compatible). Model: `qwen2.5-vl-3b-instruct`. When running in Docker, the URL becomes `http://host.docker.internal:1234/v1` (set via `.env`).
- **STT service** (Whisper, FastAPI) on port 8000. In Docker it's the `stt` container; locally it's `http://127.0.0.1:8000/transcribe`.

### LangGraph agent (`services/llm_service/agents/agent.py`)

The core logic is a compiled LangGraph `StateGraph`. Flow per turn:

```
START → prepare → route_first_turn
    ├─ [first turn] → init_question → render → END
    └─ [subsequent] → evaluate → route_after_evaluate
           ├─ [poor/off-topic answer] → followup → END
           └─ [good answer] → retrieve → select → render → END
```

- **prepare**: extracts `last_answer` from messages, sets `is_first_turn` flag
- **init_question**: picks the first question deterministically from the bank (no LLM call)
- **evaluate**: calls `evaluator.py` (rubric scoring) + `llm_drift` (on-topic check)
- **followup**: generates a clarifying question about the *same* topic using `llm_followup`
- **retrieve**: FAISS semantic search filtered by domain/difficulty/`exclude_ids`
- **select**: LLM picks the best candidate from top-5 retrieved questions
- **render**: LLM generates a short conversational prefix, then appends the raw question text

Four separate `ChatOpenAI` instances hit LM Studio: `llm_selector`, `llm_renderer`, `llm_followup`, `llm_drift`. All use `temperature=0`.

Routing thresholds (in `agent.py`):
- `DRIFT_SCORE_THRESHOLD = 0.45` — score below this → followup
- `ON_TOPIC_OVERRIDE_SCORE = 0.80` — score above this with 0 missed → always retrieve

### RAG (`services/llm_service/rag/`)

- `store.py`: builds FAISS IndexFlatIP (inner product / cosine) from `shared/questions/raw/*.json` using `sentence-transformers/all-MiniLM-L6-v2`. Outputs to `shared/questions/index/`.
- `retriever.py`: loads the index at module import time (singleton). `search()` returns `(score, metadata)` pairs.
- `api.py`: high-level `retrieve_questions()` filters by domain, difficulty (cumulative — junior ≤ middle ≤ senior), and `exclude_ids`.

### Question bank (`shared/questions/raw/`)

Six JSON files: `{ml,mlops}_{junior,middle,senior}.json`. Each question has: `id`, `domain`, `topic`, `difficulty`, `question`, `rubric` (list of expected aspects), `tags`.

Run `uv run python -m services.llm_service.rag.store` whenever questions are added or modified to rebuild the FAISS index.

### Two front-ends

- **CLI** (`services/orchestrator/main.py`): text input or audio file path → STT → `graph.invoke`
- **Gradio web** (`services/web_service/web_app.py`): supports microphone/file upload, optional open evaluation panel, session download as JSON

Both call `graph.invoke` directly — no HTTP layer between them and the agent.

### STT (`services/stt_service/stt_client.py`)

Thin HTTP client. Posts audio files to `STT_URL/transcribe`. The actual STT server (`services.stt_service.main`) is a FastAPI app (not present in this repo — runs in the `stt` Docker container).

## Правила работы с этим проектом

Это работающий сервис, а не черновик. Задача — точечные исправления,
не переписывание.

- Минимальный диff. Не рефакторь то, о чём не просили.
- Не меняй архитектуру графа LangGraph, узлы и переходы.
- Не меняй промпты, кроме случаев, где это явно указано в задании.
- Не добавляй новые зависимости, кроме явно перечисленных в задании.
- Не трогай shared/questions/ — банк вопросов и индекс оставь как есть.
- Не переименовывай существующие функции и модули без указания.
- Комментарий только там, где объясняет неочевидное решение,
  не пересказывает код.
- Русский язык в докстрингах и сообщениях, как в остальном проекте.
- После выполнения кратко перечисли изменённые файлы и суть правок,
  без пересказа кода.
