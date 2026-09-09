"""
Единая конфигурация проекта.

Значения читаются из переменных окружения (в т.ч. из .env),
дефолты соответствуют локальному запуску без .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()


LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-8b")

# Директива qwen3, отключающая режим рассуждений. Для не-reasoning
# модели гасится пустой строкой.
PROMPT_SUFFIX = os.getenv("PROMPT_SUFFIX", "/no_think")

STT_URL = os.getenv("STT_URL", "http://127.0.0.1:8000/transcribe")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

DRIFT_SCORE_THRESHOLD = float(os.getenv("DRIFT_SCORE_THRESHOLD", "0.45"))
ON_TOPIC_OVERRIDE_SCORE = float(os.getenv("ON_TOPIC_OVERRIDE_SCORE", "0.80"))
