"""
Проверки окружения при старте приложения.

Модуль зависит только от requests и shared.config: его импорт не должен
тянуть langchain и torch, чтобы проверку можно было выполнить до сборки
графа.
"""

from typing import List

import requests

from shared.config import LM_STUDIO_BASE_URL, MODEL_NAME

# короткий таймаут: проверка не должна задерживать старт
MODELS_REQUEST_TIMEOUT = 5


class ModelNotFoundError(RuntimeError):
    """Сервер доступен, но нужной модели среди загруженных нет."""


def fetch_available_models() -> List[str]:
    """
    Возвращает id моделей, загруженных в LM Studio.
    """
    resp = requests.get(
        f"{LM_STUDIO_BASE_URL}/models",
        timeout=MODELS_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or []
    return [m.get("id") for m in data if m.get("id")]


def check_model_available() -> List[str]:
    """
    Сверяет MODEL_NAME со списком загруженных моделей.

    При расхождении LM Studio не сообщает об ошибке: она либо молча
    подставляет единственную загруженную модель, либо отвечает 503 с
    пустым телом. Поэтому имя сверяем сами.

    Возвращает список доступных id. Бросает ModelNotFoundError, если
    модели нет; сетевые ошибки прокидывает наверх.
    """
    available = fetch_available_models()

    if MODEL_NAME not in available:
        listed = "\n".join(f"  - {m}" for m in available) or "  (список пуст)"
        raise ModelNotFoundError(
            f"Модель {MODEL_NAME!r} не найдена на {LM_STUDIO_BASE_URL}.\n"
            f"Доступные модели:\n{listed}\n"
            "Имя копируется из поля \"This model's API identifier\" в LM Studio "
            "и должно совпадать посимвольно. Поправь MODEL_NAME в .env."
        )

    return available


def check_model_or_warn() -> None:
    """
    Сверка модели для старта приложения.

    Недоступный сервер — не повод не подниматься: без этого отладка
    интерфейса стала бы невозможной. А вот неверное имя модели при
    работающем сервере — ошибка конфигурации, и её нужно показать сразу.
    """
    try:
        check_model_available()
    except ModelNotFoundError:
        raise
    except Exception as e:
        print(
            f"[warning] Не удалось проверить модель на {LM_STUDIO_BASE_URL}: {e}\n"
            "[warning] Интерфейс поднимется, но запросы к модели работать не будут."
        )
