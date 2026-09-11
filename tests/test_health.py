"""
Тесты сверки имени модели при старте.

Ответ /models подменяется моком, поэтому тесты не требуют сети
и запущенной LM Studio.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from shared.config import MODEL_NAME
from shared.health import (
    MODELS_REQUEST_TIMEOUT,
    ModelNotFoundError,
    check_model_available,
    check_model_or_warn,
)


def _models_response(ids):
    resp = MagicMock()
    resp.json.return_value = {"data": [{"id": i} for i in ids]}
    resp.raise_for_status.return_value = None
    return resp


class TestCheckModelAvailable:
    def test_совпадение_проходит(self):
        with patch.object(requests, "get", return_value=_models_response([MODEL_NAME])):
            assert check_model_available() == [MODEL_NAME]

    def test_совпадение_среди_нескольких_моделей(self):
        ids = ["google/gemma-3-1b", MODEL_NAME, "text-embedding-bge-m3"]

        with patch.object(requests, "get", return_value=_models_response(ids)):
            assert check_model_available() == ids

    def test_отсутствие_имени_даёт_ошибку_со_списком(self):
        ids = ["google/gemma-3-1b", "text-embedding-bge-m3"]

        with patch.object(requests, "get", return_value=_models_response(ids)):
            with pytest.raises(ModelNotFoundError) as exc:
                check_model_available()

        message = str(exc.value)
        assert MODEL_NAME in message
        for model_id in ids:
            assert model_id in message
        assert "This model's API identifier" in message

    def test_пустой_список_моделей(self):
        with patch.object(requests, "get", return_value=_models_response([])):
            with pytest.raises(ModelNotFoundError) as exc:
                check_model_available()

        assert "список пуст" in str(exc.value)

    def test_похожее_но_иное_имя_не_считается_совпадением(self):
        # LM Studio молча подставляет единственную загруженную модель,
        # поэтому сверка должна быть посимвольной
        with patch.object(requests, "get", return_value=_models_response([MODEL_NAME + "-instruct"])):
            with pytest.raises(ModelNotFoundError):
                check_model_available()

    @pytest.mark.parametrize(
        "exc",
        [
            requests.ConnectionError("сервер недоступен"),
            requests.Timeout("истекло время ожидания"),
        ],
    )
    def test_недоступный_сервер_прокидывает_сетевую_ошибку(self, exc):
        # не ModelNotFoundError: web_app различает эти случаи
        with patch.object(requests, "get", side_effect=exc):
            with pytest.raises(type(exc)):
                check_model_available()

    def test_запрос_идёт_с_коротким_таймаутом(self):
        with patch.object(requests, "get", return_value=_models_response([MODEL_NAME])) as mock_get:
            check_model_available()

        assert mock_get.call_args.kwargs["timeout"] == MODELS_REQUEST_TIMEOUT
        assert MODELS_REQUEST_TIMEOUT <= 5


class TestCheckModelOrWarn:
    """Старт падает на неверном имени, но не на недоступном сервере."""

    def test_недоступный_сервер_даёт_предупреждение(self, capsys):
        with patch(
            "shared.health.check_model_available",
            side_effect=requests.ConnectionError("нет связи"),
        ):
            check_model_or_warn()  # исключения быть не должно

        out = capsys.readouterr().out
        assert "[warning]" in out

    def test_неверное_имя_модели_роняет_старт(self):
        with patch(
            "shared.health.check_model_available",
            side_effect=ModelNotFoundError("модель не найдена"),
        ):
            with pytest.raises(ModelNotFoundError):
                check_model_or_warn()

    def test_успешная_проверка_молчит(self, capsys):
        with patch(
            "shared.health.check_model_available",
            return_value=[MODEL_NAME],
        ):
            check_model_or_warn()

        assert capsys.readouterr().out == ""
