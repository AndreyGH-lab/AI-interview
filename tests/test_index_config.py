"""
Тесты согласования FAISS-индекса с текущей конфигурацией эмбеддингов.

Настоящий индекс не строится: метаданные подставляются фикстурой,
поэтому тесты не требуют сети и загрузки модели.
"""

import pytest

from services.llm_service.rag.store import (
    REBUILD_HINT,
    check_index_config,
    current_index_config,
)


@pytest.fixture
def matching_config():
    """Конфигурация индекса, совпадающая с текущей."""
    return dict(current_index_config())


class TestCurrentIndexConfig:
    def test_содержит_модель_и_оба_префикса(self, matching_config):
        assert set(matching_config) == {
            "embedding_model_name",
            "query_prefix",
            "passage_prefix",
        }

    def test_значения_берутся_из_конфигурации(self, matching_config):
        from shared import config

        assert matching_config["embedding_model_name"] == config.EMBEDDING_MODEL_NAME
        assert matching_config["query_prefix"] == config.EMBEDDING_QUERY_PREFIX
        assert matching_config["passage_prefix"] == config.EMBEDDING_PASSAGE_PREFIX


class TestCheckIndexConfig:
    def test_совпадение_проходит(self, matching_config):
        check_index_config(matching_config)

    def test_расхождение_по_модели(self, matching_config):
        matching_config["embedding_model_name"] = "sentence-transformers/all-MiniLM-L6-v2"

        with pytest.raises(ValueError) as exc:
            check_index_config(matching_config)

        message = str(exc.value)
        assert "модель" in message
        assert "all-MiniLM-L6-v2" in message
        assert REBUILD_HINT in message

    def test_расхождение_по_префиксу_запроса(self, matching_config):
        matching_config["query_prefix"] = ""

        with pytest.raises(ValueError) as exc:
            check_index_config(matching_config)

        message = str(exc.value)
        assert "префикс запроса" in message
        assert REBUILD_HINT in message

    def test_расхождение_по_префиксу_документа(self, matching_config):
        matching_config["passage_prefix"] = "иной префикс: "

        with pytest.raises(ValueError) as exc:
            check_index_config(matching_config)

        message = str(exc.value)
        assert "префикс документа" in message
        assert REBUILD_HINT in message

    @pytest.mark.parametrize("value", [None, {}])
    def test_индекс_без_сведений_о_модели(self, value):
        # формат до появления сверки: meta был голым списком вопросов
        with pytest.raises(ValueError) as exc:
            check_index_config(value)

        message = str(exc.value)
        assert "старой версией кода" in message
        assert REBUILD_HINT in message

    def test_отсутствующий_ключ_считается_расхождением(self, matching_config):
        del matching_config["passage_prefix"]

        with pytest.raises(ValueError) as exc:
            check_index_config(matching_config)

        assert "префикс документа" in str(exc.value)

    def test_сообщение_показывает_обе_стороны(self, matching_config):
        matching_config["embedding_model_name"] = "чужая/модель"

        with pytest.raises(ValueError) as exc:
            check_index_config(matching_config)

        message = str(exc.value)
        # и та модель, которой собран индекс, и та, что в конфигурации
        assert "чужая/модель" in message
        assert current_index_config()["embedding_model_name"] in message
