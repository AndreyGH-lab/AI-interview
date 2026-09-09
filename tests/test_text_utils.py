"""
Юнит-тесты чистых функций из text_utils.

Модуль намеренно не импортирует ничего, что тянет langchain, — тесты
должны запускаться без LM Studio и без собранного FAISS-индекса.
"""

import json

import pytest

from services.llm_service.agents.text_utils import (
    ensure_question_prefix,
    safe_json_parse,
    sanitize_prefix,
)


class TestSafeJsonParse:
    def test_чистый_json(self):
        assert safe_json_parse('{"score": 0.5}') == {"score": 0.5}

    def test_json_в_markdown_обёртке(self):
        raw = '```json\n{"covered": ["a"], "score": 1.0}\n```'
        assert safe_json_parse(raw) == {"covered": ["a"], "score": 1.0}

    def test_обёртка_без_указания_языка(self):
        assert safe_json_parse('```\n{"a": 1}\n```') == {"a": 1}

    def test_текст_до_и_после_json(self):
        raw = 'Вот результат оценки:\n{"score": 0.7}\nНадеюсь, помог.'
        assert safe_json_parse(raw) == {"score": 0.7}

    def test_многострочный_json_с_вложенностью(self):
        raw = 'Ответ:\n{\n  "missed": [],\n  "nested": {"k": "v"}\n}\nконец'
        assert safe_json_parse(raw) == {"missed": [], "nested": {"k": "v"}}

    @pytest.mark.parametrize(
        "raw",
        [
            "просто текст без json",
            "",
            "{не валидный json}",
            "{'одинарные': 'кавычки'}",
        ],
    )
    def test_невалидный_вход_бросает_JSONDecodeError(self, raw):
        with pytest.raises(json.JSONDecodeError):
            safe_json_parse(raw)


class TestEnsureQuestionPrefix:
    def test_префикс_не_дублируется(self):
        assert ensure_question_prefix("Вопрос: Что такое дрейф?") == "Вопрос: Что такое дрейф?"

    def test_префикс_добавляется(self):
        assert ensure_question_prefix("Что такое дрейф?") == "Вопрос: Что такое дрейф?"

    def test_переносы_схлопываются_в_пробел(self):
        assert ensure_question_prefix("Первая\nвторая\nтретья") == "Вопрос: Первая вторая третья"

    def test_переносы_схлопываются_при_имеющемся_префиксе(self):
        assert ensure_question_prefix("Вопрос: раз\nдва") == "Вопрос: раз два"

    def test_пустая_строка(self):
        assert ensure_question_prefix("") == "Вопрос: "

    def test_окружающие_пробелы_обрезаются(self):
        assert ensure_question_prefix("   Что такое MLOps?   ") == "Вопрос: Что такое MLOps?"


class TestSanitizePrefix:
    QUESTION = "Из каких этапов состоит типичный ML-пайплайн в продакшене?"

    @pytest.mark.parametrize("prefix", ["", "   ", "\n"])
    def test_пустой_вход(self, prefix):
        assert sanitize_prefix(prefix, self.QUESTION) == ""

    def test_длиннее_120_символов(self):
        assert sanitize_prefix("а" * 121, self.QUESTION) == ""

    def test_ровно_120_символов_проходит(self):
        assert sanitize_prefix("а" * 120, self.QUESTION) == "а" * 120 + "."

    def test_содержит_вопросительный_знак(self):
        assert sanitize_prefix("А теперь такой момент?", self.QUESTION) == ""

    def test_содержит_многоточие(self):
        assert sanitize_prefix("Ну что ж...", self.QUESTION) == ""

    @pytest.mark.parametrize(
        "prefix",
        [
            "Что такое подобное явление",
            "Из каких частей это состоит",
            "Зачем нам это нужно",
            "Почему так вышло",
            "Какие есть варианты",
            "MLOps в целом",
            "ML-пайплайн вообще",
            "Пайплайн как таковой",
            "Это важная тема",
            "— это важная тема",
            "- это важная тема",
        ],
    )
    def test_начинается_с_запрещённого_слова(self, prefix):
        assert sanitize_prefix(prefix, self.QUESTION) == ""

    def test_запрещённое_слово_в_другом_регистре(self):
        assert sanitize_prefix("ЗАЧЕМ нам это", self.QUESTION) == ""

    def test_пересечение_с_началом_вопроса(self):
        # первые 20 символов вопроса ("из каких этапов сост") входят в префикс
        assert sanitize_prefix("Напомню, из каких этапов состоит он", self.QUESTION) == ""

    def test_пересечение_с_концом_вопроса(self):
        # последние 20 символов вопроса входят в префикс
        question = "Из каких этапов состоит типичный ML-пайплайн в продакшене"
        assert sanitize_prefix("Речь про ML-пайплайн в продакшене", question) == ""

    def test_нормальный_случай_добавляется_точка(self):
        assert sanitize_prefix("Хорошо, идём дальше", self.QUESTION) == "Хорошо, идём дальше."

    def test_точка_не_дублируется(self):
        assert sanitize_prefix("Хорошо, идём дальше.", self.QUESTION) == "Хорошо, идём дальше."

    @pytest.mark.parametrize("ending", [":", "—"])
    def test_другие_допустимые_окончания_не_меняются(self, ending):
        prefix = f"Хорошо, идём дальше{ending}"
        assert sanitize_prefix(prefix, self.QUESTION) == prefix

    def test_переносы_схлопываются(self):
        assert sanitize_prefix("Хорошо,\nидём дальше", self.QUESTION) == "Хорошо, идём дальше."

    def test_короткий_вопрос_не_проверяется_на_пересечение(self):
        # вопрос короче 20 символов — правило пересечения не применяется
        assert sanitize_prefix("Ясно, теперь дальше", "Что дальше") == "Ясно, теперь дальше."
