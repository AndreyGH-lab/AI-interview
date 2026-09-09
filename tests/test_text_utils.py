"""
Юнит-тесты чистых функций из text_utils.

Модуль намеренно не импортирует ничего, что тянет langchain, — тесты
должны запускаться без LM Studio и без собранного FAISS-индекса.
"""

import json

import pytest

from services.llm_service.agents.text_utils import (
    MIN_FOLLOWUP_LENGTH,
    build_followup_fallback,
    ensure_question_prefix,
    safe_json_parse,
    sanitize_prefix,
    strip_reasoning,
)


class TestStripReasoning:
    def test_закрытый_блок_вырезается(self):
        raw = "<think>Прикидываю варианты</think>Полезный ответ"
        assert strip_reasoning(raw) == "Полезный ответ"

    def test_блок_с_атрибутами_в_теге(self):
        raw = '<think type="internal">рассуждение</think>Ответ'
        assert strip_reasoning(raw) == "Ответ"

    def test_регистр_тега_игнорируется(self):
        assert strip_reasoning("<THINK>шум</THINK>Ответ") == "Ответ"

    def test_многострочный_блок(self):
        raw = "<think>\nстрока раз\nстрока два\n</think>\nОтвет"
        assert strip_reasoning(raw) == "Ответ"

    def test_несколько_блоков_подряд(self):
        raw = "<think>раз</think>А<think>два</think>Б"
        assert strip_reasoning(raw) == "АБ"

    def test_незакрытый_think_даёт_пустую_строку(self):
        assert strip_reasoning("<think>генерация оборвалась на полуслове") == ""

    def test_закрытый_и_следом_незакрытый(self):
        # первый блок вырезан, второй оборван — полезного текста нет
        raw = "<think>раз</think>Ответ<think>обрыв"
        assert strip_reasoning(raw) == ""

    def test_текст_без_think_не_меняется(self):
        assert strip_reasoning("Просто ответ без рассуждений") == "Просто ответ без рассуждений"

    def test_json_без_think_не_меняется(self):
        assert strip_reasoning('{"score": 1.0}') == '{"score": 1.0}'

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_пустой_вход(self, value):
        assert strip_reasoning(value) == ""

    def test_окружающие_пробелы_обрезаются(self):
        assert strip_reasoning("  <think>шум</think>  Ответ  ") == "Ответ"


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

    def test_think_с_фигурными_скобками_перед_json(self):
        # без вырезания <think> поиск r"\{.*\}" выдернул бы мусор из рассуждения
        raw = (
            '<think>Возьмём формат {"score": 0.1} для примера, но это черновик</think>\n'
            '{"score": 0.9, "covered": ["a"]}'
        )
        assert safe_json_parse(raw) == {"score": 0.9, "covered": ["a"]}

    def test_think_и_json_в_markdown_обёртке(self):
        raw = '<think>прикидываю</think>\n```json\n{"score": 0.5}\n```'
        assert safe_json_parse(raw) == {"score": 0.5}

    def test_незакрытый_think_бросает_JSONDecodeError(self):
        with pytest.raises(json.JSONDecodeError):
            safe_json_parse('<think>оборвалось на {"score":')


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

    def test_think_вырезается_до_добавления_префикса(self):
        raw = "<think>надо уточнить про мониторинг</think>Вопрос: А как мониторишь дрейф?"
        assert ensure_question_prefix(raw) == "Вопрос: А как мониторишь дрейф?"

    def test_think_без_префикса_получает_префикс(self):
        raw = "<think>рассуждение</think>А как мониторишь дрейф?"
        assert ensure_question_prefix(raw) == "Вопрос: А как мониторишь дрейф?"

    def test_незакрытый_think_даёт_только_префикс(self):
        assert ensure_question_prefix("<think>оборвалось") == "Вопрос: "


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


class TestBuildFollowupFallback:
    QUESTION = "Из каких этапов состоит типичный ML-пайплайн в продакшене?"

    def test_пустой_missed_повторяет_вопрос(self):
        result = build_followup_fallback(self.QUESTION, [])
        assert result == f"Вопрос: Давай вернёмся к вопросу. {self.QUESTION}"
        assert "Отдельно остановись" not in result

    @pytest.mark.parametrize("missed", [None, [], ["", "   "]])
    def test_missed_без_полезных_пунктов(self, missed):
        result = build_followup_fallback(self.QUESTION, missed)
        assert "Отдельно остановись" not in result

    def test_один_пункт_missed(self):
        result = build_followup_fallback(self.QUESTION, ["мониторинг дрейфа"])
        assert result == (
            f"Вопрос: Давай вернёмся к вопросу. {self.QUESTION}"
            " Отдельно остановись на: мониторинг дрейфа"
        )

    def test_несколько_пунктов_берётся_первый(self):
        result = build_followup_fallback(self.QUESTION, ["деплой", "мониторинг", "откат"])
        assert result.endswith("Отдельно остановись на: деплой")
        assert "мониторинг" not in result

    def test_первый_пустой_пункт_пропускается(self):
        result = build_followup_fallback(self.QUESTION, ["  ", "деплой"])
        assert result.endswith("Отдельно остановись на: деплой")

    def test_вопрос_без_знака_вопроса(self):
        question = "Расскажи про версионирование данных"
        result = build_followup_fallback(question, ["DVC"])
        assert result == (
            f"Вопрос: Давай вернёмся к вопросу. {question}"
            " Отдельно остановись на: DVC"
        )

    def test_пустой_вопрос(self):
        assert build_followup_fallback("", []) == "Вопрос: Давай вернёмся к предыдущему вопросу."

    def test_нестроковый_пункт_missed(self):
        result = build_followup_fallback(self.QUESTION, [42])
        assert result.endswith("Отдельно остановись на: 42")

    def test_всегда_начинается_с_префикса(self):
        for missed in ([], ["деплой"]):
            assert build_followup_fallback(self.QUESTION, missed).startswith("Вопрос: ")

    def test_результат_проходит_порог_длины(self):
        # fallback обязан быть длиннее порога, иначе followup_node зациклится на нём
        for question, missed in [(self.QUESTION, []), ("", []), ("Коротко?", ["a"])]:
            result = build_followup_fallback(question, missed)
            assert len(result.removeprefix("Вопрос:").strip()) >= MIN_FOLLOWUP_LENGTH


class TestSanitizePrefixRubric:
    QUESTION = "Почему важно версионировать модели и данные?"
    RUBRIC = [
        "версионирование моделей и данных",
        "воспроизводимость экспериментов",
        "откат к предыдущей версии",
    ]

    def test_случай_из_прогона_отбраковывается(self):
        # реальный вывод модели: подводка пересказывала rubric этого же вопроса
        prefix = "Версионирование помогает сохранять историю изменений"
        assert sanitize_prefix(prefix, self.QUESTION, self.RUBRIC) == ""

    def test_другая_форма_слова_отбраковывается(self):
        # "версионирование" в подводке против "версионировать" в rubric
        assert sanitize_prefix("Версионирование это полезно", self.QUESTION, ["версионировать модели"]) == ""

    @pytest.mark.parametrize(
        "prefix",
        [
            "Воспроизводимость тут ключевая",
            "Речь про откат изменений",
            "Эксперименты стоит фиксировать",
        ],
    )
    def test_любой_пункт_rubric_отбраковывает(self, prefix):
        assert sanitize_prefix(prefix, self.QUESTION, self.RUBRIC) == ""

    @pytest.mark.parametrize(
        "prefix",
        [
            "Понятно",
            "Хорошо, идём дальше",
            "Спасибо, зафиксировал",
            "Ясно, тогда следующий момент",
        ],
    )
    def test_нейтральная_связка_проходит(self, prefix):
        result = sanitize_prefix(prefix, self.QUESTION, self.RUBRIC)
        assert result == prefix + "."

    def test_rubric_none_ведёт_себя_как_раньше(self):
        prefix = "Версионирование помогает сохранять историю"
        # без rubric содержательный фильтр не применяется
        assert sanitize_prefix(prefix, self.QUESTION, None) == prefix + "."
        assert sanitize_prefix(prefix, self.QUESTION) == prefix + "."

    def test_пустой_rubric(self):
        prefix = "Версионирование помогает сохранять историю"
        assert sanitize_prefix(prefix, self.QUESTION, []) == prefix + "."

    def test_пункт_из_коротких_слов_не_отбраковывает(self):
        # слова короче RUBRIC_MIN_WORD_LENGTH значимыми не считаются
        assert sanitize_prefix("Ну да, тут всё ясно", self.QUESTION, ["ROC AUC", "F1"]) == "Ну да, тут всё ясно."

    def test_граница_длины_значимого_слова(self):
        # "дрейф" — ровно RUBRIC_MIN_WORD_LENGTH символов, значимое
        assert sanitize_prefix("Дрейф тут важен", self.QUESTION, ["дрейф данных"]) == ""
        # "ETL" короче порога — совпадение не считается
        assert sanitize_prefix("ETL тут важен", self.QUESTION, ["ETL"]) == "ETL тут важен."

    def test_rubric_с_нестроковыми_пунктами(self):
        assert sanitize_prefix("Хорошо, дальше", self.QUESTION, [None, 42]) == "Хорошо, дальше."

    def test_форма_проверяется_раньше_содержания(self):
        # вопросительный знак отсекается до проверки rubric
        assert sanitize_prefix("Версионирование?", self.QUESTION, self.RUBRIC) == ""
