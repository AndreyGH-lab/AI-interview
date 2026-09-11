"""
Тесты деградации при недоступной модели.

Клиенты подменяются моками, которые бросают исключение, поэтому тесты
не требуют ни сети, ни запущенной LM Studio.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestEvaluatorUnavailable:
    RUBRIC = ["переобучение", "плохое обобщение"]

    def _evaluate_with_failure(self, exc):
        from services.llm_service.evaluation import evaluator

        client = MagicMock()
        client.invoke.side_effect = exc

        with patch.object(evaluator, "get_evaluator_llm", return_value=client):
            return evaluator.evaluate_answer(
                question="Что такое overfitting?",
                answer="Модель слишком сложная",
                rubric=self.RUBRIC,
            )

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("LM Studio недоступна"),
            RuntimeError("Error code: 503"),
            TimeoutError("истекло время ожидания"),
        ],
    )
    def test_отказ_модели_даёт_отличимый_comment(self, exc):
        result = self._evaluate_with_failure(exc)

        assert result["comment"] == "Модель недоступна"
        assert result["available"] is False

    def test_структура_совпадает_с_ошибкой_разбора(self):
        result = self._evaluate_with_failure(ConnectionError("boom"))

        assert result["covered"] == []
        assert result["missed"] == self.RUBRIC
        assert result["score"] == 0.0
        assert "error" in result

    def test_невалидный_json_не_помечается_недоступностью(self):
        from services.llm_service.evaluation import evaluator

        client = MagicMock()
        client.invoke.return_value = MagicMock(content="не json вовсе")

        with patch.object(evaluator, "get_evaluator_llm", return_value=client):
            result = evaluator.evaluate_answer(
                question="q", answer="a", rubric=self.RUBRIC
            )

        # недоступность сервера и мусор в ответе должны различаться
        assert result["comment"] != "Модель недоступна"
        assert "available" not in result

    def test_успешный_ответ_не_помечается_недоступностью(self):
        from services.llm_service.evaluation import evaluator

        client = MagicMock()
        client.invoke.return_value = MagicMock(
            content='{"covered": ["переобучение"], "missed": [], "score": 0.9, "comment": "ок"}'
        )

        with patch.object(evaluator, "get_evaluator_llm", return_value=client):
            result = evaluator.evaluate_answer(
                question="q", answer="a", rubric=self.RUBRIC
            )

        assert result["score"] == 0.9
        assert result["comment"] == "ок"
        assert "available" not in result


class TestSelectNodeUnavailable:
    def _candidates(self):
        return [
            {
                "id": f"q{i}",
                "domain": "ml",
                "topic": "t",
                "difficulty": "middle",
                "question": f"вопрос {i}",
                "rubric": [],
                "score": score,
            }
            for i, score in enumerate([0.9, 0.7, 0.5])
        ]

    def _state(self):
        return {
            "candidates": self._candidates(),
            "asked_ids": [],
            "difficulty": "middle",
            "last_answer": "ответ",
            "last_evaluation": {},
        }

    def test_отказ_модели_даёт_лучшего_по_сходству(self):
        import services.llm_service.agents.agent as agent

        client = MagicMock()
        client.invoke.side_effect = ConnectionError("503")

        with patch.object(agent, "get_selector_llm", return_value=client):
            result = agent.select_node(self._state())

        # candidates[0] — максимальное косинусное сходство
        assert result["last_question_id"] == "q0"
        assert result["selected"]["score"] == 0.9

    def test_успешный_выбор_модели_соблюдается(self):
        import services.llm_service.agents.agent as agent

        client = MagicMock()
        client.invoke.return_value = MagicMock(content='{"question_id": "q2"}')

        with patch.object(agent, "get_selector_llm", return_value=client):
            result = agent.select_node(self._state())

        assert result["last_question_id"] == "q2"

    def test_мусор_в_ответе_даёт_лучшего_по_сходству(self):
        import services.llm_service.agents.agent as agent

        client = MagicMock()
        client.invoke.return_value = MagicMock(content="<think>оборвалось")

        with patch.object(agent, "get_selector_llm", return_value=client):
            result = agent.select_node(self._state())

        assert result["last_question_id"] == "q0"


class TestRouteAfterEvaluate:
    def _state(self, evaluation):
        return {
            "last_evaluation": evaluation,
            "last_rubric": ["аспект один", "аспект два"],
            "last_answer": "какой-то ответ кандидата",
        }

    def test_недоступность_модели_ведёт_в_retrieve(self):
        import services.llm_service.agents.agent as agent

        # без этой ветки нулевой score увёл бы в followup
        evaluation = {
            "covered": [],
            "missed": ["аспект один", "аспект два"],
            "score": 0.0,
            "comment": "Модель недоступна",
            "available": False,
        }

        assert agent.route_after_evaluate(self._state(evaluation)) == "retrieve"

    def test_плохой_ответ_по_прежнему_ведёт_в_followup(self):
        import services.llm_service.agents.agent as agent

        evaluation = {
            "covered": [],
            "missed": ["аспект один", "аспект два"],
            "score": 0.0,
            "comment": "ответ не раскрывает тему",
            "on_topic": False,
        }

        assert agent.route_after_evaluate(self._state(evaluation)) == "followup"

    def test_хороший_ответ_ведёт_в_retrieve(self):
        import services.llm_service.agents.agent as agent

        evaluation = {
            "covered": ["аспект один", "аспект два"],
            "missed": [],
            "score": 0.95,
            "on_topic": True,
        }

        assert agent.route_after_evaluate(self._state(evaluation)) == "retrieve"

    def test_available_true_не_меняет_маршрут(self):
        import services.llm_service.agents.agent as agent

        evaluation = {
            "covered": [],
            "missed": ["аспект один", "аспект два"],
            "score": 0.0,
            "on_topic": False,
            "available": True,
        }

        assert agent.route_after_evaluate(self._state(evaluation)) == "followup"
