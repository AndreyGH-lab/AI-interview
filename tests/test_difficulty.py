"""
Юнит-тесты difficulty_allowed: разрешены вопросы текущего уровня и ниже.
"""

import pytest

from services.llm_service.rag.api import difficulty_allowed


@pytest.mark.parametrize(
    "q_diff, current_diff, expected",
    [
        # кандидат junior — только junior
        ("junior", "junior", True),
        ("middle", "junior", False),
        ("senior", "junior", False),
        # кандидат middle — junior и middle
        ("junior", "middle", True),
        ("middle", "middle", True),
        ("senior", "middle", False),
        # кандидат senior — всё
        ("junior", "senior", True),
        ("middle", "senior", True),
        ("senior", "senior", True),
    ],
)
def test_все_девять_комбинаций(q_diff, current_diff, expected):
    assert difficulty_allowed(q_diff, current_diff) is expected


@pytest.mark.parametrize(
    "q_diff, current_diff",
    [
        ("unknown", "middle"),
        ("junior", "unknown"),
        ("unknown", "unknown"),
        ("", ""),
        (None, "middle"),
    ],
)
def test_неизвестное_значение_не_падает(q_diff, current_diff):
    # неизвестный уровень трактуется как 0 (junior), исключения быть не должно
    assert isinstance(difficulty_allowed(q_diff, current_diff), bool)
