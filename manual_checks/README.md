# Ручные проверки

Интеграционные проверки, которые ходят в живые сервисы: требуют запущенной
LM Studio (`http://127.0.0.1:1234/v1`) с загруженной моделью и собранного
FAISS-индекса. Запускаются вручную из корня проекта, в CI не входят.

```bash
uv run python -m manual_checks.check_agent
uv run python -m manual_checks.check_evaluator
uv run python -m manual_checks.check_api
```

Это не pytest-тесты: файлы намеренно названы `check_*.py`, чтобы pytest их
не собирал. Автоматические тесты живут в `tests/`.
