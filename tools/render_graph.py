# -*- coding: utf-8 -*-
"""
Рендер графа LangGraph из скомпилированного объекта.

Схема берётся из самого графа, а не рисуется руками, — значит она не может
разойтись с кодом.

Импорт agent.py не создаёт ни одного клиента к языковой модели: все они
инициализируются лениво через lru_cache. Поэтому скрипт работает без
запущенного LM Studio.

Запуск:
    uv run python -m tools.render_graph
"""
from __future__ import annotations

from pathlib import Path

from services.llm_service.agents.agent import graph

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    drawable = graph.get_graph()

    mermaid_path = ASSETS / "graph.mermaid"
    mermaid_path.write_text(drawable.draw_mermaid(), encoding="utf-8")
    print(f"mermaid: {mermaid_path}")

    png_path = ASSETS / "graph.png"
    try:
        # рендер идёт через внешний сервис mermaid.ink, нужна сеть
        png_path.write_bytes(drawable.draw_mermaid_png())
        print(f"png:     {png_path}")
    except Exception as e:
        print(f"png не отрендерен ({type(e).__name__}: {e})")
        print("Текст схемы сохранён — его можно вставить в README как блок mermaid,")
        print("GitHub отрендерит его сам.")


if __name__ == "__main__":
    main()
