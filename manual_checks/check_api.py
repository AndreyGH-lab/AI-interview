from services.llm_service.rag.api import retrieve_questions


def main():
    test_answer = (
        "Модель хорошо работает на обучающих данных, "
        "но на тестовых качество сильно падает."
    )

    results = retrieve_questions(
        test_answer,
        domain="ml",
        difficulty="middle",
        exclude_ids=[],
        top_k=5,
    )

    print("\n--- Retrieved questions ---\n")

    for r in results:
        print(f"[{r.score:.3f}] ({r.difficulty}) {r.question}")


if __name__ == "__main__":
    main()
