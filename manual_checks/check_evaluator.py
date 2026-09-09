from services.llm_service.evaluation.evaluator import evaluate_answer

def main():
    question = "Что такое overfitting и почему он возникает?"

    answer = (
        "Модель слишком сложная, поэтому она хорошо работает на обучающих данных, "
        "но плохо обобщает на новых. Также данных может быть недостаточно."
    )

    rubric = [
        "переобучение на тренировочных данных",
        "плохое обобщение",
        "слишком сложная модель",
        "мало данных",
    ]

    # ✅ КОРРЕКТНЫЙ ВЫЗОВ
    result = evaluate_answer(
        question=question,
        answer=answer,
        rubric=rubric,
    )

    print("\n=== EVALUATION RESULT ===")
    print("covered:", result.get("covered"))
    print("missed:", result.get("missed"))
    print("score:", result.get("score"))
    print("comment:", result.get("comment"))
    print("\nRAW MODEL OUTPUT:\n", result.get("raw"))

if __name__ == "__main__":
    main()
