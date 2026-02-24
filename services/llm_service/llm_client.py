from langchain_openai import ChatOpenAI

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

SYSTEM_PROMPT = """
Ты — ИИ-интервьюер на техническом собеседовании (программирование, ML).
Отвечай по-русски, задавай уточняющие вопросы по ответу кандидата.
Не отвечай за кандидата, а веди диалог как интервьюер.
"""

llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    model=MODEL_NAME,
    temperature=0,
)

def interview_reply(history: list[str], user_text: str) -> str:

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Очень простая история: всё как user, кроме ответов модели
    # (для начала хватит, потом можно сделать нормальную структуру сообщений)
    for msg in history:
        messages.append({"role": "user", "content": msg})

    messages.append({"role": "user", "content": user_text})

    resp = llm.invoke(messages)
    return resp.content
