import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.config import EMBEDDING_MODEL_NAME



BASE_DIR = Path(__file__).resolve().parents[3]

QUESTIONS_RAW_DIR = BASE_DIR / "shared" / "questions" / "raw"
INDEX_DIR = BASE_DIR / "shared" / "questions" / "index"

FAISS_INDEX_PATH = INDEX_DIR / "questions.faiss"
META_PATH = INDEX_DIR / "questions_meta.json"


#Data
@dataclass
class Question:
    id: str
    domain: str
    topic: str
    difficulty: str
    question: str
    rubric: List[str]
    tags: List[str]
    embedding_text: str


#Utils

def build_embedding_text(q: dict) -> str:
    rubric_text = "; ".join(q.get("rubric", []))
    tags_text = ", ".join(q.get("tags", []))

    return (
        f"Вопрос: {q['question']}\n"
        f"Домен: {q['domain']}\n"
        f"Тема: {q['topic']}\n"
        f"Уровень: {q['difficulty']}\n"
        f"Ключевые понятия: {rubric_text}\n"
        f"Теги: {tags_text}"
    )


def load_questions() -> List[Question]:
    questions: List[Question] = []

    for file_path in QUESTIONS_RAW_DIR.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_q in data:
            questions.append(
                Question(
                    id=raw_q["id"],
                    domain=raw_q["domain"],
                    topic=raw_q["topic"],
                    difficulty=raw_q["difficulty"],
                    question=raw_q["question"],
                    rubric=raw_q.get("rubric", []),
                    tags=raw_q.get("tags", []),
                    embedding_text=build_embedding_text(raw_q),
                )
            )

    if not questions:
        raise ValueError("Вопросы не найдены")

    return questions


#Index

def build_and_save_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [q.embedding_text for q in questions]
    embeddings = model.encode(texts, normalize_embeddings=True)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    meta = [
        {
            "id": q.id,
            "domain": q.domain,
            "topic": q.topic,
            "difficulty": q.difficulty,
            "question": q.question,
            "rubric": q.rubric,
            "tags": q.tags,
        }
        for q in questions
    ]

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Индекс построен: {len(questions)} вопросов")
    print(f"FAISS: {FAISS_INDEX_PATH}")
    print(f"Meta: {META_PATH}")


def load_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS индекс не найден по пути {FAISS_INDEX_PATH}. Собери его командой:\n"
            "python -m services.llm_service.rag.store"
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, meta



if __name__ == "__main__":
    build_and_save_index()
