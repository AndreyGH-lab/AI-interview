# -*- coding: utf-8 -*-
"""
Бенчмарк ретривала: сравнение моделей эмбеддингов на банке вопросов.

Измеряемая задача — прокси, и это осознанно. В продакшене запросом выступает
ответ кандидата, а найти нужно СЛЕДУЮЩИЙ вопрос; объективной разметки для
этого не существует, потому что "хороший следующий вопрос" — суждение,
а не факт.

Поэтому измеряется то, что от модели эмбеддингов действительно зависит:
способность сопоставить ответ кандидата с вопросом, НА КОТОРЫЙ он отвечает.
Здесь разметка объективна — правильный ответ ровно один.

Эталонные ответы написаны разговорным языком, без дословных формулировок
из rubric: иначе поиск становится тривиальным и бенчмарк перестаёт
различать модели.

Запуск:
    uv run python -m benchmark.retrieval_benchmark            # все модели
    uv run python -m benchmark.retrieval_benchmark --quick    # только лёгкие
    uv run python -m benchmark.retrieval_benchmark --models e5-small LaBSE

Модели скачиваются с HuggingFace при первом запуске: полный набор — около
4 ГБ, --quick — около 700 МБ.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from services.llm_service.rag.store import load_questions

BENCH_DIR = Path(__file__).resolve().parent
ANSWERS_PATH = BENCH_DIR / "answers.json"
RESULTS_PATH = BENCH_DIR / "results.json"

K_VALUES = (1, 3, 5)


@dataclass
class ModelSpec:
    name: str
    # Префиксы обязательны для семейства e5: модель обучена различать роль
    # текста, и без них качество заметно падает — сравнение вышло бы нечестным.
    query_prefix: str = ""
    passage_prefix: str = ""
    heavy: bool = False


MODELS: List[ModelSpec] = [
    ModelSpec("sentence-transformers/all-MiniLM-L6-v2"),
    ModelSpec("cointegrated/rubert-tiny2"),
    ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ModelSpec("intfloat/multilingual-e5-small", "query: ", "passage: "),
    ModelSpec("intfloat/multilingual-e5-base", "query: ", "passage: ", heavy=True),
    ModelSpec("sentence-transformers/LaBSE", heavy=True),
    ModelSpec("intfloat/multilingual-e5-large", "query: ", "passage: ", heavy=True),
]


def load_answers() -> Dict[str, str]:
    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(spec: ModelSpec, questions, answers: Dict[str, str]) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec.name, device="cpu")
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    ids = [q.id for q in questions]
    passages = [spec.passage_prefix + q.embedding_text for q in questions]

    pairs = [(qid, answers[qid]) for qid in ids]
    queries = [spec.query_prefix + text for _, text in pairs]
    truth = [qid for qid, _ in pairs]

    t0 = time.perf_counter()
    passage_emb = model.encode(passages, normalize_embeddings=True, batch_size=32)
    query_emb = model.encode(queries, normalize_embeddings=True, batch_size=32)
    encode_seconds = time.perf_counter() - t0

    passage_emb = np.asarray(passage_emb, dtype="float32")
    query_emb = np.asarray(query_emb, dtype="float32")

    result = _score(
        model_name=spec.name,
        params_m=params_m,
        ids=ids,
        passage_emb=passage_emb,
        query_emb=query_emb,
        encode_seconds=encode_seconds,
    )
    result["tokens_per_query"] = mean_tokens(model, [t for _, t in pairs])
    return result


def mean_tokens(model, texts: List[str]) -> float:
    """
    Среднее число токенов на текст.

    Объясняет разрыв между моделями напрямую: словарь, не рассчитанный на
    русский, рубит слова на множество мелких кусков — отсюда и длинные
    последовательности, и потеря смысла.
    """
    tok = model.tokenizer
    lengths = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    return round(float(np.mean(lengths)), 1)


def _score(*, model_name: str, params_m: float, ids: List[str],
           passage_emb: np.ndarray, query_emb: np.ndarray,
           encode_seconds: float) -> dict:
    """
    Считает Recall@k и MRR. Правильный ответ для i-го запроса — ids[i].

    Корректность проверена двумя контролями: при query == passage выходит
    recall@1 = 1.0, а MRR случайного ранжирования совпадает с
    аналитическим sum(1/i)/n.
    """
    # то же, что в продакшене: нормализованные векторы + внутреннее произведение
    index = faiss.IndexFlatIP(passage_emb.shape[1])
    index.add(passage_emb)
    _, full_idx = index.search(query_emb, len(ids))

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []
    misses: List[dict] = []

    for row, correct_id in enumerate(ids):
        ranking = [ids[i] for i in full_idx[row]]
        rank = ranking.index(correct_id) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        for k in K_VALUES:
            if rank <= k:
                hits[k] += 1
        if rank > max(K_VALUES):
            misses.append({"id": correct_id, "rank": rank, "top1": ranking[0]})

    n = len(ids)
    return {
        "model": model_name,
        "params_m": round(params_m, 1),
        "dim": int(passage_emb.shape[1]),
        "n": n,
        **{f"recall@{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "encode_seconds": round(encode_seconds, 2),
        # ранг правильного вопроса для каждого запроса: нужен, чтобы
        # считать значимость парным бутстрепом, не пересчитывая эмбеддинги
        "ranks": ranks,
        "misses": misses,
    }



def evaluate_tfidf_baseline(questions, answers: Dict[str, str]) -> dict:
    """
    Лексический бейзлайн без нейросети: символьные n-граммы + TF-IDF.

    Нужен как нижняя планка. Модель эмбеддингов, которая не оторвалась от
    этого числа заметно, не оправдывает своего веса в образе и времени на
    кодирование. Работает без сети.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = [q.id for q in questions]
    passages = [q.embedding_text for q in questions]
    queries = [answers[qid] for qid in ids]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    t0 = time.perf_counter()
    passage_emb = vec.fit_transform(passages).toarray().astype("float32")
    query_emb = vec.transform(queries).toarray().astype("float32")
    encode_seconds = time.perf_counter() - t0

    passage_emb /= np.linalg.norm(passage_emb, axis=1, keepdims=True) + 1e-9
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9

    return _score(
        model_name="TF-IDF char 3-5gram (бейзлайн, без нейросети)",
        params_m=0.0,
        ids=ids,
        passage_emb=passage_emb,
        query_emb=query_emb,
        encode_seconds=encode_seconds,
    )



def _metric_from_ranks(ranks: np.ndarray, metric: str) -> float:
    if metric == "mrr":
        return float(np.mean(1.0 / ranks))
    k = int(metric.split("@")[1])
    return float(np.mean(ranks <= k))


def paired_bootstrap(ranks_a: List[int], ranks_b: List[int], metric: str,
                     iterations: int = 20000, seed: int = 0) -> dict:
    """
    Парный бутстреп разницы метрики между двумя моделями.

    Парный, а не независимый: обе модели отвечали на ОДНИ И ТЕ ЖЕ 90 запросов,
    поэтому пересэмплировать нужно запросы, а не модели — иначе доверительный
    интервал получится шире реального.

    p — двусторонняя оценка: доля ресэмплов, где разница меняет знак или
    обращается в ноль, умноженная на два. На 90 примерах разрешающая
    способность ограничена, и это ограничение самого замера, а не метода.
    """
    a = np.asarray(ranks_a, dtype=float)
    b = np.asarray(ranks_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("ранги должны быть по одним и тем же запросам")

    observed = _metric_from_ranks(a, metric) - _metric_from_ranks(b, metric)

    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        diffs[i] = (_metric_from_ranks(a[idx], metric)
                    - _metric_from_ranks(b[idx], metric))

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    if observed >= 0:
        p = 2.0 * float(np.mean(diffs <= 0))
    else:
        p = 2.0 * float(np.mean(diffs >= 0))

    return {
        "metric": metric,
        "diff": round(observed, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "p": round(min(p, 1.0), 4),
    }


def print_significance(results: List[dict]) -> None:
    """
    Сравнивает лучшую по recall@5 модель со всеми остальными.

    Смотрим на recall@5, а не на recall@1: в продакшене retrieve_node берёт
    top-3 по каждому домену, а в селектор уходят первые пять кандидатов —
    значит достаточно, чтобы нужный вопрос попал в пятёрку.
    """
    scored = [r for r in results if "ranks" in r]
    if len(scored) < 2:
        return

    best = max(scored, key=lambda r: r["recall@5"])
    print()
    print(f"Значимость против лучшей модели ({best['model']}), "
          f"парный бутстреп по {best['n']} запросам:")
    print()
    print(f"| {'Модель':<56} | Δ recall@5 | 95% ДИ            |     p |")
    print("|" + "-" * 58 + "|" + "-" * 12 + "|" + "-" * 19 + "|" + "-" * 7 + "|")

    rows = []
    for r in scored:
        if r["model"] == best["model"]:
            continue
        st = paired_bootstrap(best["ranks"], r["ranks"], "recall@5")
        rows.append((r["model"], st))

    for name, st in sorted(rows, key=lambda x: x[1]["diff"]):
        ci = f"[{st['ci_low']:+.3f}, {st['ci_high']:+.3f}]"
        print(f"| {name:<56} | {st['diff']:>+10.3f} | {ci:<17} | {st['p']:>5.3f} |")

    print()
    print("Положительная разница означает преимущество лучшей модели.")
    print("Доверительный интервал, накрывающий ноль, — разница неотличима от шума.")


def print_table(results: List[dict]) -> None:
    print()
    print(f"| {'Модель':<56} | Параметров | Recall@1 | Recall@3 | Recall@5 "
          f"|   MRR | Токенов | Кодирование |")
    print("|" + "-" * 58 + "|" + "-" * 12 + "|" + "-" * 10 + "|" + "-" * 10
          + "|" + "-" * 10 + "|" + "-" * 7 + "|" + "-" * 9 + "|" + "-" * 13 + "|")
    for r in results:
        tokens = r.get("tokens_per_query")
        tokens_str = f"{tokens:>7.1f}" if tokens is not None else f"{'—':>7}"
        print(f"| {r['model']:<56} | {r['params_m']:>9.1f}M | "
              f"{r['recall@1']:>8.3f} | {r['recall@3']:>8.3f} | "
              f"{r['recall@5']:>8.3f} | {r['mrr']:>5.3f} | {tokens_str} | "
              f"{r['encode_seconds']:>10.1f}s |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="подстроки имён моделей; по умолчанию — все")
    parser.add_argument("--quick", action="store_true",
                        help="только лёгкие модели (без heavy)")
    args = parser.parse_args()

    questions = load_questions()
    answers = load_answers()

    missing = [q.id for q in questions if q.id not in answers]
    if missing:
        raise SystemExit(f"нет эталонных ответов для: {missing}")

    specs = MODELS
    if args.quick:
        specs = [s for s in specs if not s.heavy]
    if args.models:
        specs = [s for s in specs if any(m in s.name for m in args.models)]
    if not specs:
        raise SystemExit("под фильтр не попала ни одна модель")

    print(f"вопросов в банке: {len(questions)}, эталонных ответов: {len(answers)}")

    results = []
    print("--- лексический бейзлайн (TF-IDF)")
    baseline = evaluate_tfidf_baseline(questions, answers)
    results.append(baseline)
    print(f"    recall@1={baseline['recall@1']} recall@5={baseline['recall@5']} "
          f"mrr={baseline['mrr']}")

    for spec in specs:
        print(f"--- {spec.name}")
        try:
            res = evaluate_model(spec, questions, answers)
        except Exception as e:
            print(f"    пропущена: {type(e).__name__}: {e}")
            continue
        results.append(res)
        print(f"    params={res['params_m']}M dim={res['dim']} "
              f"recall@1={res['recall@1']} recall@5={res['recall@5']} "
              f"mrr={res['mrr']} encode={res['encode_seconds']}s")

    if len(results) == 1:
        print("\nни одна нейросетевая модель не посчиталась — "
              "вероятно, нет доступа к HuggingFace")

    results.sort(key=lambda r: r["mrr"], reverse=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_table(results)
    print_significance(results)
    print(f"\nподробные результаты (ранги, промахи, токены): {RESULTS_PATH}")


if __name__ == "__main__":
    main()
