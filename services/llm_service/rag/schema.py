from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedQuestion:
    id: str
    domain: str
    topic: str
    difficulty: str
    question: str
    rubric: List[str]
    tags: List[str]
    score: float
