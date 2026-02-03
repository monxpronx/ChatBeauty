from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Candidate:
    item_id: str
    retrieval_score: float
    metadata: dict


@dataclass
class RerankSample:
    query: str
    candidates: List[Candidate]
    labels: Optional[List[float]] = None