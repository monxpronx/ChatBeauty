import json
from dataclasses import dataclass
from typing import List, Optional, Generator

@dataclass
class Candidate:
    item_id: str
    retrieval_score: float
    metadata: dict

@dataclass
class RerankSample:
    query_keywords: List[str]
    candidates: List[Candidate]
    labels: Optional[List[float]] = None

def iter_samples(path: str, limit: Optional[int] = None) -> Generator[RerankSample, None, None]:
    """리스트를 만들지 않고 한 줄씩 읽어서 바로 Sample 객체로 반환합니다."""
    print(f"Streaming samples from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            
            row = json.loads(line)
            target_id = row["parent_asin"]
            kw_list = [str(k).lower() for k in row.get("keywords", [])]

            candidates = []
            labels = []
            for c in row["candidates"]:
                item_id = c["item_asin"]
                candidates.append(Candidate(
                    item_id=item_id,
                    retrieval_score=float(c.get("score", 0.0)),
                    metadata=c
                ))
                labels.append(1.0 if item_id == target_id else 0.0)

            yield RerankSample(
                query_keywords=kw_list,
                candidates=candidates,
                labels=labels
            )