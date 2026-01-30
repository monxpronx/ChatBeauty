import json
from dataclasses import dataclass
from typing import List, Optional


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
    
    

def load_jsonl(path: str, limit: int = 10000) -> List[dict]:
    rows = []
    print(f"Loading first {limit} rows from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return rows

def parse_samples_from_rows(rows: List[dict]) -> List[RerankSample]:
    samples = []
    for row in rows:
        target_id = row["parent_asin"]
        kw_list = [k.lower() for k in row.get("keywords", [])]

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

        samples.append(RerankSample(
            query_keywords=kw_list,
            candidates=candidates,
            labels=labels
        ))
    return samples