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
    query: str
    candidates: List[Candidate]
    labels: Optional[List[float]] = None
    
    

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def parse_samples_from_rows(rows: List[dict]) -> List[RerankSample]:
    samples = []

    for row in rows:
        candidates = [
            Candidate(
                item_id=c["item_id"],
                retrieval_score=float(c.get("retrieval_score", 0.0)),
                metadata={k: v for k, v in c.items()
                          if k not in ["item_id", "retrieval_score"]}
            )
            for c in row["candidates"]
        ]

        sample = RerankSample(
            query=row["query"],
            candidates=candidates,
            labels=row.get("labels"),
        )
        samples.append(sample)

    return samples