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


def iter_samples(
    path: str,
    limit: Optional[int] = None
) -> Generator[RerankSample, None, None]:
    """
    JSONL 파일을 한 줄씩 읽어서 RerankSample을 생성한다.
    dataset 단계에서는 feature engineering을 하지 않는다.
    """

    print(f"[INFO] Streaming samples from {path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            row = json.loads(line)

            target_id = row.get("parent_asin")
            raw_keywords = row.get("keywords", [])
            query_keywords = [
                k.lower() for k in raw_keywords if isinstance(k, str)
            ]

            candidates: List[Candidate] = []
            labels: List[float] = []

            for c in row.get("candidates", []):
                item_id = c.get("item_asin")

                candidates.append(
                    Candidate(
                        item_id=item_id,
                        retrieval_score=float(c.get("score", 0.0)),
                        metadata=c,
                    )
                )

                labels.append(1.0 if item_id == target_id else 0.0)

            yield RerankSample(
                query_keywords=query_keywords,
                candidates=candidates,
                labels=labels,
            )