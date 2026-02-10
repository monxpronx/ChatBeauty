import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]  # backend/ml
ITEM_FEAT_PATH = BASE_DIR / "data/cleaned/item_v1/item_v1.csv"
ITEM_FEAT_DF = pd.read_csv(ITEM_FEAT_PATH)
ITEM_FEAT_MAP = ITEM_FEAT_DF.set_index("parent_asin").to_dict(orient="index")

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


def iter_samples(data_path: str, limit: Optional[int] = None) -> Generator[RerankSample, None, None]:
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skip broken line at {i}")
                continue

            target_id = row.get("parent_asin")
            raw_keywords = row.get("keywords") or []
            query_keywords = [k.lower() for k in raw_keywords if isinstance(k, str)]

            candidates = []
            labels = []

            for c in row.get("candidates", []):
                item_id = c["item_asin"]
                item_feat = ITEM_FEAT_MAP.get(item_id, {})

                metadata = {
                    **{k: v for k, v in c.items() if isinstance(v, (int, float))},
                    **item_feat
                }

                candidates.append(
                    Candidate(
                        item_id=item_id,
                        retrieval_score=float(c.get("score", 0.0)),
                        metadata=metadata
                    )
                )

                labels.append(1.0 if item_id == target_id else 0.0)

            yield RerankSample(
                query_keywords=query_keywords,
                candidates=candidates,
                labels=labels
            )