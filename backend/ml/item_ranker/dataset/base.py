import json
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Generator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ITEM_FEAT_PATH = BASE_DIR / "data/cleaned/item_v1/item_v1.csv"

if ITEM_FEAT_PATH.exists():
    ITEM_FEAT_DF = pd.read_csv(ITEM_FEAT_PATH).fillna(0)
    ITEM_FEAT_MAP = ITEM_FEAT_DF.set_index("parent_asin").to_dict(orient="index")
else:
    ITEM_FEAT_MAP = {}
    print(f"[WARN] item feature not found: {ITEM_FEAT_PATH}")


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
    data_path: str,
    limit: Optional[int] = None
) -> Generator[RerankSample, None, None]:

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            row = json.loads(line)

            target_id = row.get("parent_asin")
            keywords = [
                str(k).lower()
                for k in row.get("keywords", [])
                if isinstance(k, str)
            ]

            candidates, labels = [], []

            for c in row.get("candidates", []):
                item_id = c["item_asin"]
                item_feat = ITEM_FEAT_MAP.get(item_id, {})

                metadata = {
                    **c,
                    **item_feat,
                }

                candidates.append(
                    Candidate(
                        item_id=item_id,
                        retrieval_score=float(c.get("score", 0.0)),
                        metadata=metadata,
                    )
                )

                if target_id is not None:
                    labels.append(1.0 if item_id == target_id else 0.0)

            yield RerankSample(
                query_keywords=keywords,
                candidates=candidates,
                labels=labels if target_id else None,
            )
