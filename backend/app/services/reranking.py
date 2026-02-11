from item_ranker.dataset.base import Candidate, RerankSample
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.modeling.predict.lgbm import LGBMTreeReranker
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RERANK_MODEL_PATH = (
    BASE_DIR / "ml" / "model" / "reranking" / "lgbm_reranker_current_features_v1.pkl"
)

ITEM_FEAT_PATH = (
    BASE_DIR / "ml" / "features" / "item_features_v1.csv"
)

feature_builder = TreeFeatureBuilder(str(ITEM_FEAT_PATH))

reranker = LGBMTreeReranker(
    model_path=str(RERANK_MODEL_PATH),
    feature_builder=feature_builder,
)


def rerank_items(query: str, candidates: list[dict], top_k: int):
    query_keywords = query.lower().split()

    candidate_objs = [
        Candidate(
            item_id=c["item_id"],
            retrieval_score=c["score"],
            metadata=c,
        )
        for c in candidates
    ]

    sample = RerankSample(
        query_keywords=query_keywords,
        candidates=candidate_objs,
        labels=None,
    )

    scores = reranker.score(sample)

    reranked = []
    for c, s in zip(candidates, scores):
        item = c.copy()
        item["score"] = float(s)
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]
