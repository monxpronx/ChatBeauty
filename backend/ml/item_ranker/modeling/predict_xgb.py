import xgboost as xgb
import numpy as np
from typing import List

from item_ranker.dataset import iter_samples
from item_ranker.features import FeatureBuilder


def rerank_candidates(
    data_path: str,
    model_path: str,
    limit: int = 1000,
    topk: int = 10,
):
    model = xgb.Booster()
    model.load_model(model_path)

    feature_builder = FeatureBuilder()
    results = []

    for sample in iter_samples(data_path, limit=limit):
        X = np.array(feature_builder.build(sample), dtype=np.float32)
        dtest = xgb.DMatrix(X)

        scores = model.predict(dtest)

        ranked_idx = np.argsort(scores)[::-1][:topk]
        reranked_items = [
            sample.candidates[i].item_id for i in ranked_idx
        ]

        results.append({
            "items": reranked_items
        })

    return results