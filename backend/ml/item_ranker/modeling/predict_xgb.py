import pickle
import numpy as np
import xgboost as xgb

from backend.ml.item_ranker.dataset_xgb import iter_samples
from backend.ml.item_ranker.features_xgb import FeatureBuilder


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

        results.append({"items": reranked_items})

    return results


class XGBReranker:
    def __init__(self, model_path: str):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.feature_builder = FeatureBuilder()

    def score(self, sample):
        X = np.array(self.feature_builder.build(sample), dtype=np.float32)
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)
        return scores.tolist()