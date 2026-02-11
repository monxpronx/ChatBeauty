from typing import List
import lightgbm as lgb

from item_ranker.features_lgbm import FeatureBuilder
from item_ranker.dataset_lgbm import RerankSample
from item_ranker.modeling.predict_lgbm import Reranker


class TreeReranker(Reranker):
    def __init__(self, model: lgb.LGBMRanker, feature_builder: FeatureBuilder):
        self.model = model
        self.feature_builder = feature_builder

    def score(self, sample: RerankSample) -> List[float]:
        X = self.feature_builder.build(sample)
        return self.model.predict(X)
