import lightgbm as lgb
from typing import List
from item_ranker.dataset import RerankSample
from .base_tree import BaseTreeReranker

class LGBMTreeReranker(BaseTreeReranker):
    def __init__(self, model: lgb.LGBMRanker, feature_builder: TreeFeatureBuilder):
        super().__init__(feature_builder)
        self.model = model

    def score(self, sample: RerankSample) -> List[float]:
        X = self._build_features(sample)
        return self.model.predict(X).tolist()