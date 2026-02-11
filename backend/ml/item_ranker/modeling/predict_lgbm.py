import pickle
import numpy as np
from item_ranker.features_lgbm import FeatureBuilder
from item_ranker.dataset_lgbm import RerankSample


class LGBMReranker:
    def __init__(self, model_path: str, feature_builder: FeatureBuilder):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.feature_builder = feature_builder

    def score(self, sample: RerankSample):
        X = np.array(
            self.feature_builder.build(sample),
            dtype=np.float32
        )
        return self.model.predict(X)
