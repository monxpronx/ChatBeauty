import pickle
from typing import List

import lightgbm as lgb

from item_ranker.dataset import RerankSample
from item_ranker.features import FeatureBuilder


def build_lgbm_data(samples, feature_builder):
    X, y, group = [], [], []

    for sample in samples:
        assert sample.labels is not None, "Training requires labels"

        feats = feature_builder.build(sample)
        labels = sample.labels

        assert len(feats) == len(labels)

        X.extend(feats)
        y.extend(labels)
        group.append(len(feats))

    return X, y, group


def train_reranker(
    samples: List[RerankSample],
    model_path: str,
):
    feature_builder = FeatureBuilder()
    X, y, group = build_lgbm_data(samples, feature_builder)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )

    model.fit(X, y, group=group)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {model_path}")
    return model
