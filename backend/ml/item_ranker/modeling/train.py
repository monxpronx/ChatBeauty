import pickle
from typing import List

import lightgbm as lgb
import pandas as pd

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


def train_reranker(samples: List[RerankSample], model_path: str):
    feature_builder = FeatureBuilder()
    X, y, group = build_lgbm_data(samples, feature_builder)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )

    model.fit(
        X, y, 
        group=group,
    )

    feature_names = [
        "retrieval_score", "original_idx", "rating", "price", 
        "overlap_count", "jaccard", "coverage", "title_len", "has_cheap"
    ]
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    
    print("\n" + "="*30)
    print("[Feature Importance - Gain]")
    print(importances.sort_values(ascending=False))
    print("="*30 + "\n")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {model_path}")
    return model
