import xgboost as xgb
import numpy as np
from typing import Optional

from backend.ml.item_ranker.dataset import iter_samples
from backend.ml.item_ranker.features import FeatureBuilder


def train_reranker_xgb(
    data_path: str,
    model_path: str,
    limit: Optional[int] = None,
):
    feature_builder = FeatureBuilder()

    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        feats = feature_builder.build(sample)
        labels = sample.labels

        X_list.append(np.array(feats, dtype=np.float32))
        y_list.append(np.array(labels, dtype=np.float32))
        group.append(len(feats))

        if len(group) % 1000 == 0:
            print(f"Processed {len(group)} queries")

    print("Concatenating features...")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 6,
        "tree_method": "hist",
        "seed": 42,
    }

    print("Training XGBoost Ranker...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
    )

    model.save_model(model_path)
    print(f"[OK] Model saved to {model_path}")

    return model