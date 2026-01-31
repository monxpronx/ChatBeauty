import pickle
import numpy as np
from xgboost import XGBRanker

from item_ranker.dataset import iter_samples
from item_ranker.features import FeatureBuilder


def train_reranker_xgb(data_path: str, model_path: str, limit=None):
    feature_builder = FeatureBuilder()

    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        feats = feature_builder.build(sample)      # (n_candidates, n_features)
        labels = sample.labels                     # (n_candidates,)

        X_list.append(np.array(feats, dtype=np.float32))
        y_list.append(np.array(labels, dtype=np.float32))
        group.append(len(feats))

        if len(group) % 1000 == 0:
            print(f"Processed {len(group)} queries")

    print("Concatenating features...")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    model = XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        random_state=42,
    )

    print("Training XGBoost Ranker...")
    model.fit(X, y, group=group)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {model_path}")
    return model