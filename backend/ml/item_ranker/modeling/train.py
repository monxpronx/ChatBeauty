import pickle
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import ndcg_score
from item_ranker.features import FeatureBuilder
from item_ranker.dataset import iter_samples
from typing import Optional, Dict, Tuple


def train_reranker(
    data_path: str,
    model_path: str,
    limit: Optional[int] = None
) -> Tuple[lgb.LGBMRanker, Dict[str, float]]:

    feature_builder = FeatureBuilder()

    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        feats = feature_builder.build(sample)
        labels = sample.labels

        X_list.append(np.array(feats, dtype=np.float32))
        y_list.append(np.array(labels, dtype=np.float32))
        group.append(len(feats))

        if len(group) % 10000 == 0:
            print(f"Processed {len(group)} queries...")

    print("Concatenating features into matrix...")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    del X_list, y_list

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )

    print("Starting LightGBM training...")
    model.fit(X, y, group=group)

    y_pred = model.predict(X)

    ndcg_scores = []
    start = 0
    for g in group:
        end = start + g
        ndcg_scores.append(
            ndcg_score(
                [y[start:end]],
                [y_pred[start:end]],
                k=10
            )
        )
        start = end

    mean_ndcg_10 = float(np.mean(ndcg_scores))

    feature_names = [
        "retrieval_score", "original_idx", "rating", "price",
        "overlap_count", "jaccard", "coverage", "title_len", "has_cheap",
        "vp_ratio", "recent_review_cnt", "rating_std", "log_median_price"
    ]

    importances = pd.Series(model.feature_importances_, index=feature_names)
    print("\n" + "=" * 30)
    print("[Feature Importance - Gain]")
    print(importances.sort_values(ascending=False))
    print("=" * 30 + "\n")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {model_path}")
    print(f"[Metric] mean_ndcg_10 = {mean_ndcg_10:.4f}")

    return model, {
        "ndcg_10": mean_ndcg_10
    }
