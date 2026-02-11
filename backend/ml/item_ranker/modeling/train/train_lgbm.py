import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from typing import Optional, Dict, Tuple
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset import iter_samples

def train_reranker(
    data_path: str,
    model_path: str,
    feature_builder: TreeFeatureBuilder,
    limit: Optional[int] = None,
) -> Tuple[lgb.LGBMRanker, Dict[str, float]]:

    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        df = feature_builder.build(sample)
        labels = sample.labels

        X_list.append(df)
        y_list.extend(labels)
        group.append(len(df))

    X = pd.concat(X_list, ignore_index=True)
    y = y_list

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )

    print("[Train] Start LightGBM training...")
    model.fit(X, y, group=group)

    y_pred = model.predict(X)

    ndcg_scores = []
    start = 0
    for g in group:
        end = start + g
        ndcg_scores.append(
            ndcg_score([y[start:end]], [y_pred[start:end]], k=10)
        )
        start = end

    mean_ndcg_10 = float(sum(ndcg_scores) / len(ndcg_scores))

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {model_path}")
    print(f"[Metric] train_ndcg@10 = {mean_ndcg_10:.4f}")

    return model, {"train_ndcg_10": mean_ndcg_10}