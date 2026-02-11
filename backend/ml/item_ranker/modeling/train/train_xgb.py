import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from sklearn.metrics import ndcg_score
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset import iter_samples


def train_reranker_xgb(
    data_path: str,
    model_path: str,
    feature_builder: TreeFeatureBuilder,
    limit: Optional[int] = None,
) -> Tuple[xgb.Booster, Dict, float]:
 
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 3,
        "tree_method": "hist",
        "seed": 42,
    }

    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        df = feature_builder.build(sample)
        labels = sample.labels

        X_list.append(df)
        y_list.extend(labels)
        group.append(len(df))

    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list, dtype=np.float32)

    dtrain = xgb.DMatrix(X.values, label=y, feature_names=X.columns.tolist())
    dtrain.set_group(group)

    print("[Train] Start XGBoost training...")
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=300)
    model.save_model(model_path)
    print(f"[OK] Model saved to {model_path}")

    y_pred = model.predict(dtrain)
    ndcg_scores = []
    start = 0
    for g in group:
        end = start + g
        ndcg_scores.append(ndcg_score([y[start:end]], [y_pred[start:end]], k=10))
        start = end
    mean_ndcg_10 = float(sum(ndcg_scores) / len(ndcg_scores))
    print(f"[Metric] train_ndcg@10 = {mean_ndcg_10:.4f}")

    return model, params, mean_ndcg_10
