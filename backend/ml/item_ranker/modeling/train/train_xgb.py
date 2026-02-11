import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from sklearn.metrics import ndcg_score
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset.base import iter_samples


응. XGB도 동일하게 수정해야 해.
지금 구조는 LightGBM이랑 똑같이 train만으로 학습 + train으로 평가라서, valid를 넣어야 제대로 튜닝이 가능해.

아래처럼 바꾸면 돼.

1️⃣ 데이터 빌드 함수 공통화 (재사용 권장)
def build_dataset(data_path, feature_builder, limit=None):
    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        df = feature_builder.build(sample)
        labels = sample.labels

        if not labels or sum(labels) == 0:
            continue

        X_list.append(df)
        y_list.extend(labels)
        group.append(len(df))

    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list, dtype=np.float32)
    return X, y, group

2️⃣ train_reranker_xgb 수정 (valid + early stopping)
def train_reranker_xgb(
    train_path: str,
    valid_path: str,
    model_path: str,
    feature_builder: TreeFeatureBuilder,
    limit: Optional[int] = None,
):

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 3,
        "tree_method": "hist",
        "seed": 42,
    }

    X_train, y_train, group_train = build_dataset(train_path, feature_builder, limit)
    X_valid, y_valid, group_valid = build_dataset(valid_path, feature_builder)

    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=X_train.columns.tolist())
    dtrain.set_group(group_train)

    dvalid = xgb.DMatrix(X_valid.values, label=y_valid, feature_names=X_valid.columns.tolist())
    dvalid.set_group(group_valid)

    print("[Train] Start XGBoost training...")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,                # 크게 잡고
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,            # 핵심
        verbose_eval=50
    )

    model.save_model(model_path)
    print(f"[OK] Model saved to {model_path}")

    return model, params

def build_dataset(data_path, feature_builder, limit=None):
    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        df = feature_builder.build(sample)
        labels = sample.labels

        if not labels or sum(labels) == 0:
            continue

        X_list.append(df)
        y_list.extend(labels)
        group.append(len(df))

    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list, dtype=np.float32)
    return X, y, group