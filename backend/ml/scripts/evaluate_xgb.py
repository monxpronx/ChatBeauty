import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset.base import iter_samples

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "evaluation", "retrieval_candidates_train.jsonl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "reranking", "xgb_reranker_current_features_v1.json")
ITEM_FEAT_PATH = os.path.join(BASE_DIR, "features", "item_features_v1.csv")
EXPERIMENT_NAME = "Reranker_XGB_Current_Features"

def evaluate_ndcg(k: int = 10):
    feature_builder = TreeFeatureBuilder(ITEM_FEAT_PATH)

    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print(f"[OK] Loaded model from {MODEL_PATH}")

    X_list, y_list, group = [], [], []
    for sample in iter_samples(DATA_PATH):
        df = feature_builder.build(sample)
        labels = sample.labels

        if not labels or sum(labels) == 0:
            continue

        X_list.append(df)
        y_list.extend(labels)
        group.append(len(df))

    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list, dtype=np.float32)

    dmatrix = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
    dmatrix.set_group(group)

    y_pred = model.predict(dmatrix)

    ndcg_scores = []
    start = 0
    for g in group:
        end = start + g
        ndcg_scores.append(ndcg_score([y[start:end]], [y_pred[start:end]], k=k))
        start = end

    mean_ndcg = float(np.mean(ndcg_scores))
    print(f"[Metric] train_ndcg@{k} = {mean_ndcg:.4f}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"eval_xgb_ndcg_k{k}"):
        mlflow.log_metric(f"train_ndcg_{k}", mean_ndcg)
        mlflow.xgboost.log_model(model, artifact_path="model")

    print("[OK] NDCG evaluation logged to MLflow")
    return mean_ndcg


if __name__ == "__main__":
    evaluate_ndcg(k=5)
