import mlflow
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from backend.ml.item_ranker.dataset_xgb import iter_samples
from backend.ml.item_ranker.modeling.predict_xgb import XGBReranker


def evaluate(data_path: str, model_path: str, k: int = 10):
    reranker = XGBReranker(model_path)

    ndcg_list = []
    baseline_list = []

    for sample in tqdm(iter_samples(data_path)):
        if not sample.labels or sum(sample.labels) == 0:
            continue

        scores = reranker.score(sample)

        y_true = np.array([sample.labels])
        y_score = np.array([scores])

        ndcg_list.append(ndcg_score(y_true, y_score, k=k))

        baseline_scores = [c.retrieval_score for c in sample.candidates]
        baseline_list.append(
            ndcg_score(y_true, np.array([baseline_scores]), k=k)
        )

    mean_ndcg = float(np.mean(ndcg_list))
    mean_baseline = float(np.mean(baseline_list))

    # MLflow 기록
    mlflow.log_metric(f"ndcg_{k}", mean_ndcg)
    mlflow.log_metric(f"baseline_ndcg_{k}", mean_baseline)
    mlflow.log_metric(
        f"ndcg_improvement_pct_{k}",
        (mean_ndcg - mean_baseline) / mean_baseline * 100
    )


    print("=" * 40)
    print(f"NDCG@{k} (baseline): {mean_baseline:.4f}")
    print(f"NDCG@{k} (XGB)     : {mean_ndcg:.4f}")
    print(f"Improvement (%)  : {(mean_ndcg - mean_baseline) / mean_baseline * 100:.2f}%")
    print("=" * 40)

    return mean_ndcg


if __name__ == "__main__":
    VALID_PATH = "backend/ml/data/processed/retrieval_candidates_valid.jsonl"
    MODEL_PATH = "backend/ml/model/reranking/xgb_ranker.pkl"

    evaluate(VALID_PATH, MODEL_PATH, k=10)