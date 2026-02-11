import numpy as np
import mlflow
from tqdm import tqdm
from sklearn.metrics import ndcg_score

from item_ranker.dataset_lgbm import iter_samples
from item_ranker.modeling.predict_lgbm import LGBMReranker
from item_ranker.features_lgbm import FeatureBuilder


def evaluate(data_path: str, model_path: str, feature_builder: FeatureBuilder, k: int = 10):
    reranker = LGBMReranker(model_path, feature_builder)

    ndcg_list = []
    baseline_list = []

    print(f"Evaluating on {data_path}...")
    print(f"Model: {model_path}")
    print(f"Disabled features: {feature_builder.disable_features}")

    for sample in tqdm(iter_samples(data_path)):
        if not sample.labels or sum(sample.labels) == 0:
            continue

        scores = reranker.score(sample)

        y_true = np.array([sample.labels])
        y_score = np.array([scores])

        ndcg_list.append(ndcg_score(y_true, y_score, k=k))

        baseline_scores = [c.retrieval_score for c in sample.candidates]
        baseline_list.append(
            ndcg_score(y_true, [baseline_scores], k=k)
        )

    mean_ndcg = float(np.mean(ndcg_list))
    mean_baseline = float(np.mean(baseline_list))
    improvement = (mean_ndcg - mean_baseline) / mean_baseline * 100

    if mlflow.active_run():
        mlflow.log_metric(f"ndcg_at_{k}", mean_ndcg)
        mlflow.log_metric("baseline_ndcg", mean_baseline)
        mlflow.log_metric("improvement_pct", improvement)

    print("\n" + "=" * 40)
    print(f"Baseline NDCG@{k}: {mean_baseline:.4f}")
    print(f"Model     NDCG@{k}: {mean_ndcg:.4f}")
    print(f"Improvement       : {improvement:+.2f}%")
    print("=" * 40 + "\n")

    return mean_ndcg


if __name__ == "__main__":
    VALID_PATH = "data/processed/retrieval_candidates_valid.jsonl"
    mlflow.set_experiment("Reranker_Feature_Expansion")

    with mlflow.start_run(run_name="Eval_baseline", nested=True):
        evaluate(
            VALID_PATH,
            "model/reranking/baseline_retrieval_only.pkl",
            FeatureBuilder(use_only_retrieval=True),
        )

    with mlflow.start_run(run_name="Eval_full", nested=True):
        evaluate(
            VALID_PATH,
            "model/reranking/reranker_full_13_features.pkl",
            FeatureBuilder(),
        )

    with mlflow.start_run(run_name="Eval_no_item_stats", nested=True):
        evaluate(
            VALID_PATH,
            "model/reranking/reranker_no_item_stats.pkl",
            FeatureBuilder(disable_features=[
                "vp_ratio",
                "recent_review_cnt",
                "rating_std",
                "log_median_price",
            ]),
        )

    with mlflow.start_run(run_name="Eval_no_text_match", nested=True):
        evaluate(
            VALID_PATH,
            "model/reranking/reranker_no_text_match.pkl",
            FeatureBuilder(disable_features=[
                "overlap_count",
                "jaccard",
                "coverage",
            ]),
        )
