import os
import mlflow
import mlflow.lightgbm
from item_ranker.modeling.train import train_reranker


def main():
    os.makedirs("model/reranking", exist_ok=True)

    mlflow.set_experiment("Reranker_Feature_Expansion")

    with mlflow.start_run(run_name="13_features_item_stats") as run:
        data_path = "data/processed/retrieval_candidates_train.jsonl"
        model_path = "model/reranking/rerankinglgbm_reranker.pkl"

        mlflow.log_param("data_path", data_path)
        mlflow.log_param("num_features", 13)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 5)

        model, metrics = train_reranker(
            data_path=data_path,
            model_path=model_path,
            limit=None
        )

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        feature_names = [
            "retrieval_score", "original_idx", "rating", "price",
            "overlap_count", "jaccard", "coverage", "title_len", "has_cheap",
            "vp_ratio", "recent_review_cnt", "rating_std", "log_median_price"
        ]
        
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            feature_names=feature_names
        )

        print(f"\n[MLflow] Run ID: {run.info.run_id}")
        print("[MLflow] Model logged & ready for Registry")


if __name__ == "__main__":
    main()