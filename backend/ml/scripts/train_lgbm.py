import os
import mlflow
import mlflow.lightgbm

from item_ranker.features_lgbm import FeatureBuilder
from item_ranker.modeling.train_lgbm import train_reranker


def run_experiment(
    run_name: str,
    feature_builder: FeatureBuilder,
    num_features: int,
):
    with mlflow.start_run(run_name=run_name):
        data_path = "data/processed/retrieval_candidates_train.jsonl"
        model_path = f"model/reranking/{run_name}.pkl"

        mlflow.log_param("num_features", num_features)

        model, metrics = train_reranker(
            data_path=data_path,
            model_path=model_path,
            feature_builder=feature_builder
        )

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.lightgbm.log_model(model, artifact_path="model")


def main():
    os.makedirs("model/reranking", exist_ok=True)
    mlflow.set_experiment("Reranker_Feature_Expansion")

    run_experiment(
        run_name="baseline_retrieval_only",
        feature_builder=FeatureBuilder(use_only_retrieval=True),
        num_features=1,
    )

    run_experiment(
        run_name="reranker_full_13_features",
        feature_builder=FeatureBuilder(),
        num_features=13,
    )

    run_experiment(
        run_name="reranker_no_item_stats",
        feature_builder=FeatureBuilder(
            disable_features=[
                "vp_ratio",
                "recent_review_cnt",
                "rating_std",
                "log_median_price",
            ]
        ),
        num_features=9,
    )

    run_experiment(
        run_name="reranker_no_text_match",
        feature_builder=FeatureBuilder(
            disable_features=[
                "overlap_count",
                "jaccard",
                "coverage",
            ]
        ),
        num_features=10,
    )


if __name__ == "__main__":
    main()
