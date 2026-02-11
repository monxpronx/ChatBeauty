import os
import mlflow
import mlflow.xgboost
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.modeling.train.train_xgb import train_reranker_xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "evaluation", "retrieval_candidates_train.jsonl")
MODEL_DIR = os.path.join(BASE_DIR, "model", "reranking")
ITEM_FEAT_PATH = os.path.join(BASE_DIR, "features", "item_features_v1.csv")


def run_experiment(run_name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{run_name}.json")

    feature_builder = TreeFeatureBuilder(ITEM_FEAT_PATH)

    with mlflow.start_run(run_name=run_name):
        model, params, mean_ndcg_10 = train_reranker_xgb(
            data_path=DATA_PATH,
            model_path=model_path,
            feature_builder=feature_builder,
        )

        mlflow.log_params(params)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("num_features", len(feature_builder.FEATURE_NAMES))
        mlflow.log_metric("train_ndcg_10", mean_ndcg_10)

        mlflow.xgboost.log_model(model, artifact_path="model")


def main():
    mlflow.set_experiment("Reranker_XGB_Current_Features")
    run_experiment("xgb_reranker_current_features_v1")


if __name__ == "__main__":
    main()
