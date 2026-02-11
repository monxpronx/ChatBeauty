import os
import mlflow
import mlflow.xgboost
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.modeling.train.train_xgb import train_reranker_xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "evaluation", "retrieval_candidates_train.jsonl")
VALID_PATH = os.path.join(BASE_DIR, "data", "evaluation", "retrieval_candidates_valid.jsonl")
MODEL_DIR = os.path.join(BASE_DIR, "model", "reranking")
ITEM_FEAT_PATH = os.path.join(BASE_DIR, "features", "item_features_v1.csv")


def run_experiment(run_name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{run_name}.json")

    feature_builder = TreeFeatureBuilder(ITEM_FEAT_PATH)

    with mlflow.start_run(run_name=run_name):
        model, params, evals_result = train_reranker_xgb(
            train_path=TRAIN_PATH,
            valid_path=VALID_PATH,
            model_path=model_path,
            feature_builder=feature_builder,
        )

        best_iter = model.best_iteration

        best_ndcg_5 = evals_result["valid"]["ndcg@5"][best_iter]
        best_ndcg_10 = evals_result["valid"]["ndcg@10"][best_iter]

        mlflow.log_params(params)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("num_features", len(feature_builder.FEATURE_NAMES))
        mlflow.log_param("best_iteration", best_iter)

        mlflow.log_metric("best_valid_ndcg_5", best_ndcg_5)
        mlflow.log_metric("best_valid_ndcg_10", best_ndcg_10)

        mlflow.xgboost.log_model(model, artifact_path="model")

def main():
    mlflow.set_experiment("Reranker_XGB_Current_Features")
    run_experiment("xgb_reranker_current_features_v1")


if __name__ == "__main__":
    main()
