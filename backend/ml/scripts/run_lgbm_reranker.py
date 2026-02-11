import os
import mlflow
import mlflow.lightgbm
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.modeling.train.train_lgbm import train_reranker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "evaluation", "retrieval_candidates_train.jsonl")
VALID_PATH = os.path.join(BASE_DIR, "data", "evaluation", "retrieval_candidates_valid.jsonl")

MODEL_DIR = os.path.join(BASE_DIR, "model", "reranking")
ITEM_FEAT_PATH = os.path.join(BASE_DIR, "features", "item_features_v1.csv")

def run_experiment(run_name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{run_name}.pkl")

    feature_builder = TreeFeatureBuilder(ITEM_FEAT_PATH)

    with mlflow.start_run(run_name=run_name):
        model = train_reranker(
            train_path=TRAIN_PATH,
            valid_path=VALID_PATH,
            model_path=model_path,
            feature_builder=feature_builder
        )

        mlflow.log_param("num_features", len(feature_builder.FEATURE_NAMES))
        mlflow.lightgbm.log_model(model, artifact_path="model")

def main():
    mlflow.set_experiment("Reranker_LGBM_Current_Features")
    run_experiment("lgbm_reranker_current_features_v1")

if __name__ == "__main__":
    main()