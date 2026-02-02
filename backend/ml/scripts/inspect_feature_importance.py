import pickle
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import os

EXPERIMENT_NAME = "Reranker_Feature_Expansion"

MODEL_PATH = "model/reranking/reranker_full_13_features.pkl"
RUN_NAME = "Feature_Importance_full"

FEATURE_NAMES = [
    "retrieval_score",
    "original_idx",
    "rating",
    "price",
    "overlap_count",
    "jaccard",
    "coverage",
    "title_len",
    "has_cheap",
    "vp_ratio",
    "recent_review_cnt",
    "rating_std",
    "log_median_price",
]

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "feature_importance.csv")
PNG_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        importances = pd.Series(
            model.feature_importances_,
            index=FEATURE_NAMES
        ).sort_values(ascending=False)

        print("\n[Feature Importance — Gain]")
        print(importances)

        importances.to_csv(CSV_PATH)
        mlflow.log_artifact(CSV_PATH)

        plt.figure(figsize=(8, 6))
        importances.sort_values().plot.barh()
        plt.title("Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig(PNG_PATH)
        plt.close()

        mlflow.log_artifact(PNG_PATH)

        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("num_features", len(FEATURE_NAMES))

        print(f"\n[OK] Feature importance logged to MLflow")
        print(f" - CSV : {CSV_PATH}")
        print(f" - PNG : {PNG_PATH}")


if __name__ == "__main__":
    main()
