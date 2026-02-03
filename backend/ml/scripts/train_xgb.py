import os
from backend.ml.item_ranker.modeling.train_xgb import train_reranker_xgb

def main():
    data_path = "backend/ml/data/processed/retrieval_candidates_train.jsonl"
    model_path = "backend/ml/model/reranking/xgb_ranker.pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_reranker_xgb(
        data_path=data_path,
        model_path=model_path,
        run_name="baseline_(max_depth=5)", ########### -> 이름 설정!
        limit=None
    )

if __name__ == "__main__":
    main()