# backend/ml/item_ranker/scripts/train_xgb.py

import os
from item_ranker.modeling import train_reranker

def main():
    data_path = "backend/ml/data/processed/retrieval_candidates_train.jsonl"
    model_path = "backend/ml/model/reranking/xgb_ranker.pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_reranker(
        data_path=data_path,
        model_path=model_path,
        limit=None
    )

if __name__ == "__main__":
    main()