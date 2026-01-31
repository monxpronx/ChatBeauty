import os
from item_ranker_xgb.modeling.train_xgb import train_reranker_xgb


def main():
    os.makedirs("model/reranking_xgb", exist_ok=True)

    data_path = "data/processed/retrieval_candidates_train.jsonl"
    model_path = "model/reranking_xgb/xgb_reranker.pkl"

    train_reranker_xgb(
        data_path=data_path,
        model_path=model_path,
        limit=None
    )


if __name__ == "__main__":
    main()