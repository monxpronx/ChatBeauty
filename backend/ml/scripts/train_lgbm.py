import os
from item_ranker.modeling.train import train_reranker

def main():
    os.makedirs("model", exist_ok=True)
    
    data_path = "data/processed/retrieval_candidates_train.jsonl"
    model_path = "model/reranking/rerankinglgbm_reranker.pkl"

    train_reranker(
        data_path=data_path,
        model_path=model_path,
        limit=None
    )

if __name__ == "__main__":
    main()