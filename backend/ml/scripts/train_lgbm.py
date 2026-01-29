from item_ranker.dataset import load_jsonl, parse_samples_from_rows
from item_ranker.modeling.train import train_reranker


def main():
    rows = load_jsonl("ml/data/raw/train.jsonl")
    samples = parse_samples_from_rows(rows)

    train_reranker(
        samples=samples,
        model_path="ml/models/lgbm_reranker.pkl",
    )


if __name__ == "__main__":
    main()
