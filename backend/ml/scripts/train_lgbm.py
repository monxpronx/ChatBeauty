from item_ranker.dataset import load_jsonl, parse_samples_from_rows
from item_ranker.modeling.train import train_reranker


def main():
    rows = load_jsonl("data/processed/retrieval_candidates_train.jsonl")
    samples = parse_samples_from_rows(rows)

    print(f"#samples = {len(samples)}")

    num_pos = sum(sum(s.labels) for s in samples)
    print(f"#positive labels = {num_pos}")

    avg_group = sum(len(s.candidates) for s in samples) / len(samples)
    print(f"avg group size = {avg_group:.2f}")

    train_reranker(
        samples=samples,
        model_path="model/rerankinglgbm_reranker.pkl",
    )


if __name__ == "__main__":
    main()
