import os
import numpy as np
from sklearn.metrics import ndcg_score
from item_ranker.dataset.base import iter_samples

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "evaluation", "retrieval_candidates_valid.jsonl"
)

def evaluate_retrieval_ndcg(k: int = 10):
    ndcg_scores = []

    for sample in iter_samples(DATA_PATH):
        if not sample.labels or sum(sample.labels) == 0:
            continue

        y_true = np.array(sample.labels, dtype=np.float32)

        y_pred = np.array(
            [c.retrieval_score for c in sample.candidates],
            dtype=np.float32
        )

        score = ndcg_score([y_true], [y_pred], k=k)
        ndcg_scores.append(score)

    mean_ndcg = float(np.mean(ndcg_scores))
    print(f"[Baseline] Retrieval NDCG@{k} = {mean_ndcg:.4f}")

    return mean_ndcg


if __name__ == "__main__":
    evaluate_retrieval_ndcg(k=5)
