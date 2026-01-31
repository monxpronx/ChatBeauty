import numpy as np
import mlflow
from tqdm import tqdm
from sklearn.metrics import ndcg_score

from item_ranker.dataset import iter_samples
from item_ranker.modeling.predict import LGBMReranker

def evaluate(data_path: str, model_path: str, k: int = 10):
    reranker = LGBMReranker(model_path)
    
    ndcg_list = []
    baseline_list = []
    
    print(f"Evaluating on {data_path}...")
    
    for sample in tqdm(iter_samples(data_path)):
        if not sample.labels or sum(sample.labels) == 0:
            continue
            
        scores = reranker.score(sample)
        y_true = np.array([sample.labels])
        y_score = np.array([scores])
        
        score = ndcg_score(y_true, y_score, k=k)
        ndcg_list.append(score)
        
        baseline_scores = [c.retrieval_score for c in sample.candidates]
        b_score = ndcg_score(y_true, [baseline_scores], k=k)
        baseline_list.append(b_score)
    
    mean_ndcg = np.mean(ndcg_list)
    mean_baseline = np.mean(baseline_list)
    improvement = (mean_ndcg - mean_baseline) / mean_baseline * 100
    
    if mlflow.active_run():
        mlflow.log_metric(f"ndcg_at_{k}", mean_ndcg)
        mlflow.log_metric("baseline_ndcg", mean_baseline)
        mlflow.log_metric("improvement_pct", improvement)
    
    print("\n" + "="*40)
    print(f"Results for {data_path}")
    print(f"Baseline NDCG@{k}: {mean_baseline:.4f}")
    print(f"Model     NDCG@{k}: {mean_ndcg:.4f}")
    print(f"Improvement    : {improvement:+.2f}%")
    print("="*40 + "\n")
    
    return mean_ndcg

if __name__ == "__main__":
    MODEL_PATH = "model/reranking/rerankinglgbm_reranker.pkl"
    VALID_PATH = "data/processed/retrieval_candidates_valid.jsonl"
    
    mlflow.set_experiment("Reranker_Feature_Expansion")
    with mlflow.start_run(run_name="Evaluation_Only", nested=True):
        evaluate(VALID_PATH, MODEL_PATH, k=10)