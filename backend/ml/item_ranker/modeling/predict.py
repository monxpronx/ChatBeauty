import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from abc import ABC, abstractmethod

from item_ranker.dataset import RerankSample, Candidate
from item_ranker.features import FeatureBuilder

class Reranker(ABC):
    @abstractmethod
    def score(self, sample: RerankSample) -> List[float]:
        pass

class LGBMReranker(Reranker):
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        self.feature_builder = FeatureBuilder()
        
        self.feature_names = [
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
            "recent_cnt",
            "rating_std",
            "log_price"
        ]

    def score(self, sample: RerankSample) -> List[float]:
        feats = self.feature_builder.build(sample) 
        
        X = pd.DataFrame(feats, columns=self.feature_names)
        
        scores = self.model.predict(X)
        return scores.tolist()

def rerank(sample: RerankSample, reranker: Reranker) -> List[Tuple[Candidate, float]]:
    scores = reranker.score(sample)
    ranked = sorted(
        zip(sample.candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked