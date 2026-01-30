import pandas as pd
import os
from item_ranker.dataset import RerankSample

class FeatureBuilder:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.dirname(current_dir)
        item_feat_path = os.path.join(ml_dir, "features", "item_features_v1.csv")
        
        if os.path.exists(item_feat_path):
            df = pd.read_csv(item_feat_path)
            df = df.fillna(0)
            self.item_stats = df.set_index("parent_asin").to_dict(orient="index")
            print(f"[OK] Loaded item features from {item_feat_path}")
        else:
            print(f"[Error] CSV not found at: {item_feat_path}")
            self.item_stats = {}

    def build(self, sample: RerankSample):
        query_words = set(sample.query_keywords)
        features = []

        for idx, c in enumerate(sample.candidates):
            asin = c.metadata.get("item_asin")
            
            title = (c.metadata.get("title") or "").lower()
            title_words = set(title.split())
            overlap_count = len(query_words & title_words)
            jaccard = overlap_count / len(query_words | title_words) if (query_words | title_words) else 0.0
            coverage = overlap_count / len(query_words) if query_words else 0.0
            rating = float(c.metadata.get("average_rating", 0.0) or 0.0)
            price = float(c.metadata.get("price", 0.0) or 0.0)
            title_len = len(title_words)
            has_cheap = 1.0 if "cheap" in query_words else 0.0

            stats = self.item_stats.get(asin, {})
            vp_ratio = float(stats.get("vp_ratio", 0.0))
            recent_cnt = float(stats.get("recent_review_cnt", 0.0))
            rating_std = float(stats.get("rating_std", 0.0))
            log_price = float(stats.get("log_median_price", 0.0))

            row_feat = [
                c.retrieval_score,
                idx,
                rating,
                price,
                overlap_count,
                jaccard,
                coverage,
                title_len,
                has_cheap,
                vp_ratio,
                recent_cnt,
                rating_std,
                log_price
            ]
            features.append(row_feat)

        return features