class FeatureBuilder:
    def build(self, sample):
        """
        sample: RerankSample
        return: List[List[float]]  (n_candidates, n_features)
        """

        features = []

        query_words = set(sample.query_keywords)

        for idx, c in enumerate(sample.candidates):
            title = (c.metadata.get("title") or "").lower()
            title_words = set(title.split())

            overlap = len(query_words & title_words)

            row = [
                c.retrieval_score,   # retrieval score
                idx,                 # original rank
                overlap,             # token overlap

                # --- item-level features ---
                c.metadata.get("avg_rating", 0.0),
                c.metadata.get("rating_std", 0.0),
                c.metadata.get("review_cnt", 0),
                c.metadata.get("vp_ratio", 0.0),
                c.metadata.get("recent_review_cnt", 0),
                c.metadata.get("log_median_price", 0.0),
            ]

            features.append(row)

        return features