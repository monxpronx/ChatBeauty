from item_ranker.dataset import RerankSample


class FeatureBuilder:
    def build(self, sample: RerankSample):
        query_words = set(sample.query_keywords)
        features = []

        for idx, c in enumerate(sample.candidates):
            title = (c.metadata.get("title") or "").lower()
            title_words = set(title.split())

            common_words = query_words & title_words
            overlap_count = len(common_words)

            union_words = query_words | title_words
            jaccard = overlap_count / len(union_words) if union_words else 0.0
            coverage = overlap_count / len(query_words) if query_words else 0.0

            rating = float(c.metadata.get("average_rating", 0.0) or 0.0)
            price = float(c.metadata.get("price", 0.0) or 0.0)

            title_len = len(title_words)

            has_cheap = 1.0 if "cheap" in query_words else 0.0

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
            ]

            features.append(row_feat)

        return features
