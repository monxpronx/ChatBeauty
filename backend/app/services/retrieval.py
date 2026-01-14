def retrieve_candidates(n: int = 20):
    return [
        {"item_id" : f"item_{i}", "score":1.0}
        for i in range(n)
    ]