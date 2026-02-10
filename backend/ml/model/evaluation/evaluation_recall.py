"""
Evaluate retrieval quality (Recall@K, MRR) from retrieval_candidates_{split}.jsonl.

Usage:
    python ml/evaluation/evaluate_recall.py SPLIT=valid
    python ml/evaluation/evaluate_recall.py SPLIT=test
"""

import json
import sys
from pathlib import Path


def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def main():
    cli_args = parse_args()
    split = cli_args.get('SPLIT', 'test')

    # File is at: backend/ml/model/evaluation/evaluation_recall.py
    # Need to go up to backend/ml/
    base_dir = Path(__file__).parent.parent.parent  # backend/ml/
    candidates_path = base_dir / f'data/evaluation/retrieval_candidates_{split}.jsonl'

    if not candidates_path.exists():
        print(f"Error: {candidates_path} not found")
        return

    k_values = [1, 5, 10, 20, 50, 100]
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total = 0

    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            ground_truth = item['parent_asin']
            candidate_asins = [c['item_asin'] for c in item['candidates']]

            # Find rank of ground truth (1-indexed)
            rank = None
            for i, asin in enumerate(candidate_asins):
                if asin == ground_truth:
                    rank = i + 1
                    break

            for k in k_values:
                if rank is not None and rank <= k:
                    hits[k] += 1

            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            total += 1

    if total == 0:
        print("No queries found.")
        return

    print("=" * 40)
    print(f"Split: {split} | Queries: {total:,}")
    print("=" * 40)
    for k in k_values:
        recall = hits[k] / total
        print(f"  Recall@{k:<4d} {recall:.4f}  ({hits[k]:,}/{total:,})")
    mrr = sum(reciprocal_ranks) / total
    print(f"  MRR        {mrr:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()