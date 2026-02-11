


# ------------------- 정상 구조로 정리 -------------------
import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from tqdm import tqdm

def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args

def main():
    cli_args = parse_args()

    random.seed(42)

    # Configuration
    MODEL_NAME = cli_args.get('MODEL', 'BAAI/bge-m3')
    EPOCHS = int(cli_args.get('EPOCHS', 1))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 16))
    LEARNING_RATE = float(cli_args.get('LR', 2e-5))
    WARMUP_RATIO = float(cli_args.get('WARMUP_RATIO', 0.1))
    USE_FP16 = cli_args.get('USE_FP16', 'true').lower() == 'true'

    # File paths
    base_dir = Path(__file__).parent.parent.parent  # backend/ml/
    matched_jsonl_path = base_dir / 'data/processed/matched_query_item.jsonl'
    output_dir = base_dir / 'model/retrieval' / f'bge-m3-finetuned-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print("=" * 60)
    print("BGE-M3 MultipleNegativesRankingLoss Fine-tuning")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Output: {output_dir}")
    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")
        USE_FP16 = False
    print()

    # Load model
    print("=== Loading model ===")
    model = SentenceTransformer(MODEL_NAME)

    # Prepare training pairs (query, item_title)
    print("\n=== Preparing training pairs ===")
    query_item_texts = []
    error_count = 0

    with open(matched_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                query = data.get('query')
                item = data.get('item', {})
                asin = item.get('asin')
                title = item.get('title', '')

                if query and asin and title:
                    query_item_texts.append((query, title))
            except json.JSONDecodeError:
                error_count += 1
                continue

    if error_count > 0:
        print(f"   Skipped {error_count} lines due to JSON errors", flush=True)
    print(f"   Loaded {len(query_item_texts)} positive pairs", flush=True)

    print("2. Creating training examples...", flush=True)
    train_examples = [InputExample(texts=[query, item_title]) for query, item_title in query_item_texts]
    random.shuffle(train_examples)
    print(f"   Total training examples: {len(train_examples)}", flush=True)

    train_dataloader = DataLoader(train_examples, batch_size=BATCH_SIZE, shuffle=True)

    # MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Calculate training steps
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train!
    print("\n=== Starting training ===")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=USE_FP16
    )

    print(f"\n=== Training complete ===")
    print(f"Model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Rebuild ChromaDB with fine-tuned model:")
    print(f"   python save_to_chromadb.py MODEL_PATH={output_dir}")
    print("2. Evaluate retrieval performance on test set")

if __name__ == '__main__':
    main()