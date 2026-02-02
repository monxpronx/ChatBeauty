


# ------------------- 정상 구조로 정리 -------------------
import sys
import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
from tqdm import tqdm
from .prepare_training_pairs import prepare_training_pairs

def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args

def main():
    cli_args = parse_args()

    # Configuration
    MODEL_NAME = cli_args.get('MODEL', 'BAAI/bge-m3')
    EPOCHS = int(cli_args.get('EPOCHS', 1))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 16))
    LEARNING_RATE = float(cli_args.get('LR', 2e-5))
    WARMUP_RATIO = float(cli_args.get('WARMUP_RATIO', 0.1))
    USE_FP16 = cli_args.get('USE_FP16', 'true').lower() == 'true'

    # File paths
    base_dir = Path(__file__).parent.parent.parent  # backend/
    matched_jsonl_path = base_dir / 'ml/data/processed/matched_query_item.jsonl'
    chromadb_dir = base_dir / 'ml/data/chromadb'
    output_dir = base_dir / 'models' / f'bge-m3-finetuned-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print("=" * 60)
    print("BGE-M3 CosineSimilarity Fine-tuning")
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

    # Prepare training pairs (query_emb, item_emb, label)
    print("\n=== Preparing training pairs ===")
    pairs = prepare_training_pairs(
        str(matched_jsonl_path),
        str(chromadb_dir),
        collection_name="items",
        n_neg_per_query=100,
        seed=42
    )
    print(f"Total training pairs: {len(pairs)}")

    # Custom Dataset
    class EmbeddingPairDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            q_emb, i_emb, label = self.pairs[idx]
            return {'features_a': q_emb, 'features_b': i_emb, 'label': label}

    train_dataset = EmbeddingPairDataset(pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # CosineSimilarityLoss
    train_loss = losses.CosineSimilarityLoss(model)

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
