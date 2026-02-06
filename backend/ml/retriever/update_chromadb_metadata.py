"""
One-time script to update ChromaDB metadata without regenerating embeddings.

This script adds new metadata fields to existing ChromaDB items:
- rating_number: Number of ratings (popularity signal)
- details: Product details as JSON string
- image: Main product image URL (MAIN variant, large)
- description: Full product description
- features: Full product features (not truncated)
- top_reviews: Top 3 verified reviews sorted by helpful_vote
- total_helpful_votes: Sum of helpful_votes for the item

Also removes deprecated fields:
- categories (all empty)
- description_summary (replaced by description)

Usage:
    python update_chromadb_metadata.py
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import chromadb


def load_raw_metadata(meta_path: str) -> dict:
    """Load raw metadata and extract new fields"""
    metadata = {}

    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading raw metadata"):
            item = json.loads(line)
            parent_asin = item.get('parent_asin')
            if not parent_asin:
                continue

            # Extract full description
            desc_list = item.get('description', [])
            description = ' '.join(str(d) for d in desc_list if d) if desc_list else ''

            # Extract full features (pipe-separated)
            feat_list = item.get('features', [])
            features = ' | '.join(str(f) for f in feat_list if f) if feat_list else ''

            # Extract details as JSON string
            details = item.get('details', {})
            details_str = json.dumps(details, ensure_ascii=False) if details else ''

            # Extract main image URL (MAIN variant, large)
            image = ''
            images = item.get('images', [])
            if images:
                for img in images:
                    if isinstance(img, dict) and img.get('variant') == 'MAIN':
                        image = img.get('large', img.get('thumb', '')) or ''
                        break
                if not image and isinstance(images[0], dict):
                    image = images[0].get('large', images[0].get('thumb', '')) or ''

            metadata[parent_asin] = {
                'rating_number': item.get('rating_number', 0) or 0,
                'details': details_str,
                'image': image,
                'description': description,
                'features': features,
            }

    return metadata


def aggregate_reviews(reviews_path: str) -> dict:
    """
    Aggregate top reviews and total helpful votes per item.

    Returns dict: parent_asin -> {top_reviews: str, total_helpful_votes: int}
    """
    # Collect all reviews per item
    item_reviews = defaultdict(list)

    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            review = json.loads(line)
            parent_asin = review.get('parent_asin') or review.get('asin')
            if not parent_asin:
                continue

            text = review.get('text', '')
            if not text or len(text.strip()) < 10:
                continue

            item_reviews[parent_asin].append({
                'text': text,
                'helpful_vote': review.get('helpful_vote', 0) or 0,
                'verified_purchase': review.get('verified_purchase', False),
            })

    # Aggregate per item
    result = {}
    for parent_asin, reviews in tqdm(item_reviews.items(), desc="Aggregating reviews"):
        # Calculate total helpful votes
        total_helpful = sum(r['helpful_vote'] for r in reviews)

        # Filter verified purchases, sort by helpful_vote, take top 3
        verified_reviews = [r for r in reviews if r['verified_purchase']]
        if not verified_reviews:
            # Fallback to all reviews if no verified ones
            verified_reviews = reviews

        sorted_reviews = sorted(verified_reviews, key=lambda x: x['helpful_vote'], reverse=True)
        top_3 = sorted_reviews[:3]

        # Join top reviews with separator
        top_reviews_text = ' ||| '.join(r['text'] for r in top_3)

        result[parent_asin] = {
            'top_reviews': top_reviews_text,
            'total_helpful_votes': total_helpful,
        }

    return result


def update_chromadb(
    chromadb_path: str,
    collection_name: str,
    raw_metadata: dict,
    review_data: dict,
    batch_size: int = 1000
):
    """Update ChromaDB metadata for all items"""

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_collection(name=collection_name)

    total_items = collection.count()
    print(f"Total items in collection: {total_items}")

    # Get all IDs
    all_results = collection.get(include=[])
    all_ids = all_results['ids']

    print(f"Updating metadata for {len(all_ids)} items...")

    # Update in batches
    updated = 0
    skipped = 0

    for i in tqdm(range(0, len(all_ids), batch_size), desc="Updating ChromaDB"):
        batch_ids = all_ids[i:i + batch_size]
        batch_metadatas = []
        valid_ids = []

        # Get existing metadata for this batch
        existing = collection.get(ids=batch_ids, include=['metadatas'])

        for j, asin in enumerate(batch_ids):
            old_meta = existing['metadatas'][j] if existing['metadatas'] else {}

            # Get new data
            meta = raw_metadata.get(asin, {})
            reviews = review_data.get(asin, {})

            # Build new metadata: 11 fields total
            new_meta = {
                # Re-ranking features (5)
                'price': float(old_meta.get('price', 0) or 0),
                'average_rating': float(old_meta.get('average_rating', 0) or 0),
                'rating_number': int(meta.get('rating_number', 0) or 0),
                'store': old_meta.get('store', ''),
                'total_helpful_votes': int(reviews.get('total_helpful_votes', 0) or 0),

                # Explanation text (5)
                'title': old_meta.get('title', ''),
                'description': meta.get('description', ''),
                'features': meta.get('features', ''),
                'top_reviews': reviews.get('top_reviews', ''),
                'details': meta.get('details', ''),

                # Display (1)
                'image': meta.get('image', ''),
            }

            # Deprecated fields removed: categories, description_summary, review_keywords

            batch_metadatas.append(new_meta)
            valid_ids.append(asin)

        if valid_ids:
            collection.update(
                ids=valid_ids,
                metadatas=batch_metadatas
            )
            updated += len(valid_ids)

    print(f"\n=== Update Complete ===")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent  # backend/ml/

    meta_path = base_dir / 'data/raw/meta_All_Beauty.jsonl'
    reviews_path = base_dir / 'data/raw/All_Beauty.jsonl'
    chromadb_path = base_dir / 'data/chromadb'
    collection_name = "beauty_products"

    print("=" * 60)
    print("ChromaDB Metadata Updater")
    print("=" * 60)
    print(f"ChromaDB path: {chromadb_path}")
    print(f"Collection: {collection_name}")
    print()

    # Check files exist
    if not meta_path.exists():
        print(f"Error: Metadata file not found: {meta_path}")
        return
    if not reviews_path.exists():
        print(f"Error: Reviews file not found: {reviews_path}")
        return
    if not chromadb_path.exists():
        print(f"Error: ChromaDB not found: {chromadb_path}")
        return

    # Step 1: Load raw metadata
    print("\n=== Step 1: Loading raw metadata ===")
    raw_metadata = load_raw_metadata(str(meta_path))
    print(f"Loaded metadata for {len(raw_metadata)} items")

    # Step 2: Aggregate reviews
    print("\n=== Step 2: Aggregating reviews ===")
    review_data = aggregate_reviews(str(reviews_path))
    print(f"Aggregated reviews for {len(review_data)} items")

    # Step 3: Update ChromaDB
    print("\n=== Step 3: Updating ChromaDB ===")
    update_chromadb(
        chromadb_path=str(chromadb_path),
        collection_name=collection_name,
        raw_metadata=raw_metadata,
        review_data=review_data
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
