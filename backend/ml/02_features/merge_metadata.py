"""
Merge item metadata with extracted keywords for creating item embeddings.

Preprocessing:
- Title: Raw (as-is)
- Features: Raw (as-is)
- Review Keywords: Already extracted keywords
- Description: Summarized using LLaMA

Supports both Ollama and vLLM backends.

Usage:
    # Ollama (default)
    python merge_metadata.py

    # vLLM (auto GPU detection)
    python merge_metadata.py BACKEND=vllm

    # Skip description summarization
    python merge_metadata.py USE_LLM=false

Output format:
[Title] Coleman Sundome Tent [Review Keywords] Heavy Rain Protection, Easy Setup [Description Summary] Family Camping, Comfortable [Features] Waterproof 3000mm, Weight: 2kg, Polyester
"""

import json
import time
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.utils.llm_client import LLMClient


class DescriptionSummarizer:
    """Summarize product descriptions using LLM"""

    def __init__(self, client: LLMClient):
        self.client = client

    def _build_prompt(self, description: str) -> str:
        """Build prompt for description summarization"""
        return f"""Summarize this product description into 2-5 short keywords or phrases that capture the main use cases and benefits. Focus on WHO would use it and WHAT situations it's good for.

Description: {description}

Respond with ONLY valid JSON, no markdown:
{{"summary": ["keyword1", "keyword2", "keyword3"]}}"""

    def summarize(self, description: str) -> Optional[List[str]]:
        """Summarize a single description"""
        if not description or len(description.strip()) < 10:
            return None

        result = self.client.generate_json(self._build_prompt(description))
        if result:
            return result.get("summary", [])
        return None

    def summarize_batch(self, descriptions: List[str]) -> List[Optional[List[str]]]:
        """Summarize multiple descriptions (batched for vLLM efficiency)"""
        # Filter valid descriptions
        valid_indices = []
        prompts = []
        for i, desc in enumerate(descriptions):
            if desc and len(desc.strip()) >= 10:
                valid_indices.append(i)
                prompts.append(self._build_prompt(desc))

        if not prompts:
            return [None] * len(descriptions)

        # Generate responses
        responses = self.client.generate_batch(prompts)

        # Map results back
        results = [None] * len(descriptions)
        for idx, response in zip(valid_indices, responses):
            if response:
                try:
                    text = response
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    parsed = json.loads(text.strip())
                    results[idx] = parsed.get("summary", [])
                except json.JSONDecodeError:
                    pass

        return results


def load_metadata(meta_path: str) -> Dict:
    """Load metadata and create lookup by parent_asin"""
    metadata = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading metadata"):
            item = json.loads(line)
            parent_asin = item.get('parent_asin')
            if parent_asin:
                metadata[parent_asin] = {
                    'title': item.get('title', ''),
                    'features': item.get('features', []),
                    'description': item.get('description', []),
                    # Additional metadata for ChromaDB
                    'price': item.get('price'),
                    'average_rating': item.get('average_rating'),
                    'store': item.get('store', ''),
                    'categories': item.get('categories', []),
                    'main_category': item.get('main_category', ''),
                }
    return metadata


def format_description(description: list) -> str:
    """Convert description list to single string"""
    if not description:
        return ''
    return ' '.join(description)


def create_embedding_text(
    title: str,
    review_keywords: List[str],
    description_summary: List[str],
    features: List[str]
) -> str:
    """
    Create final embedding text in format:
    [Title] X [Review Keywords] Y [Description Summary] Z [Features] W
    """
    parts = []

    if title:
        parts.append(f"[Title] {title}")

    if review_keywords:
        parts.append(f"[Review Keywords] {', '.join(review_keywords)}")

    if description_summary:
        parts.append(f"[Description Summary] {', '.join(description_summary)}")

    if features:
        parts.append(f"[Features] {', '.join(features)}")

    return ' '.join(parts)


def process_descriptions_with_llm(
    keywords_path: str,
    metadata: Dict,
    cache_path: str,
    client: LLMClient,
    batch_size: int = 1,
    delay: float = 0.3
) -> Dict[str, List[str]]:
    """
    Summarize descriptions for items using LLM and cache results.
    Returns dict: asin -> summary keywords
    """
    # Load which items have keywords
    keywords_asins = set()
    with open(keywords_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            keywords_asins.add(item.get('asin'))

    print(f"Found {len(keywords_asins)} items with keywords")

    cache_file = Path(cache_path)

    # Load existing cache
    summaries = {}
    if cache_file.exists():
        print(f"Loading cached summaries from {cache_path}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                summaries[item['asin']] = item['summary']
        print(f"Loaded {len(summaries)} cached summaries")

    # Find items needing processing
    to_process = []
    for asin in keywords_asins:
        if asin not in summaries:
            meta = metadata.get(asin, {})
            desc = format_description(meta.get('description', []))
            if desc and len(desc.strip()) > 10:
                to_process.append((asin, desc))

    if not to_process:
        print("All descriptions already summarized (cached)")
        return summaries

    print(f"Need to summarize {len(to_process)} descriptions")

    summarizer = DescriptionSummarizer(client)

    # Process and cache
    with open(cache_file, 'a', encoding='utf-8') as f:
        if batch_size == 1:
            # Single item processing
            for asin, desc in tqdm(to_process, desc="Summarizing"):
                summary = summarizer.summarize(desc)
                if summary:
                    summaries[asin] = summary
                    f.write(json.dumps({'asin': asin, 'summary': summary}) + '\n')
                time.sleep(delay)
        else:
            # Batch processing
            for i in tqdm(range(0, len(to_process), batch_size), desc="Summarizing batches"):
                batch = to_process[i:i + batch_size]
                asins = [item[0] for item in batch]
                descs = [item[1] for item in batch]

                results = summarizer.summarize_batch(descs)

                for asin, summary in zip(asins, results):
                    if summary:
                        summaries[asin] = summary
                        f.write(json.dumps({'asin': asin, 'summary': summary}) + '\n')

    return summaries


def merge_all(
    keywords_path: str,
    meta_path: str,
    output_path: str,
    summary_cache_path: str,
    client: Optional[LLMClient] = None,
    batch_size: int = 1,
    delay: float = 0.3
):
    """Main merge function"""

    # Step 1: Load metadata
    print(f"\n=== Step 1: Loading metadata ===")
    metadata = load_metadata(meta_path)
    print(f"Loaded {len(metadata)} items")

    # Step 2: Summarize descriptions with LLM
    print(f"\n=== Step 2: Summarizing descriptions ===")
    if client:
        description_summaries = process_descriptions_with_llm(
            keywords_path, metadata, summary_cache_path, client, batch_size, delay
        )
    else:
        print("Skipping LLM summarization (no client provided)")
        description_summaries = {}

    # Step 3: Merge with keywords
    print(f"\n=== Step 3: Merging with keywords ===")
    matched = 0
    unmatched = 0

    with open(keywords_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Merging data"):
            item = json.loads(line)
            asin = item.get('asin')

            meta = metadata.get(asin, {})
            if meta:
                matched += 1
            else:
                unmatched += 1

            title = meta.get('title', '')
            review_keywords = item.get('keywords', [])
            description_summary = description_summaries.get(asin, [])
            features = meta.get('features', [])

            embedding_text = create_embedding_text(
                title=title,
                review_keywords=review_keywords,
                description_summary=description_summary,
                features=features
            )

            output_item = {
                'asin': asin,
                'title': title,
                'review_keywords': review_keywords,
                'description_summary': description_summary,
                'features': features,
                'embedding_text': embedding_text,
                # Additional metadata for ChromaDB
                'price': meta.get('price'),
                'average_rating': meta.get('average_rating'),
                'store': meta.get('store', ''),
                'categories': meta.get('categories', []),
                'main_category': meta.get('main_category', ''),
            }

            f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"\n=== Merge Complete ===")
    print(f"  Matched with metadata: {matched}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Output saved to: {output_path}")


def parse_args() -> Dict[str, str]:
    """Parse KEY=VALUE arguments from command line"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def main():
    """
    Main execution function

    Usage:
        python merge_metadata.py                          # Ollama (default)
        python merge_metadata.py BACKEND=vllm             # vLLM with auto GPU
        python merge_metadata.py BACKEND=vllm GPU=0,1     # vLLM with specific GPUs
        python merge_metadata.py USE_LLM=false            # Skip description summarization
        python merge_metadata.py BATCH_SIZE=32            # Batch size for vLLM
    """
    cli_args = parse_args()

    # Backend configuration
    BACKEND = cli_args.get('BACKEND', 'ollama').lower()
    MODEL = cli_args.get('MODEL')
    GPU = cli_args.get('GPU')
    USE_LLM = cli_args.get('USE_LLM', 'true').lower() != 'false'

    # Parse GPU IDs
    gpu_ids = None
    if GPU:
        gpu_ids = [int(g.strip()) for g in GPU.split(',')]

    # File paths
    base_dir = Path(__file__).parent.parent.parent  # backend/

    keywords_path = base_dir / 'data/processed/keywords_output.jsonl'
    meta_path = base_dir / 'data/raw/meta_All_Beauty.jsonl'
    output_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    summary_cache_path = base_dir / 'data/processed/description_summaries_cache.jsonl'

    # Processing options
    DELAY = float(cli_args.get('DELAY', 0.3 if BACKEND == 'ollama' else 0.0))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 1 if BACKEND == 'ollama' else 64))

    print("=" * 60)
    print("Metadata Merger + Description Summarizer")
    print("=" * 60)
    print(f"Backend: {BACKEND}")
    print(f"Use LLM: {USE_LLM}")
    if MODEL:
        print(f"Model: {MODEL}")
    if gpu_ids:
        print(f"GPUs: {gpu_ids}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Initialize LLM client if needed
    client = None
    if USE_LLM:
        client_kwargs = {"backend": BACKEND}
        if MODEL:
            client_kwargs["model"] = MODEL
        if gpu_ids:
            client_kwargs["gpu_ids"] = gpu_ids

        client = LLMClient(**client_kwargs)

        if not client.verify_connection():
            print("Failed to connect to LLM backend. Proceeding without description summarization.")
            client = None

    merge_all(
        keywords_path=str(keywords_path),
        meta_path=str(meta_path),
        output_path=str(output_path),
        summary_cache_path=str(summary_cache_path),
        client=client,
        batch_size=BATCH_SIZE,
        delay=DELAY
    )


if __name__ == '__main__':
    main()
