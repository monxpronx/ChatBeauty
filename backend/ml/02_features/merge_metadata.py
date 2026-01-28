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
        """Build prompt for description summarization using Llama 3.1 Template"""
        
        system_content = """You are a product content editor. 
Summarize the provided product description into a concise list of 2-5 keywords or short phrases.
Focus on:
- Main use cases (Who uses it?)
- Key benefits (Why use it?)
- Situations (When to use it?)

Respond with ONLY valid JSON, no markdown, no conversational text.
Example format: {"summary": ["feature 1", "benefit 2", "use case 3"]}"""

        user_content = f"""Description: {description}"""

        # Llama 3.1 Chat Template (Critical for vLLM)
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def summarize(self, description: str) -> Optional[List[str]]:
        """Summarize a single description"""
        if not description or len(description.strip()) < 10:
            return None

        # 단일 처리도 batch 로직을 재사용하는 것이 안전함
        return self.summarize_batch([description])[0]

    def summarize_batch(self, descriptions: List[str]) -> List[Optional[List[str]]]:
        """Summarize multiple descriptions (batched for vLLM efficiency)"""
        import re

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

        # --- DEBUG LOGGING (Check if model works) ---
        if responses and len(responses) > 0:
            print(f"\n[DEBUG] Raw Output Example: {str(responses[0])[:100]}...")
        # --------------------------------------------

        # Map results back
        results = [None] * len(descriptions)

        for idx, response in zip(valid_indices, responses):
            if response:
                try:
                    text = response.strip()
                    # --- Smart Parsing: 텍스트 안에서 JSON 객체만 찾아내기 ---
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    
                    if json_match:
                        clean_json = json_match.group(0)
                        parsed = json.loads(clean_json)
                        results[idx] = parsed.get("summary", [])
                    else:
                        # 디버깅: JSON을 못 찾았을 때 로그 출력
                        # print(f"[DEBUG] Parsing Failed. Raw: {text[:100]}...")
                        results[idx] = []
                        
                except (json.JSONDecodeError, AttributeError):
                    results[idx] = []
            else:
                results[idx] = []

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


def aggregate_keywords_by_item(keywords_path: str) -> Dict[str, List[str]]:
    """
    Aggregate all review keywords per item (asin).

    Since multiple reviews exist per item, we need to merge all keywords
    into a single list per item for creating one embedding vector per item.

    Returns:
        Dict mapping asin -> list of unique keywords (deduplicated, ordered by frequency)
    """
    from collections import Counter

    item_keywords: Dict[str, Counter] = {}

    with open(keywords_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Aggregating keywords by item"):
            item = json.loads(line)
            asin = item.get('asin')
            keywords = item.get('keywords', [])

            if asin not in item_keywords:
                item_keywords[asin] = Counter()

            # Count keyword occurrences across reviews
            for kw in keywords:
                if kw is not None and kw != '':  # skip empty/None values
                    item_keywords[asin][str(kw).strip()] += 1

    # Convert to sorted list (most frequent first, then alphabetically)
    aggregated = {}
    for asin, kw_counter in item_keywords.items():
        # Sort by frequency (desc), then alphabetically for ties
        sorted_keywords = sorted(kw_counter.keys(), key=lambda k: (-kw_counter[k], k))
        aggregated[asin] = sorted_keywords

    print(f"Aggregated keywords for {len(aggregated)} unique items")
    return aggregated


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
        # 리스트 안에 숫자가 있어도 강제로 문자열로 변환 (str(k))
        cleaned_keywords = [str(k) for k in review_keywords if k is not None]
        parts.append(f"[Review Keywords] {', '.join(cleaned_keywords)}")

    if description_summary:
        # 문자열 변환 추가
        cleaned_summary = [str(s) for s in description_summary if s is not None]
        parts.append(f"[Description Summary] {', '.join(cleaned_summary)}")

    if features:
        # 문자열 변환 추가
        cleaned_features = [str(f) for f in features if f is not None]
        parts.append(f"[Features] {', '.join(cleaned_features)}")

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

    # Step 3: Aggregate keywords from all reviews per item
    print(f"\n=== Step 3: Aggregating review keywords per item ===")
    aggregated_keywords = aggregate_keywords_by_item(keywords_path)

    # Step 4: Merge all data and create embedding text per item
    # Iterate over ALL metadata items (not just those with keywords)
    # so items without train-split reviews still get embedded via title/features/description
    print(f"\n=== Step 4: Creating item embeddings ===")
    with_keywords = 0
    without_keywords = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for asin in tqdm(metadata.keys(), desc="Merging data"):
            meta = metadata[asin]

            title = meta.get('title', '')
            review_keywords = aggregated_keywords.get(asin, [])
            description_summary = description_summaries.get(asin, [])
            features = meta.get('features', [])

            if review_keywords:
                with_keywords += 1
            else:
                without_keywords += 1

            embedding_text = create_embedding_text(
                title=title,
                review_keywords=review_keywords,
                description_summary=description_summary,
                features=features
            )

            # Skip items with no usable text at all
            if not embedding_text.strip():
                continue

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
    print(f"  With review keywords: {with_keywords}")
    print(f"  Without review keywords (metadata only): {without_keywords}")
    print(f"  Total: {with_keywords + without_keywords}")
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
        python merge_metadata.py BACKEND=vllm GPU_MEM=0.5 # vLLM with 50% GPU memory
        python merge_metadata.py USE_LLM=false            # Skip description summarization
        python merge_metadata.py BATCH_SIZE=32            # Batch size for vLLM
    """
    cli_args = parse_args()

    # Backend configuration
    BACKEND = cli_args.get('BACKEND', 'ollama').lower()
    MODEL = cli_args.get('MODEL')
    GPU = cli_args.get('GPU')
    GPU_MEM = cli_args.get('GPU_MEM')  # GPU memory utilization (0.0-1.0)
    USE_LLM = cli_args.get('USE_LLM', 'true').lower() != 'false'

    # Parse GPU IDs
    gpu_ids = None
    if GPU:
        gpu_ids = [int(g.strip()) for g in GPU.split(',')]

    # Parse GPU memory utilization
    gpu_memory_utilization = 0.8  # default
    if GPU_MEM:
        gpu_memory_utilization = float(GPU_MEM)

    # File paths
    base_dir = Path(__file__).parent.parent.parent  # backend/

    keywords_path = base_dir / 'data/processed/keywords_train.jsonl'
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
    if BACKEND == 'vllm':
        print(f"GPU memory utilization: {gpu_memory_utilization}")
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
        if BACKEND == 'vllm':
            client_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

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
