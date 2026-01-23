"""
Merge item metadata with extracted keywords for creating item embeddings.

Preprocessing:
- Title: Raw (as-is)
- Features: Raw (as-is)
- Review Keywords: Already extracted keywords
- Description: Summarized using Llama 3.1:8B

Output format:
[Title] Coleman Sundome Tent [Review Keywords] Heavy Rain Protection, Easy Setup [Description Summary] Family Camping, Comfortable [Features] Waterproof 3000mm, Weight: 2kg, Polyester
"""

import json
import requests
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List


class DescriptionSummarizer:
    """Summarize product descriptions using Llama via Ollama"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.1:8b",
        temperature: float = 0.3,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.api_endpoint = f"{self.ollama_url}/api/generate"

    def _verify_connection(self) -> bool:
        """Verify Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if any(self.model_name in name for name in model_names):
                    print(f"Connected to Ollama, model '{self.model_name}' available")
                    return True
                print(f"Model '{self.model_name}' not found. Available: {model_names}")
                return False
            return False
        except Exception as e:
            print(f"Cannot connect to Ollama: {e}")
            return False

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

        prompt = self._build_prompt(description)

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()

                # Clean up markdown if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                extracted = json.loads(response_text)
                return extracted.get("summary", [])
            return None

        except Exception:
            return None


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
                }
    return metadata


def format_features(features: list) -> str:
    """Convert features list to comma-separated string"""
    if not features:
        return ''
    return ', '.join(features)


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

    # Title (raw)
    if title:
        parts.append(f"[Title] {title}")

    # Review Keywords
    if review_keywords:
        parts.append(f"[Review Keywords] {', '.join(review_keywords)}")

    # Description Summary (from Llama)
    if description_summary:
        parts.append(f"[Description Summary] {', '.join(description_summary)}")

    # Features (raw)
    if features:
        parts.append(f"[Features] {', '.join(features)}")

    return ' '.join(parts)


def process_descriptions_with_llama(
    keywords_path: str,
    metadata: Dict,
    cache_path: str,
    delay: float = 0.3
) -> Dict[str, List[str]]:
    """
    Summarize descriptions for items that have keywords using Llama and cache results.
    Returns dict: asin -> summary keywords
    """
    # First, load which items have keywords
    keywords_asins = set()
    with open(keywords_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            keywords_asins.add(item.get('asin'))
    
    print(f"Found {len(keywords_asins)} items with keywords")
    
    cache_file = Path(cache_path)

    # Load existing cache if available
    summaries = {}
    if cache_file.exists():
        print(f"Loading cached summaries from {cache_path}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                summaries[item['asin']] = item['summary']
        print(f"Loaded {len(summaries)} cached summaries")

    # Find items that need processing (only those with keywords)
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

    print(f"Need to summarize {len(to_process)} descriptions with Llama")

    summarizer = DescriptionSummarizer()
    if not summarizer._verify_connection():
        print("Skipping Llama summarization - Ollama not available")
        return summaries

    # Process and append to cache
    with open(cache_file, 'a', encoding='utf-8') as f:
        for asin, desc in tqdm(to_process, desc="Summarizing descriptions"):
            summary = summarizer.summarize(desc)
            if summary:
                summaries[asin] = summary
                f.write(json.dumps({'asin': asin, 'summary': summary}) + '\n')
            time.sleep(delay)

    return summaries


def merge_all(
    keywords_path: str,
    meta_path: str,
    output_path: str,
    summary_cache_path: str,
    use_llama: bool = True
):
    """Main merge function"""

    # Step 1: Load metadata
    print(f"\n=== Step 1: Loading metadata ===")
    metadata = load_metadata(meta_path)
    print(f"Loaded {len(metadata)} items")

    # Step 2: Summarize descriptions with Llama (with caching)
    print(f"\n=== Step 2: Summarizing descriptions ===")
    if use_llama:
        description_summaries = process_descriptions_with_llama(
            keywords_path, metadata, summary_cache_path
        )
    else:
        print("Skipping Llama summarization (use_llama=False)")
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

            # Get components
            title = meta.get('title', '')
            review_keywords = item.get('keywords', [])
            description_summary = description_summaries.get(asin, [])
            features = meta.get('features', [])

            # Create embedding text
            embedding_text = create_embedding_text(
                title=title,
                review_keywords=review_keywords,
                description_summary=description_summary,
                features=features
            )

            # Output record
            output_item = {
                'asin': asin,
                'title': title,
                'review_keywords': review_keywords,
                'description_summary': description_summary,
                'features': features,
                'embedding_text': embedding_text
            }

            f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"\n=== Merge Complete ===")
    print(f"  Matched with metadata: {matched}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Output saved to: {output_path}")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent.parent

    keywords_path = base_dir / 'data/processed/keywords_output.jsonl'
    meta_path = base_dir / 'data/raw/meta_All_Beauty.jsonl'
    output_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    summary_cache_path = base_dir / 'data/processed/description_summaries_cache.jsonl'
    merge_all(
        keywords_path=str(keywords_path),
        meta_path=str(meta_path),
        output_path=str(output_path),
        summary_cache_path=str(summary_cache_path),
        use_llama=True  # Set to False to skip Llama and just merge existing data
    )
