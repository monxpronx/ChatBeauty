"""
Extract keywords and use scenarios from All_Beauty.jsonl using LLaMA

Supports both Ollama and vLLM backends.

Usage:
    # Ollama (default)
    python extract_keywords_with_llama.py

    # vLLM (auto GPU detection)
    python extract_keywords_with_llama.py BACKEND=vllm

    # With options
    python extract_keywords_with_llama.py BACKEND=vllm MAX_ITEMS=100 DELAY=0.3
    python extract_keywords_with_llama.py BACKEND=vllm MODEL=meta-llama/Llama-3.1-8B-Instruct
    python extract_keywords_with_llama.py BACKEND=vllm GPU=0,1
"""

import json
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.utils.llm_client import LLMClient


class KeywordExtractor:
    """Extract keywords and scenarios from product reviews using LLM"""

    def __init__(self, client: LLMClient):
        self.client = client

    def _build_prompt(self, product_title: str, review_text: str) -> str:
        """Build prompt for contextual keyword extraction with Llama 3 formatting"""
        
        system_content = """You are a product recommendation expert. Analyze this product review to extract contextual keywords that describe:
- Product features and characteristics
- Benefits and improvements provided
- Use cases and situations where it's useful
- Target user types and their needs

Combine all these aspects into a single efficient list of keywords that capture WHO would use this product, WHEN, and WHY.

Please respond with ONLY valid JSON, no markdown, no extra text:
{
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}"""

        user_content = f"""Product: {product_title}
Review: {review_text}"""

        # Llama 3.1 Chat Template
        # This is what vLLM needs to see to act as an assistant
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return formatted_prompt

    def extract_keywords(self, product_title: str, review_text: str) -> Optional[List[str]]:
        """Extract contextual keywords from a single review"""
        prompt = self._build_prompt(product_title, review_text)
        result = self.client.generate_json(prompt)

        if result:
            return result.get("keywords", [])
        return None

    def extract_keywords_batch(
        self, items: List[Dict]
    ) -> List[List[str]]:
        """Extract keywords from multiple reviews (batched for vLLM efficiency)"""
        prompts = [
            self._build_prompt(item["title"], item["text"])
            for item in items
        ]

        responses = self.client.generate_batch(prompts)
        print(f"DEBUG RAW OUTPUT: {responses[0]}")

        results = []
        for response in responses:
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
                    results.append(parsed.get("keywords", []))
                except json.JSONDecodeError:
                    results.append([])
            else:
                results.append([])

        return results

    def process_jsonl_file(
        self,
        input_file: str,
        output_file: str,
        max_items: Optional[int] = None,
        delay: float = 0.5,
        batch_size: int = 1
    ) -> None:
        """Process all products in a JSONL file and extract keywords"""
        input_path = Path(input_file).resolve()
        output_path = Path(output_file).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return

        # Load all items first
        items_to_process = []
        with open(input_path, "r") as inf:
            for line in inf:
                try:
                    product = json.loads(line)
                    text = product.get("text", "")
                    if text and len(text.strip()) >= 10:
                        items_to_process.append(product)
                        if max_items and len(items_to_process) >= max_items:
                            break
                except json.JSONDecodeError:
                    continue

        print(f"Processing {len(items_to_process)} items from {input_file}")
        print(f"Results will be saved to {output_file}")
        print(f"Batch size: {batch_size}\n")

        processed = 0
        successful = 0
        failed = 0

        with open(output_path, "w") as outf:
            # Process in batches
            for i in range(0, len(items_to_process), batch_size):
                batch = items_to_process[i:i + batch_size]

                if batch_size == 1:
                    # Single item processing
                    product = batch[0]
                    title = product.get("title", "Unknown Product")
                    print(f"[{i+1}] Processing: {title[:50]}...", end=" ", flush=True)

                    keywords = self.extract_keywords(title, product.get("text", ""))

                    if keywords:
                        output_record = {
                            "asin": product.get("asin", ""),
                            "title": title,
                            "review_text": product.get("text", "")[:500],
                            "rating": product.get("rating"),
                            "keywords": keywords,
                        }
                        outf.write(json.dumps(output_record) + "\n")
                        print("OK")
                        successful += 1
                    else:
                        print("FAIL")
                        failed += 1

                    processed += 1
                    time.sleep(delay)

                else:
                    # Batch processing (for vLLM efficiency)
                    print(f"Processing batch {i//batch_size + 1} ({len(batch)} items)...", end=" ", flush=True)

                    batch_items = [
                        {"title": p.get("title", "Unknown"), "text": p.get("text", "")}
                        for p in batch
                    ]
                    keywords_list = self.extract_keywords_batch(batch_items)

                    batch_successful = 0
                    for j, (product, keywords) in enumerate(zip(batch, keywords_list)):
                        
                        # Ensure keywords is a list
                        final_keywords = keywords if keywords is not None else []
                        
                        # Always write, regardless of whether list is empty
                        output_record = {
                            "asin": product.get("asin", ""),
                            "title": product.get("title", ""),
                            "review_text": product.get("text", "")[:500],
                            "rating": product.get("rating"),
                            "keywords": keywords,
                        }
                        outf.write(json.dumps(output_record) + "\n")

                        # Count success if we got any keywords
                        if final_keywords:
                            batch_successful += 1
                        
                        # track "empty" vs "error" separately
                        successful += 1

                    processed += len(batch)
                    print(f"OK ({batch_successful}/{len(batch)})")

                # Progress update
                if processed % 100 == 0:
                    print(f"   Progress: {processed}/{len(items_to_process)} ({successful} successful)")

        print(f"\n" + "=" * 60)
        print(f"Processing Complete!")
        print(f"  Total processed: {processed}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Results saved to: {output_path}")
        print("=" * 60)


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
        python extract_keywords_with_llama.py                          # Ollama (default)
        python extract_keywords_with_llama.py BACKEND=vllm             # vLLM with auto GPU
        python extract_keywords_with_llama.py BACKEND=vllm GPU=0,1     # vLLM with specific GPUs
        python extract_keywords_with_llama.py MAX_ITEMS=100            # Limit items
        python extract_keywords_with_llama.py BATCH_SIZE=32            # Batch size for vLLM
    """
    cli_args = parse_args()

    # Backend configuration
    BACKEND = cli_args.get('BACKEND', 'ollama').lower()
    MODEL = cli_args.get('MODEL')
    GPU = cli_args.get('GPU')

    # Parse GPU IDs
    gpu_ids = None
    if GPU:
        gpu_ids = [int(g.strip()) for g in GPU.split(',')]

    # File paths
    script_dir = Path(__file__).parent.parent.parent  # backend/
    INPUT_FILE = script_dir / "data/raw/All_Beauty.jsonl"
    OUTPUT_FILE = script_dir / "data/processed/keywords_output.jsonl"

    # Processing options
    MAX_ITEMS = int(cli_args['MAX_ITEMS']) if 'MAX_ITEMS' in cli_args else None
    DELAY = float(cli_args.get('DELAY', 0.5 if BACKEND == 'ollama' else 0.0))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 1 if BACKEND == 'ollama' else 64))

    print("=" * 60)
    print("LLaMA Keyword & Scenario Extractor")
    print("=" * 60)
    print(f"Backend: {BACKEND}")
    if MODEL:
        print(f"Model: {MODEL}")
    if gpu_ids:
        print(f"GPUs: {gpu_ids}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Initialize LLM client
    client_kwargs = {"backend": BACKEND}
    if MODEL:
        client_kwargs["model"] = MODEL
    if gpu_ids:
        client_kwargs["gpu_ids"] = gpu_ids

    client = LLMClient(**client_kwargs)

    if not client.verify_connection():
        print("Failed to connect to LLM backend. Exiting.")
        return

    # Initialize extractor and process
    extractor = KeywordExtractor(client)
    extractor.process_jsonl_file(
        input_file=str(INPUT_FILE),
        output_file=str(OUTPUT_FILE),
        max_items=MAX_ITEMS,
        delay=DELAY,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    main()
