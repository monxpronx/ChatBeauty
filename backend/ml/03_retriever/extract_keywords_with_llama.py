"""
Extract keywords and use scenarios from All_Beauty.jsonl using LLaMA 3.1 8B via Ollama

This script connects to Ollama running locally and uses LLaMA 3.1 8B model to:
1. Extract relevant keywords from product reviews
2. Identify use cases and scenarios where the product is useful
3. Save results to a structured output file

Requirements:
    - Ollama installed and running locally (default: http://localhost:11434)
    - LLaMA 3.1 8B model pulled: ollama pull llama2:7b or ollama pull llama3.1:8b
    - Python packages: requests, jsonl (or just use standard json and jsonlines)
"""

import json
import requests
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path


class LLaMAKeywordExtractor:
    """Extract keywords and scenarios from product reviews using LLaMA via Ollama"""
    
    def __init__(
        self, 
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.1:8b",
        temperature: float = 0.3,
        top_p: float = 0.9
    ):
        """
        Initialize the extractor
        
        Args:
            ollama_url: URL where Ollama is running (default: localhost:11434)
            model_name: Model to use. Common options:
                - llama3.1:8b (recommended for balance of speed/quality)
                - llama2:7b (faster, less accurate)
                - mistral:7b (faster, good for keywords)
            temperature: Model creativity (0-2). Lower = more focused/repetitive
            top_p: Diversity parameter (0-1). Lower = more focused responses
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.api_endpoint = f"{self.ollama_url}/api/generate"
        
        # Verify Ollama is running
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if not any(self.model_name in name for name in model_names):
                    print(f"⚠️  Model '{self.model_name}' not found in Ollama.")
                    print(f"Available models: {model_names}")
                    print(f"To download: ollama pull {self.model_name}")
                    return False
                
                print(f"✓ Connected to Ollama at {self.ollama_url}")
                print(f"✓ Model '{self.model_name}' is available")
                return True
            else:
                print(f"❌ Failed to connect to Ollama at {self.ollama_url}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot reach Ollama at {self.ollama_url}")
            print("Make sure Ollama is running: ollama serve")
            return False
        except Exception as e:
            print(f"❌ Error verifying connection: {e}")
            return False
    
    def _build_prompt(self, product_title: str, review_text: str) -> str:
        """
        Build a well-structured prompt for contextual keyword extraction
        
        The prompt guides LLaMA to:
        1. Understand the use situations and target user from the review
        2. Generate contextual keywords that combine features, benefits, scenarios, and user types
        3. Return a single comprehensive list of efficient, meaningful keywords
        
        Args:
            product_title: Title/name of the product
            review_text: Review text from the customer
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a product recommendation expert. Analyze this product review to extract contextual keywords that describe:
- Product features and characteristics
- Benefits and improvements provided
- Use cases and situations where it's useful
- Target user types and their needs

Combine all these aspects into a single efficient list of keywords that capture WHO would use this product, WHEN, and WHY.

Product: {product_title}

Review: {review_text}

Please respond with ONLY valid JSON, no markdown, no extra text:
{{
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}"""
        return prompt
    
    def extract_keywords(self, product_title: str, review_text: str) -> Optional[List[str]]:
        """
        Extract contextual keywords from a single review
        
        Args:
            product_title: Title of the product
            review_text: Review text to analyze
            
        Returns:
            List of contextual keywords, or None if failed
        """
        prompt = self._build_prompt(product_title, review_text)
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Try to parse JSON response
                try:
                    # Clean up response if needed
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.startswith("```"):
                        response_text = response_text[3:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    
                    extracted = json.loads(response_text)
                    # Return just the keywords list
                    return extracted.get("keywords", [])
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse JSON response: {response_text[:100]}")
                    return None
            else:
                print(f"❌ API error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏱️  Request timeout - model may be processing large data")
            return None
        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            return None
    
    def process_jsonl_file(
        self,
        input_file: str,
        output_file: str,
        max_items: Optional[int] = None,
        delay: float = 0.5
    ) -> None:
        """
        Process all products in a JSONL file and extract keywords
        
        Args:
            input_file: Path to input JSONL file (All_Beauty.jsonl)
            output_file: Path to output JSONL file for results
            max_items: Maximum number of items to process (None = all)
            delay: Delay between requests in seconds (default 0.5 to avoid overload)
        """
        input_path = Path(input_file).resolve()
        output_path = Path(output_file).resolve()
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            return
        
        processed = 0
        successful = 0
        failed = 0
        
        print(f"📂 Processing {input_file}...")
        print(f"💾 Results will be saved to {output_file}\n")
        
        with open(output_path, "w") as outf:
            with open(input_path, "r") as inf:
                for line_num, line in enumerate(inf, 1):
                    if max_items and processed >= max_items:
                        print(f"\n✓ Reached max items limit ({max_items})")
                        break
                    
                    try:
                        product = json.loads(line)
                        title = product.get("title", "Unknown Product")
                        text = product.get("text", "")
                        asin = product.get("asin", "")
                        
                        # Skip if no text
                        if not text or len(text.strip()) < 10:
                            continue
                        
                        # Extract keywords
                        print(f"[{line_num}] Processing: {title[:50]}...", end=" ", flush=True)
                        extracted = self.extract_keywords(title, text)
                        
                        if extracted:
                            # Combine original product data with extracted keywords
                            output_record = {
                                "asin": asin,
                                "title": title,
                                "review_text": text[:500],  # Store first 500 chars
                                "rating": product.get("rating"),
                                "keywords": extracted,  # List of contextual keywords
                            }
                            outf.write(json.dumps(output_record) + "\n")
                            print("✓")
                            successful += 1
                        else:
                            print("✗")
                            failed += 1
                        
                        processed += 1
                        
                        # Rate limiting to avoid overwhelming Ollama
                        if processed % 10 == 0:
                            print(f"   Progress: {processed} items processed ({successful} successful)")
                        
                        time.sleep(delay)
                        
                    except json.JSONDecodeError:
                        print("✗ (Invalid JSON)")
                        failed += 1
                    except Exception as e:
                        print(f"✗ ({str(e)[:30]})")
                        failed += 1
        
        print(f"\n" + "="*60)
        print(f"✓ Processing Complete!")
        print(f"  Total processed: {processed}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Results saved to: {output_path}")
        print("="*60)


def main():
    """Main execution function"""
    
    # Configuration
    OLLAMA_URL = "http://localhost:11434"  # Change if Ollama runs elsewhere
    MODEL = "llama3.1:8b"                  # or "llama2:7b", "mistral:7b"
    
    # Use absolute paths based on script location
    script_dir = Path(__file__).parent.parent.parent   # Go up to backend/
    
    INPUT_FILE = script_dir / "data/raw/All_Beauty.jsonl"
    OUTPUT_FILE = script_dir / "data/processed/keywords_output.jsonl"

    MAX_ITEMS = 50  # Set to None to process all items (takes longer)
    DELAY = 0.5     # Delay between requests in seconds
    
    print("="*60)
    print("🚀 LLaMA Keyword & Scenario Extractor")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print()
    
    # Initialize extractor
    extractor = LLaMAKeywordExtractor(
        ollama_url=OLLAMA_URL,
        model_name=MODEL,
        temperature=0.3,  # Lower = more consistent, less creative
        top_p=0.9
    )
    
    # Process file
    extractor.process_jsonl_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        max_items=MAX_ITEMS,
        delay=DELAY
    )


if __name__ == "__main__":
    main()
