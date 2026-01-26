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
import random
import json
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

from vllm import LLM, SamplingParams


class KeywordExtractor:
    """Extract keywords and scenarios from product reviews using LLM"""

    def __init__(self, llm, sampling_params):
        self.llm = llm
        self.sampling_params = sampling_params

    def _build_prompt(self, query: str) -> str:
        """Build prompt for contextual keyword extraction with Llama 3 formatting (query only)"""
        system_content = """You are a product recommendation expert. Analyze this user query to extract contextual keywords that describe:
    - Product features and characteristics
    - Benefits and improvements provided
    - Use cases and situations where it's useful
    - Target user types and their needs

    Combine all these aspects into a single efficient list of keywords that capture WHO would use this product, WHEN, and WHY.

    Please respond with ONLY valid JSON, no markdown, no extra text:
    {
        "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    }"""
        user_content = f"Query: {query}"
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return formatted_prompt

    def extract_keywords(self, query: str) -> Optional[List[str]]:
        prompt = self._build_prompt(query)
        outputs = self.llm.generate([prompt], self.sampling_params)
        text = outputs[0].outputs[0].text
        try:
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text.strip())
            return result.get("keywords", [])
        except Exception:
            return []

    def extract_keywords_batch(self, queries: List[str]) -> List[List[str]]:
        prompts = [self._build_prompt(query) for query in queries]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for o in outputs:
            text = o.outputs[0].text
            try:
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                parsed = json.loads(text.strip())
                results.append(parsed.get("keywords", []))
            except Exception:
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

        # Load all queries first
        queries = []
        with open(input_path, "r") as inf:
            for line in inf:
                try:
                    product = json.loads(line)
                    query = product.get("generated_query", "")
                    if query and len(query.strip()) >= 5 and query != "INSUFFICIENT_INFO":
                        queries.append(query)
                        if max_items and len(queries) >= max_items:
                            break
                except json.JSONDecodeError:
                    continue

        print(f"Processing {len(queries)} queries from {input_file}")
        print(f"Results will be saved to {output_file}")
        print(f"Batch size: {batch_size}\n")

        processed = 0
        successful = 0
        failed = 0

        with open(output_path, "w") as outf:
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]

                if batch_size == 1:
                    query = batch[0]
                    print(f"[{i+1}] Processing: {query[:50]}...", end=" ", flush=True)
                    keywords = self.extract_keywords(query)
                    if keywords:
                        output_record = {
                            "query": query[:500],
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
                    print(f"Processing batch {i//batch_size + 1} ({len(batch)} items)...", end=" ", flush=True)
                    keywords_list = self.extract_keywords_batch(batch)
                    batch_successful = 0
                    for query, keywords in zip(batch, keywords_list):
                        final_keywords = keywords if keywords is not None else []
                        output_record = {
                            "query": query[:500],
                            "keywords": keywords,
                        }
                        outf.write(json.dumps(output_record) + "\n")
                        if final_keywords:
                            batch_successful += 1
                        successful += 1
                    processed += len(batch)
                    print(f"OK ({batch_successful}/{len(batch)})")
                if processed % 100 == 0:
                    print(f"   Progress: {processed}/{len(queries)} ({successful} successful)")
                        

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
            import traceback
            print("[DEBUG] main() 진입")
            try:
                cli_args = parse_args()
                print("[DEBUG] parse_args() 완료")

                # Backend configuration
                BACKEND = cli_args.get('BACKEND', 'ollama').lower()
                MODEL = cli_args.get('MODEL')
                GPU = cli_args.get('GPU')

                # Parse GPU IDs
                gpu_ids = None
                if GPU:
                    gpu_ids = [int(g.strip()) for g in GPU.split(',')]

                # File paths
                script_dir = Path(__file__).parent
                INPUT_FILE = script_dir / "generated_queries_train.jsonl"
                OUTPUT_FILE = script_dir / "keywords_output.jsonl"
                print(f"[DEBUG] INPUT_FILE: {INPUT_FILE}")

                # Processing options
                MAX_ITEMS = int(cli_args['MAX_ITEMS']) if 'MAX_ITEMS' in cli_args else None
                DELAY = float(cli_args.get('DELAY', 0.5 if BACKEND == 'ollama' else 0.0))
                BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 1 if BACKEND == 'ollama' else 64))

                print("=" * 60)
                print("LLaMA Keyword & Scenario Extractor")
                print("=" * 60)
                SAMPLED_OUTPUT_FILE = Path("/data/ephemeral/home/sampled_aspect_output.jsonl")
                print(f"Backend: {BACKEND}")
                if MODEL:
                    print(f"Model: {MODEL}")
                if gpu_ids:
                    print(f"GPUs: {gpu_ids}")
                print(f"Batch size: {BATCH_SIZE}")
                print()

                print("[DEBUG] LLM 객체 생성 시작")
                # LLM 직접 생성 (extract_aspect.py 방식)
                llm = LLM(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    dtype="float16",
                    max_model_len=4096,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=8
                )
                print("[DEBUG] LLM 객체 생성 완료")
                sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
                extractor = KeywordExtractor(llm, sampling_params)
                print("[DEBUG] KeywordExtractor 생성 완료")
                # 1. 샘플링: INSUFFICIENT_INFO가 아닌 query 10개 무작위 추출
                queries = []
                with open(INPUT_FILE, "r") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            query = obj.get("generated_query", "")
                            if query and query != "INSUFFICIENT_INFO" and len(query.strip()) > 0:
                                queries.append(query)
                        except Exception:
                            continue
                print(f"[DEBUG] 샘플링 대상 쿼리 개수: {len(queries)}")
                if not queries:
                    print("[ERROR] 샘플링할 쿼리가 없습니다. generated_queries_train.jsonl 파일을 확인하세요.")
                    return
                sampled_queries = random.sample(queries, min(10, len(queries)))
                print(f"[DEBUG] 샘플링된 쿼리 개수: {len(sampled_queries)}")
                print("\n[샘플링된 10개 query]")
                for idx, q in enumerate(sampled_queries, 1):
                    print(f"[{idx}] {q}")
                # 2. aspect 추출
                print("\n[샘플 10개 query에 대해 aspect 추출]")
                # LLM 원본 출력도 함께 출력
                prompts = [extractor._build_prompt(query) for query in sampled_queries]
                outputs = llm.generate(prompts, sampling_params)
                print("\n[LLM 원본 출력 결과]")
                for idx, (q, o) in enumerate(zip(sampled_queries, outputs), 1):
                    print(f"[{idx}] query: {q}")
                    print(f"    raw_output: {o.outputs[0].text}")
                aspects_list = extractor.extract_keywords_batch(sampled_queries)
                print("\n[LLM 파싱된 결과]")
                for idx, (q, aspects) in enumerate(zip(sampled_queries, aspects_list), 1):
                    print(f"[{idx}] query: {q}")
                    print(f"    aspects: {aspects}")
                # 3. 결과 저장
                with open(SAMPLED_OUTPUT_FILE, "w") as out_f:
                    for query, aspects in zip(sampled_queries, aspects_list):
                        out_f.write(json.dumps({"query": query, "keywords": aspects}, ensure_ascii=False) + "\n")
                print(f"\n[INFO] 샘플 10개 결과가 {SAMPLED_OUTPUT_FILE} 파일에 저장되었습니다.")
            except Exception as e:
                print("[ERROR] main()에서 예외 발생:")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()