import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from vllm import LLM, SamplingParams

import sys
sys.path.insert(0, str(Path(__file__).parent))
import generate_query as gq


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _skip_iter(it: Iterable[dict], n: int) -> Iterable[dict]:
    # Consume n items from iterator.
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            return iter(())
    return it


def _make_llm(model_name: str, hf_token: str, gpu_memory_utilization: float, max_model_len: int) -> LLM:
    # vLLM reads token from env; set both common names.
    os.environ.setdefault("HF_TOKEN", hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    # V100 does not support bf16; use fp16.
    return LLM(
        model=model_name,
        dtype="half",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=int(max_model_len),
        trust_remote_code=False,
        enforce_eager=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--out", required=True, help="Output .jsonl path")
    p.add_argument("--resume", action="store_true", help="Resume by appending and skipping already written lines")
    p.add_argument("--limit", type=int, default=0, help="Optional cap for debugging (0 = no limit)")
    p.add_argument("--progress_every", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32, help="How many prompts to send to vLLM per batch")
    p.add_argument("--max_new_tokens", type=int, default=140)
    p.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Override max sequence length for vLLM (smaller = much lower KV cache memory).",
    )
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".jsonl":
        raise SystemExit("--out must end with .jsonl for streaming writes")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build iterator (same eligibility logic as Generate_query.py)
    it = gq.iter_items_from_merged_jsonl(args.input_jsonl)

    already = 0
    mode = "w"
    if args.resume:
        already = _count_lines(out_path)
        mode = "a"
        if already:
            it = iter(it)
            it = _skip_iter(it, already)

    hf_token = gq._get_hf_token()
    print(f"📦 input={args.input_jsonl}")
    print(f"📝 out={out_path} (mode={mode})")
    print(f"↩️  resume={args.resume} already_written={already}")

    print("🔄 vLLM 엔진 로딩 중...", flush=True)
    llm = _make_llm(
        gq.model_name,
        hf_token,
        gpu_memory_utilization=float(args.gpu_mem_util),
        max_model_len=int(args.max_model_len),
    )
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None
    print("✅ vLLM 엔진 로딩 완료", flush=True)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_new_tokens),
    )

    processed = already
    written = 0

    truncated_prompts = 0

    pending_items: list[dict] = []
    pending_prompts: list[str] = []

    def _prompt_token_len(text: str) -> int:
        if tokenizer is None:
            # Best-effort fallback based on character count.
            return max(1, len(text) // 4)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _build_prompt_fit(item: dict) -> str:
        """Build a prompt that fits within max_model_len.

        vLLM validates that prompt_len <= max_model_len; it may also require
        prompt_len + max_new_tokens <= max_model_len. We enforce the latter.
        """

        nonlocal truncated_prompts

        max_prompt_tokens = int(args.max_model_len) - int(args.max_new_tokens) - 8
        # Safety floor: avoid negative/too-small budgets.
        max_prompt_tokens = max(256, max_prompt_tokens)

        # Fast path.
        prompt = gq.build_prompt(item)
        if _prompt_token_len(prompt) <= max_prompt_tokens:
            return prompt

        # Slow path: iteratively truncate long fields (usually details/description).
        item2 = dict(item)
        fields_order = ["details", "description", "features", "review", "title"]

        changed = False
        for _ in range(40):
            prompt = gq.build_prompt(item2)
            if _prompt_token_len(prompt) <= max_prompt_tokens:
                break

            shrunk_any = False
            for k in fields_order:
                s = (item2.get(k) or "")
                if not isinstance(s, str):
                    s = str(s)
                s = s.strip()

                # Keep at least some content; if already short, skip.
                if len(s) <= 240:
                    continue

                new_len = max(240, int(len(s) * 0.7))
                if new_len >= len(s):
                    continue
                item2[k] = s[:new_len]
                shrunk_any = True
                changed = True
                break

            if not shrunk_any:
                break

        prompt = gq.build_prompt(item2)
        if _prompt_token_len(prompt) > max_prompt_tokens:
            # Last resort: drop verbose metadata entirely, keep a short title/review.
            item3 = dict(item2)
            for k in ("details", "description", "features"):
                item3[k] = ""
            r = (item3.get("review") or "")
            t = (item3.get("title") or "")
            item3["review"] = (r[:1200]).strip()
            item3["title"] = (t[:240]).strip()
            prompt = gq.build_prompt(item3)
            changed = True

        # Ultra-last resort: hard-truncate tokens (keeps header; may cut fields).
        if tokenizer is not None and _prompt_token_len(prompt) > max_prompt_tokens:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids = ids[:max_prompt_tokens]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
            # Ensure there's an output cue (tiny append; should still fit).
            if "Output:" not in prompt[-120:]:
                prompt = prompt.rstrip() + "\n\nOutput:"
            changed = True

        if changed:
            truncated_prompts += 1
            if truncated_prompts <= 5 or (truncated_prompts % 200 == 0):
                print(
                    f"[warn] truncated long prompt to fit context (count={truncated_prompts})",
                    flush=True,
                )
        return prompt

    def flush_batch(f):
        nonlocal pending_items, pending_prompts, processed, written
        if not pending_items:
            return

        outputs = llm.generate(pending_prompts, sampling)
        if len(outputs) != len(pending_items):
            raise RuntimeError("vLLM output count mismatch")

        for item, out in zip(pending_items, outputs):
            item_id = item.get("id")
            parent_asin_text = (item.get("parent_asin") or "").strip()
            average_rating = item.get("average_rating")
            rating_number = item.get("rating_number")
            title_text = (item.get("title") or "").strip()
            review_text = (item.get("review") or "").strip()

            raw = ""
            if out.outputs:
                raw = (out.outputs[0].text or "").strip()

            cleaned = gq._sanitize_query(raw)
            if cleaned != "INSUFFICIENT_INFO":
                if gq._force_generate_mode(item) and not gq._is_query_consistent_with_item(item, cleaned):
                    forced = gq._sanitize_query(gq._rule_based_query(item))
                    if forced != "INSUFFICIENT_INFO":
                        cleaned = forced
            else:
                if gq._force_generate_mode(item):
                    forced = gq._sanitize_query(gq._rule_based_query(item))
                    if forced != "INSUFFICIENT_INFO":
                        cleaned = forced

            row = {
                "id": item_id,
                "parent_asin": parent_asin_text,
                "average_rating": average_rating,
                "rating_number": rating_number,
                "title": title_text,
                "review": review_text,
                "features": (item.get("features") or "").strip(),
                "description": (item.get("description") or "").strip(),
                "details": (item.get("details") or "").strip(),
                "generated_query": cleaned,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            processed += 1

            if processed % max(1, int(args.progress_every)) == 0:
                f.flush()
                print(f"[progress] processed={processed} (+{written} this run)", flush=True)

        pending_items = []
        pending_prompts = []

    with out_path.open(mode, encoding="utf-8") as f:
        for item in it:
            if args.limit and (processed - already) >= args.limit:
                break

            if not gq.should_generate_query(item):
                # Still write a row, but don't send to vLLM.
                row = {
                    "id": item.get("id"),
                    "parent_asin": (item.get("parent_asin") or "").strip(),
                    "average_rating": item.get("average_rating"),
                    "rating_number": item.get("rating_number"),
                    "title": (item.get("title") or "").strip(),
                    "review": (item.get("review") or "").strip(),
                    "features": (item.get("features") or "").strip(),
                    "description": (item.get("description") or "").strip(),
                    "details": (item.get("details") or "").strip(),
                    "generated_query": "INSUFFICIENT_INFO",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed += 1
                written += 1
                if processed % max(1, int(args.progress_every)) == 0:
                    f.flush()
                    print(f"[progress] processed={processed} (+{written} this run)", flush=True)
                continue

            pending_items.append(item)
            pending_prompts.append(_build_prompt_fit(item))

            if len(pending_items) >= max(1, int(args.batch_size)):
                flush_batch(f)

        flush_batch(f)
        f.flush()

    print(f"✅ done: processed={processed} written_this_run={written}")


if __name__ == "__main__":
    main()
