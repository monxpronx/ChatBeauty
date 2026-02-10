import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ITEMS = [
    {
        "id": 1,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!",
        "features": "",
        "description": "If given the choice, weÕd leave most telltale signs of the beachÑsunburns, sandy toes, crab claw pinches, etc.Ñat the beach where they belong. The one thing wish we could take with us? That salty sea breeze. This all-natural spray manages, magically, to bottle the effect of sea mist so you can have it all the time. A quick spray in your hair imparts that impossibly perfect texture that comes from a swim in the ocean and a sprawl in the sun.",
        "details": "'Hair Type': 'Wavy', 'Material Type Free': 'Dairy Free', 'Scent': 'Coconut', 'Liquid Volume': '8 Fluid Ounces', 'Item Form': 'Spray', 'Is Discontinued By Manufacturer': 'No', 'Package Dimensions': '9.57 x 3.27 x 3.15 inches; 15.84 Ounces', 'UPC': '632709433010', 'Manufacturer': 'Herbivore Botanicals'",
    },
    {
        "id": 2,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "This product does what I need it to do, I just wish it was odorless or had a soft coconut smell. Having my head smell like an orange coffee is offputting. (granted, I did know the smell was described but I was hoping it would be light)",
        "features": "",
        "description": "",
        "details": "'Brand': 'Two Goats Apothecary', 'Item Form': 'Powder', 'Age Range (Description)': 'Adult', 'Unit Count': '2.0 Ounce', 'Package Dimensions': '6.6 x 4.2 x 1.5 inches; 5.61 Ounces'",
    },
    {
        "id": 3,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "Smells good, feels great!",
        "features": "Same Great Product, NEW PACKAGING., MOISTURIZED SKIN: New Road Beauty Paraffin Wax is outstanding for moisturizing dry skin, nourishing finger and toe nails and providing massage therapy. Having cosmetic and therapeutic benefits, it is often used in skin-softening treatment on the hands, cuticles, and feet. Each Paraffin Wax bar 1lb., QUALITY INGREDIENTS: Our Paraffin wax is a soft, solid wax and is made from saturated hydrocarbon. The wax is a natural emollient, helping make skin supple and soft. When applied to the skin, it adds moisture and continues to boost the moisture levels of the skin barrier after the treatment is complete. It can also help pores and remove dead skin cells, that may help make the skin look fresher and feel smoother. You can repeat wax treatments daily., HEAT THERAPY: When heat from a Paraffin Wax treatment is applied gently and evenly around the affected muscle, it will instantly release any tightness or tension that we are often unknowingly holding onto., MADE IN THE USA: New Road Beauty Paraffin Wax has been proudly manufactured in the USA for 4+ years.",
        "description": "New Road Beauty Paraffin Wax is recommended for Beauty and Medical Application. It is outstanding for moisturizing dry skin, nourishing finger and toe nails and providing massage therapy. Having cosmetic and therapeutic benefits, it is often used in skin-softening treatment on the hands, cuticles, and feet. Paraffin wax is a soft solid wax and is made from saturated hydrocarbon. The wax is a natural emollient, helping make skin supple and soft. When applied to the skin, it adds moisture and continues to boost the moisture levels of the skin after the treatment is complete. It can also help pores and remove dead skin cells, that may help make the skin look fresher and feel smoother. New Road Beauty Paraffin Wax is available in Unscented as well as seven different fragrances, including Wintergreen, Peach, Lavender, Vanilla, Coconut, Mango, and Pineapple. Each fragrance comes in a 1 lb. size. Each scent comes in a 1 pack, 3 pack, or 6 pack. Try one of our 3 or 6 combination packs that are available to you. You should not use paraffin wax if you have: Poor blood circulation Numbness in your hands or feet Any rashes or open sores If you have very sensitive skin, paraffin wax may cause heat rash. Heat rash results in small red bumps on the skin that can be itchy and uncomfortable.",
        "details": "'Package Dimensions': '10.5 x 6.4 x 1.6 inches; 2.6 Pounds', 'UPC': '695924647044', 'Manufacturer': 'New Road Beauty'",
    },
    {
        "id": 4,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "Felt synthetic",
        "features": "?Hair Bundle Material?:Brazilian Virgin Human Hair,Body Wave 3 Bundle,Ombre Black To Grey Color., ?Hair Bundle Quality?:Double Machine Weft(More Strong,Less Shedding).Full & Bouncy,Tangle & Shedding Free,No Lices And Smell, Soft And Thick,No Split Ends., ?Hair Bundle?:Bundle is Body Wave Remy Hair,100g/Bundle and Length is real enough.Generally Speaking 3 Bundle Can Be Made Into a Wig., ?Body Wave Bundles?:Can Be Straightened And Restyled as You Wish,But We recommend to Perform Perm as Little as Possible, Which Will Damage The Hair Quality., ?Delivery Time?:Same Day, Next Day And Second Day Delivery For , 30 Days No Reason To Return,Please be Careful Not to Damage The Product Packaging., Hair type: Wave,Body Wave, Unit count type: Ounce",
        "description": "Hair Material: Brazilian Virgin Human Hair Bundle Hair Weft: Machine Double Weft, Tight and Neat Hair Color: 1B/GREY Black To Grey Hair Length:8-30 inch in Stock Hair Weight:100g(+-5g)/Pcs, 300g/Pack，3 Bundles Enough For Making A Full Head Hair Quality: Ture To Length,Soft and Full,No Smell,Shedding & Tangle Free, No Dry Split Ends,Can Be Permed and Curled Delivery&Return Policy:Amazon Prime 1-3 Days Arrival，No Reason To Return Within 30 Days FQA: Q1:Can hair be permed？ A1:The Can Be Straightened And Restyled as You Wish，Please Be Careful Not to Exceed 200°F, Otherwise It Will Damage The Hair.Q2： Hair Smell? A2： Unprocessed Hair, No Chemical Smell.But We Use Shampoo and Oil To Keep The Hair Soft,Some Wavy Or Curly Hair May Have a Little Hairspray Smell,and It'Ll Disappear After Washing. Q3: Hair Tangle? A3:Hair Tangle Due To Dry. Please Put Oil/Dirt Built Up, Salt-Water, Chlorine and Not Comb Daily. Make Sure Wash And Condition With Your Hair Every Two Weeks. Use Olive Oil To Moisten Smoothly.",
        "details": "'Brand': 'muaowig', 'Material': 'Human Hair', 'Extension Length': '12 Inches', 'Hair Type': 'Wavy', 'Material Feature': 'Natural', 'Package Dimensions': '13.94 x 10.43 x 2.32 inches; 13.09 Ounces', 'UPC': '764799744377'",
    },
   
    {
        "id": 5,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "The polish was quiet thick and did not apply smoothly. I let dry overnight before adding a second coat since it was so thick",
        "features": "Light lavender pink nail color with golden shimmer, From the China Glaze Road Trip Collection, Gives long lasting manicures, Dries quickly on nails",
        "description": "China Glaze Nail Polish, Wanderlust, 1381, .50 fl. oz.Light lavender pink nail color with golden shimmer.China Glaze Road Trip Collection.",
        "details": "'Brand': 'China Glaze', 'Item Form': 'Liquid', 'Color': 'Pink', 'Finish Type': 'Shimmery', 'Special Feature': 'Long Lasting', 'Age Range (Description)': 'Adult', 'Number of Items': '1', 'Number of Pieces': '1', 'Liquid Volume': '14 Milliliters', 'Item Dimensions LxWxH': '1.38 x 1.38 x 3.35 inches', 'Is Discontinued By Manufacturer': 'No', 'Product Dimensions': '1.38 x 1.38 x 3.35 inches; 2.33 Ounces', 'Item model number': '82384', 'UPC': '019965823845', 'Manufacturer': 'American International Industries'",
    },
    {
        "id": 6,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "Love it",    
        "features": "",
        "description": "",
        "details": "'Package Dimensions': '8.5 x 3.82 x 2.24 inches; 9.14 Ounces'",	
    },
    {
        "id": 7,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "Great for many tasks. I purchased these for makeup removal. No makeup on your washcloths. Disposable, so great for travel. Soft. Absorbant.",
        "features": "",
        "description": "",
        "details": "'Package Dimensions': '8.58 x 4.37 x 3.27 inches; 8.54 Ounces'",
    },
    {
        "id": 8,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "These were lightweight and soft but much too small for my liking. I would have preferred two of these together to make one loc. For that reason I will not be repurchasing.",
        "features": "",
        "description": "",
        "details": "'Brand': 'Niseyo', 'Extension Length': '24 Inches', 'Hair Type': 'All', 'Material Feature': 'Natural', 'Product Dimensions': '15 x 8 x 3 inches; 1.57 Pounds', 'UPC': '782931053590'"
    },
    {
        "id": 9,
        "parent_asin": "",
        "average_rating": None,
        "rating_number": None,
        "title": "",
        "review": "This is perfect for my between salon visits. I have been using this now twice a week for over a month and I absolutely love it! My skin looks amazing and feels super smooth and silky. This is also super easy to use (just follow instructions). I can see already that I will begin expanding the time between visits which will definitely help me save money in the long run. Highly recommend!",
        "features": "",
        "description": "This is perfect for my between salon visits. I have been using this now twice a week for over a month and I absolutely love it! My skin looks amazing and feels super smooth and silky. This is also super easy to use (just follow instructions). I can see already that I will begin expanding the time between visits which will definitely help me save money in the long run. Highly recommend!",
        "details": "'Skin Type': 'Dry', 'Product Benefits': 'Hydration', 'Use for': 'Face', 'Scent': 'Aloe', 'Brand': 'Nira'"
    },
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None

model_name = "meta-llama/Llama-3.1-8B-Instruct"


def _load_env_file(env_path: Path) -> None:
    """Minimal .env loader (KEY=VALUE, supports quotes). Does not override existing env vars."""

    try:
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Do not override environment variables already set.
            os.environ.setdefault(key, value)
    except Exception:
        # Best-effort only; if parsing fails we fall back to env / HF cache.
        return


def _get_hf_token() -> str:
    _load_env_file(Path(__file__).with_name(".env"))
    # Convenience: also read .env.example if present (does not override existing env vars).
    _load_env_file(Path(__file__).with_name(".env.example"))

    token = os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
    except Exception:
        token = None

    if not token:
        raise SystemExit(
            "HF_TOKEN is not set and no cached Hugging Face token was found. "
            "Set it with `export HF_TOKEN=...`, or create a .env file with HF_TOKEN=..., or run `hf auth login`, then retry."
        )
    return token


def _stringify_text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            parts.append(f"{k}: {s}")
        return "; ".join(parts).strip()
    return str(value).strip()


def _jsonl_iter(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _reservoir_sample(records: Iterable[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    seen = 0
    for rec in records:
        seen += 1
        if len(sample) < k:
            sample.append(rec)
        else:
            j = rng.randrange(seen)
            if j < k:
                sample[j] = rec
    return sample


def _record_to_item(rec: dict, item_id: int) -> dict:
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}

    parent_asin = _stringify_text_field(rec.get("parent_asin"))

    average_rating = _safe_float(meta.get("average_rating"))
    rating_number = _safe_int(meta.get("rating_number"))

    review_title = _stringify_text_field(rec.get("title"))
    review_body = _stringify_text_field(rec.get("text"))

    # Use review title as item's title, and keep review body separate.
    # If body is missing, fall back to title as review text.
    review = (review_body or review_title).strip()

    features = _stringify_text_field(meta.get("features"))
    description = _stringify_text_field(meta.get("description"))
    details = _stringify_text_field(meta.get("details"))

    return {
        "id": item_id,
        "parent_asin": parent_asin,
        "average_rating": average_rating,
        "rating_number": rating_number,
        "title": review_title,
        "review": review,
        "features": features,
        "description": description,
        "details": details,
    }


def load_items_from_merged_jsonl(path: str, sample_size: int, seed: int) -> list[dict]:
    raw_records = _jsonl_iter(path)

    def eligible() -> Iterable[dict]:
        for rec in raw_records:
            meta = rec.get("meta")
            if not isinstance(meta, dict):
                continue
            # We require at least a review body or title, and at least one meta field.
            has_review = bool(_stringify_text_field(rec.get("text")) or _stringify_text_field(rec.get("title")))
            has_meta_any = any(
                _stringify_text_field(meta.get(k))
                for k in ("title", "features", "description", "details")
            )
            if has_review and has_meta_any:
                yield rec

    sampled = _reservoir_sample(eligible(), k=sample_size, seed=seed)
    return [_record_to_item(rec, i + 1) for i, rec in enumerate(sampled)]


def iter_items_from_merged_jsonl(path: str) -> Iterable[dict]:
    """Stream eligible items from merged JSONL without sampling.

    This is the only practical way to process large splits end-to-end.
    """

    raw_records = _jsonl_iter(path)

    def eligible() -> Iterable[dict]:
        for rec in raw_records:
            meta = rec.get("meta")
            if not isinstance(meta, dict):
                continue
            # We require at least a review body or title, and at least one meta field.
            has_review = bool(_stringify_text_field(rec.get("text")) or _stringify_text_field(rec.get("title")))
            has_meta_any = any(
                _stringify_text_field(meta.get(k))
                for k in ("title", "features", "description", "details")
            )
            if has_review and has_meta_any:
                yield rec

    for i, rec in enumerate(eligible(), start=1):
        yield _record_to_item(rec, i)


def build_prompt(item: dict) -> str:
    review = (item.get("review") or "").strip()
    title = _strip_overly_specific_metadata((item.get("title") or "").strip())
    features = _strip_overly_specific_metadata((item.get("features") or "").strip())
    description = _strip_overly_specific_metadata((item.get("description") or "").strip())
    details = _strip_overly_specific_metadata((item.get("details") or "").strip())

    has_any_extra = any((item.get(k) or "").strip() for k in ("features", "description", "details"))
    # Force generation when we have enough signal to produce a broad query.
    force_generate = bool(
        _infer_product_phrase(item) != "a product"
        or _extract_high_level_attributes(item)
        or _extract_general_needs_from_review(review)
        or (has_any_extra and _is_vague_emotion_only(review))
    )

    output_options = (
        "- An English recommendation-request query\n"
        if force_generate
        else "- An English recommendation-request query, OR\n- Exactly: INSUFFICIENT_INFO\n"
    )

    return f"""You generate ONE user-style product-search query.

Output MUST be exactly ONE line and be either:
{output_options}

Hard requirements for a generated query:
- It MUST start with: "My situation is"
- It MUST contain: "can you recommend"
- It MUST be one line only (no newlines)
- It MUST follow this structure (exact idea, flexible wording):
    "My situation is <situation>, and I'm looking for <product type + attributes>—can you recommend something?"

Rules:
1. Use ONLY information explicitly stated in the input texts below.
2. If the review is vague/emotional only (e.g., "Smells good, feels great!", "Love it"):
    - Treat the review as low-signal and do NOT treat it as product facts.
    - Use Features/Description/Details to identify the product type and a broad use-case.
    - If Features/Description/Details contain any concrete product facts, you MUST output a query (do NOT output INSUFFICIENT_INFO).
3. Keep the query generic and search-like:
    - Prefer: product type + 0–2 broad attributes + a common use-case.
    - Avoid overly concrete aspects like exact colors/shades, exact materials, exact sizes, exact counts, exact temperatures, or other technical specs.
    - Avoid overly specific proper nouns and identifiers: brand names, UPCs, model numbers, manufacturer names, exact dimensions, shipping/return policy.
    - Ignore warnings/contraindications/disclaimers and do NOT mention medical conditions unless the REVIEW explicitly says the user has that condition.
5. Use mostly lower-case for product terms/attributes (only capitalize the first word of the sentence).
6. The output MUST end with a single question mark (?) and contain nothing after it.
7. Do not output links, brackets, usernames, UI text, or any extra words after the query.

Mini examples (style only, do NOT copy words):
- If review is "Smells good, feels great!" but Features/Description describe a moisturizing wax for hands/feet with multiple scents → generate a query about a moisturizing wax treatment for dry hands/feet and preferred scent.

Now read the input texts and output ONE line.

Input texts (use ONLY what is explicitly written below):

Title:

{title}

Review:

{review}

Features:
{features}

Description:
{description}

Details:
{details}

Output:"""


def _compose_input_text(item: dict) -> str:
    parts: list[str] = []
    title = (item.get("title") or "").strip()
    review = (item.get("review") or "").strip()
    features = (item.get("features") or "").strip()
    description = (item.get("description") or "").strip()
    details = (item.get("details") or "").strip()

    if title:
        parts.append(f"Title: {title}")
    if review:
        parts.append(f"Review: {review}")
    if features:
        parts.append(f"Features: {features}")
    if description:
        parts.append(f"Description: {description}")
    if details:
        parts.append(f"Details: {details}")

    return "\n".join(parts).strip()


_NEGATIVE_PATTERNS = [
    r"\btoo\s+thick\b",
    r"\btoo\s+thin\b",
    r"\b(thick|thin)\b.*\b(not|didn['’]?t)\b.*\b(apply|spread)\b",
    r"\bdid\s+not\s+apply\s+smoothly\b",
    r"\bnot\s+apply\s+smoothly\b",
    r"\b(offputting|gross|weird|synthetic)\b",
    r"\b(irritated|irritating|rash|itch|burn)\b",
    r"\b(terrible|awful|bad|horrible)\b",
    r"\b(disappointed|waste\s+of\s+money)\b",
    r"\bdoesn['’]?t\s+work\b",
    r"\bdidn['’]?t\s+work\b",
]

_VAGUE_EMOTION_ONLY = [
    r"^\s*(great|nice|amazing|love\s+it|good|awesome|perfect)\s*[.!]*\s*$",
    r"^\s*smells\s+good\s*,\s*feels\s+great\s*[.!]*\s*$",
    r"^\s*felt\s+synthetic\s*[.!]*\s*$",
]


_OVERLY_SPECIFIC_KEYS = [
    "brand",
    "upc",
    "manufacturer",
    "asin",
    "item model number",
    "model number",
    "sku",
    "mpn",
    "gtin",
    "ean",
    "isbn",
    "package dimensions",
    "product dimensions",
    "date first available",
    "country of origin",
    "is discontinued",
    "department",
]


def _strip_overly_specific_metadata(text: str) -> str:
    """Remove common ID-ish metadata that users don't search for (UPC, dimensions, etc.)."""

    if not text:
        return ""

    cleaned_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        normalized_line = line
        # Many scraped "Details" are a single comma-separated key/value string.
        # Convert key/value pair separators into semicolons so we can filter fields.
        if "': '" in normalized_line and "', '" in normalized_line:
            normalized_line = normalized_line.replace("', '", "'; '")

        # Some fields are semi-colon separated (common in scraped "Details").
        segments = [seg.strip() for seg in normalized_line.split(";") if seg.strip()]
        kept_segments = []
        for seg in segments:
            seg_l = seg.lower()
            if any(k in seg_l for k in _OVERLY_SPECIFIC_KEYS):
                continue
            kept_segments.append(seg)

        if not kept_segments:
            continue

        kept_line = "; ".join(kept_segments)
        # Drop long numeric identifiers that tend to be UPC/IDs.
        kept_line = re.sub(r"\b\d{8,}\b", "", kept_line)
        kept_line = re.sub(r"\s{2,}", " ", kept_line).strip()
        if kept_line:
            cleaned_parts.append(kept_line)

    return "\n".join(cleaned_parts)


def _is_short_review(review: str, min_chars: int = 50) -> bool:
    return len(review.strip()) < min_chars


def _is_negative_review(review: str) -> bool:
    text = review.lower()
    return any(re.search(p, text) for p in _NEGATIVE_PATTERNS)


def _is_vague_emotion_only(review: str) -> bool:
    text = review.strip().lower()
    return any(re.match(p, text) for p in _VAGUE_EMOTION_ONLY)


def _has_desired_alternative(review: str) -> bool:
    """Detect whether a negative review explicitly states what the user wants instead."""

    text = (review or "").lower()
    if not text.strip():
        return False

    # Preference / desire cues
    if not re.search(r"\b(wish|prefer|would\s+rather|instead|if\s+only|i\s+want|i\s+need)\b", text):
        return False

    # A small set of common desired-attribute terms (extendable)
    desired_terms = [
        r"\bodorless\b",
        r"\bunscented\b",
        r"\blight\s+scent\b",
        r"\bsoft\s+\w+\s+smell\b",
        r"\bcoconut\b",
        r"\bfragrance[-\s]?free\b",
        r"\bnot\s+so\s+strong\b",
        r"\bmild\b",
    ]
    return any(re.search(p, text) for p in desired_terms)


def _force_generate_mode(item: dict) -> bool:
    """Whether we should force a query even if the review itself is low-signal."""

    review = (item.get("review") or "").strip()
    has_any_extra = any((item.get(k) or "").strip() for k in ("features", "description", "details"))
    # We generate even for negative reviews, as long as we can form a broad query.
    return bool(
        _infer_product_phrase(item) != "a product"
        or _extract_high_level_attributes(item)
        or _extract_general_needs_from_review(review)
        or (has_any_extra and _is_vague_emotion_only(review))
    )


def _extract_general_needs_from_review(review: str) -> list[str]:
    """Extract broad needs from the review (avoid concrete specs like exact size/color/length)."""

    text = (review or "").lower()
    needs: list[str] = []

    if re.search(r"\b(break|broke|broken|snap|snapped|tear|tore|flimsy|cheap)\b", text):
        needs.append("durable")
    if re.search(r"\b(stuck|snag|snags|pull|pulls|hurts|pain)\b", text):
        needs.append("won't snag or pull hair")
    if re.search(r"\b(smell|odor|offputting|strong)\b", text):
        needs.append("a mild scent")
    if re.search(r"\b(irritat|rash|itch|burn)\b", text):
        needs.append("gentle")
    if re.search(r"\b(dry|drying)\b", text) and "skin" in text:
        needs.append("hydrating")

    return needs[:2]


def _infer_product_phrase(item: dict) -> str:
    """Infer a generic product phrase from explicit text (no brands/IDs)."""

    blob = " ".join(
        [
            (item.get("title") or ""),
            (item.get("review") or ""),
            _strip_overly_specific_metadata((item.get("features") or "")),
            _strip_overly_specific_metadata((item.get("description") or "")),
            _strip_overly_specific_metadata((item.get("details") or "")),
        ]
    ).lower()

    # Hair accessories / tools
    if re.search(r"\b(scrunchie|scrunchies)\b", blob):
        return "hair scrunchies"
    if re.search(r"\b(elastic|rubber\s+band|hair\s+tie|hair\s+ties)\b", blob) and "hair" in blob:
        return "hair ties"
    if re.search(r"\b(headband|hair\s+band)\b", blob):
        return "a headband"

    if "paraffin" in blob and "wax" in blob:
        return "a paraffin wax treatment"
    if "nail" in blob and "polish" in blob:
        return "a nail polish"
    if "human hair" in blob and ("bundle" in blob or "weft" in blob or "extension" in blob):
        return "human hair extensions"
    if "detangl" in blob and ("brush" in blob or "comb" in blob):
        return "a detangling brush"
    if "wide tooth" in blob and "comb" in blob:
        return "a wide-tooth comb"
    if "hair" in blob and "spray" in blob:
        return "a hair spray"
    if "powder" in blob:
        return "a powder personal care product"
    return "a product"


def _extract_high_level_attributes(item: dict) -> list[str]:
    blob = " ".join(
        [
            (item.get("title") or ""),
            (item.get("review") or ""),
            _strip_overly_specific_metadata((item.get("features") or "")),
            _strip_overly_specific_metadata((item.get("description") or "")),
            _strip_overly_specific_metadata((item.get("details") or "")),
        ]
    ).lower()

    attrs: list[str] = []

    # Keep attributes broad (avoid concrete shades/materials/specs).
    if re.search(r"\bodorless\b|\bunscented\b|\bfragrance[-\s]?free\b", blob):
        attrs.append("fragrance free")
    # Only treat as skincare dryness if it explicitly mentions skin or moisturizing/hydration.
    if re.search(r"\bmoisturiz\w*\b|\bhydrat\w*\b|\bdry\s+skin\b", blob):
        attrs.append("for dry skin")
    if re.search(r"\bsensitive\s+skin\b|\bgentle\b", blob):
        attrs.append("for sensitive skin")
    if re.search(r"\bdetangl\w*\b", blob) and "hair" in blob:
        attrs.append("for detangling")
    if re.search(r"\blong\s+last\w*\b", blob):
        attrs.append("long lasting")

    return attrs[:2]


def _rule_based_query(item: dict) -> str:
    """Deterministic fallback that produces a single query line."""

    review = (item.get("review") or "").strip().lower()
    product_phrase = _infer_product_phrase(item)
    attrs = _extract_high_level_attributes(item)

    needs = _extract_general_needs_from_review(review)

    # If negative review explicitly expresses a preference, use that preference.
    if _is_negative_review(review) and _has_desired_alternative(review):
        situation = "i'm sensitive to strong scents"
        looking_for = f"{product_phrase} that is fragrance free"
        return f"My situation is {situation}, and i'm looking for {looking_for}—can you recommend something?"

    # Prefer broad needs/attributes.
    descriptors: list[str] = []
    for v in attrs:
        if v and v not in descriptors:
            descriptors.append(v)
    for v in needs:
        if v and v not in descriptors:
            descriptors.append(v)

    situation = "i'm looking to replace my current one" if _is_negative_review(review) else "i'm shopping for something similar"
    if descriptors:
        looking_for = f"{product_phrase} that is {' and '.join(descriptors)}"
    else:
        looking_for = product_phrase
    return f"My situation is {situation}, and i'm looking for {looking_for}—can you recommend something?"


def should_generate_query(item: dict) -> bool:
    """Option B: minimal hard filters. Only block obvious low-signal inputs.

    If extra fields (features/description/details) exist, we don't want to block
    just because the review itself is short.
    """

    combined = _compose_input_text(item)
    if not combined:
        return False

    # Rating-based skip rule (only when the review seems negative):
    # if rating_number < 20 AND average_rating < 4, do not generate a query.
    review = (item.get("review") or "").strip()
    if _is_negative_review(review):
        avg = _safe_float(item.get("average_rating"))
        rn = _safe_int(item.get("rating_number"))
        if rn is not None and avg is not None and rn < 20 and avg < 4:
            return False

    # If ONLY the review exists and it's short/vague, block.
    has_any_extra = any((item.get(k) or "").strip() for k in ("features", "description", "details"))

    if not has_any_extra:
        if _is_short_review(review):
            return False
        if _is_vague_emotion_only(review):
            return False

    return True


def _sanitize_query(text: str) -> str:
    q = text.strip()
    if not q:
        return "INSUFFICIENT_INFO"
    if any(bad in q for bad in ("```", "def ", "import ")):
        return "INSUFFICIENT_INFO"
    # Force single-line output
    q = q.replace("\r", " ").replace("\n", " ").strip()
    q = re.sub(r"\s+", " ", q).strip()

    # Strip URLs and javascript-ish fragments.
    q = re.sub(r"https?://\S+", "", q)
    q = re.sub(r"javascript:\S+", "", q, flags=re.IGNORECASE)
    # Remove any trailing example-like artifacts if they appear
    for marker in (
        "Review:",
        "Output:",
        "Examples:",
        "Here's your input:",
        "Here's your input",
        "ANSWER:",
        "Answer:",
        "ANSWER",
        "Bookmarklet.js",
        "Bookmark this",
        "Show activity",
        "Share",
        "edit",
        "Delete",
        "Flag",
        "Switch to mobile view",
        "Try Chrome",
        "Firefox Add-on",
    ):
        if marker in q:
            q = q.split(marker)[0].strip()

    # If the model tried to include markdown links, drop the tail after the first closing paren.
    if "](" in q and ")" in q:
        q = q.split(")")[0] + ")"

    # Extract only the first well-formed query segment.
    m = re.match(r"^(my situation is.*?\bcan\s+you\s+recommend\b.*?\?)\s*.*$", q, flags=re.IGNORECASE)
    if m:
        q = m.group(1).strip()
    else:
        # If it's a single query but forgot the final '?', trim noise and add it.
        if q.lower().startswith("my situation is") and re.search(r"\bcan\s+you\s+recommend\b", q, flags=re.IGNORECASE):
            q = re.sub(r"\s+", " ", q).strip()
            # Hard cap to avoid accidental long spillover.
            q = q[:240].rstrip(" .,-—–")
            if not q.endswith("?"):
                q = q + "?"
    # If it looks like it cut off mid-phrase, don't emit it.
    if q.lower().endswith((" for", " that", " and", " with", " to")):
        return "INSUFFICIENT_INFO"
    if len(q) < 10:
        return "INSUFFICIENT_INFO"
    # Must be English (reject Hangul) and must ask for recommendations.
    if re.search(r"[\u3131-\u318E\uAC00-\uD7A3]", q):
        return "INSUFFICIENT_INFO"
    if not re.search(r"[A-Za-z]", q):
        return "INSUFFICIENT_INFO"
    if not q.lower().startswith("my situation is"):
        return "INSUFFICIENT_INFO"
    if not re.search(r"\bcan\s+you\s+recommend\b", q, flags=re.IGNORECASE):
        return "INSUFFICIENT_INFO"
    if not re.search(
        r"\band\s+(?:i['’]?m|im|i\s+am)\s+looking\s+for\b",
        q,
        flags=re.IGNORECASE,
    ):
        return "INSUFFICIENT_INFO"
    if not q.endswith("?"):
        return "INSUFFICIENT_INFO"

    # Ensure exactly one query: trim after the first '?'.
    if q.count("?") > 1:
        q = q.split("?", 1)[0].strip() + "?"
    return q


def _is_query_consistent_with_item(item: dict, query: str) -> bool:
    """Conservative guardrail: ensure generated query aligns with inferred product type."""

    q = (query or "").lower()
    if not q or q == "insufficient_info":
        return False

    # Hard reject obvious out-of-scope supplement-like outputs.
    if re.search(r"\b(supplement|vitamin|capsule|capsules|gummy|gummies|pill|pills)\b", q):
        return False

    product_phrase = _infer_product_phrase(item).lower()
    if product_phrase == "a product":
        return True

    # Minimal keyword checks by product phrase.
    if "extensions" in product_phrase and not re.search(r"\b(extension|extensions)\b", q):
        return False
    if "hair ties" in product_phrase and not re.search(r"\b(hair\s+tie|hair\s+ties|tie|ties|ponytail\s+holder)\b", q):
        return False
    if "detangling brush" in product_phrase and not re.search(r"\b(brush|comb|detangl)\b", q):
        return False
    if "headband" in product_phrase and "headband" not in q:
        return False

    return True

def generate_query(item: dict):
    global tokenizer
    global model

    if not should_generate_query(item):
        return "INSUFFICIENT_INFO"

    prompt = build_prompt(item)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    raw = text.split("Output:")[-1].strip()
    # Keep the full segment; sanitizer will collapse newlines and extract the first valid query.
    raw = raw.strip() if raw else ""
    cleaned = _sanitize_query(raw)
    if cleaned != "INSUFFICIENT_INFO":
        # If the model produced a syntactically valid query but it's off-category
        # (e.g., wrong product type), prefer deterministic fallback.
        if _force_generate_mode(item) and not _is_query_consistent_with_item(item, cleaned):
            forced = _sanitize_query(_rule_based_query(item))
            if forced != "INSUFFICIENT_INFO":
                return forced
        return cleaned

    # If we are in forced-generation mode, fall back to deterministic query.
    if _force_generate_mode(item):
        forced = _sanitize_query(_rule_based_query(item))
        if forced != "INSUFFICIENT_INFO":
            return forced

    # Second-pass: if the model produced something close but not in the exact template,
    # ask it to rewrite into the required format using ONLY explicit info.
    title_text = (item.get("title") or "").strip()
    review_text = (item.get("review") or "").strip()
    features_text = _strip_overly_specific_metadata((item.get("features") or "").strip())
    description_text = _strip_overly_specific_metadata((item.get("description") or "").strip())
    details_text = _strip_overly_specific_metadata((item.get("details") or "").strip())
    if raw and re.search(r"[A-Za-z]", raw) and len(raw) > 20:
        rewrite_prompt = (
            "Rewrite the query into EXACTLY this template and one line only:\n"
            "My situation is <situation>, and I'm looking for <product type + attributes>—can you recommend something?\n\n"
            "Rules:\n"
            "- Use ONLY information explicitly present in the input texts below. Do NOT add new attributes.\n"
            "- Avoid overly specific proper nouns and IDs (brand names, UPCs, model numbers, manufacturer names, exact dimensions).\n"
            "- Keep it generic and search-like (product type + broad attributes + use-case).\n"
            "- Avoid overly concrete aspects like exact colors/shades, exact materials, exact sizes, exact counts, exact temperatures, or other technical specs.\n\n"
            "- Use mostly lower-case for product terms/attributes (only capitalize the first word).\n\n"
            "- Ignore warnings/contraindications/disclaimers and do NOT mention medical conditions unless the REVIEW explicitly says the user has that condition.\n\n"
            f"Title: {title_text}\n"
            f"Review: {review_text}\n"
            f"Features: {features_text}\n"
            f"Description: {description_text}\n"
            f"Details: {details_text}\n\n"
            f"Bad query: {raw}\n\n"
            "Rewritten query:" 
        )
        rewrite_inputs = tokenizer(rewrite_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            rewrite_out = model.generate(
                **rewrite_inputs,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        rewrite_text = tokenizer.decode(rewrite_out[0], skip_special_tokens=True)
        rewritten = rewrite_text.split("Rewritten query:")[-1].strip()
        return _sanitize_query(rewritten)

    return "INSUFFICIENT_INFO"


tokenizer: AutoTokenizer
model: AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="",
        help="Merged JSONL to sample from (e.g., All_Beauty_inner_merged.jsonl).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="How many items to sample from input_jsonl. Set to 0 to use the full dataset (no sampling).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on how many items to process (0 = no limit). Useful for debugging.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N processed items.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("generated_queries.json")),
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    write_jsonl = out_path.suffix.lower() == ".jsonl"

    if args.input_jsonl:
        if args.sample_size and args.sample_size > 0:
            items: Iterable[dict] = load_items_from_merged_jsonl(
                args.input_jsonl, sample_size=args.sample_size, seed=args.seed
            )
            if not items:
                raise SystemExit(f"No eligible records sampled from {args.input_jsonl}.")
            print(f"📦 Loaded {len(list(items))} sampled items from {args.input_jsonl}", flush=True)
            # Recreate iterable after materializing length.
            items = load_items_from_merged_jsonl(args.input_jsonl, sample_size=args.sample_size, seed=args.seed)
        else:
            items = iter_items_from_merged_jsonl(args.input_jsonl)
            mode = "full (no sampling)"
            print(f"📦 Loaded items from {args.input_jsonl} in {mode} mode", flush=True)
            if not write_jsonl:
                print(
                    "[WARN] You are processing a full dataset but output is .json. "
                    "This may use a lot of memory. Consider using --out ...jsonl",
                    flush=True,
                )
    else:
        items = list(DEFAULT_ITEMS)
        print(f"📦 Using DEFAULT_ITEMS ({len(items)} items)", flush=True)

    token = _get_hf_token()

    global tokenizer
    global model
    print("🔄 토크나이저 로딩 중...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    print("✅ 토크나이저 로딩 완료", flush=True)
    print("🔄 모델 로딩 중... (시간이 걸릴 수 있습니다)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    print("✅ 모델 로딩 완료", flush=True)
    model.eval()

    results: list[dict] = []
    seen_ids: set[int] = set()
    processed = 0

    out_f = None
    if args.input_jsonl and write_jsonl:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    try:
        for item in items:
            if args.limit and processed >= args.limit:
                break

            processed += 1
            item_id = item.get("id")
            if item_id in seen_ids:
                print(f"[WARN] Duplicate id {item_id} detected; skipping duplicate.", flush=True)
                continue
            seen_ids.add(item_id)

            title_text = (item.get("title") or "").strip()
            review_text = (item.get("review") or "").strip()
            parent_asin_text = (item.get("parent_asin") or "").strip()
            average_rating = item.get("average_rating")
            rating_number = item.get("rating_number")

            print(f"\n[{item_id}] 쿼리 생성 중...", flush=True)
            q = generate_query(item)
            if parent_asin_text:
                print(f"[{item_id}] parent_asin: {parent_asin_text}", flush=True)
            if average_rating is not None or rating_number is not None:
                print(f"[{item_id}] avg_rating: {average_rating} | rating_number: {rating_number}", flush=True)
            if title_text:
                print(f"[{item_id}] Title: {title_text}", flush=True)
            print(f"[{item_id}] Review: {review_text}", flush=True)
            print(f"    → Query: {q}\n", flush=True)

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
                "generated_query": q,
            }

            if out_f is not None:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if processed % max(1, int(args.progress_every)) == 0:
                    out_f.flush()
            else:
                results.append(row)

            if processed % max(1, int(args.progress_every)) == 0:
                print(f"[progress] processed={processed}", flush=True)

    finally:
        if out_f is not None:
            out_f.close()

    if out_f is None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        total_written = len(results)
    else:
        total_written = processed

    print(f"\n✅ 생성된 쿼리가 '{args.out}' 파일에 저장되었습니다.", flush=True)
    print(f"   총 {total_written}개의 쿼리가 생성되었습니다.", flush=True)


if __name__ == "__main__":
    main()
