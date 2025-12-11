import re
import json
from typing import Any, Dict, List, Tuple
from rapidfuzz import fuzz

# ---------------------------
# Configuration / Blacklist
# ---------------------------

# Blacklist phrases (case-insensitive). If any of these appear (very-strict fuzzy match)
# inside a markdown section, the whole markdown key will be removed.
BLACKLIST_PHRASES = [
    "Important safety information",
    "Contraindicated",
    "References",
    "Prescribing Information",
    "All rights reserved"
]

# fuzzy threshold: "very strict" as requested (90-100)
FUZZY_THRESHOLD = 90


# ---------------------------
# Helpers - normalization
# ---------------------------
def normalize_for_matching(s: str) -> str:
    """Normalize text for fuzzy matching: lower-case, remove punctuation, collapse spaces."""
    s = s.lower()
    # replace punctuation with spaces
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------
# Link/Markdown removal helpers
# ---------------------------
# Patterns for markdown links/images and raw URLs
MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\((?:\s*<?(.*?)>?\s*)\)", flags=re.IGNORECASE)  # [text](url)
MD_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\((?:\s*<?(.*?)>?\s*)\)", flags=re.IGNORECASE)  # ![alt](url)
RAW_URL_PATTERN = re.compile(
    r"""(?xi)
    \b
    (?:http|https)://[^\s<>"'()]+
    """,
)
ANGLE_LINK_PATTERN = re.compile(r"<(https?://[^>]+)>", flags=re.IGNORECASE)  # <http://...>
HTML_LINK_PATTERN = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', flags=re.IGNORECASE)


def remove_links_and_images(text: str) -> str:
    """
    Remove markdown links, images, raw URLs, and simple HTML anchor tags.
    Per your instruction: remove anchor text also (i.e., remove the whole [text](url) and keep nothing).
    For anchor-like HTML we also remove the label.
    """
    if not text:
        return text

    # Remove markdown images entirely: ![alt](url) -> remove whole
    text = MD_IMAGE_PATTERN.sub("", text)

    # Remove markdown links entirely (remove anchor text as requested)
    text = MD_LINK_PATTERN.sub("", text)

    # Remove HTML anchors including their displayed text
    text = HTML_LINK_PATTERN.sub("", text)

    # Remove angle bracket links <http...>
    text = ANGLE_LINK_PATTERN.sub("", text)

    # Remove raw URLs
    text = RAW_URL_PATTERN.sub("", text)

    # remove leftover stray parentheses or extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# Content checks
# ---------------------------
def contains_blacklist_phrase(text: str) -> bool:
    """
    Check whether the text contains any blacklist phrase using fuzzy matching.
    Returns True if any phrase matches with score >= FUZZY_THRESHOLD.
    We check phrase-by-phrase using token_set_ratio which is robust to word order.
    """
    if not text:
        return False

    norm_text = normalize_for_matching(text)

    for phrase in BLACKLIST_PHRASES:
        norm_phrase = normalize_for_matching(phrase)
        # Use token_set_ratio for phrase match
        score = fuzz.token_set_ratio(norm_phrase, norm_text)
        if score >= FUZZY_THRESHOLD:
            return True
    return False


def is_link_or_image_only(text: str) -> bool:
    """
    Determine whether the paragraph contains only images/links and no other meaningful text.
    Strategy:
      - Remove markdown link/image constructs and raw URLs using patterns.
      - After removal, if the remaining text is empty or extremely short (<= 3 chars),
        consider it link-only.
    """
    if text is None:
        return True

    # If there's nothing but whitespace
    if not text.strip():
        return True

    # Remove links/images but preserve anchor text? (We are removing anchor text as requested)
    cleaned = remove_links_and_images(text)

    # If after removing links we have no textual characters (letters/digits), it's link-only
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return True

    # If remaining length is tiny (e.g., remnants like punctuation), treat as link-only
    if len(cleaned.strip()) <= 3:
        return True

    return False


# ---------------------------
# JSON traversal utilities
# ---------------------------
def extract_all_markdown(data: Any) -> List[str]:
    """Recursively collect all markdown strings from the JSON."""
    results: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "markdown" and isinstance(v, str):
                results.append(v)
            else:
                results.extend(extract_all_markdown(v))
    elif isinstance(data, list):
        for item in data:
            results.extend(extract_all_markdown(item))
    return results


def process_and_clean_markdown_inplace(data: Any) -> List[str]:
    """
    Traverse JSON and:
      - If a "markdown" key is present, decide to remove it or replace with cleaned text.
    Returns the list of cleaned markdown strings that will be kept (for final paragraph).
    Mutates the input 'data' dict/list: removes markdown keys or replaces values.
    """
    kept_texts: List[str] = []

    if isinstance(data, dict):
        keys = list(data.keys())  # copy to allow mutation
        for k in keys:
            v = data.get(k)
            if k == "markdown" and isinstance(v, str):
                # 1. If contains blacklist phrase -> delete this key
                if contains_blacklist_phrase(v):
                    del data[k]
                    continue

                # 2. If link/image-only -> delete key
                if is_link_or_image_only(v):
                    del data[k]
                    continue

                # 3. Else, remove links/images (including anchor text) and keep the cleaned text
                cleaned = remove_links_and_images(v).strip()

                # If cleaned becomes empty after link removal -> delete key
                if not cleaned:
                    del data[k]
                    continue

                # Replace original markdown with cleaned text (or remove key? user earlier wanted markdown removed,
                # but they also wanted the cleaned text preserved. Here we REPLACE markdown value with cleaned text.)
                # If you want to remove markdown key entirely but store text elsewhere, change logic.
                data[k] = cleaned
                kept_texts.append(cleaned)

            else:
                # recurse
                kept_texts.extend(process_and_clean_markdown_inplace(v))

    elif isinstance(data, list):
        # iterate and process in place
        for idx, item in enumerate(data):
            kept_texts.extend(process_and_clean_markdown_inplace(item))

    return kept_texts


# ---------------------------
# Master function
# ---------------------------
def clean_bda_markdown_json(bda_json: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Process a BDA JSON object:
      - Remove or clean markdown keys as per rules
      - Return (combined_paragraph_string, cleaned_json)
    Behavior summary:
      - Blacklist fuzzy match -> remove markdown key entirely
      - Link/image-only -> remove markdown key entirely
      - Else -> remove links/images (including anchor text) and keep the cleaned text (replace markdown value)
    """
    # Work on a deep copy if you do not want to mutate original. For performance, mutate in place:
    import copy
    data_copy = copy.deepcopy(bda_json)

    # Process in-place and gather kept cleaned markdown strings
    kept_list = process_and_clean_markdown_inplace(data_copy)

    # Combine into one paragraph. Normalize spaces and collapse multiple sentences into single paragraph.
    if kept_list:
        paragraph = " ".join(p.strip() for p in kept_list if p and p.strip())
        # final cleanups
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
    else:
        paragraph = ""

    return paragraph, data_copy


# ---------------------------
# Usage example
# ---------------------------
if __name__ == "__main__":
    # Example BDA json snippet
    sample = {
        "document": {
            "pages": [
                {
                    "blocks": [
                        {"markdown": "Important Safety Information: this drug causes X."},
                        {"markdown": "This is a clinical note. Dose: 10 mg daily. Click here [Full PI](https://pi.example.com)"},
                        {"markdown": "[Click here](https://marketing)"},
                        {"markdown": "sincerely, Synthroid representative"},
                        {"markdown": "Visit https://example.com for more info"},
                        {"markdown": "Study A: showed benefit in 70% of patients."}
                    ]
                }
            ]
        }
    }

    para, cleaned = clean_bda_markdown_json(sample)
    print("FINAL PARAGRAPH:")
    print(para)
    print("\nCLEANED JSON MARKDOWN KEYS:")
    print(json.dumps(cleaned, indent=2))
