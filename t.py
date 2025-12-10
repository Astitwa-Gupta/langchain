import re
from typing import Any, Dict, List
from openai import OpenAI

client = OpenAI(api_key="YOUR_ILIAD_KEY")

# -----------------------------
# RULE-BASED RELEVANT MARKDOWN FILTER
# -----------------------------
def is_relevant_markdown(text: str) -> bool:
    if not text or len(text.strip()) < 15:
        return False

    noise_patterns = [
        r"^figure\s*\d+",
        r"^table\s*\d+",
        r"^page\s*\d+",
        r"copyright",
        r"all rights reserved",
        r"references?",
        r"^toc$",
        r"^contents$",
        r"^index$",
        r"^author",
        r"http[s]?:\/\/",
        r"^image:\s*",
    ]

    for pat in noise_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return False

    return True


# ------------------------------------------------------
# REMOVE ALL UNNECESSARY MARKDOWN KEYS FROM BDA RESPONSE
# ------------------------------------------------------
def remove_unnecessary_markdown(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "markdown":
                if not is_relevant_markdown(v):
                    continue  # remove it completely
            new_obj[k] = remove_unnecessary_markdown(v)
        return new_obj

    elif isinstance(obj, list):
        return [remove_unnecessary_markdown(i) for i in obj]

    return obj


# --------------------------------------
# EXTRACT ALL REMAINING MARKDOWN STRINGS
# --------------------------------------
def collect_markdown(obj: Any) -> List[str]:
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "markdown" and is_relevant_markdown(v):
                results.append(v)
            else:
                results += collect_markdown(v)
    elif isinstance(obj, list):
        for i in obj:
            results += collect_markdown(i)
    return results


# -------------------------------
# LLM SEMANTIC KEY MESSAGE FILTER
# -------------------------------
def llm_extract_key_messages(text: str) -> str:
    prompt = f"""
You are an expert medical summarizer.

Extract ONLY key clinical messages from the text below.
Ignore: page numbers, image captions, TOC, references, legal lines, boilerplate, formatting noise.

Return one clean, concise paragraph.

TEXT:
{text}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # Or Iliad medical-tuned model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2
    )

    return resp.choices[0].message.content.strip()


# -------------------------------
# MASTER FUNCTION
# -------------------------------
def extract_clean_medical_paragraph(bda_json: Dict[str, Any]) -> str:
    # 1. Remove all unnecessary markdown keys
    cleaned_json = remove_unnecessary_markdown(bda_json)

    # 2. Collect remaining markdown blocks
    md_blocks = collect_markdown(cleaned_json)

    if not md_blocks:
        return ""

    combined_text = "\n".join(md_blocks)

    # 3. LLM filter to extract final key messages
    final_paragraph = llm_extract_key_messages(combined_text)

    return final_paragraph
