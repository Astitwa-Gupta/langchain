import re
from typing import List
from rapidfuzz import fuzz

# optional imports for NLP models (load when available)
try:
    import spacy
except Exception:
    spacy = None

try:
    import pysbd
    SBD = pysbd.Segmenter(language="en", clean=False)
except Exception:
    SBD = None

# -------------------------
# Configuration / Blacklist
# -------------------------
BLACKLIST_PHRASES = [
    "Important safety information",
    "Contraindicated",
    "References",
    "Prescribing Information",
    "All rights reserved",
    # extended aggressive terms
    "unsubscribe", "privacy policy", "terms of use", "click here", "learn more",
    "sincerely", "regards", "representative", "support program", "visit",
    "full prescribing", "pi", "prescribing information", "copyright"
]
FUZZY_THRESHOLD = 90  # very strict

# Medical keyword hints (keeps paragraphs when brand appears)
MEDICAL_KEYWORDS = [
    "mg","mcg","μg","mcg/day","dose","dosage","treatment","therapy","trial","study",
    "patients","efficacy","safety","adverse","side effect","contraindication",
    "hypothyroid","hyperthyroid","tsh","levothyroxine","levothyroxine sodium",
    "treatment emergent", "placebo", "randomized", "n=", "cohort", "follow-up", "baseline"
]

# Regex for dosage and study patterns
DOSAGE_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|μg|g|ml|units)\b", flags=re.IGNORECASE)
STUDY_RE = re.compile(r"\bN\s*=\s*\d+|\b[nN]=\d+|\b\d+\s*patients\b", flags=re.IGNORECASE)

# Heuristic brand detection markers (keeps true brand mentions)
BRAND_MARKERS = [r"®", r"™"]  # trademark symbols
# You can extend with explicit brand list if you have one:
KNOWN_BRANDS = set()  # e.g. {"synthroid", "levothyroxine"} (lowercase) — optionally populate

# -------------------------
# Helpers
# -------------------------
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def contains_blacklist(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    # exact substring or fuzzy
    for phrase in BLACKLIST_PHRASES:
        if phrase.lower() in low:
            return True
        # fuzzy match
        if fuzz.token_set_ratio(phrase.lower(), low) >= FUZZY_THRESHOLD:
            return True
    return False

def remove_markdown_heading_or_bullet(line: str) -> str:
    # Remove starting heading markers and bullets; return cleaned line
    line = re.sub(r"^\s*#{1,6}\s*", "", line)
    line = re.sub(r"^\s*[-*+\u2022]\s*", "", line)  # -, *, +, bullet
    line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)  # 1. or 1)
    return line.strip()

def remove_links_and_images(text: str) -> str:
    # Remove markdown images and links, raw urls, angle-bracket urls
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)          # ![alt](url)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)           # [text](url)  -- remove anchor text too per requirement
    text = re.sub(r"<https?://[^>]+>", "", text)         # <http...>
    text = re.sub(r"https?://\S+", "", text)             # raw urls
    text = re.sub(r"www\.[^\s]+", "", text)
    return normalize_whitespace(text)

def is_link_or_image_only(text: str) -> bool:
    cleaned = remove_links_and_images(text)
    # if nothing alphanumeric remains it's link-only
    return not bool(re.search(r"[A-Za-z0-9]", cleaned))

def detect_medical_signals(text: str) -> bool:
    """
    Strong but conservative test for medical content:
     - presence of MEDICAL_KEYWORDS or dosage pattern or study pattern
     - or named entities from available NLP models (scispacy)
    """
    low = text.lower()
    # quick keyword
    for kw in MEDICAL_KEYWORDS:
        if kw in low:
            return True
    # dosage/study regex
    if DOSAGE_RE.search(text) or STUDY_RE.search(text):
        return True

    # spaCy / SciSpacy entity checks (if models available)
    if spacy is not None:
        # try SciSpacy models if installed (stronger clinical NER)
        for model_name in ("en_ner_bc5cdr_md", "en_core_sci_sm", "en_core_web_trf", "en_core_web_md", "en_core_web_sm"):
            try:
                nlp = spacy.load(model_name)
            except Exception:
                nlp = None
            if nlp:
                try:
                    doc = nlp(text)
                    if getattr(doc, "ents", None):
                        # If any entity exists, check their labels for clinical sense (very permissive)
                        if len(doc.ents) > 0:
                            return True
                except Exception:
                    pass
    return False

def detect_brand_presence(text: str) -> bool:
    """
    Heuristic brand detection:
     - presence of trademark symbols ® or ™
     - or presence of a token in KNOWN_BRANDS
     - or capitalized PROPN-like tokens (heuristic)
    """
    if any(sym in text for sym in BRAND_MARKERS):
        return True
    low = text.lower()
    for b in KNOWN_BRANDS:
        if b in low:
            return True

    # heuristic: any token that is Titlecase and not at sentence start and length>2
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    # if there are many proper-looking Titlecase words, likely brand-like; but be conservative
    if words:
        # reduce false positives: if only one Titlecase word that's a standard start-of-sentence word, skip
        # instead, if 2+ Titlecase words or trademark symbol present -> brand presence
        if len(words) >= 2:
            return True
        # else if a single Titlecase word appears with trademark-like siblings (e.g., ®) -> brand
    return False

# -------------------------
# Paragraph segmentation (robust)
# -------------------------
def smart_paragraph_split(text: str) -> List[str]:
    """
    Hybrid paragraph splitter:
      - split on blank lines first
      - split on obvious markdown structural markers
      - otherwise group short lines together to reconstruct paragraphs
    """
    if not text:
        return []

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]

    paragraphs = []
    current = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # remove heading/bullet prefix immediately (we will drop headings/bullets per user's request)
        cleaned_line = remove_markdown_heading_or_bullet(line_stripped)
        # if cleaned_line becomes empty (heading only), skip adding it
        if not cleaned_line:
            # Treat heading/bullet as paragraph boundary (do not keep)
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        # If original line looked like heading or bullet, treat it as boundary but include the cleaned content
        if re.match(r"^\s*#{1,6}\s", line_stripped) or re.match(r"^\s*[-*+\u2022]\s", line_stripped) or re.match(r"^\s*\d+[\.\)]\s", line_stripped):
            # boundary
            if current:
                paragraphs.append(" ".join(current))
                current = []
            current.append(cleaned_line)
            paragraphs.append(" ".join(current))
            current = []
            continue

        # Heuristics to decide merging vs boundary
        next_line = lines[i+1].strip() if i+1 < len(lines) else None

        # If line is short (<5 tokens) — likely broken fragment: merge
        if len(cleaned_line.split()) < 5:
            current.append(cleaned_line)
            continue

        # If current is empty: start new
        if not current:
            current.append(cleaned_line)
            continue

        # If current last ends with punctuation that suggests a paragraph end and next starts with uppercase -> boundary
        last = current[-1]
        if last.endswith(('.', '?', '!')) and cleaned_line[0].isupper():
            paragraphs.append(" ".join(current))
            current = [cleaned_line]
            continue

        # Otherwise merge
        current.append(cleaned_line)

    if current:
        paragraphs.append(" ".join(current))

    # Final normalization
    paragraphs = [normalize_whitespace(p) for p in paragraphs if p.strip()]
    return paragraphs

# -------------------------
# Master cleaner function
# -------------------------
def clean_markdown_paragraphs_hybrid(markdown_text: str) -> str:
    """
    Main entrypoint. Input: raw markdown string from BDA 'markdown' key.
    Output: cleaned text string where only whole paragraphs that pass all tests are kept.
    Paragraphs joined with double newline.
    """
    paragraphs = smart_paragraph_split(markdown_text)
    kept = []

    for para in paragraphs:
        # 1) Remove headings & pure bullets (already not preserved by splitter)
        # 2) If any blacklist phrase found -> drop paragraph
        if contains_blacklist(para):
            continue

        # 3) If paragraph is link-only or image-only -> drop
        if is_link_or_image_only(para):
            continue

        # 4) If paragraph contains brand(s): require medical signals to keep (C3)
        has_brand = detect_brand_presence(para)
        has_medical = detect_medical_signals(para)

        if has_brand and not has_medical:
            continue  # drop paragraph (brand but not medical)

        # 5) If paragraph has no brand, still require medical signals (strict mode)
        #    — this enforces that only clinically relevant paragraphs remain
        if (not has_brand) and (not has_medical):
            continue

        # 6) Final sanitation: remove links/images anchor text from kept paragraph (we keep clinical content only)
        cleaned = remove_links_and_images(para)
        cleaned = normalize_whitespace(cleaned)
        if cleaned:
            kept.append(cleaned)

    # join kept paragraphs into a single paragraph string separated by blank lines
    # If you prefer a single single-line paragraph, use " ".join(kept)
    return "\n\n".join(kept)
