import re
import spacy
from rapidfuzz import fuzz

# Load spaCy once (fast after initial load)
nlp = spacy.load("en_core_web_sm")

BLACKLIST = [
    "important safety information",
    "contraindicated",
    "references",
    "prescribing information",
    "all rights reserved"
]

def contains_blacklist(text):
    """Return True if text contains any strict or fuzzy blacklisted term."""
    t = text.lower()
    for term in BLACKLIST:
        if term in t:
            return True
        if fuzz.partial_ratio(term, t) > 85:
            return True
    return False


def remove_links(text):
    """Remove markdown links and raw URLs, including anchor text."""
    # Remove markdown anchor-style links: [text](url)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # Remove naked URLs
    text = re.sub(r"https?://\S+", "", text)

    return text.strip()


def heuristic_paragraph_split(text):
    """
    Hybrid markdown-aware paragraph extraction.
    - Separate headings (#)
    - Separate bullets/lists
    - Otherwise accumulate lines.
    """
    lines = text.split("\n")
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()

        # Markdown headings → standalone paragraph
        if re.match(r"^#{1,6}\s", stripped):
            if current:
                paragraphs.append("\n".join(current))
                current = []
            paragraphs.append(stripped)
            continue

        # Bullets or numbered lists → own paragraph
        if (
            re.match(r"^[-*+]\s", stripped) or
            re.match(r"^\d+[\.\)]\s", stripped)
        ):
            if current:
                paragraphs.append("\n".join(current))
                current = []
            paragraphs.append(stripped)
            continue

        # Normal text
        if stripped:
            current.append(stripped)

    if current:
        paragraphs.append("\n".join(current))

    return paragraphs


def semantic_sentences(text):
    """Split paragraph into spaCy sentences."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def is_meaningful(text):
    """
    Optional: Check if text contains meaningful content.
    - Has nouns/verbs
    - Not too short
    """
    if not text or len(text) < 3:
        return False

    doc = nlp(text)
    for token in doc:
        if token.pos_ in ("NOUN", "VERB", "PROPN"):
            return True

    return False


def clean_markdown_text(markdown_text: str) -> str:
    """Final function: input raw markdown string → output cleaned markdown string."""
    
    paragraphs = heuristic_paragraph_split(markdown_text)
    cleaned_paragraphs = []

    for para in paragraphs:

        # Step 1 — Remove blacklist at paragraph level
        if contains_blacklist(para):
            continue

        # Step 2 — Sentence-level processing
        sentences = semantic_sentences(para)
        cleaned_sentences = []

        for sent in sentences:
            sent = remove_links(sent)

            if not sent:
                continue

            if contains_blacklist(sent):
                continue

            if is_meaningful(sent):
                cleaned_sentences.append(sent)

        cleaned_para = " ".join(cleaned_sentences).strip()

        if cleaned_para:
            cleaned_paragraphs.append(cleaned_para)

    # Join paragraphs with blank line (markdown friendly)
    return "\n\n".join(cleaned_paragraphs)
