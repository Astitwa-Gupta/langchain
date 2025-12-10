def is_relevant_markdown(text: str) -> bool:
    """
    Returns True only if markdown contains clinically relevant content.
    Filters out marketing, disclaimers, signatures, links, and footer noise.
    """

    if not text or len(text.strip()) < 10:
        return False

    text_lower = text.lower().strip()

    # -------------------------
    # REMOVE MARKETING / PROMO
    # -------------------------
    marketing_phrases = [
        "click here",
        "learn more",
        "prescribing information",
        "full prescribing",
        "visit",
        "support program",
        "more information",
        "order now",
        "contact us",
        "for more info",
    ]

    # -------------------------
    # EMAIL SIGNATURE / REP
    # -------------------------
    signature_phrases = [
        "sincerely",
        "regards",
        "representative",
        "best wishes",
        "thank you",
        "team",
    ]

    # -------------------------
    # DISCLAIMERS / FOOTERS
    # -------------------------
    footer_phrases = [
        "this email has been sent",
        "we respect your privacy",
        "if you no longer wish",
        "unsubscribe",
        "email preferences",
        "privacy policy",
        "terms of use",
    ]

    # -------------------------
    # LEGAL / COPYRIGHT
    # -------------------------
    legal_phrases = [
        "all rights reserved",
        "copyright",
        "©",
        "™",
        "®",
    ]

    # -------------------------
    # COMBINE ALL BLOCKERS
    # -------------------------
    blocklist = (
        marketing_phrases
        + signature_phrases
        + footer_phrases
        + legal_phrases
    )

    if any(p in text_lower for p in blocklist):
        return False

    # -------------------------
    # REMOVE LINK-HEAVY CONTENT
    # -------------------------
    if "http://" in text_lower or "https://" in text_lower:
        return False

    # If more than 20% of text is URL characters
    if text.count("http") >= 1:
        return False

    # -------------------------
    # REMOVE NOISY EMAIL FOOTER PATTERNS
    # -------------------------
    footer_regex = [
        r"unsubscribe",
        r"^sincerely,?",
        r"^regards,?",
        r"this email has been",
    ]

    for pattern in footer_regex:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False

    # -------------------------
    # OTHERWISE — RELEVANT
    # -------------------------
    return True
