def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()