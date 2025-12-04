import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x0c", " ")  # common OCR junk
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII noise
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip()


def mask_pii(text: str) -> str:
    if not text:
        return ""

    # Mask emails
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "***@***",
        text,
    )

    # Mask phone numbers
    text = re.sub(
        r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
        "XXX-XXX-XXXX",
        text,
    )

    return text
