import re
from app.core import config


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    # splits on sentence boundaries, packs sentences up to ~size chars,
    # and carries the tail of the previous chunk over so context isn't cut mid thought
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        # a single huge "sentence" (no punctuation) gets hard split
        while len(s) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(s[:size])
            s = s[size - overlap:] if overlap < size else s[size:]
        if current and len(current) + len(s) + 1 > size:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
            # start the new chunk on a word boundary
            current = current[current.find(" ") + 1:] if " " in current else current
        current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks
