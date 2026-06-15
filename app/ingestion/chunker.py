from app.core.config import settings


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """
    Split page texts into overlapping character-level chunks.

    Returns list of {chunk_id, page_num, text} dicts.
    Pages shorter than chunk_size are kept as a single chunk.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        text: str = page["text"]
        page_num: int = page["page_num"]

        if len(text) <= chunk_size:
            chunks.append({"chunk_id": chunk_id, "page_num": page_num, "text": text})
            chunk_id += 1
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {"chunk_id": chunk_id, "page_num": page_num, "text": chunk_text}
                )
                chunk_id += 1
            if end >= len(text):
                break
            start = end - overlap

    return chunks
