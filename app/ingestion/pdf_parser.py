import io
import pdfplumber


def parse_pdf(file_bytes: bytes) -> list[dict]:
    """Return a list of {page_num, text} dicts, one per non-empty page."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_num": page_num, "text": text})
    return pages
