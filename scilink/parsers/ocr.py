"""Vision-OCR fallback for scanned / image-only PDFs.

When a PDF page carries no usable embedded text layer, the page is rendered
to an image (PyMuPDF ``get_pixmap``) and transcribed with a vision LLM.

This module imports only ``fitz`` — the vision model is **injected** by the
caller (any object with a ``generate_content`` method), so ``scilink/parsers/``
stays free of dependencies on the rest of ``scilink``. When no model is
supplied, callers simply skip OCR and keep whatever sparse text the page had.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple


# A page whose embedded text layer yields fewer than this many characters is
# treated as scanned/sparse and becomes a candidate for vision-OCR.
SPARSE_PAGE_CHAR_THRESHOLD = 50

# Render resolution for OCR. 200 DPI is a reasonable balance of fine-print
# fidelity against image size / token cost.
DEFAULT_OCR_DPI = 200

# Cost guard: OCR at most this many pages per document by default.
MAX_OCR_PAGES = 50

# Provenance marker prefixed to vision-OCR'd pages in flat text output — a
# vision model can transcribe plausibly-but-wrong, so OCR'd content is flagged
# rather than silently merged with exact text-layer extraction.
OCR_MARKER = "[OCR — transcribed from a scanned page image; verify figures/numerics]"

OCR_PROMPT = (
    "This is an image of a single page from a scanned document. "
    "Transcribe ALL of its text content verbatim and in reading order. "
    "Render any tables as GitHub-flavored Markdown tables. "
    "Do not summarize, explain, translate, or add commentary — output only "
    "the transcribed page content. If the page has no readable text, output "
    "nothing."
)


def render_pdf_page(page: Any, dpi: int = DEFAULT_OCR_DPI) -> bytes:
    """Render a PyMuPDF page to JPEG image bytes at the given DPI."""
    import fitz

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("jpeg")


def transcribe_image(image_bytes: bytes, model: Any) -> str:
    """Transcribe a single page image with the injected vision model.

    Returns the transcription, or an empty string on failure / empty output —
    the caller is expected to fall back to whatever sparse text it already has.
    """
    try:
        response = model.generate_content(
            [OCR_PROMPT, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        text = getattr(response, "text", "") or ""
        return text.strip()
    except Exception as e:  # noqa: BLE001 - one bad page must not abort the doc
        logging.warning(f"Vision-OCR transcription failed: {e}")
        return ""


def ocr_pdf_pages(pdf_path: str,
                  page_numbers: List[int],
                  model: Any,
                  dpi: int = DEFAULT_OCR_DPI,
                  max_ocr_pages: int = MAX_OCR_PAGES
                  ) -> Tuple[Dict[int, str], List[int]]:
    """Render and transcribe the requested (1-indexed) pages of a PDF.

    At most ``max_ocr_pages`` pages are transcribed; any beyond the cap are
    returned as skipped. Per-page failures are isolated — a page that fails to
    render or transcribe is simply absent from the result dict.

    Returns ``(transcriptions, skipped)`` where ``transcriptions`` maps
    1-indexed page number -> transcribed text, and ``skipped`` is the list of
    1-indexed page numbers not attempted because of the cap.
    """
    import fitz

    ordered = sorted(page_numbers)
    to_ocr = ordered[:max_ocr_pages]
    skipped = ordered[max_ocr_pages:]

    if skipped:
        print(f"  - ⚠️  Vision-OCR cap reached ({max_ocr_pages} pages); "
              f"{len(skipped)} scanned page(s) left un-transcribed.")

    transcriptions: Dict[int, str] = {}
    if not to_ocr:
        return transcriptions, skipped

    print(f"  - 👁️  Vision-OCR: transcribing {len(to_ocr)} scanned page(s) "
          f"at {dpi} DPI...")
    doc = fitz.open(pdf_path)
    try:
        for page_num in to_ocr:
            try:
                image_bytes = render_pdf_page(doc[page_num - 1], dpi=dpi)
            except Exception as e:  # noqa: BLE001 - isolate per-page render errors
                logging.warning(f"Vision-OCR: could not render page {page_num}: {e}")
                continue
            text = transcribe_image(image_bytes, model)
            if text:
                transcriptions[page_num] = text
            else:
                print(f"    - ⚠️  Page {page_num}: no text transcribed.")
    finally:
        doc.close()

    print(f"  - ✓ Vision-OCR transcribed {len(transcriptions)}/{len(to_ocr)} page(s).")
    return transcriptions, skipped
