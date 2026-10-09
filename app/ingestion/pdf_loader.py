from pathlib import Path

import pymupdf

from app.ingestion.models import Document, DocumentPage


def load_pdf(file_path: str | Path) -> Document:
    """
    Load a PDF and preserve its page boundaries.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A Document containing the source path and extracted pages.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    document = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        pages.append(
            DocumentPage(
                page_number=page_number,
                text=page.get_text(),
            )
        )

    document.close()

    return Document(
        source=str(path),
        pages=pages,
    )