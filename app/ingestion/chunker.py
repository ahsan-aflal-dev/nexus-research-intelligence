import re

from app.ingestion.models import Document, DocumentChunk


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using blank lines."""
    paragraphs = re.split(r"\n\s*\n", text.strip())

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 1,
) -> list[DocumentChunk]:
    """
    Split a document into paragraph-aware, sentence-based chunks.

    Paragraph boundaries are respected whenever possible.

    Args:
        document: Document to split.
        chunk_size: Target maximum number of characters per chunk.
        overlap: Number of sentences shared between neighboring chunks.

    Returns:
        A list of document chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    chunks = []

    for page in document.pages:
        paragraphs = split_into_paragraphs(page.text)

        current_sentences = []

        for paragraph in paragraphs:
            sentences = split_into_sentences(paragraph)

            for sentence in sentences:
                current_text = " ".join(current_sentences)

                if (
                    current_sentences
                    and len(current_text) + 1 + len(sentence) > chunk_size
                ):
                    chunk_number = len(chunks) + 1

                    chunks.append(
                        DocumentChunk(
                            chunk_id=(
                                f"{document.source}:"
                                f"p{page.page_number}:"
                                f"c{chunk_number}"
                            ),
                            text=current_text,
                            source=document.source,
                            page_number=page.page_number,
                        )
                    )

                    if overlap > 0:
                        current_sentences = current_sentences[-overlap:]
                    else:
                        current_sentences = []

                current_sentences.append(sentence)

        if current_sentences:
            chunk_number = len(chunks) + 1

            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{document.source}:"
                        f"p{page.page_number}:"
                        f"c{chunk_number}"
                    ),
                    text=" ".join(current_sentences),
                    source=document.source,
                    page_number=page.page_number,
                )
            )

    return chunks