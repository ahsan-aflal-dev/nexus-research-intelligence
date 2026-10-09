from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.search import SearchResult, SemanticSearch


def build_search_index(
    file_path: str | Path,
) -> SemanticSearch:
    """Load a document, chunk it, embed it, and build a search index."""

    document = load_pdf(file_path)

    chunks = chunk_document(document)

    embedding_model = EmbeddingModel()

    search = SemanticSearch(embedding_model)
    search.add_chunks(chunks)

    return search


def search_document(
    file_path: str | Path,
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search a document for evidence relevant to a question."""

    search = build_search_index(file_path)

    return search.search(
        query=query,
        top_k=top_k,
    )