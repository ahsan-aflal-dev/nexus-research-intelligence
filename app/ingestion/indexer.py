from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.index import VectorIndex


def index_document(
    file_path: str | Path,
    index_directory: str | Path,
) -> VectorIndex:
    """Process a document and persist its vector index."""

    document = load_pdf(file_path)

    chunks = chunk_document(document)

    embedding_model = EmbeddingModel()

    index = VectorIndex(embedding_model)

    index.add_chunks(chunks)

    index.save(index_directory)

    return index