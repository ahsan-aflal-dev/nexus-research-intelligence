
import json
from datetime import datetime, timezone
from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.index import VectorIndex


def _registry_path(index_directory: str | Path) -> Path:
    return Path(index_directory) / "documents.json"


def _load_registry(index_directory: str | Path) -> dict:
    path = _registry_path(index_directory)

    if not path.exists():
        return {"version": 1, "documents": []}

    with path.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if not isinstance(registry.get("documents"), list):
        raise ValueError("Invalid document registry: documents must be a list.")

    return registry


def _save_registry(index_directory: str | Path, registry: dict) -> None:
    directory = Path(index_directory)
    directory.mkdir(parents=True, exist_ok=True)

    destination = _registry_path(directory)
    temporary = destination.with_suffix(".json.tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, ensure_ascii=False)

    temporary.replace(destination)


def _load_or_create_index(
    index_directory: str | Path,
    embedding_model: EmbeddingModel,
) -> VectorIndex:
    directory = Path(index_directory)

    embeddings_path = directory / "embeddings.npy"
    metadata_path = directory / "metadata.json"

    if embeddings_path.exists() and metadata_path.exists():
        return VectorIndex.load(directory, embedding_model)

    if embeddings_path.exists() or metadata_path.exists():
        raise ValueError(
            f"Incomplete index in {directory}. "
            "Restore both index files or rebuild the index."
        )

    return VectorIndex(embedding_model)


def ingest_document(
    file_path: str | Path,
    index_directory: str | Path = "data/index",
    embedding_model: EmbeddingModel | None = None,
) -> dict:
    """Add a PDF to the persistent index without replacing other documents."""

    source_path = Path(file_path).expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Document not found: {source_path}")

    if source_path.suffix.lower() != ".pdf":
        raise ValueError("NEXUS currently supports PDF files only.")

    directory = Path(index_directory)
    directory.mkdir(parents=True, exist_ok=True)

    stat = source_path.stat()
    source_key = str(source_path)
    registry = _load_registry(directory)

    existing = next(
        (
            item for item in registry["documents"]
            if item["source_path"] == source_key
        ),
        None,
    )

    unchanged = (
        existing is not None
        and existing["size_bytes"] == stat.st_size
        and existing["modified_ns"] == stat.st_mtime_ns
    )

    if unchanged:
        return {
            "status": "skipped",
            "source_path": source_key,
            "chunk_count": existing["chunk_count"],
            "message": "This unchanged document is already indexed.",
        }

    model = embedding_model or EmbeddingModel()
    index = _load_or_create_index(directory, model)

    if existing is None:
            legacy_sources = {
                chunk.source
                for chunk in index.chunks
                if Path(chunk.source).expanduser().resolve() == source_path
            }
            for legacy_source in legacy_sources:
                index.remove_source(legacy_source)

    # Remove old chunks first when a registered source has changed.
    if existing is not None:
        index.remove_source(source_key)
        registry["documents"].remove(existing)

    document = load_pdf(source_path)
    chunks = chunk_document(document)

    if not chunks:
        raise ValueError(f"No extractable text found in {source_path}")

    index.add_chunks(chunks)
    index.save(directory)

    registry["documents"].append(
        {
            "source_path": source_key,
            "file_name": source_path.name,
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
            "chunk_count": len(chunks),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    registry["documents"].sort(key=lambda item: item["source_path"])
    _save_registry(directory, registry)

    return {
        "status": "indexed",
        "source_path": source_key,
        "chunk_count": len(chunks),
        "message": "Document indexed successfully.",
    }


def list_documents(index_directory: str | Path = "data/index") -> list[dict]:
    """Return the registry of indexed documents."""

    return _load_registry(index_directory)["documents"]


def remove_document(
    file_path: str | Path,
    index_directory: str | Path = "data/index",
    embedding_model: EmbeddingModel | None = None,
) -> dict:
    """Remove a document and its chunks from the persistent index."""

    source_path = Path(file_path).expanduser().resolve()
    source_key = str(source_path)
    directory = Path(index_directory)
    registry = _load_registry(directory)

    existing = next(
        (
            item for item in registry["documents"]
            if item["source_path"] == source_key
        ),
        None,
    )

    if existing is None:
        raise ValueError(f"Document is not registered: {source_key}")

    model = embedding_model or EmbeddingModel()
    index = VectorIndex.load(directory, model)
    index.remove_source(source_key)

    registry["documents"].remove(existing)

    if index.chunks:
        index.save(directory)
    else:
        # Keep an empty, valid index so future ingestion can start cleanly.
        index.save(directory)

    _save_registry(directory, registry)

    return {
        "status": "removed",
        "source_path": source_key,
        "message": "Document removed from the index.",
    }


def rebuild_index(
    index_directory: str | Path = "data/index",
    embedding_model: EmbeddingModel | None = None,
) -> dict:
    """Rebuild the index from all registered source PDFs."""

    directory = Path(index_directory)
    registry = _load_registry(directory)
    model = embedding_model or EmbeddingModel()
    rebuilt = VectorIndex(model)
    updated_documents = []

    for item in registry["documents"]:
        source_path = Path(item["source_path"])

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Cannot rebuild: source PDF is missing: {source_path}"
            )

        stat = source_path.stat()
        document = load_pdf(source_path)
        chunks = chunk_document(document)

        if not chunks:
            raise ValueError(f"No extractable text found in {source_path}")

        rebuilt.add_chunks(chunks)

        updated_documents.append(
            {
                **item,
                "file_name": source_path.name,
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "chunk_count": len(chunks),
                "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    if not rebuilt.chunks:
        raise ValueError(
            "No registered documents to rebuild. Ingest a PDF first."
        )

    rebuilt.save(directory)
    registry["documents"] = updated_documents
    _save_registry(directory, registry)

    return {
        "status": "rebuilt",
        "document_count": len(updated_documents),
        "chunk_count": len(rebuilt.chunks),
        "message": "Index rebuilt successfully.",
    }
