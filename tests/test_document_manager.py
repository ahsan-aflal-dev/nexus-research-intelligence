
from unittest.mock import Mock

import pytest

from app.ingestion.manager import (
    ingest_document,
    list_documents,
    rebuild_index,
    remove_document,
)
from app.ingestion.models import Document, DocumentChunk, DocumentPage
from app.retrieval.index import VectorIndex


@pytest.fixture
def setup_manager(tmp_path, monkeypatch):
    import app.ingestion.manager as manager

    model = Mock()
    model.dimension = 2
    model.embed_texts.side_effect = (
        lambda texts: [[1.0, 0.0] for _ in texts]
    )
    model.embed_text.return_value = [1.0, 0.0]

    def fake_load_pdf(path):
        path = str(path)
        return Document(
            source=path,
            pages=[
                DocumentPage(
                    page_number=1,
                    text=f"Research content from {path}",
                )
            ],
        )

    def fake_chunk_document(document):
        return [
            DocumentChunk(
                chunk_id=f"{document.source}:p1:c1",
                text=document.pages[0].text,
                source=document.source,
                page_number=1,
            )
        ]

    monkeypatch.setattr(manager, "load_pdf", fake_load_pdf)
    monkeypatch.setattr(manager, "chunk_document", fake_chunk_document)

    index_directory = tmp_path / "index"

    return manager, model, index_directory


def test_ingest_adds_document_to_registry(tmp_path, setup_manager):
    _, model, index_directory = setup_manager
    pdf = tmp_path / "research.pdf"
    pdf.write_bytes(b"fake pdf")

    result = ingest_document(pdf, index_directory, model)

    assert result["status"] == "indexed"
    assert result["chunk_count"] == 1
    assert len(list_documents(index_directory)) == 1

    index = VectorIndex.load(index_directory, model)
    assert len(index.chunks) == 1
    assert index.chunks[0].source == str(pdf.resolve())


def test_ingesting_same_unchanged_document_skips_duplicates(
    tmp_path, setup_manager
):
    _, model, index_directory = setup_manager
    pdf = tmp_path / "research.pdf"
    pdf.write_bytes(b"fake pdf")

    first = ingest_document(pdf, index_directory, model)
    second = ingest_document(pdf, index_directory, model)

    assert first["status"] == "indexed"
    assert second["status"] == "skipped"

    index = VectorIndex.load(index_directory, model)
    assert len(index.chunks) == 1


def test_multiple_documents_share_one_index(tmp_path, setup_manager):
    _, model, index_directory = setup_manager
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    first_pdf.write_bytes(b"first")
    second_pdf.write_bytes(b"second")

    ingest_document(first_pdf, index_directory, model)
    ingest_document(second_pdf, index_directory, model)

    assert len(list_documents(index_directory)) == 2

    index = VectorIndex.load(index_directory, model)
    assert len(index.chunks) == 2
    assert {
        chunk.source for chunk in index.chunks
    } == {
        str(first_pdf.resolve()),
        str(second_pdf.resolve()),
    }


def test_remove_document_keeps_other_documents(
    tmp_path, setup_manager
):
    _, model, index_directory = setup_manager
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    first_pdf.write_bytes(b"first")
    second_pdf.write_bytes(b"second")

    ingest_document(first_pdf, index_directory, model)
    ingest_document(second_pdf, index_directory, model)

    result = remove_document(first_pdf, index_directory, model)

    assert result["status"] == "removed"
    documents = list_documents(index_directory)
    assert len(documents) == 1
    assert documents[0]["source_path"] == str(second_pdf.resolve())

    index = VectorIndex.load(index_directory, model)
    assert len(index.chunks) == 1
    assert index.chunks[0].source == str(second_pdf.resolve())


def test_remove_unknown_document_raises_error(
    tmp_path, setup_manager
):
    _, model, index_directory = setup_manager
    index_directory.mkdir()
    pdf = tmp_path / "unknown.pdf"
    pdf.write_bytes(b"unknown")

    with pytest.raises(ValueError, match="not registered"):
        remove_document(pdf, index_directory, model)


def test_rebuild_index_recreates_vectors(tmp_path, setup_manager):
    _, model, index_directory = setup_manager
    pdf = tmp_path / "research.pdf"
    pdf.write_bytes(b"fake pdf")

    ingest_document(pdf, index_directory, model)
    result = rebuild_index(index_directory, model)

    assert result["status"] == "rebuilt"
    assert result["document_count"] == 1
    assert result["chunk_count"] == 1

    index = VectorIndex.load(index_directory, model)
    assert len(index.chunks) == 1


def test_rebuild_fails_if_registered_pdf_is_missing(
    tmp_path, setup_manager
):
    _, model, index_directory = setup_manager
    pdf = tmp_path / "research.pdf"
    pdf.write_bytes(b"fake pdf")

    ingest_document(pdf, index_directory, model)
    pdf.unlink()

    with pytest.raises(FileNotFoundError, match="source PDF is missing"):
        rebuild_index(index_directory, model)
