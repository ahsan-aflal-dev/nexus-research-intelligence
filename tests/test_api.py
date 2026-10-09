from fastapi.testclient import TestClient

import app.api as api


client = TestClient(api.app)


def test_root_returns_api_information():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "NEXUS API"
    assert response.json()["docs"] == "/docs"


def test_health_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_returns_answer_and_sources(monkeypatch):
    def fake_answer_question(question, index_directory, top_k):
        return (
            "Recall at K measures relevant results retrieved [S1].",
            [
                type(
                    "FakeResult",
                    (),
                    {
                        "chunk": type(
                            "FakeChunk",
                            (),
                            {
                                "source": "example.pdf",
                                "page_number": 1,
                                "chunk_id": "example.pdf:p1:c1",
                                "text": "Recall at K measures retrieval coverage.",
                            },
                        )(),
                        "score": 0.92,
                    },
                )()
            ],
        )

    monkeypatch.setattr(api, "answer_question", fake_answer_question)

    response = client.post(
        "/query",
        json={
            "question": "What does Recall at K measure?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "What does Recall at K measure?"
    assert "[S1]" in data["answer"]
    assert data["source_count"] == 1
    assert data["sources"][0]["citation"] == "S1"
    assert data["sources"][0]["source"] == "example.pdf"
    assert data["sources"][0]["page_number"] == 1
    assert data["sources"][0]["similarity"] == 0.92


def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={"question": "", "top_k": 3},
    )

    assert response.status_code == 422


def test_query_rejects_invalid_top_k():
    response = client.post(
        "/query",
        json={
            "question": "What is retrieval?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_document_ingestion_rejects_missing_file(monkeypatch):
    def fake_ingest_document(file_path, index_directory):
        raise FileNotFoundError("PDF file not found")

    monkeypatch.setattr(api, "ingest_document", fake_ingest_document)

    response = client.post(
        "/documents",
        json={"file_path": "missing.pdf"},
    )

    assert response.status_code == 404
    assert "PDF file not found" in response.json()["detail"]

def test_delete_managed_upload_deletes_pdf(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    pdf = upload_dir / "managed.pdf"
    pdf.write_bytes(b"%PDF-test")

    monkeypatch.setattr(api, "UPLOAD_DIRECTORY", upload_dir)
    monkeypatch.setattr(api, "DEFAULT_INDEX_DIRECTORY", tmp_path / "index")
    monkeypatch.setattr(
        api,
        "remove_document",
        lambda file_path, index_directory: {"status": "removed"},
    )

    response = client.request(
        "DELETE", "/documents", json={"file_path": str(pdf)}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "removed"
    assert response.json()["file_deleted"] is True
    assert not pdf.exists()


def test_delete_external_document_preserves_original(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    original_pdf = tmp_path / "research.pdf"
    original_pdf.write_bytes(b"%PDF-original")

    monkeypatch.setattr(api, "UPLOAD_DIRECTORY", upload_dir)
    monkeypatch.setattr(api, "DEFAULT_INDEX_DIRECTORY", tmp_path / "index")
    monkeypatch.setattr(
        api,
        "remove_document",
        lambda file_path, index_directory: {"status": "removed"},
    )

    response = client.request(
        "DELETE", "/documents", json={"file_path": str(original_pdf)}
    )

    assert response.status_code == 200
    assert response.json()["file_deleted"] is False
    assert original_pdf.exists()
    assert original_pdf.read_bytes() == b"%PDF-original"


def test_delete_unknown_document_returns_404_without_deleting_file(
    monkeypatch, tmp_path
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    pdf = upload_dir / "unknown.pdf"
    pdf.write_bytes(b"%PDF-unknown")

    monkeypatch.setattr(api, "UPLOAD_DIRECTORY", upload_dir)
    monkeypatch.setattr(api, "DEFAULT_INDEX_DIRECTORY", tmp_path / "index")

    def reject_unknown_document(file_path, index_directory):
        raise ValueError("Document is not registered")

    monkeypatch.setattr(api, "remove_document", reject_unknown_document)

    response = client.request(
        "DELETE", "/documents", json={"file_path": str(pdf)}
    )

    assert response.status_code == 404
    assert pdf.exists()


def test_documents_delete_route_is_registered_once():
    routes = [
        route
        for route in api.app.routes
        if getattr(route, "path", None) == "/documents"
        and "DELETE" in getattr(route, "methods", set())
    ]

    assert len(routes) == 1
